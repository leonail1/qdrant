use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use common::types::PointOffsetType;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use super::shader_builder::ShaderBuilderParameters;
use crate::common::check_stopped;
use crate::common::operation_error::{OperationError, OperationResult};
use crate::index::hnsw_index::gpu::GPU_TIMEOUT;
use crate::index::hnsw_index::graph_layers::GraphLayersBase;
use crate::index::hnsw_index::graph_layers_builder::GraphLayersBuilder;

/// Size of transfer buffer for links.
const LINKS_TRANSFER_BUFFER_SIZE: usize = 32 * 1024 * 1024;

#[derive(FromBytes, Immutable, IntoBytes, KnownLayout)]
#[repr(C)]
struct GpuLinksParamsBuffer {
    m: u32,
    links_capacity: u32,
}

/// GPU resources for links.
pub struct GpuLinks {
    m: usize,
    links_capacity: usize,
    max_patched_points: usize,
    device: Arc<gpu::Device>,
    links_buffer: Arc<gpu::Buffer>,
    offsets_buffer: Option<Arc<gpu::Buffer>>,
    params_buffer: Arc<gpu::Buffer>,
    patch_buffer: Option<Arc<gpu::Buffer>>,
    patched_points: Vec<(PointOffsetType, usize)>,
    descriptor_set_layout: Arc<gpu::DescriptorSetLayout>,
    descriptor_set: Arc<gpu::DescriptorSet>,
}

impl ShaderBuilderParameters for GpuLinks {
    fn shader_includes(&self) -> HashMap<String, String> {
        HashMap::from([(
            "links.comp".to_string(),
            include_str!("shaders/links.comp").to_string(),
        )])
    }

    fn shader_defines(&self) -> HashMap<String, Option<String>> {
        let mut defines = HashMap::new();
        if self.offsets_buffer.is_some() {
            defines.insert("LINKS_COMPACT".to_owned(), None);
        } else {
            defines.insert(
                "LINKS_CAPACITY".to_owned(),
                Some(self.links_capacity.to_string()),
            );
        }
        defines
    }
}

impl GpuLinks {
    pub fn max_links_on_level(
        graph: &impl GraphLayersBase,
        points_count: usize,
        level: usize,
    ) -> usize {
        (0..points_count as PointOffsetType)
            .map(|point_id| {
                let mut count = 0;
                graph.for_each_link(point_id, level, |_| count += 1);
                count
            })
            .max()
            .unwrap_or(0)
    }

    pub fn new(
        device: Arc<gpu::Device>,
        m: usize,
        links_capacity: usize,
        points_count: usize,
    ) -> gpu::GpuResult<Self> {
        let links_buffer = gpu::Buffer::new(
            device.clone(),
            "Links buffer",
            gpu::BufferType::Storage,
            points_count * (links_capacity + 1) * std::mem::size_of::<PointOffsetType>(),
        )?;
        let params_buffer = gpu::Buffer::new(
            device.clone(),
            "Links params buffer",
            gpu::BufferType::Uniform,
            std::mem::size_of::<GpuLinksParamsBuffer>(),
        )?;

        let max_patched_points = LINKS_TRANSFER_BUFFER_SIZE
            / ((links_capacity + 1) * std::mem::size_of::<PointOffsetType>());
        let links_patch_capacity =
            max_patched_points * (links_capacity + 1) * std::mem::size_of::<PointOffsetType>();
        let patch_buffer = gpu::Buffer::new(
            device.clone(),
            "Links patch buffer",
            gpu::BufferType::CpuToGpu,
            links_patch_capacity + std::mem::size_of::<GpuLinksParamsBuffer>(),
        )?;

        let params = GpuLinksParamsBuffer {
            m: m as u32,
            links_capacity: links_capacity as u32,
        };
        patch_buffer.upload(&params, 0)?;

        let mut upload_context = gpu::Context::new(device.clone())?;
        upload_context.copy_gpu_buffer(
            patch_buffer.clone(),
            params_buffer.clone(),
            0,
            0,
            std::mem::size_of::<GpuLinksParamsBuffer>(),
        )?;
        upload_context.clear_buffer(links_buffer.clone())?;
        upload_context.run()?;
        upload_context.wait_finish(GPU_TIMEOUT)?;

        let descriptor_set_layout = gpu::DescriptorSetLayout::builder()
            .add_uniform_buffer(0)
            .add_storage_buffer(1)
            .build(device.clone())?;

        let descriptor_set = gpu::DescriptorSet::builder(descriptor_set_layout.clone())
            .add_uniform_buffer(0, params_buffer.clone())
            .add_storage_buffer(1, links_buffer.clone())
            .build()?;

        Ok(Self {
            m,
            links_capacity,
            max_patched_points,
            device,
            links_buffer,
            offsets_buffer: None,
            params_buffer,
            patch_buffer: Some(patch_buffer),
            patched_points: vec![],
            descriptor_set_layout,
            descriptor_set,
        })
    }

    /// Build an immutable CSR projection for filtered graph search. GPU HNSW
    /// construction continues to use [`Self::new`]'s mutable fixed-width
    /// representation.
    pub fn new_compact(
        device: Arc<gpu::Device>,
        graph: &impl GraphLayersBase,
        points_count: usize,
        stopped: &AtomicBool,
    ) -> OperationResult<Self> {
        let mut offsets = Vec::with_capacity(points_count + 1);
        offsets.push(0u32);
        let mut max_links = 0usize;
        for point_id in 0..points_count {
            if point_id % 4_096 == 0 {
                check_stopped(stopped)?;
            }
            let mut links_count = 0usize;
            graph.for_each_link(point_id as PointOffsetType, 0, |_| links_count += 1);
            max_links = max_links.max(links_count);
            let next_offset = (*offsets.last().unwrap() as usize)
                .checked_add(links_count)
                .and_then(|offset| u32::try_from(offset).ok())
                .ok_or_else(|| {
                    OperationError::service_error(
                        "GPU compact graph has more links than u32 offsets can address",
                    )
                })?;
            offsets.push(next_offset);
        }
        let total_links = usize::try_from(*offsets.last().unwrap()).unwrap();
        if max_links == 0 || total_links == 0 {
            return Err(OperationError::service_error(
                "GPU filtered graph has no level-0 links",
            ));
        }

        let offsets_bytes = std::mem::size_of_val(offsets.as_slice());
        let links_bytes = total_links
            .checked_mul(std::mem::size_of::<PointOffsetType>())
            .ok_or_else(|| OperationError::service_error("GPU compact links size overflow"))?;
        let dense_bytes = points_count
            .checked_mul(max_links + 1)
            .and_then(|count| count.checked_mul(std::mem::size_of::<PointOffsetType>()))
            .ok_or_else(|| OperationError::service_error("GPU dense links size overflow"))?;

        let links_buffer = gpu::Buffer::new(
            device.clone(),
            "Compact links neighbors",
            gpu::BufferType::Storage,
            links_bytes,
        )?;
        let offsets_buffer = gpu::Buffer::new(
            device.clone(),
            "Compact links offsets",
            gpu::BufferType::Storage,
            offsets_bytes,
        )?;
        let params_buffer = gpu::Buffer::new(
            device.clone(),
            "Links params buffer",
            gpu::BufferType::Uniform,
            std::mem::size_of::<GpuLinksParamsBuffer>(),
        )?;
        let upload_buffer = gpu::Buffer::new(
            device.clone(),
            "Compact links upload",
            gpu::BufferType::CpuToGpu,
            LINKS_TRANSFER_BUFFER_SIZE,
        )?;
        let mut upload_context = gpu::Context::new(device.clone())?;

        let params = GpuLinksParamsBuffer {
            m: graph.get_m(0) as u32,
            links_capacity: 0,
        };
        upload_buffer.upload(&params, 0)?;
        upload_context.copy_gpu_buffer(
            upload_buffer.clone(),
            params_buffer.clone(),
            0,
            0,
            std::mem::size_of::<GpuLinksParamsBuffer>(),
        )?;
        upload_context.run()?;
        upload_context.wait_finish(GPU_TIMEOUT)?;

        let transfer_elements = LINKS_TRANSFER_BUFFER_SIZE / std::mem::size_of::<u32>();
        for (chunk_index, chunk) in offsets.chunks(transfer_elements).enumerate() {
            upload_u32_slice(
                &mut upload_context,
                &upload_buffer,
                &offsets_buffer,
                chunk,
                chunk_index * transfer_elements,
            )?;
        }

        let mut links_chunk = Vec::with_capacity(transfer_elements);
        let mut uploaded_links = 0usize;
        for point_id in 0..points_count {
            if point_id % 4_096 == 0 {
                check_stopped(stopped)?;
            }
            let expected_links = offsets[point_id + 1] as usize - offsets[point_id] as usize;
            if expected_links > transfer_elements {
                return Err(OperationError::service_error(
                    "A GPU compact adjacency list exceeds the staging buffer",
                ));
            }
            if !links_chunk.is_empty()
                && links_chunk.len().saturating_add(expected_links) > transfer_elements
            {
                upload_u32_slice(
                    &mut upload_context,
                    &upload_buffer,
                    &links_buffer,
                    &links_chunk,
                    uploaded_links,
                )?;
                uploaded_links += links_chunk.len();
                links_chunk.clear();
            }
            let before = links_chunk.len();
            graph.for_each_link(point_id as PointOffsetType, 0, |link| {
                links_chunk.push(link)
            });
            if links_chunk.len() - before != expected_links {
                return Err(OperationError::service_error(
                    "GPU graph changed while building the compact projection",
                ));
            }
        }
        if !links_chunk.is_empty() {
            upload_u32_slice(
                &mut upload_context,
                &upload_buffer,
                &links_buffer,
                &links_chunk,
                uploaded_links,
            )?;
            uploaded_links += links_chunk.len();
        }
        if uploaded_links != total_links {
            return Err(OperationError::service_error(
                "GPU compact graph upload did not cover every link",
            ));
        }

        let descriptor_set_layout = gpu::DescriptorSetLayout::builder()
            .add_uniform_buffer(0)
            .add_storage_buffer(1)
            .add_storage_buffer(2)
            .build(device.clone())?;
        let descriptor_set = gpu::DescriptorSet::builder(descriptor_set_layout.clone())
            .add_uniform_buffer(0, params_buffer.clone())
            .add_storage_buffer(1, links_buffer.clone())
            .add_storage_buffer(2, offsets_buffer.clone())
            .build()?;

        log::info!(
            "GPU compact links projection: points={points_count}, links={total_links}, max_degree={max_links}, compact_bytes={}, dense_bytes={dense_bytes}",
            links_bytes + offsets_bytes,
        );
        Ok(Self {
            m: graph.get_m(0),
            links_capacity: 0,
            max_patched_points: 0,
            device,
            links_buffer,
            offsets_buffer: Some(offsets_buffer),
            params_buffer,
            patch_buffer: None,
            patched_points: Vec::new(),
            descriptor_set_layout,
            descriptor_set,
        })
    }

    pub fn update_params(
        &mut self,
        gpu_context: &mut gpu::Context,
        m: usize,
    ) -> OperationResult<()> {
        self.m = m;

        let params = GpuLinksParamsBuffer {
            m: m as u32,
            links_capacity: self.links_capacity as u32,
        };
        let links_patch_capacity = self.max_patched_points
            * (self.links_capacity + 1)
            * std::mem::size_of::<PointOffsetType>();
        let patch_buffer = self.patch_buffer()?;
        patch_buffer.upload(&params, links_patch_capacity)?;

        gpu_context.copy_gpu_buffer(
            patch_buffer,
            self.params_buffer.clone(),
            links_patch_capacity,
            0,
            std::mem::size_of::<GpuLinksParamsBuffer>(),
        )?;
        gpu_context.run()?;
        gpu_context.wait_finish(GPU_TIMEOUT)?;
        Ok(())
    }

    /// Release the CPU-to-GPU transfer buffer after an immutable graph has
    /// been uploaded. Query shaders retain only the resident links and params
    /// buffers, so the staging allocation is not needed by the search cache.
    pub fn release_patch_buffer(&mut self) -> OperationResult<()> {
        if !self.patched_points.is_empty() {
            return Err(OperationError::service_error(
                "Cannot release GPU links patch buffer with pending patches",
            ));
        }
        self.patch_buffer = None;
        Ok(())
    }

    pub fn clear(&mut self, gpu_context: &mut gpu::Context) -> OperationResult<()> {
        if !self.patched_points.is_empty() {
            self.patched_points.clear();
        }
        gpu_context.clear_buffer(self.links_buffer.clone())?;
        gpu_context.run()?;
        gpu_context.wait_finish(GPU_TIMEOUT)?;
        Ok(())
    }

    pub fn links_buffer(&self) -> Arc<gpu::Buffer> {
        self.links_buffer.clone()
    }

    pub fn upload_links(
        &mut self,
        level: usize,
        graph_layers_builder: &GraphLayersBuilder,
        gpu_context: &mut gpu::Context,
        stopped: &AtomicBool,
    ) -> OperationResult<()> {
        self.update_params(gpu_context, graph_layers_builder.get_m(level))?;
        self.clear(gpu_context)?;

        let timer = std::time::Instant::now();
        let points: Vec<_> = (0..graph_layers_builder.links_layers().len())
            .filter(|&point_id| {
                graph_layers_builder.get_point_level(point_id as PointOffsetType) >= level
            })
            .filter(|&point_id| {
                !graph_layers_builder.links_layers()[point_id][level]
                    .read()
                    .links()
                    .is_empty()
            })
            .collect();

        for points_slice in points.chunks(self.max_patched_points) {
            check_stopped(stopped)?;

            for &point_id in points_slice {
                let links = graph_layers_builder.links_layers()[point_id][level].read();
                self.set_links(point_id as PointOffsetType, links.links())?;
            }
            self.apply_gpu_patches(gpu_context)?;
            gpu_context.run()?;
            gpu_context.wait_finish(GPU_TIMEOUT)?;
        }

        log::trace!("Upload links on level {level} time: {:?}", timer.elapsed());
        Ok(())
    }

    /// Upload an immutable, already-built Qdrant graph layer.
    ///
    /// Unlike [`Self::upload_links`], this consumes [`GraphLayersBase`] rather
    /// than the mutable construction graph. The GPU buffer is only a resident
    /// projection of Qdrant's native graph and is never persisted separately.
    pub fn upload_graph_layer(
        &mut self,
        level: usize,
        graph: &impl GraphLayersBase,
        points_count: usize,
        gpu_context: &mut gpu::Context,
        stopped: &AtomicBool,
    ) -> OperationResult<()> {
        self.update_params(gpu_context, graph.get_m(level))?;
        self.clear(gpu_context)?;

        for start in (0..points_count).step_by(self.max_patched_points) {
            check_stopped(stopped)?;
            let end = (start + self.max_patched_points).min(points_count);
            for point_id in start..end {
                let mut links = Vec::new();
                graph.for_each_link(point_id as PointOffsetType, level, |link| links.push(link));
                if links.len() > self.links_capacity {
                    return Err(OperationError::service_error(format!(
                        "GPU graph link capacity exceeded at point {point_id}: {} > {}",
                        links.len(),
                        self.links_capacity,
                    )));
                }
                if !links.is_empty() {
                    self.set_links(point_id as PointOffsetType, &links)?;
                }
            }
            self.apply_gpu_patches(gpu_context)?;
            gpu_context.run()?;
            gpu_context.wait_finish(GPU_TIMEOUT)?;
        }
        Ok(())
    }

    pub fn download_links(
        &mut self,
        level: usize,
        graph_layers_builder: &GraphLayersBuilder,
        gpu_context: &mut gpu::Context,
        stopped: &AtomicBool,
    ) -> OperationResult<()> {
        let timer = std::time::Instant::now();
        // Collect bad links to check if there are any errors in the links.
        let mut bad_links = Vec::new();

        let links_patch_capacity = self.max_patched_points
            * (self.links_capacity + 1)
            * std::mem::size_of::<PointOffsetType>();
        let download_buffer = gpu::Buffer::new(
            self.device.clone(),
            "Download links staging buffer",
            gpu::BufferType::GpuToCpu,
            links_patch_capacity,
        )?;

        let points = (0..graph_layers_builder.links_layers().len() as PointOffsetType)
            .filter(|&point_id| graph_layers_builder.get_point_level(point_id) >= level)
            .collect::<Vec<_>>();

        for chunk_index in 0..points.len().div_ceil(self.max_patched_points) {
            check_stopped(stopped)?;

            let start = chunk_index * self.max_patched_points;
            let end = (start + self.max_patched_points).min(points.len());
            let chunk_size = end - start;
            for (i, &point_id) in points[start..end].iter().enumerate() {
                let links_size = (self.links_capacity + 1) * std::mem::size_of::<PointOffsetType>();
                gpu_context.copy_gpu_buffer(
                    self.links_buffer.clone(),
                    download_buffer.clone(),
                    point_id as usize * links_size,
                    i * links_size,
                    links_size,
                )?;
            }
            gpu_context.run()?;
            gpu_context.wait_finish(GPU_TIMEOUT)?;

            let links = download_buffer
                .download_vec::<PointOffsetType>(0, chunk_size * (self.links_capacity + 1))?;

            for (index, chunk) in links.chunks(self.links_capacity + 1).enumerate() {
                let point_id = points[start + index] as usize;
                let links_count = chunk[0] as usize;
                let links = &chunk[1..=links_count];
                let mut dst = graph_layers_builder.links_layers()[point_id][level].write();
                dst.fill_from(links.iter().copied().filter(|&other_point_id| {
                    let is_correct_link =
                        level < graph_layers_builder.links_layers()[other_point_id as usize].len();
                    if !is_correct_link {
                        bad_links.push(other_point_id);
                    }
                    is_correct_link
                }));
            }
        }

        if !bad_links.is_empty() {
            log::warn!(
                "Incorrect links on level {} were found. Amount of incorrect links: {}, zeroes: {}",
                level,
                bad_links.len(),
                bad_links.iter().filter(|&&point_id| point_id == 0).count()
            );
        }

        log::trace!(
            "Download links for level {} in time {:?}",
            level,
            timer.elapsed()
        );
        Ok(())
    }

    pub fn descriptor_set_layout(&self) -> Arc<gpu::DescriptorSetLayout> {
        self.descriptor_set_layout.clone()
    }

    pub fn descriptor_set(&self) -> Arc<gpu::DescriptorSet> {
        self.descriptor_set.clone()
    }

    fn apply_gpu_patches(&mut self, gpu_context: &mut gpu::Context) -> OperationResult<()> {
        let patch_buffer = self.patch_buffer()?;
        for (i, &(patched_point_id, patched_links_count)) in self.patched_points.iter().enumerate()
        {
            let patch_start_index =
                i * (self.links_capacity + 1) * std::mem::size_of::<PointOffsetType>();
            let patch_size = (patched_links_count + 1) * std::mem::size_of::<PointOffsetType>();
            let links_start_index = patched_point_id as usize
                * (self.links_capacity + 1)
                * std::mem::size_of::<PointOffsetType>();
            gpu_context.copy_gpu_buffer(
                patch_buffer.clone(),
                self.links_buffer.clone(),
                patch_start_index,
                links_start_index,
                patch_size,
            )?;
        }
        self.patched_points.clear();
        Ok(())
    }

    fn set_links(
        &mut self,
        point_id: PointOffsetType,
        links: &[PointOffsetType],
    ) -> OperationResult<()> {
        if self.patched_points.len() >= self.max_patched_points {
            return Err(OperationError::service_error("Gpu links patches are full"));
        }

        let mut patch_start_index = self.patched_points.len()
            * (self.links_capacity + 1)
            * std::mem::size_of::<PointOffsetType>();
        let patch_buffer = self.patch_buffer()?;
        patch_buffer.upload(&(links.len() as u32), patch_start_index)?;
        patch_start_index += std::mem::size_of::<PointOffsetType>();
        patch_buffer.upload(links, patch_start_index)?;
        self.patched_points.push((point_id, links.len()));

        Ok(())
    }

    fn patch_buffer(&self) -> OperationResult<Arc<gpu::Buffer>> {
        self.patch_buffer.clone().ok_or_else(|| {
            OperationError::service_error("GPU links patch buffer has already been released")
        })
    }
}

fn upload_u32_slice(
    context: &mut gpu::Context,
    staging: &Arc<gpu::Buffer>,
    target: &Arc<gpu::Buffer>,
    values: &[u32],
    target_element_offset: usize,
) -> OperationResult<()> {
    let bytes = std::mem::size_of_val(values);
    if bytes > staging.size() {
        return Err(OperationError::service_error(
            "GPU compact links upload chunk exceeds the staging buffer",
        ));
    }
    let target_byte_offset = target_element_offset
        .checked_mul(std::mem::size_of::<u32>())
        .filter(|offset| offset.saturating_add(bytes) <= target.size())
        .ok_or_else(|| OperationError::service_error("GPU compact links upload is out of range"))?;
    staging.upload(values, 0)?;
    context.copy_gpu_buffer(
        staging.clone(),
        target.clone(),
        0,
        target_byte_offset,
        bytes,
    )?;
    context.run()?;
    context.wait_finish(GPU_TIMEOUT)?;
    Ok(())
}
