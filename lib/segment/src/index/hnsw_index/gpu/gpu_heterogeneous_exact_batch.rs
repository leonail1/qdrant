use std::sync::Arc;

use common::top_k::TopK;
use common::types::{PointOffsetType, ScoredPointOffset};

use super::GPU_TIMEOUT;
use super::gpu_vector_storage::GpuVectorStorage;
use super::shader_builder::ShaderBuilder;
use crate::common::operation_error::{OperationError, OperationResult};

const TILE_CANDIDATES: usize = 128;
const BLOCK_TOPK: usize = 16;
const METADATA_WORDS: usize = 4;
const MAX_DEVICE_BUFFER_BYTES: usize = 64 * 1024 * 1024;

pub struct HeterogeneousExactRequest<'a> {
    pub query: &'a [f32],
    pub candidates: &'a [PointOffsetType],
}

#[derive(Debug, Default)]
pub struct HeterogeneousExactTimings {
    pub total_candidates: usize,
    pub prepare_ns: u64,
    pub command_record_ns: u64,
    pub queue_submit_ns: u64,
    pub fence_wait_ns: u64,
    pub download_ns: u64,
    pub cpu_topk_ns: u64,
}

/// A single bounded broker context for batches whose queries use different
/// payload predicates. Candidate IDs are packed for each submission so the
/// GPU does not need a fixed-size per-predicate atlas. This keeps the device
/// memory ceiling explicit; a resident atlas can replace the H2D step later
/// if the measured breakdown proves that it is worthwhile.
pub struct GpuHeterogeneousExactBatchContext {
    context: gpu::Context,
    pipeline: Arc<gpu::Pipeline>,
    descriptor_set_layout: Arc<gpu::DescriptorSetLayout>,
    vector_storage: Arc<GpuVectorStorage>,
    candidate_buffer: Arc<gpu::Buffer>,
    candidate_staging: Arc<gpu::Buffer>,
    query_buffer: Arc<gpu::Buffer>,
    query_staging: Arc<gpu::Buffer>,
    metadata_buffer: Arc<gpu::Buffer>,
    metadata_staging: Arc<gpu::Buffer>,
    result_buffer: Arc<gpu::Buffer>,
    result_staging: Arc<gpu::Buffer>,
    zero_visibility: Arc<gpu::Buffer>,
    candidate_capacity: usize,
    batch_capacity: usize,
    query_capacity: usize,
    resident_device_bytes: usize,
}

impl std::fmt::Debug for GpuHeterogeneousExactBatchContext {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GpuHeterogeneousExactBatchContext")
            .field("candidate_capacity", &self.candidate_capacity)
            .field("batch_capacity", &self.batch_capacity)
            .field("resident_device_bytes", &self.resident_device_bytes)
            .finish_non_exhaustive()
    }
}

impl GpuHeterogeneousExactBatchContext {
    pub fn new(
        vector_storage: Arc<GpuVectorStorage>,
        candidate_capacity: usize,
        batch_capacity: usize,
        queue_index: usize,
    ) -> OperationResult<Self> {
        if candidate_capacity == 0 || batch_capacity < 2 {
            return Err(OperationError::service_error(
                "GPU heterogeneous exact batch capacities are invalid",
            ));
        }
        let device = vector_storage.device();
        let packed_candidate_capacity = candidate_capacity
            .checked_mul(batch_capacity)
            .ok_or_else(|| OperationError::service_error("GPU candidate capacity overflow"))?;
        let candidate_bytes = packed_candidate_capacity
            .checked_mul(std::mem::size_of::<PointOffsetType>())
            .ok_or_else(|| OperationError::service_error("GPU candidate byte size overflow"))?;
        let query_capacity = vector_storage.vector_capacity();
        let query_bytes = query_capacity
            .checked_mul(batch_capacity)
            .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| OperationError::service_error("GPU query byte size overflow"))?;
        let metadata_bytes = batch_capacity * METADATA_WORDS * std::mem::size_of::<u32>();
        let result_capacity = candidate_capacity
            .div_ceil(TILE_CANDIDATES)
            .checked_mul(BLOCK_TOPK)
            .and_then(|value| value.checked_mul(batch_capacity))
            .ok_or_else(|| OperationError::service_error("GPU result capacity overflow"))?;
        let result_bytes = result_capacity
            .checked_mul(std::mem::size_of::<ScoredPointOffset>())
            .ok_or_else(|| OperationError::service_error("GPU result byte size overflow"))?;
        let visibility_bytes = vector_storage
            .num_vectors()
            .div_ceil(u32::BITS as usize)
            .max(1)
            * std::mem::size_of::<u32>();
        let resident_device_bytes = candidate_bytes
            .saturating_add(query_bytes)
            .saturating_add(metadata_bytes)
            .saturating_add(result_bytes)
            .saturating_add(visibility_bytes);
        if resident_device_bytes > MAX_DEVICE_BUFFER_BYTES {
            return Err(OperationError::service_error(format!(
                "GPU heterogeneous exact batch would reserve {resident_device_bytes} bytes, above the {MAX_DEVICE_BUFFER_BYTES}-byte memory gate",
            )));
        }

        let descriptor_set_layout = gpu::DescriptorSetLayout::builder()
            .add_storage_buffer(0)
            .add_storage_buffer(1)
            .add_storage_buffer(2)
            .add_storage_buffer(3)
            .add_storage_buffer(4)
            .build(device.clone())?;
        let shader = ShaderBuilder::new(device.clone())
            .with_shader_code(include_str!(
                "shaders/run_exact_filtered_heterogeneous_block_topk.comp"
            ))
            .with_parameters(vector_storage.as_ref())
            .build("run_exact_filtered_heterogeneous_block_topk.comp")?;
        let pipeline = gpu::Pipeline::builder()
            .add_descriptor_set_layout(0, descriptor_set_layout.clone())
            .add_descriptor_set_layout(1, vector_storage.descriptor_set_layout())
            .add_shader(shader)
            .build(device.clone())?;

        let candidate_buffer = gpu::Buffer::new(
            device.clone(),
            "Heterogeneous exact candidate IDs",
            gpu::BufferType::Storage,
            candidate_bytes,
        )?;
        let candidate_staging = gpu::Buffer::new(
            device.clone(),
            "Heterogeneous exact candidate upload",
            gpu::BufferType::CpuToGpu,
            candidate_bytes,
        )?;
        let query_buffer = gpu::Buffer::new(
            device.clone(),
            "Heterogeneous exact queries",
            gpu::BufferType::Storage,
            query_bytes,
        )?;
        let query_staging = gpu::Buffer::new(
            device.clone(),
            "Heterogeneous exact query upload",
            gpu::BufferType::CpuToGpu,
            query_bytes,
        )?;
        let metadata_buffer = gpu::Buffer::new(
            device.clone(),
            "Heterogeneous exact metadata",
            gpu::BufferType::Storage,
            metadata_bytes,
        )?;
        let metadata_staging = gpu::Buffer::new(
            device.clone(),
            "Heterogeneous exact metadata upload",
            gpu::BufferType::CpuToGpu,
            metadata_bytes,
        )?;
        let result_buffer = gpu::Buffer::new(
            device.clone(),
            "Heterogeneous exact block results",
            gpu::BufferType::Storage,
            result_bytes,
        )?;
        let result_staging = gpu::Buffer::new(
            device.clone(),
            "Heterogeneous exact result download",
            gpu::BufferType::GpuToCpu,
            result_bytes,
        )?;
        let zero_visibility = gpu::Buffer::new(
            device.clone(),
            "Heterogeneous exact zero visibility",
            gpu::BufferType::Storage,
            visibility_bytes,
        )?;
        let mut context = gpu::Context::new_with_queue_index(device, queue_index)?;
        context.clear_buffer(zero_visibility.clone())?;
        context.run()?;
        context.wait_finish(GPU_TIMEOUT)?;

        log::info!(
            "Initialized heterogeneous GPU exact batch: max_candidates={}, max_queries={}, resident_device_bytes={}",
            candidate_capacity,
            batch_capacity,
            resident_device_bytes,
        );
        Ok(Self {
            context,
            pipeline,
            descriptor_set_layout,
            vector_storage,
            candidate_buffer,
            candidate_staging,
            query_buffer,
            query_staging,
            metadata_buffer,
            metadata_staging,
            result_buffer,
            result_staging,
            zero_visibility,
            candidate_capacity,
            batch_capacity,
            query_capacity,
            resident_device_bytes,
        })
    }

    pub fn resident_device_bytes(&self) -> usize {
        self.resident_device_bytes
    }

    pub fn search(
        &mut self,
        requests: &[HeterogeneousExactRequest<'_>],
        top: usize,
        visibility_buffer: Option<Arc<gpu::Buffer>>,
    ) -> OperationResult<(Vec<Vec<ScoredPointOffset>>, HeterogeneousExactTimings)> {
        if requests.is_empty() || requests.len() > self.batch_capacity {
            return Err(OperationError::service_error(
                "GPU heterogeneous exact query batch capacity exceeded",
            ));
        }
        if top == 0 || top > BLOCK_TOPK {
            return Err(OperationError::service_error(
                "GPU heterogeneous exact top-k is unsupported",
            ));
        }
        if requests.iter().any(|request| {
            request.query.len() != self.vector_storage.dim()
                || request.candidates.is_empty()
                || request.candidates.len() > self.candidate_capacity
        }) {
            return Err(OperationError::service_error(
                "GPU heterogeneous exact request is invalid",
            ));
        }

        let prepare_started = std::time::Instant::now();
        let total_candidates = requests
            .iter()
            .map(|request| request.candidates.len())
            .sum::<usize>();
        if total_candidates > self.candidate_capacity * self.batch_capacity {
            return Err(OperationError::service_error(
                "GPU heterogeneous packed candidate capacity exceeded",
            ));
        }
        let mut candidates = Vec::with_capacity(total_candidates);
        let mut queries = vec![0.0f32; self.query_capacity * requests.len()];
        let mut metadata = Vec::with_capacity(METADATA_WORDS * requests.len());
        let mut output_offsets = Vec::with_capacity(requests.len() + 1);
        output_offsets.push(0usize);
        let mut max_block_count = 0usize;
        for (query_index, request) in requests.iter().enumerate() {
            let candidate_offset = candidates.len();
            candidates.extend_from_slice(request.candidates);
            let query_offset = query_index * self.query_capacity;
            queries[query_offset..query_offset + request.query.len()]
                .copy_from_slice(request.query);
            let block_count = request.candidates.len().div_ceil(TILE_CANDIDATES);
            max_block_count = max_block_count.max(block_count);
            let output_offset = *output_offsets.last().unwrap();
            let next_output_offset = output_offset + block_count * BLOCK_TOPK;
            output_offsets.push(next_output_offset);
            metadata.extend_from_slice(&[
                u32::try_from(candidate_offset).map_err(|_| {
                    OperationError::service_error("GPU candidate offset exceeds u32")
                })?,
                u32::try_from(request.candidates.len()).map_err(|_| {
                    OperationError::service_error("GPU candidate count exceeds u32")
                })?,
                u32::try_from(output_offset)
                    .map_err(|_| OperationError::service_error("GPU result offset exceeds u32"))?,
                0,
            ]);
        }
        self.candidate_staging.upload(candidates.as_slice(), 0)?;
        self.query_staging.upload(queries.as_slice(), 0)?;
        self.metadata_staging.upload(metadata.as_slice(), 0)?;
        let visibility_buffer = visibility_buffer.unwrap_or_else(|| self.zero_visibility.clone());
        let descriptor_set = gpu::DescriptorSet::builder(self.descriptor_set_layout.clone())
            .add_storage_buffer(0, self.candidate_buffer.clone())
            .add_storage_buffer(1, self.query_buffer.clone())
            .add_storage_buffer(2, self.result_buffer.clone())
            .add_storage_buffer(3, self.metadata_buffer.clone())
            .add_storage_buffer(4, visibility_buffer)
            .build()?;
        let output_count = *output_offsets.last().unwrap();
        let output_bytes = output_count * std::mem::size_of::<ScoredPointOffset>();
        let prepare_ns = prepare_started.elapsed().as_nanos() as u64;

        let command_record_started = std::time::Instant::now();
        self.context.copy_gpu_buffer(
            self.candidate_staging.clone(),
            self.candidate_buffer.clone(),
            0,
            0,
            std::mem::size_of_val(candidates.as_slice()),
        )?;
        self.context.copy_gpu_buffer(
            self.query_staging.clone(),
            self.query_buffer.clone(),
            0,
            0,
            std::mem::size_of_val(queries.as_slice()),
        )?;
        self.context.copy_gpu_buffer(
            self.metadata_staging.clone(),
            self.metadata_buffer.clone(),
            0,
            0,
            std::mem::size_of_val(metadata.as_slice()),
        )?;
        self.context.barrier_buffers(&[
            self.candidate_buffer.clone(),
            self.query_buffer.clone(),
            self.metadata_buffer.clone(),
        ])?;
        self.context.bind_pipeline(
            self.pipeline.clone(),
            &[descriptor_set, self.vector_storage.descriptor_set()],
        )?;
        self.context.dispatch(max_block_count, requests.len(), 1)?;
        self.context
            .barrier_buffers(std::slice::from_ref(&self.result_buffer))?;
        self.context.copy_gpu_buffer(
            self.result_buffer.clone(),
            self.result_staging.clone(),
            0,
            0,
            output_bytes,
        )?;
        let command_record_ns = command_record_started.elapsed().as_nanos() as u64;
        let queue_submit_started = std::time::Instant::now();
        self.context.run()?;
        let queue_submit_ns = queue_submit_started.elapsed().as_nanos() as u64;
        let fence_wait_started = std::time::Instant::now();
        self.context.wait_finish(GPU_TIMEOUT)?;
        let fence_wait_ns = fence_wait_started.elapsed().as_nanos() as u64;

        let download_started = std::time::Instant::now();
        let block_results = self
            .result_staging
            .download_vec::<ScoredPointOffset>(0, output_count)?;
        let download_ns = download_started.elapsed().as_nanos() as u64;
        let cpu_topk_started = std::time::Instant::now();
        let mut results = Vec::with_capacity(requests.len());
        for output_range in output_offsets.windows(2) {
            let mut queue = TopK::new(top);
            for &point in &block_results[output_range[0]..output_range[1]] {
                if point.idx != PointOffsetType::MAX {
                    queue.push(point);
                }
            }
            results.push(queue.into_vec());
        }
        let cpu_topk_ns = cpu_topk_started.elapsed().as_nanos() as u64;
        Ok((
            results,
            HeterogeneousExactTimings {
                total_candidates,
                prepare_ns,
                command_record_ns,
                queue_submit_ns,
                fence_wait_ns,
                download_ns,
                cpu_topk_ns,
            },
        ))
    }
}
