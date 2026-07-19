use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use common::bitvec::BitSlice;
use common::types::{PointOffsetType, ScoredPointOffset};
use parking_lot::Mutex;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use super::GPU_TIMEOUT;
use super::gpu_exact_search::GpuVisibilitySnapshot;
use super::gpu_links::GpuLinks;
use super::gpu_vector_storage::GpuVectorStorage;
use super::gpu_visited_flags::GpuVisitedFlags;
use super::shader_builder::{ShaderBuilder, ShaderBuilderParameters};
use crate::common::operation_error::{OperationError, OperationResult};
use crate::index::hnsw_index::graph_layers::GraphLayersBase;

const MIN_POINTS_FOR_BINARY_HEAP: usize = 512;
const RESIDENT_PREDICATE_LIMIT: usize = 256;
const RESIDENT_VISIBILITY_LIMIT: usize = 64;

#[derive(Clone, Copy, FromBytes, Immutable, IntoBytes, KnownLayout)]
#[repr(C)]
struct GpuFilteredGraphRequest {
    entry: PointOffsetType,
    top: u32,
    point_count: u32,
    reserved: u32,
}

struct GpuFilteredGraphShaderParameters {
    ef: usize,
}

impl ShaderBuilderParameters for GpuFilteredGraphShaderParameters {
    fn shader_includes(&self) -> HashMap<String, String> {
        HashMap::from([
            (
                "shared_buffer.comp".to_string(),
                include_str!("shaders/shared_buffer.comp").to_string(),
            ),
            (
                "bheap.comp".to_string(),
                include_str!("shaders/bheap.comp").to_string(),
            ),
            (
                "search_context.comp".to_string(),
                include_str!("shaders/search_context.comp").to_string(),
            ),
        ])
    }

    fn shader_defines(&self) -> HashMap<String, Option<String>> {
        let mut defines = HashMap::from([("EF".to_string(), Some(self.ef.to_string()))]);
        if self.ef < MIN_POINTS_FOR_BINARY_HEAP {
            defines.insert("BHEAP_LINEAR".to_string(), None);
        }
        defines
    }
}

pub struct GpuFilteredGraphSearchContext {
    context: gpu::Context,
    pipeline: Arc<gpu::Pipeline>,
    descriptor_set_layout: Arc<gpu::DescriptorSetLayout>,
    vector_storage: Arc<GpuVectorStorage>,
    links: Arc<GpuLinks>,
    visited: GpuVisitedFlags,
    query_buffer: Arc<gpu::Buffer>,
    query_staging: Arc<gpu::Buffer>,
    result_buffer: Arc<gpu::Buffer>,
    result_staging: Arc<gpu::Buffer>,
    predicate_buffer: Arc<gpu::Buffer>,
    predicate_staging: Arc<gpu::Buffer>,
    deleted_buffer: Arc<gpu::Buffer>,
    deleted_staging: Arc<gpu::Buffer>,
    request_buffer: Arc<gpu::Buffer>,
    request_staging: Arc<gpu::Buffer>,
    point_count: usize,
    word_count: usize,
    query_capacity: usize,
    ef: usize,
}

impl GpuFilteredGraphSearchContext {
    pub fn new(
        device: Arc<gpu::Device>,
        vector_storage: Arc<GpuVectorStorage>,
        graph: &impl GraphLayersBase,
        point_count: usize,
        ef: usize,
        stopped: &AtomicBool,
    ) -> OperationResult<Self> {
        if point_count == 0 || ef == 0 || point_count != vector_storage.num_vectors() {
            return Err(OperationError::service_error(
                "GPU filtered graph capacities are inconsistent",
            ));
        }
        let links = Arc::new(GpuLinks::new_compact(
            device.clone(),
            graph,
            point_count,
            stopped,
        )?);

        Self::new_with_resources(device, vector_storage, links, point_count, ef, 0)
    }

    fn new_with_resources(
        device: Arc<gpu::Device>,
        vector_storage: Arc<GpuVectorStorage>,
        links: Arc<GpuLinks>,
        point_count: usize,
        ef: usize,
        queue_index: usize,
    ) -> OperationResult<Self> {
        if point_count == 0 || ef == 0 || point_count != vector_storage.num_vectors() {
            return Err(OperationError::service_error(
                "GPU filtered graph capacities are inconsistent",
            ));
        }

        let visited = GpuVisitedFlags::new(device.clone(), 1, point_count, 1..=1)?;
        let query_capacity = vector_storage.vector_capacity();
        let query_bytes = query_capacity * std::mem::size_of::<f32>();
        let result_bytes = ef * std::mem::size_of::<ScoredPointOffset>();
        let word_count = point_count.div_ceil(u32::BITS as usize);
        let word_bytes = word_count * std::mem::size_of::<u32>();
        let request_bytes = std::mem::size_of::<GpuFilteredGraphRequest>();

        let query_buffer = gpu::Buffer::new(
            device.clone(),
            "Filtered graph query",
            gpu::BufferType::Storage,
            query_bytes,
        )?;
        let query_staging = gpu::Buffer::new(
            device.clone(),
            "Filtered graph query upload",
            gpu::BufferType::CpuToGpu,
            query_bytes,
        )?;
        let result_buffer = gpu::Buffer::new(
            device.clone(),
            "Filtered graph results",
            gpu::BufferType::Storage,
            result_bytes,
        )?;
        let result_staging = gpu::Buffer::new(
            device.clone(),
            "Filtered graph result download",
            gpu::BufferType::GpuToCpu,
            result_bytes,
        )?;
        let predicate_buffer = gpu::Buffer::new(
            device.clone(),
            "Filtered graph predicate words",
            gpu::BufferType::Storage,
            word_bytes,
        )?;
        let predicate_staging = gpu::Buffer::new(
            device.clone(),
            "Filtered graph predicate upload",
            gpu::BufferType::CpuToGpu,
            word_bytes,
        )?;
        let deleted_buffer = gpu::Buffer::new(
            device.clone(),
            "Filtered graph deleted words",
            gpu::BufferType::Storage,
            word_bytes,
        )?;
        let deleted_staging = gpu::Buffer::new(
            device.clone(),
            "Filtered graph deleted upload",
            gpu::BufferType::CpuToGpu,
            word_bytes,
        )?;
        let request_buffer = gpu::Buffer::new(
            device.clone(),
            "Filtered graph request",
            gpu::BufferType::Storage,
            request_bytes,
        )?;
        let request_staging = gpu::Buffer::new(
            device.clone(),
            "Filtered graph request upload",
            gpu::BufferType::CpuToGpu,
            request_bytes,
        )?;

        let descriptor_set_layout = gpu::DescriptorSetLayout::builder()
            .add_storage_buffer(0)
            .add_storage_buffer(1)
            .add_storage_buffer(2)
            .add_storage_buffer(3)
            .add_storage_buffer(4)
            .build(device.clone())?;
        let shader_parameters = GpuFilteredGraphShaderParameters { ef };
        let shader = ShaderBuilder::new(device.clone())
            .with_shader_code(include_str!("shaders/run_filtered_graph_search.comp"))
            .with_parameters(vector_storage.as_ref())
            .with_parameters(links.as_ref())
            .with_parameters(&visited)
            .with_parameters(&shader_parameters)
            .build("run_filtered_graph_search.comp")?;
        let pipeline = gpu::Pipeline::builder()
            .add_descriptor_set_layout(0, descriptor_set_layout.clone())
            .add_descriptor_set_layout(1, vector_storage.descriptor_set_layout())
            .add_descriptor_set_layout(2, links.descriptor_set_layout())
            .add_descriptor_set_layout(3, visited.descriptor_set_layout())
            .add_shader(shader)
            .build(device.clone())?;

        Ok(Self {
            context: gpu::Context::new_with_queue_index(device, queue_index)?,
            pipeline,
            descriptor_set_layout,
            vector_storage,
            links,
            visited,
            query_buffer,
            query_staging,
            result_buffer,
            result_staging,
            predicate_buffer,
            predicate_staging,
            deleted_buffer,
            deleted_staging,
            request_buffer,
            request_staging,
            point_count,
            word_count,
            query_capacity,
            ef,
        })
    }

    fn upload_resident_words(
        &mut self,
        name: &str,
        words: &[u32],
    ) -> OperationResult<Arc<gpu::Buffer>> {
        if words.len() != self.word_count {
            return Err(OperationError::service_error(
                "GPU filtered graph resident bitset has an invalid length",
            ));
        }
        let bytes = std::mem::size_of_val(words);
        let buffer = gpu::Buffer::new(
            self.vector_storage.device(),
            name,
            gpu::BufferType::Storage,
            bytes,
        )?;
        self.predicate_staging.upload(words, 0)?;
        self.context.copy_gpu_buffer(
            self.predicate_staging.clone(),
            buffer.clone(),
            0,
            0,
            bytes,
        )?;
        self.context.run()?;
        self.context.wait_finish(GPU_TIMEOUT)?;
        Ok(buffer)
    }

    pub fn search(
        &mut self,
        query: &[f32],
        entry: PointOffsetType,
        included: &[PointOffsetType],
        deleted: Option<&BitSlice>,
        top: usize,
    ) -> OperationResult<Vec<ScoredPointOffset>> {
        // GpuVectorStorage::dim() is the device-aligned vector capacity. Dense
        // query vectors keep their logical dimension and are zero-padded into
        // that capacity before upload.
        if query.len() > self.query_capacity || top == 0 || top > self.ef {
            return Err(OperationError::service_error(
                "GPU filtered graph query shape is unsupported",
            ));
        }
        if entry as usize >= self.point_count {
            return Err(OperationError::service_error(
                "GPU filtered graph entry is out of range",
            ));
        }
        let mut predicate_words = vec![0u32; self.word_count];
        for &point_id in included {
            if point_id as usize >= self.point_count {
                return Err(OperationError::service_error(
                    "GPU filtered graph predicate contains an out-of-range point",
                ));
            }
            predicate_words[point_id as usize / u32::BITS as usize] |=
                1u32 << (point_id as usize % u32::BITS as usize);
        }
        let mut deleted_words = vec![0u32; self.word_count];
        if let Some(deleted) = deleted {
            for point_id in deleted
                .iter_ones()
                .filter(|&point_id| point_id < self.point_count)
            {
                deleted_words[point_id / u32::BITS as usize] |=
                    1u32 << (point_id % u32::BITS as usize);
            }
        }
        let entry_word = entry as usize / u32::BITS as usize;
        let entry_mask = 1u32 << (entry as usize % u32::BITS as usize);
        if predicate_words[entry_word] & entry_mask == 0
            || deleted_words[entry_word] & entry_mask != 0
        {
            return Err(OperationError::service_error(
                "GPU filtered graph entry is not visible in the predicate",
            ));
        }

        self.predicate_staging
            .upload(predicate_words.as_slice(), 0)?;
        self.deleted_staging.upload(deleted_words.as_slice(), 0)?;
        let visibility_uploads = [
            (
                self.predicate_staging.clone(),
                self.predicate_buffer.clone(),
            ),
            (self.deleted_staging.clone(), self.deleted_buffer.clone()),
        ];
        for (source, target) in &visibility_uploads {
            self.context
                .copy_gpu_buffer(source.clone(), target.clone(), 0, 0, target.size())?;
        }
        self.context.barrier_buffers(
            &visibility_uploads
                .iter()
                .map(|(_, target)| target.clone())
                .collect::<Vec<_>>(),
        )?;
        self.search_with_resident_words(
            query,
            entry,
            self.predicate_buffer.clone(),
            self.deleted_buffer.clone(),
            top,
        )
    }

    fn search_with_resident_words(
        &mut self,
        query: &[f32],
        entry: PointOffsetType,
        predicate_buffer: Arc<gpu::Buffer>,
        deleted_buffer: Arc<gpu::Buffer>,
        top: usize,
    ) -> OperationResult<Vec<ScoredPointOffset>> {
        if query.len() > self.query_capacity
            || top == 0
            || top > self.ef
            || entry as usize >= self.point_count
        {
            return Err(OperationError::service_error(
                "GPU filtered graph resident query shape is unsupported",
            ));
        }
        let mut padded_query = vec![0.0f32; self.query_capacity];
        padded_query[..query.len()].copy_from_slice(query);
        let request = GpuFilteredGraphRequest {
            entry,
            top: top as u32,
            point_count: self.point_count as u32,
            reserved: 0,
        };
        self.query_staging.upload(padded_query.as_slice(), 0)?;
        self.request_staging.upload(&request, 0)?;
        self.visited.clear(&mut self.context)?;
        let uploads = [
            (self.query_staging.clone(), self.query_buffer.clone()),
            (self.request_staging.clone(), self.request_buffer.clone()),
        ];
        for (source, target) in &uploads {
            self.context
                .copy_gpu_buffer(source.clone(), target.clone(), 0, 0, target.size())?;
        }
        self.context.barrier_buffers(
            &uploads
                .iter()
                .map(|(_, target)| target.clone())
                .collect::<Vec<_>>(),
        )?;
        let descriptor_set = gpu::DescriptorSet::builder(self.descriptor_set_layout.clone())
            .add_storage_buffer(0, self.query_buffer.clone())
            .add_storage_buffer(1, self.result_buffer.clone())
            .add_storage_buffer(2, predicate_buffer)
            .add_storage_buffer(3, deleted_buffer)
            .add_storage_buffer(4, self.request_buffer.clone())
            .build()?;
        self.context.bind_pipeline(
            self.pipeline.clone(),
            &[
                descriptor_set,
                self.vector_storage.descriptor_set(),
                self.links.descriptor_set(),
                self.visited.descriptor_set(),
            ],
        )?;
        self.context.dispatch(1, 1, 1)?;
        self.context
            .barrier_buffers(std::slice::from_ref(&self.result_buffer))?;
        self.context.copy_gpu_buffer(
            self.result_buffer.clone(),
            self.result_staging.clone(),
            0,
            0,
            top * std::mem::size_of::<ScoredPointOffset>(),
        )?;
        self.context.run()?;
        self.context.wait_finish(GPU_TIMEOUT)?;
        Ok(self
            .result_staging
            .download_vec::<ScoredPointOffset>(0, top)?
            .into_iter()
            .filter(|point| point.idx != PointOffsetType::MAX)
            .collect())
    }
}

struct ResidentPredicate {
    _candidates: Arc<Vec<PointOffsetType>>,
    buffer: Arc<gpu::Buffer>,
}

/// Segment-local filtered HNSW executor. Vector and graph projections are
/// shared, while each in-flight query owns its command context, visited flags,
/// query/result buffers and queue lane.
pub struct GpuFilteredGraphSearchCache {
    contexts: Mutex<Vec<GpuFilteredGraphSearchContext>>,
    resident_predicates: Mutex<HashMap<usize, ResidentPredicate>>,
    resident_visibility: Mutex<HashMap<(u64, usize), Arc<gpu::Buffer>>>,
    zero_visibility: Arc<gpu::Buffer>,
    point_count: usize,
    ef: usize,
}

impl std::fmt::Debug for GpuFilteredGraphSearchCache {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GpuFilteredGraphSearchCache")
            .field("point_count", &self.point_count)
            .field("ef", &self.ef)
            .finish_non_exhaustive()
    }
}

impl GpuFilteredGraphSearchCache {
    pub fn new(
        device: Arc<gpu::Device>,
        vector_storage: Arc<GpuVectorStorage>,
        graph: &impl GraphLayersBase,
        point_count: usize,
        ef: usize,
        context_count: usize,
        stopped: &AtomicBool,
    ) -> OperationResult<Self> {
        if context_count == 0 {
            return Err(OperationError::service_error(
                "GPU filtered graph context count must be positive",
            ));
        }
        let first = GpuFilteredGraphSearchContext::new(
            device.clone(),
            vector_storage,
            graph,
            point_count,
            ef,
            stopped,
        )?;
        let shared_vectors = first.vector_storage.clone();
        let shared_links = first.links.clone();
        let mut contexts = Vec::with_capacity(context_count);
        contexts.push(first);
        for queue_index in 1..context_count {
            contexts.push(GpuFilteredGraphSearchContext::new_with_resources(
                device.clone(),
                shared_vectors.clone(),
                shared_links.clone(),
                point_count,
                ef,
                queue_index,
            )?);
        }
        let word_count = point_count.div_ceil(u32::BITS as usize);
        let zero_words = vec![0u32; word_count];
        let zero_visibility =
            contexts[0].upload_resident_words("Filtered graph zero visibility", &zero_words)?;
        Ok(Self {
            contexts: Mutex::new(contexts),
            resident_predicates: Mutex::new(HashMap::new()),
            resident_visibility: Mutex::new(HashMap::new()),
            zero_visibility,
            point_count,
            ef,
        })
    }

    fn predicate_buffer(
        &self,
        context: &mut GpuFilteredGraphSearchContext,
        candidates: Arc<Vec<PointOffsetType>>,
    ) -> OperationResult<Arc<gpu::Buffer>> {
        let key = Arc::as_ptr(&candidates) as usize;
        if let Some(entry) = self.resident_predicates.lock().get(&key) {
            return Ok(entry.buffer.clone());
        }
        let mut words = vec![0u32; self.point_count.div_ceil(u32::BITS as usize)];
        for &point_id in candidates.iter() {
            if point_id as usize >= self.point_count {
                return Err(OperationError::service_error(
                    "GPU filtered graph predicate contains an out-of-range point",
                ));
            }
            words[point_id as usize / u32::BITS as usize] |=
                1u32 << (point_id as usize % u32::BITS as usize);
        }
        let buffer = context.upload_resident_words("Filtered graph predicate", &words)?;
        let mut resident = self.resident_predicates.lock();
        if resident.len() >= RESIDENT_PREDICATE_LIMIT {
            resident.clear();
        }
        resident.insert(
            key,
            ResidentPredicate {
                _candidates: candidates,
                buffer: buffer.clone(),
            },
        );
        Ok(buffer)
    }

    fn visibility_buffer(
        &self,
        context: &mut GpuFilteredGraphSearchContext,
        visibility: Option<GpuVisibilitySnapshot<'_>>,
    ) -> OperationResult<Arc<gpu::Buffer>> {
        let Some(visibility) = visibility else {
            return Ok(self.zero_visibility.clone());
        };
        if visibility.generation == 0
            || visibility.point_count != self.point_count
            || visibility.deleted.len() > visibility.point_count
        {
            return Err(OperationError::service_error(
                "GPU filtered graph visibility snapshot identity is invalid",
            ));
        }
        let key = (visibility.generation, visibility.point_count);
        if let Some(buffer) = self.resident_visibility.lock().get(&key) {
            return Ok(buffer.clone());
        }
        let mut words = vec![0u32; self.point_count.div_ceil(u32::BITS as usize)];
        for point_id in visibility
            .deleted
            .iter_ones()
            .filter(|&point_id| point_id < self.point_count)
        {
            words[point_id / u32::BITS as usize] |= 1u32 << (point_id % u32::BITS as usize);
        }
        let buffer = context.upload_resident_words("Filtered graph visibility", &words)?;
        let mut resident = self.resident_visibility.lock();
        if resident.len() >= RESIDENT_VISIBILITY_LIMIT {
            resident.clear();
        }
        resident.insert(key, buffer.clone());
        Ok(buffer)
    }

    pub fn search(
        &self,
        query: &[f32],
        entry: PointOffsetType,
        candidates: Arc<Vec<PointOffsetType>>,
        visibility: Option<GpuVisibilitySnapshot<'_>>,
        top: usize,
        ef: usize,
    ) -> OperationResult<Option<Vec<ScoredPointOffset>>> {
        if ef != self.ef {
            return Ok(None);
        }
        let Some(mut context) = self.contexts.lock().pop() else {
            return Ok(None);
        };
        let result = (|| {
            let predicate = self.predicate_buffer(&mut context, candidates)?;
            let visibility = self.visibility_buffer(&mut context, visibility)?;
            context
                .search_with_resident_words(query, entry, predicate, visibility, top)
                .map(Some)
        })();
        self.contexts.lock().push(context);
        result
    }

    pub fn clear_resident_filters(&self) {
        self.resident_predicates.lock().clear();
        self.resident_visibility.lock().clear();
    }
}

#[cfg(test)]
mod tests {
    use common::bitvec::BitVec;
    use common::counter::hardware_counter::HardwareCounterCell;

    use super::*;
    use crate::data_types::vectors::{QueryVector, VectorElementType, VectorInternal};
    use crate::index::hnsw_index::HnswM;
    use crate::index::hnsw_index::gpu::tests::create_gpu_graph_test_data;
    use crate::index::hnsw_index::point_scorer::FilteredScorer;
    use crate::vector_storage::VectorStorageRead;

    #[test]
    fn filtered_graph_gpu_half_precision_shader_compiles() {
        const COUNT: usize = 128;
        const DIM: usize = 64;
        const EF: usize = 32;

        let test = create_gpu_graph_test_data(COUNT, DIM, HnswM::new2(8), EF, 1);
        let instance = gpu::Instance::builder().build().unwrap();
        let device = gpu::Device::new(instance.clone(), &instance.physical_devices()[0]).unwrap();
        let stopped = AtomicBool::new(false);
        let gpu_storage = Arc::new(
            GpuVectorStorage::new(device.clone(), &test.vector_storage, None, true, &stopped)
                .unwrap(),
        );
        assert!(gpu_storage.resident_vector_bytes() > 0);
        GpuFilteredGraphSearchContext::new(
            device,
            gpu_storage,
            &test.graph_layers_builder,
            COUNT,
            EF,
            &stopped,
        )
        .unwrap();
    }

    #[test]
    fn filtered_graph_gpu_matches_cpu_level_zero_with_visibility() {
        const COUNT: usize = 1_024;
        const DIM: usize = 64;
        const EF: usize = 32;
        const TOP: usize = 10;

        let test = create_gpu_graph_test_data(COUNT, DIM, HnswM::new2(8), EF, 1);
        // SegmentEntry preprocesses API queries before vector-index search.
        // Reproduce that contract here; cosine preprocessing is idempotent for
        // the CPU reference scorer.
        let query = test
            .vector_storage
            .distance()
            .preprocess_vector::<VectorElementType>(test.search_vectors[0].clone());
        let included = (0..COUNT as PointOffsetType)
            .filter(|point_id| point_id % 3 != 1)
            .collect::<Vec<_>>();
        let mut deleted = BitVec::repeat(false, COUNT);
        for point_id in (6..COUNT).step_by(29) {
            deleted.set(point_id, true);
        }
        let mut effective_deleted = BitVec::repeat(true, COUNT);
        for &point_id in &included {
            effective_deleted.set(point_id as usize, deleted[point_id as usize]);
        }

        let query_vector = QueryVector::Nearest(VectorInternal::Dense(query.clone()));
        let mut scorer = FilteredScorer::new(
            query_vector,
            &test.vector_storage,
            None,
            None,
            &effective_deleted,
            HardwareCounterCell::new(),
        )
        .unwrap();
        let entry = ScoredPointOffset {
            idx: 0,
            score: scorer.score_point(0),
        };
        let expected = test
            .graph_layers_builder
            .search_on_level(entry, 0, EF, &mut scorer, &AtomicBool::new(false))
            .unwrap()
            .into_sorted_vec()
            .into_iter()
            .take(TOP)
            .collect::<Vec<_>>();

        let instance = gpu::Instance::builder().build().unwrap();
        let device = gpu::Device::new(instance.clone(), &instance.physical_devices()[0]).unwrap();
        let stopped = AtomicBool::new(false);
        let gpu_storage = Arc::new(
            GpuVectorStorage::new(device.clone(), &test.vector_storage, None, false, &stopped)
                .unwrap(),
        );
        let mut context = GpuFilteredGraphSearchContext::new(
            device,
            gpu_storage,
            &test.graph_layers_builder,
            COUNT,
            EF,
            &stopped,
        )
        .unwrap();
        for _ in 0..2 {
            let actual = context
                .search(&query, 0, &included, Some(&deleted), TOP)
                .unwrap();
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(&expected) {
                assert_eq!(actual.idx, expected.idx);
                let delta = (actual.score - expected.score).abs();
                assert!(
                    delta < 1e-5,
                    "score mismatch for point {}: GPU={}, CPU={}, delta={delta}",
                    actual.idx,
                    actual.score,
                    expected.score,
                );
            }
        }
    }

    #[test]
    fn filtered_graph_cache_reuses_shared_graph_and_predicate() {
        const COUNT: usize = 1_024;
        const DIM: usize = 64;
        const EF: usize = 32;
        const TOP: usize = 10;

        let test = create_gpu_graph_test_data(COUNT, DIM, HnswM::new2(8), EF, 1);
        let query = test
            .vector_storage
            .distance()
            .preprocess_vector::<VectorElementType>(test.search_vectors[0].clone());
        let candidates = Arc::new(
            (0..COUNT as PointOffsetType)
                .filter(|point_id| point_id % 3 != 1)
                .collect::<Vec<_>>(),
        );
        let query_vector = QueryVector::Nearest(VectorInternal::Dense(query.clone()));
        let mut effective_deleted = BitVec::repeat(true, COUNT);
        for &point_id in candidates.iter() {
            effective_deleted.set(point_id as usize, false);
        }
        let mut scorer = FilteredScorer::new(
            query_vector,
            &test.vector_storage,
            None,
            None,
            &effective_deleted,
            HardwareCounterCell::new(),
        )
        .unwrap();
        let entry = ScoredPointOffset {
            idx: 0,
            score: scorer.score_point(0),
        };
        let expected = test
            .graph_layers_builder
            .search_on_level(entry, 0, EF, &mut scorer, &AtomicBool::new(false))
            .unwrap()
            .into_sorted_vec()
            .into_iter()
            .take(TOP)
            .collect::<Vec<_>>();

        let instance = gpu::Instance::builder().build().unwrap();
        let device = gpu::Device::new(instance.clone(), &instance.physical_devices()[0]).unwrap();
        let stopped = AtomicBool::new(false);
        let gpu_storage = Arc::new(
            GpuVectorStorage::new(device.clone(), &test.vector_storage, None, false, &stopped)
                .unwrap(),
        );
        let cache = GpuFilteredGraphSearchCache::new(
            device,
            gpu_storage,
            &test.graph_layers_builder,
            COUNT,
            EF,
            2,
            &stopped,
        )
        .unwrap();
        for _ in 0..2 {
            let actual = cache
                .search(&query, 0, candidates.clone(), None, TOP, EF)
                .unwrap()
                .unwrap();
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(&expected) {
                assert_eq!(actual.idx, expected.idx);
                assert!((actual.score - expected.score).abs() < 1e-5);
            }
        }
    }
}
