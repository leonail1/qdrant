use std::fmt;
use std::sync::Arc;

use common::fixed_length_priority_queue::FixedLengthPriorityQueue;
use common::types::{PointOffsetType, ScoredPointOffset};
use parking_lot::Mutex;

use super::GPU_TIMEOUT;
use super::gpu_vector_storage::GpuVectorStorage;
use super::shader_builder::ShaderBuilder;
use crate::common::operation_error::{OperationError, OperationResult};
use crate::vector_storage::VectorStorageRead;

struct GpuExactSearchContext {
    context: gpu::Context,
    pipeline: Arc<gpu::Pipeline>,
    descriptor_set: Arc<gpu::DescriptorSet>,
    vector_storage: Arc<GpuVectorStorage>,
    candidate_buffer: Arc<gpu::Buffer>,
    candidate_staging: Arc<gpu::Buffer>,
    query_buffer: Arc<gpu::Buffer>,
    query_staging: Arc<gpu::Buffer>,
    score_buffer: Arc<gpu::Buffer>,
    score_staging: Arc<gpu::Buffer>,
    candidate_capacity: usize,
    query_capacity: usize,
}

impl GpuExactSearchContext {
    fn new(
        vector_storage: Arc<GpuVectorStorage>,
        pipeline: Arc<gpu::Pipeline>,
        descriptor_set_layout: Arc<gpu::DescriptorSetLayout>,
        candidate_capacity: usize,
        queue_index: usize,
    ) -> OperationResult<Self> {
        let device = vector_storage.device();
        let candidate_bytes = candidate_capacity * std::mem::size_of::<PointOffsetType>();
        let score_bytes = candidate_capacity * std::mem::size_of::<f32>();
        let query_capacity = vector_storage.vector_capacity();
        let query_bytes = query_capacity * std::mem::size_of::<f32>();

        let candidate_buffer = gpu::Buffer::new(
            device.clone(),
            "Exact filtered candidate IDs",
            gpu::BufferType::Storage,
            candidate_bytes,
        )?;
        let candidate_staging = gpu::Buffer::new(
            device.clone(),
            "Exact filtered candidate upload",
            gpu::BufferType::CpuToGpu,
            candidate_bytes,
        )?;
        let query_buffer = gpu::Buffer::new(
            device.clone(),
            "Exact filtered query",
            gpu::BufferType::Storage,
            query_bytes,
        )?;
        let query_staging = gpu::Buffer::new(
            device.clone(),
            "Exact filtered query upload",
            gpu::BufferType::CpuToGpu,
            query_bytes,
        )?;
        let score_buffer = gpu::Buffer::new(
            device.clone(),
            "Exact filtered candidate scores",
            gpu::BufferType::Storage,
            score_bytes,
        )?;
        let score_staging = gpu::Buffer::new(
            device.clone(),
            "Exact filtered score download",
            gpu::BufferType::GpuToCpu,
            score_bytes,
        )?;
        let descriptor_set = gpu::DescriptorSet::builder(descriptor_set_layout)
            .add_storage_buffer(0, candidate_buffer.clone())
            .add_storage_buffer(1, query_buffer.clone())
            .add_storage_buffer(2, score_buffer.clone())
            .build()?;

        Ok(Self {
            context: gpu::Context::new_with_queue_index(device, queue_index)?,
            pipeline,
            descriptor_set,
            vector_storage,
            candidate_buffer,
            candidate_staging,
            query_buffer,
            query_staging,
            score_buffer,
            score_staging,
            candidate_capacity,
            query_capacity,
        })
    }

    fn search(
        &mut self,
        query: &[f32],
        candidates: &[PointOffsetType],
        top: usize,
    ) -> OperationResult<Vec<ScoredPointOffset>> {
        if candidates.len() > self.candidate_capacity {
            return Err(OperationError::service_error(
                "GPU exact candidate capacity exceeded",
            ));
        }
        if query.len() != self.vector_storage.dim() {
            return Err(OperationError::service_error(
                "GPU exact query dimension mismatch",
            ));
        }
        if candidates.is_empty() || top == 0 {
            return Ok(Vec::new());
        }

        let mut padded_query = vec![0.0f32; self.query_capacity];
        padded_query[..query.len()].copy_from_slice(query);
        self.candidate_staging.upload(candidates, 0)?;
        self.query_staging.upload(padded_query.as_slice(), 0)?;
        self.context.copy_gpu_buffer(
            self.candidate_staging.clone(),
            self.candidate_buffer.clone(),
            0,
            0,
            std::mem::size_of_val(candidates),
        )?;
        self.context.copy_gpu_buffer(
            self.query_staging.clone(),
            self.query_buffer.clone(),
            0,
            0,
            padded_query.len() * std::mem::size_of::<f32>(),
        )?;
        self.context.run()?;
        self.context.wait_finish(GPU_TIMEOUT)?;

        self.context.bind_pipeline(
            self.pipeline.clone(),
            &[
                self.descriptor_set.clone(),
                self.vector_storage.descriptor_set(),
            ],
        )?;
        self.context.dispatch(candidates.len(), 1, 1)?;
        self.context
            .barrier_buffers(std::slice::from_ref(&self.score_buffer))?;
        self.context.copy_gpu_buffer(
            self.score_buffer.clone(),
            self.score_staging.clone(),
            0,
            0,
            candidates.len() * std::mem::size_of::<f32>(),
        )?;
        self.context.run()?;
        self.context.wait_finish(GPU_TIMEOUT)?;

        let scores = self.score_staging.download_vec::<f32>(0, candidates.len())?;
        let mut queue = FixedLengthPriorityQueue::new(top);
        for (&idx, score) in candidates.iter().zip(scores) {
            queue.push(ScoredPointOffset { idx, score });
        }
        Ok(queue.into_sorted_vec())
    }
}

pub struct GpuExactSearchCache {
    contexts: Mutex<Vec<GpuExactSearchContext>>,
    candidate_capacity: usize,
    context_count: usize,
}

impl fmt::Debug for GpuExactSearchCache {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GpuExactSearchCache")
            .field("candidate_capacity", &self.candidate_capacity)
            .field("context_count", &self.context_count)
            .finish_non_exhaustive()
    }
}

impl GpuExactSearchCache {
    pub fn new(
        device: Arc<gpu::Device>,
        vector_storage: &crate::vector_storage::VectorStorageEnum,
        candidate_capacity: usize,
        context_count: usize,
        stopped: &std::sync::atomic::AtomicBool,
    ) -> OperationResult<Self> {
        if candidate_capacity == 0 || context_count == 0 {
            return Err(OperationError::service_error(
                "GPU exact capacities must be positive",
            ));
        }
        if vector_storage.datatype() != crate::types::VectorStorageDatatype::Float32 {
            return Err(OperationError::service_error(
                "GPU exact currently supports float32 dense vectors only",
            ));
        }

        let vector_storage = Arc::new(GpuVectorStorage::new(
            device.clone(),
            vector_storage,
            None,
            false,
            stopped,
        )?);
        let descriptor_set_layout = gpu::DescriptorSetLayout::builder()
            .add_storage_buffer(0)
            .add_storage_buffer(1)
            .add_storage_buffer(2)
            .build(device.clone())?;
        let shader = ShaderBuilder::new(device.clone())
            .with_shader_code(include_str!("shaders/run_exact_filtered_score.comp"))
            .with_parameters(vector_storage.as_ref())
            .build("run_exact_filtered_score.comp")?;
        let pipeline = gpu::Pipeline::builder()
            .add_descriptor_set_layout(0, descriptor_set_layout.clone())
            .add_descriptor_set_layout(1, vector_storage.descriptor_set_layout())
            .add_shader(shader)
            .build(device)?;

        let contexts = (0..context_count)
            .map(|queue_index| {
                GpuExactSearchContext::new(
                    vector_storage.clone(),
                    pipeline.clone(),
                    descriptor_set_layout.clone(),
                    candidate_capacity,
                    queue_index,
                )
            })
            .collect::<OperationResult<Vec<_>>>()?;

        Ok(Self {
            contexts: Mutex::new(contexts),
            candidate_capacity,
            context_count,
        })
    }

    pub fn search(
        &self,
        query: &[f32],
        candidates: &[PointOffsetType],
        top: usize,
    ) -> OperationResult<Option<Vec<ScoredPointOffset>>> {
        let Some(mut context) = self.contexts.lock().pop() else {
            return Ok(None);
        };
        let result = context.search(query, candidates, top);
        self.contexts.lock().push(context);
        result.map(Some)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicBool;

    use common::counter::hardware_counter::HardwareCounterCell;

    use super::*;
    use crate::types::Distance;
    use crate::vector_storage::VectorStorage;
    use crate::vector_storage::dense::volatile_dense_vector_storage::new_volatile_dense_vector_storage;

    #[test]
    fn exact_filtered_gpu_matches_cpu_l2() {
        const DIM: usize = 128;
        const COUNT: usize = 1_024;
        const TARGET: usize = 137;
        const TOP: usize = 10;

        let vectors = (0..COUNT)
            .map(|row| {
                (0..DIM)
                    .map(|column| {
                        ((row * 17 + column * 13 + row * column * 7) % 65_521) as f32
                            / 65_521.0
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let query = vectors[TARGET].clone();
        let mut storage = new_volatile_dense_vector_storage(DIM, Distance::Euclid);
        let counter = HardwareCounterCell::new();
        for (idx, vector) in vectors.iter().enumerate() {
            storage
                .insert_vector(idx as PointOffsetType, vector.as_slice().into(), &counter)
                .unwrap();
        }

        let instance = gpu::Instance::builder().build().unwrap();
        let device = gpu::Device::new(instance.clone(), &instance.physical_devices()[0]).unwrap();
        let stopped = AtomicBool::new(false);
        let cache = Arc::new(
            GpuExactSearchCache::new(device, &storage, COUNT, 8, &stopped).unwrap(),
        );
        let query = Arc::new(query);
        let candidates = Arc::new((0..COUNT as PointOffsetType).collect::<Vec<_>>());
        let observed = (0..8)
            .map(|_| {
                let cache = cache.clone();
                let query = query.clone();
                let candidates = candidates.clone();
                std::thread::spawn(move || {
                    cache
                        .search(query.as_slice(), candidates.as_slice(), TOP)
                        .unwrap()
                        .unwrap()
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect::<Vec<_>>();

        let mut expected = vectors
            .iter()
            .enumerate()
            .map(|(idx, vector)| {
                let score = -vector
                    .iter()
                    .zip(query.iter())
                    .map(|(left, right)| (left - right) * (left - right))
                    .sum::<f32>();
                (idx as PointOffsetType, score)
            })
            .collect::<Vec<_>>();
        expected.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });

        for result in observed {
            assert_eq!(result[0].idx, TARGET as PointOffsetType);
            for (actual, &(expected_id, expected_score)) in result.iter().zip(&expected[..TOP]) {
                assert_eq!(actual.idx, expected_id);
                assert!((actual.score - expected_score).abs() < 1e-4);
            }
        }
    }
}
