use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{SyncSender, sync_channel};

use common::bitvec::BitSlice;
use common::top_k::TopK;
use common::types::{PointOffsetType, ScoredPointOffset};
use parking_lot::{Condvar, Mutex};

use super::GPU_TIMEOUT;
use super::gpu_vector_storage::GpuVectorStorage;
use super::shader_builder::ShaderBuilder;
use crate::common::operation_error::{OperationError, OperationResult};
use crate::types::Filter;
use crate::vector_storage::VectorStorageRead;

const FILTER_CACHE_MAX_ENTRIES: usize = 4_096;
const FILTER_CACHE_MAX_BYTES: usize = 64 * 1024 * 1024;
const BLOCK_TOPK_MIN_CANDIDATES: usize = 65_536;
const BLOCK_TOPK_LIMIT: usize = 16;
const BLOCK_TOPK_TILE_CANDIDATES: usize = 128;
pub const GPU_EXACT_SUBMISSION_BATCH_LIMIT: usize = 4;

#[derive(Clone, Copy)]
pub struct GpuVisibilitySnapshot<'a> {
    pub generation: u64,
    pub deleted: &'a BitSlice,
    /// Full vector ordinal space covered by this read view. Proxy deletion
    /// masks may be shorter because their bitvec only grows to the highest
    /// deleted ordinal; missing bits are live.
    pub point_count: usize,
}

/// Bounded, segment-local cache of payload-index candidate materializations.
///
/// The owning HNSW index only consults this cache for immutable segments and
/// payload-only filters. `payload_epoch` is advanced by every payload/index
/// mutation, so a hit always belongs to the same payload-index snapshot.
#[derive(Debug, Default)]
pub struct GpuFilterCandidateCache {
    payload_epoch: u64,
    entries: HashMap<Filter, Arc<Vec<PointOffsetType>>>,
    cached_points: usize,
}

impl GpuFilterCandidateCache {
    pub fn clear(&mut self) {
        self.entries.clear();
        self.cached_points = 0;
    }

    fn synchronize_epoch(&mut self, payload_epoch: u64) {
        if self.payload_epoch != payload_epoch {
            self.clear();
            self.payload_epoch = payload_epoch;
        }
    }

    pub fn get(
        &mut self,
        filter: &Filter,
        payload_epoch: u64,
    ) -> Option<Arc<Vec<PointOffsetType>>> {
        self.synchronize_epoch(payload_epoch);
        self.entries.get(filter).cloned()
    }

    pub fn insert(
        &mut self,
        filter: Filter,
        candidates: Arc<Vec<PointOffsetType>>,
        payload_epoch: u64,
    ) {
        self.synchronize_epoch(payload_epoch);
        let candidate_bytes = candidates
            .len()
            .saturating_mul(std::mem::size_of::<PointOffsetType>());
        if candidate_bytes > FILTER_CACHE_MAX_BYTES {
            return;
        }
        if self.entries.contains_key(&filter) {
            return;
        }
        let current_bytes = self
            .cached_points
            .saturating_mul(std::mem::size_of::<PointOffsetType>());
        if self.entries.len() >= FILTER_CACHE_MAX_ENTRIES
            || current_bytes.saturating_add(candidate_bytes) > FILTER_CACHE_MAX_BYTES
        {
            self.clear();
        }
        self.cached_points = self.cached_points.saturating_add(candidates.len());
        self.entries.insert(filter, candidates);
    }
}

#[derive(Default)]
struct GpuExactSearchStats {
    total_queries: AtomicU64,
    window_queries: AtomicU64,
    candidates: AtomicU64,
    predicate_ns: AtomicU64,
    visibility_ns: AtomicU64,
    cache_ns: AtomicU64,
    prepare_ns: AtomicU64,
    h2d_ns: AtomicU64,
    gpu_d2h_ns: AtomicU64,
    command_record_ns: AtomicU64,
    queue_submit_ns: AtomicU64,
    fence_wait_ns: AtomicU64,
    download_ns: AtomicU64,
    cpu_topk_ns: AtomicU64,
    resident_candidate_ns: AtomicU64,
    resident_visibility_ns: AtomicU64,
    postprocess_ns: AtomicU64,
    filter_cache_hits: AtomicU64,
    filter_cache_misses: AtomicU64,
    resident_candidate_hits: AtomicU64,
    resident_candidate_misses: AtomicU64,
    resident_visibility_hits: AtomicU64,
    resident_visibility_misses: AtomicU64,
    context_attempts: AtomicU64,
    context_unavailable: AtomicU64,
    submission_batches: AtomicU64,
    submission_queries: AtomicU64,
    coalesced_batches: AtomicU64,
    coalesced_queries: AtomicU64,
    coalescing_wait_ns: AtomicU64,
}

impl GpuExactSearchStats {
    fn add(atomic: &AtomicU64, value: u64) {
        atomic.fetch_add(value, Ordering::Relaxed);
    }

    fn record_inner(
        &self,
        query_count: usize,
        candidates: usize,
        prepare_ns: u64,
        h2d_ns: u64,
        gpu_d2h_ns: u64,
        command_record_ns: u64,
        queue_submit_ns: u64,
        fence_wait_ns: u64,
        download_ns: u64,
        cpu_topk_ns: u64,
        resident_candidate_ns: u64,
        resident_visibility_ns: u64,
        resident_candidate_hit: Option<bool>,
        resident_visibility_hit: Option<bool>,
    ) {
        Self::add(
            &self.candidates,
            (candidates as u64).saturating_mul(query_count as u64),
        );
        Self::add(&self.prepare_ns, prepare_ns);
        Self::add(&self.h2d_ns, h2d_ns);
        Self::add(&self.gpu_d2h_ns, gpu_d2h_ns);
        Self::add(&self.command_record_ns, command_record_ns);
        Self::add(&self.queue_submit_ns, queue_submit_ns);
        Self::add(&self.fence_wait_ns, fence_wait_ns);
        Self::add(&self.download_ns, download_ns);
        Self::add(&self.cpu_topk_ns, cpu_topk_ns);
        Self::add(&self.resident_candidate_ns, resident_candidate_ns);
        Self::add(&self.resident_visibility_ns, resident_visibility_ns);
        match resident_candidate_hit {
            Some(true) => Self::add(&self.resident_candidate_hits, query_count as u64),
            Some(false) => Self::add(&self.resident_candidate_misses, query_count as u64),
            None => {}
        }
        match resident_visibility_hit {
            Some(true) => Self::add(&self.resident_visibility_hits, query_count as u64),
            Some(false) => Self::add(&self.resident_visibility_misses, query_count as u64),
            None => {}
        }
        Self::add(&self.submission_batches, 1);
        Self::add(&self.submission_queries, query_count as u64);
    }

    fn record_outer(
        &self,
        query_count: usize,
        predicate_ns: u64,
        visibility_ns: u64,
        cache_ns: u64,
        postprocess_ns: u64,
        filter_cache_hit: Option<bool>,
    ) {
        Self::add(&self.predicate_ns, predicate_ns);
        Self::add(&self.visibility_ns, visibility_ns);
        Self::add(&self.cache_ns, cache_ns);
        Self::add(&self.postprocess_ns, postprocess_ns);
        match filter_cache_hit {
            Some(true) => Self::add(&self.filter_cache_hits, query_count as u64),
            Some(false) => Self::add(&self.filter_cache_misses, query_count as u64),
            None => {}
        }

        let query_count = query_count.max(1) as u64;
        let previous_sequence = self.total_queries.fetch_add(query_count, Ordering::Relaxed);
        let sequence = previous_sequence + query_count;
        self.window_queries
            .fetch_add(query_count, Ordering::Relaxed);
        if previous_sequence != 0 && previous_sequence / 1_024 == sequence / 1_024 {
            return;
        }

        let queries = self.window_queries.swap(0, Ordering::Relaxed).max(1);
        let take = |value: &AtomicU64| value.swap(0, Ordering::Relaxed);
        let candidates = take(&self.candidates);
        let predicate_ns = take(&self.predicate_ns);
        let visibility_ns = take(&self.visibility_ns);
        let cache_ns = take(&self.cache_ns);
        let prepare_ns = take(&self.prepare_ns);
        let h2d_ns = take(&self.h2d_ns);
        let gpu_d2h_ns = take(&self.gpu_d2h_ns);
        let command_record_ns = take(&self.command_record_ns);
        let queue_submit_ns = take(&self.queue_submit_ns);
        let fence_wait_ns = take(&self.fence_wait_ns);
        let download_ns = take(&self.download_ns);
        let cpu_topk_ns = take(&self.cpu_topk_ns);
        let resident_candidate_ns = take(&self.resident_candidate_ns);
        let resident_visibility_ns = take(&self.resident_visibility_ns);
        let postprocess_ns = take(&self.postprocess_ns);
        let filter_cache_hits = take(&self.filter_cache_hits);
        let filter_cache_misses = take(&self.filter_cache_misses);
        let filter_cache_lookups = filter_cache_hits + filter_cache_misses;
        let filter_cache_hit_rate = if filter_cache_lookups == 0 {
            0.0
        } else {
            filter_cache_hits as f64 / filter_cache_lookups as f64
        };
        let resident_candidate_hits = take(&self.resident_candidate_hits);
        let resident_candidate_misses = take(&self.resident_candidate_misses);
        let resident_candidate_lookups = resident_candidate_hits + resident_candidate_misses;
        let resident_candidate_hit_rate = if resident_candidate_lookups == 0 {
            0.0
        } else {
            resident_candidate_hits as f64 / resident_candidate_lookups as f64
        };
        let resident_visibility_hits = take(&self.resident_visibility_hits);
        let resident_visibility_misses = take(&self.resident_visibility_misses);
        let resident_visibility_lookups = resident_visibility_hits + resident_visibility_misses;
        let resident_visibility_hit_rate = if resident_visibility_lookups == 0 {
            0.0
        } else {
            resident_visibility_hits as f64 / resident_visibility_lookups as f64
        };
        let context_attempts = take(&self.context_attempts);
        let context_unavailable = take(&self.context_unavailable);
        let context_unavailable_rate = if context_attempts == 0 {
            0.0
        } else {
            context_unavailable as f64 / context_attempts as f64
        };
        let submission_batches = take(&self.submission_batches);
        let submission_queries = take(&self.submission_queries);
        let average_submission_batch = if submission_batches == 0 {
            0.0
        } else {
            submission_queries as f64 / submission_batches as f64
        };
        let coalesced_batches = take(&self.coalesced_batches);
        let coalesced_queries = take(&self.coalesced_queries);
        let average_coalesced_batch = if coalesced_batches == 0 {
            0.0
        } else {
            coalesced_queries as f64 / coalesced_batches as f64
        };
        let coalescing_wait_ns = take(&self.coalescing_wait_ns);
        let average_coalescing_wait_us = if coalesced_batches == 0 {
            0.0
        } else {
            coalescing_wait_ns as f64 / coalesced_batches as f64 / 1_000.0
        };
        let average_us = |value: u64| value as f64 / queries as f64 / 1_000.0;

        log::info!(
            "GPU exact filtered breakdown: sequence={sequence}, window_queries={queries}, \
             avg_candidates={:.1}, predicate_us={:.3}, visibility_us={:.3}, cache_us={:.3}, \
             prepare_us={:.3}, h2d_us={:.3}, gpu_d2h_us={:.3}, command_record_us={:.3}, \
             queue_submit_us={:.3}, fence_wait_us={:.3}, mapped_download_us={:.3}, \
             cpu_topk_us={:.3}, resident_candidate_us={:.3}, resident_visibility_us={:.3}, \
             postprocess_us={:.3}, filter_cache_hit_rate={:.3}, \
             resident_candidate_hit_rate={:.3}, resident_visibility_hit_rate={:.3}, \
             context_attempts={context_attempts}, context_unavailable={context_unavailable}, \
             context_unavailable_rate={context_unavailable_rate:.3}, \
             submission_batches={submission_batches}, submission_queries={submission_queries}, \
             average_submission_batch={average_submission_batch:.3}, \
             coalesced_batches={coalesced_batches}, coalesced_queries={coalesced_queries}, \
             average_coalesced_batch={average_coalesced_batch:.3}, \
             average_coalescing_wait_us={average_coalescing_wait_us:.3}",
            candidates as f64 / queries as f64,
            average_us(predicate_ns),
            average_us(visibility_ns),
            average_us(cache_ns),
            average_us(prepare_ns),
            average_us(h2d_ns),
            average_us(gpu_d2h_ns),
            average_us(command_record_ns),
            average_us(queue_submit_ns),
            average_us(fence_wait_ns),
            average_us(download_ns),
            average_us(cpu_topk_ns),
            average_us(resident_candidate_ns),
            average_us(resident_visibility_ns),
            average_us(postprocess_ns),
            filter_cache_hit_rate,
            resident_candidate_hit_rate,
            resident_visibility_hit_rate,
        );
    }
}

struct GpuResidentCandidateEntry {
    candidates: Arc<Vec<PointOffsetType>>,
    buffer: Arc<gpu::Buffer>,
}

#[derive(Default)]
struct GpuResidentCandidateCache {
    entries: HashMap<usize, GpuResidentCandidateEntry>,
    cached_bytes: usize,
}

impl GpuResidentCandidateCache {
    fn clear(&mut self) {
        self.entries.clear();
        self.cached_bytes = 0;
    }

    fn get(&self, candidates: &Arc<Vec<PointOffsetType>>) -> Option<Arc<gpu::Buffer>> {
        let key = Arc::as_ptr(candidates) as usize;
        self.entries.get(&key).and_then(|entry| {
            Arc::ptr_eq(&entry.candidates, candidates).then(|| entry.buffer.clone())
        })
    }

    fn insert(&mut self, candidates: Arc<Vec<PointOffsetType>>, buffer: Arc<gpu::Buffer>) {
        let candidate_bytes = candidates
            .len()
            .saturating_mul(std::mem::size_of::<PointOffsetType>());
        if candidate_bytes > FILTER_CACHE_MAX_BYTES {
            return;
        }
        if self.entries.len() >= FILTER_CACHE_MAX_ENTRIES
            || self.cached_bytes.saturating_add(candidate_bytes) > FILTER_CACHE_MAX_BYTES
        {
            self.clear();
        }
        let key = Arc::as_ptr(&candidates) as usize;
        if self.entries.contains_key(&key) {
            return;
        }
        self.cached_bytes = self.cached_bytes.saturating_add(candidate_bytes);
        self.entries
            .insert(key, GpuResidentCandidateEntry { candidates, buffer });
    }
}

struct GpuResidentVisibilityEntry {
    generation: u64,
    bit_len: usize,
    buffer: Arc<gpu::Buffer>,
}

#[derive(Default)]
struct GpuResidentVisibilityCache {
    entry: Option<GpuResidentVisibilityEntry>,
}

impl GpuResidentVisibilityCache {
    fn clear(&mut self) {
        self.entry = None;
    }

    fn get(&self, generation: u64, bit_len: usize) -> Option<Arc<gpu::Buffer>> {
        self.entry.as_ref().and_then(|entry| {
            (entry.generation == generation && entry.bit_len == bit_len)
                .then(|| entry.buffer.clone())
        })
    }

    fn insert(&mut self, generation: u64, bit_len: usize, buffer: Arc<gpu::Buffer>) {
        self.entry = Some(GpuResidentVisibilityEntry {
            generation,
            bit_len,
            buffer,
        });
    }
}

fn pack_deleted_words(deleted: &BitSlice, point_count: usize) -> Vec<u32> {
    debug_assert!(deleted.len() <= point_count);
    let mut words = vec![0u32; point_count.div_ceil(u32::BITS as usize)];
    for index in deleted.iter_ones() {
        words[index / u32::BITS as usize] |= 1u32 << (index % u32::BITS as usize);
    }
    if let Some(last) = words.last_mut() {
        let valid_bits = point_count % u32::BITS as usize;
        if valid_bits != 0 {
            *last |= !((1u32 << valid_bits) - 1);
        }
    }
    words
}

struct GpuExactSearchContext {
    context: gpu::Context,
    score_pipeline: Arc<gpu::Pipeline>,
    block_topk_pipeline: Arc<gpu::Pipeline>,
    visible_score_pipeline: Arc<gpu::Pipeline>,
    visible_block_topk_pipeline: Arc<gpu::Pipeline>,
    descriptor_set: Arc<gpu::DescriptorSet>,
    descriptor_set_layout: Arc<gpu::DescriptorSetLayout>,
    visibility_descriptor_set_layout: Arc<gpu::DescriptorSetLayout>,
    vector_storage: Arc<GpuVectorStorage>,
    candidate_buffer: Arc<gpu::Buffer>,
    candidate_staging: Arc<gpu::Buffer>,
    visibility_staging: Arc<gpu::Buffer>,
    visibility_capacity_words: usize,
    query_buffer: Arc<gpu::Buffer>,
    query_staging: Arc<gpu::Buffer>,
    score_buffer: Arc<gpu::Buffer>,
    score_staging: Arc<gpu::Buffer>,
    stats: Arc<GpuExactSearchStats>,
    candidate_capacity: usize,
    query_capacity: usize,
}

impl GpuExactSearchContext {
    fn new(
        vector_storage: Arc<GpuVectorStorage>,
        score_pipeline: Arc<gpu::Pipeline>,
        block_topk_pipeline: Arc<gpu::Pipeline>,
        visible_score_pipeline: Arc<gpu::Pipeline>,
        visible_block_topk_pipeline: Arc<gpu::Pipeline>,
        descriptor_set_layout: Arc<gpu::DescriptorSetLayout>,
        visibility_descriptor_set_layout: Arc<gpu::DescriptorSetLayout>,
        candidate_capacity: usize,
        queue_index: usize,
        stats: Arc<GpuExactSearchStats>,
    ) -> OperationResult<Self> {
        let device = vector_storage.device();
        let candidate_bytes = candidate_capacity * std::mem::size_of::<PointOffsetType>();
        let scalar_score_bytes = candidate_capacity * std::mem::size_of::<f32>();
        let batched_block_score_bytes = candidate_capacity
            .div_ceil(BLOCK_TOPK_TILE_CANDIDATES)
            .saturating_mul(BLOCK_TOPK_LIMIT)
            .saturating_mul(std::mem::size_of::<ScoredPointOffset>())
            .saturating_mul(GPU_EXACT_SUBMISSION_BATCH_LIMIT);
        let score_bytes = scalar_score_bytes.max(batched_block_score_bytes);
        let query_capacity = vector_storage.vector_capacity();
        let query_bytes = query_capacity
            .saturating_mul(GPU_EXACT_SUBMISSION_BATCH_LIMIT)
            .saturating_mul(std::mem::size_of::<f32>());
        let visibility_capacity_words = vector_storage
            .num_vectors()
            .div_ceil(u32::BITS as usize)
            .max(1);
        let visibility_bytes = visibility_capacity_words * std::mem::size_of::<u32>();

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
        let visibility_staging = gpu::Buffer::new(
            device.clone(),
            "Exact filtered visibility upload",
            gpu::BufferType::CpuToGpu,
            visibility_bytes,
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
        let descriptor_set = gpu::DescriptorSet::builder(descriptor_set_layout.clone())
            .add_storage_buffer(0, candidate_buffer.clone())
            .add_storage_buffer(1, query_buffer.clone())
            .add_storage_buffer(2, score_buffer.clone())
            .build()?;

        Ok(Self {
            context: gpu::Context::new_with_queue_index(device, queue_index)?,
            score_pipeline,
            block_topk_pipeline,
            visible_score_pipeline,
            visible_block_topk_pipeline,
            descriptor_set,
            descriptor_set_layout,
            visibility_descriptor_set_layout,
            vector_storage,
            candidate_buffer,
            candidate_staging,
            visibility_staging,
            visibility_capacity_words,
            query_buffer,
            query_staging,
            score_buffer,
            score_staging,
            stats,
            candidate_capacity,
            query_capacity,
        })
    }

    fn search_batch(
        &mut self,
        queries: &[&[f32]],
        candidates: &[PointOffsetType],
        top: usize,
        resident_candidate_buffer: Option<Arc<gpu::Buffer>>,
        resident_candidate_hit: Option<bool>,
        visibility_buffer: Option<Arc<gpu::Buffer>>,
        resident_visibility_hit: Option<bool>,
        resident_candidate_ns: u64,
        resident_visibility_ns: u64,
    ) -> OperationResult<Vec<Vec<ScoredPointOffset>>> {
        if candidates.len() > self.candidate_capacity {
            return Err(OperationError::service_error(
                "GPU exact candidate capacity exceeded",
            ));
        }
        if queries.is_empty() || queries.len() > GPU_EXACT_SUBMISSION_BATCH_LIMIT {
            return Err(OperationError::service_error(
                "GPU exact query batch capacity exceeded",
            ));
        }
        if queries
            .iter()
            .any(|query| query.len() != self.vector_storage.dim())
        {
            return Err(OperationError::service_error(
                "GPU exact query dimension mismatch",
            ));
        }
        if candidates.is_empty() || top == 0 {
            return Ok((0..queries.len()).map(|_| Vec::new()).collect());
        }

        let prepare_started = std::time::Instant::now();
        let mut padded_queries = vec![0.0f32; self.query_capacity * queries.len()];
        for (query_index, query) in queries.iter().enumerate() {
            let start = query_index * self.query_capacity;
            padded_queries[start..start + query.len()].copy_from_slice(query);
        }
        if resident_candidate_buffer.is_none() {
            self.candidate_staging.upload(candidates, 0)?;
        }
        self.query_staging.upload(padded_queries.as_slice(), 0)?;
        let has_visibility = visibility_buffer.is_some();
        let descriptor_set = match (resident_candidate_buffer, visibility_buffer) {
            (None, None) => self.descriptor_set.clone(),
            (candidate_buffer, None) => {
                gpu::DescriptorSet::builder(self.descriptor_set_layout.clone())
                    .add_storage_buffer(
                        0,
                        candidate_buffer.unwrap_or_else(|| self.candidate_buffer.clone()),
                    )
                    .add_storage_buffer(1, self.query_buffer.clone())
                    .add_storage_buffer(2, self.score_buffer.clone())
                    .build()?
            }
            (candidate_buffer, Some(visibility_buffer)) => {
                gpu::DescriptorSet::builder(self.visibility_descriptor_set_layout.clone())
                    .add_storage_buffer(
                        0,
                        candidate_buffer.unwrap_or_else(|| self.candidate_buffer.clone()),
                    )
                    .add_storage_buffer(1, self.query_buffer.clone())
                    .add_storage_buffer(2, self.score_buffer.clone())
                    .add_storage_buffer(3, visibility_buffer)
                    .build()?
            }
        };
        let use_block_topk = resident_candidate_hit.is_some()
            && candidates.len() >= BLOCK_TOPK_MIN_CANDIDATES
            && top <= BLOCK_TOPK_LIMIT;
        if queries.len() > 1 && !use_block_topk {
            return Err(OperationError::service_error(
                "GPU exact multi-query submission requires block top-k",
            ));
        }
        let block_count = candidates.len().div_ceil(BLOCK_TOPK_TILE_CANDIDATES);
        let output_count_per_query = if use_block_topk {
            block_count * BLOCK_TOPK_LIMIT
        } else {
            candidates.len()
        };
        let output_count = output_count_per_query * queries.len();
        let output_bytes = if use_block_topk {
            output_count * std::mem::size_of::<ScoredPointOffset>()
        } else {
            output_count * std::mem::size_of::<f32>()
        };
        let prepare_ns = prepare_started.elapsed().as_nanos() as u64;

        let gpu_submit_started = std::time::Instant::now();
        let command_record_started = std::time::Instant::now();
        let mut uploaded_buffers = vec![self.query_buffer.clone()];
        if resident_candidate_hit.is_none() {
            self.context.copy_gpu_buffer(
                self.candidate_staging.clone(),
                self.candidate_buffer.clone(),
                0,
                0,
                std::mem::size_of_val(candidates),
            )?;
            uploaded_buffers.push(self.candidate_buffer.clone());
        }
        self.context.copy_gpu_buffer(
            self.query_staging.clone(),
            self.query_buffer.clone(),
            0,
            0,
            padded_queries.len() * std::mem::size_of::<f32>(),
        )?;
        self.context.barrier_buffers(&uploaded_buffers)?;
        let pipeline = match (has_visibility, use_block_topk) {
            (false, false) => self.score_pipeline.clone(),
            (false, true) => self.block_topk_pipeline.clone(),
            (true, false) => self.visible_score_pipeline.clone(),
            (true, true) => self.visible_block_topk_pipeline.clone(),
        };
        self.context.bind_pipeline(
            pipeline,
            &[descriptor_set, self.vector_storage.descriptor_set()],
        )?;
        self.context.dispatch(
            if use_block_topk {
                block_count
            } else {
                candidates.len()
            },
            queries.len(),
            1,
        )?;
        self.context
            .barrier_buffers(std::slice::from_ref(&self.score_buffer))?;
        self.context.copy_gpu_buffer(
            self.score_buffer.clone(),
            self.score_staging.clone(),
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
        let h2d_ns = 0;
        let gpu_d2h_ns = gpu_submit_started.elapsed().as_nanos() as u64;

        let download_started = std::time::Instant::now();
        let block_results = use_block_topk
            .then(|| {
                self.score_staging
                    .download_vec::<ScoredPointOffset>(0, output_count)
            })
            .transpose()?;
        let scores = (!use_block_topk)
            .then(|| self.score_staging.download_vec::<f32>(0, output_count))
            .transpose()?;
        let download_ns = download_started.elapsed().as_nanos() as u64;
        let cpu_topk_started = std::time::Instant::now();
        let mut results = Vec::with_capacity(queries.len());
        if let Some(block_results) = block_results {
            for query_results in block_results.chunks_exact(output_count_per_query) {
                let mut queue = TopK::new(top);
                for &point in query_results {
                    if point.idx != PointOffsetType::MAX {
                        queue.push(point);
                    }
                }
                results.push(queue.into_vec());
            }
        } else if let Some(scores) = scores {
            debug_assert_eq!(queries.len(), 1);
            let mut queue = TopK::new(top);
            for (&idx, score) in candidates.iter().zip(scores) {
                queue.push(ScoredPointOffset { idx, score });
            }
            results.push(queue.into_vec());
        }
        let cpu_topk_ns = cpu_topk_started.elapsed().as_nanos() as u64;
        self.stats.record_inner(
            queries.len(),
            candidates.len(),
            prepare_ns,
            h2d_ns,
            gpu_d2h_ns,
            command_record_ns,
            queue_submit_ns,
            fence_wait_ns,
            download_ns,
            cpu_topk_ns,
            resident_candidate_ns,
            resident_visibility_ns,
            resident_candidate_hit,
            resident_visibility_hit,
        );
        Ok(results)
    }

    fn upload_resident_candidates(
        &mut self,
        candidates: &[PointOffsetType],
    ) -> OperationResult<Arc<gpu::Buffer>> {
        let candidate_bytes = std::mem::size_of_val(candidates);
        let candidate_buffer = gpu::Buffer::new(
            self.vector_storage.device(),
            "Resident exact filtered candidate IDs",
            gpu::BufferType::Storage,
            candidate_bytes,
        )?;
        self.candidate_staging.upload(candidates, 0)?;
        self.context.copy_gpu_buffer(
            self.candidate_staging.clone(),
            candidate_buffer.clone(),
            0,
            0,
            candidate_bytes,
        )?;
        self.context.run()?;
        self.context.wait_finish(GPU_TIMEOUT)?;
        Ok(candidate_buffer)
    }

    fn upload_resident_visibility(&mut self, words: &[u32]) -> OperationResult<Arc<gpu::Buffer>> {
        if words.is_empty() || words.len() > self.visibility_capacity_words {
            return Err(OperationError::service_error(
                "GPU visibility word capacity exceeded",
            ));
        }
        let visibility_bytes = std::mem::size_of_val(words);
        let visibility_buffer = gpu::Buffer::new(
            self.vector_storage.device(),
            "Resident exact filtered visibility",
            gpu::BufferType::Storage,
            visibility_bytes,
        )?;
        self.visibility_staging.upload(words, 0)?;
        self.context.copy_gpu_buffer(
            self.visibility_staging.clone(),
            visibility_buffer.clone(),
            0,
            0,
            visibility_bytes,
        )?;
        self.context.run()?;
        self.context.wait_finish(GPU_TIMEOUT)?;
        Ok(visibility_buffer)
    }
}

type GpuSubmissionResponse = OperationResult<Option<Vec<ScoredPointOffset>>>;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct GpuSubmissionBatchKey {
    candidates: usize,
    candidate_buffer: usize,
    visibility_buffer: usize,
    top: usize,
}

struct PendingGpuQuery {
    query: Vec<f32>,
    resident_candidate_ns: u64,
    resident_visibility_ns: u64,
    response: SyncSender<GpuSubmissionResponse>,
}

struct PendingGpuSubmission {
    id: u64,
    candidates: Arc<Vec<PointOffsetType>>,
    candidate_buffer: Arc<gpu::Buffer>,
    visibility_buffer: Option<Arc<gpu::Buffer>>,
    queries: Vec<PendingGpuQuery>,
}

#[derive(Default)]
struct GpuSubmissionBatchState {
    next_id: u64,
    pending: HashMap<GpuSubmissionBatchKey, VecDeque<PendingGpuSubmission>>,
}

pub struct GpuExactSearchCache {
    contexts: Mutex<Vec<GpuExactSearchContext>>,
    resident_candidates: Mutex<GpuResidentCandidateCache>,
    resident_visibility: Mutex<GpuResidentVisibilityCache>,
    submission_batches: Mutex<GpuSubmissionBatchState>,
    submission_ready: Condvar,
    candidate_capacity: usize,
    context_count: usize,
    batch_max_queries: usize,
    batch_window: std::time::Duration,
    stats: Arc<GpuExactSearchStats>,
}

impl fmt::Debug for GpuExactSearchCache {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GpuExactSearchCache")
            .field("candidate_capacity", &self.candidate_capacity)
            .field("context_count", &self.context_count)
            .field("batch_max_queries", &self.batch_max_queries)
            .field("batch_window", &self.batch_window)
            .finish_non_exhaustive()
    }
}

impl GpuExactSearchCache {
    pub fn supports_submission_batch(
        candidate_count: usize,
        top: usize,
        reuse_candidate_buffer: bool,
    ) -> bool {
        reuse_candidate_buffer
            && candidate_count >= BLOCK_TOPK_MIN_CANDIDATES
            && top <= BLOCK_TOPK_LIMIT
    }

    fn execute_resident_batch(
        &self,
        queries: &[&[f32]],
        candidates: &[PointOffsetType],
        top: usize,
        candidate_buffer: Arc<gpu::Buffer>,
        visibility_buffer: Option<Arc<gpu::Buffer>>,
        resident_candidate_ns: u64,
        resident_visibility_ns: u64,
    ) -> OperationResult<Option<Vec<Vec<ScoredPointOffset>>>> {
        GpuExactSearchStats::add(&self.stats.context_attempts, 1);
        let Some(mut context) = self.contexts.lock().pop() else {
            GpuExactSearchStats::add(&self.stats.context_unavailable, 1);
            return Ok(None);
        };
        let resident_visibility_hit = visibility_buffer.as_ref().map(|_| true);
        let result = context.search_batch(
            queries,
            candidates,
            top,
            Some(candidate_buffer),
            Some(true),
            visibility_buffer,
            resident_visibility_hit,
            resident_candidate_ns,
            resident_visibility_ns,
        );
        self.contexts.lock().push(context);
        result.map(Some)
    }

    fn lead_coalesced_submission(&self, key: GpuSubmissionBatchKey, batch_id: u64) {
        let wait_started = std::time::Instant::now();
        let submission = {
            let mut state = self.submission_batches.lock();
            loop {
                let query_count = state
                    .pending
                    .get(&key)
                    .and_then(|batches| batches.iter().find(|batch| batch.id == batch_id))
                    .map(|batch| batch.queries.len())
                    .expect("GPU submission leader lost its pending batch");
                let elapsed = wait_started.elapsed();
                if query_count >= self.batch_max_queries || elapsed >= self.batch_window {
                    break;
                }
                self.submission_ready
                    .wait_for(&mut state, self.batch_window - elapsed);
            }

            let (submission, remove_key) = {
                let batches = state
                    .pending
                    .get_mut(&key)
                    .expect("GPU submission batch key disappeared");
                let position = batches
                    .iter()
                    .position(|batch| batch.id == batch_id)
                    .expect("GPU submission batch ID disappeared");
                let submission = batches
                    .remove(position)
                    .expect("GPU submission batch removal failed");
                (submission, batches.is_empty())
            };
            if remove_key {
                state.pending.remove(&key);
            }
            submission
        };

        let wait_ns = wait_started.elapsed().as_nanos() as u64;
        GpuExactSearchStats::add(&self.stats.coalesced_batches, 1);
        GpuExactSearchStats::add(
            &self.stats.coalesced_queries,
            submission.queries.len() as u64,
        );
        GpuExactSearchStats::add(&self.stats.coalescing_wait_ns, wait_ns);

        let resident_candidate_ns = submission
            .queries
            .iter()
            .map(|query| query.resident_candidate_ns)
            .sum();
        let resident_visibility_ns = submission
            .queries
            .iter()
            .map(|query| query.resident_visibility_ns)
            .sum();
        let query_refs = submission
            .queries
            .iter()
            .map(|query| query.query.as_slice())
            .collect::<Vec<_>>();
        let result = self.execute_resident_batch(
            query_refs.as_slice(),
            submission.candidates.as_slice(),
            key.top,
            submission.candidate_buffer,
            submission.visibility_buffer,
            resident_candidate_ns,
            resident_visibility_ns,
        );

        match result {
            Ok(Some(results)) if results.len() == submission.queries.len() => {
                for (query, result) in submission.queries.into_iter().zip(results) {
                    let _ = query.response.send(Ok(Some(result)));
                }
            }
            Ok(Some(results)) => {
                let message = format!(
                    "GPU exact batch returned {} results for {} queries",
                    results.len(),
                    submission.queries.len(),
                );
                for query in submission.queries {
                    let _ = query
                        .response
                        .send(Err(OperationError::service_error(message.clone())));
                }
            }
            Ok(None) => {
                for query in submission.queries {
                    let _ = query.response.send(Ok(None));
                }
            }
            Err(error) => {
                let message = format!("GPU exact coalesced submission failed: {error}");
                for query in submission.queries {
                    let _ = query
                        .response
                        .send(Err(OperationError::service_error(message.clone())));
                }
            }
        }
    }

    pub fn search_coalesced(
        &self,
        query: &[f32],
        candidates: Arc<Vec<PointOffsetType>>,
        top: usize,
        reuse_candidate_buffer: bool,
        visibility: Option<GpuVisibilitySnapshot<'_>>,
    ) -> OperationResult<Option<Vec<ScoredPointOffset>>> {
        if self.batch_max_queries <= 1
            || self.batch_window.is_zero()
            || !Self::supports_submission_batch(candidates.len(), top, reuse_candidate_buffer)
        {
            return self.search(query, candidates, top, reuse_candidate_buffer, visibility);
        }

        let resident_candidate_started = std::time::Instant::now();
        let candidate_buffer = self.resident_candidates.lock().get(&candidates);
        let resident_candidate_ns = resident_candidate_started.elapsed().as_nanos() as u64;
        let Some(candidate_buffer) = candidate_buffer else {
            return self.search(query, candidates, top, reuse_candidate_buffer, visibility);
        };

        let resident_visibility_started = std::time::Instant::now();
        let visibility_buffer = if let Some(visibility) = visibility {
            if visibility.generation == 0
                || visibility.point_count == 0
                || visibility.deleted.len() > visibility.point_count
            {
                return Err(OperationError::service_error(
                    "GPU visibility snapshot identity is invalid",
                ));
            }
            self.resident_visibility
                .lock()
                .get(visibility.generation, visibility.point_count)
        } else {
            None
        };
        let resident_visibility_ns = resident_visibility_started.elapsed().as_nanos() as u64;
        if visibility.is_some() && visibility_buffer.is_none() {
            return self.search(query, candidates, top, reuse_candidate_buffer, visibility);
        }

        let key = GpuSubmissionBatchKey {
            candidates: Arc::as_ptr(&candidates) as usize,
            candidate_buffer: Arc::as_ptr(&candidate_buffer) as usize,
            visibility_buffer: visibility_buffer
                .as_ref()
                .map_or(0, |buffer| Arc::as_ptr(buffer) as usize),
            top,
        };
        let (sender, receiver) = sync_channel(1);
        let mut pending_query = Some(PendingGpuQuery {
            query: query.to_vec(),
            resident_candidate_ns,
            resident_visibility_ns,
            response: sender,
        });
        let (batch_id, is_leader) = {
            let mut state = self.submission_batches.lock();
            if let Some(batch) = state
                .pending
                .get_mut(&key)
                .and_then(|batches| batches.back_mut())
                .filter(|batch| batch.queries.len() < self.batch_max_queries)
            {
                batch
                    .queries
                    .push(pending_query.take().expect("pending GPU query missing"));
                self.submission_ready.notify_all();
                (batch.id, false)
            } else {
                state.next_id = state.next_id.wrapping_add(1).max(1);
                let batch_id = state.next_id;
                state
                    .pending
                    .entry(key)
                    .or_default()
                    .push_back(PendingGpuSubmission {
                        id: batch_id,
                        candidates,
                        candidate_buffer,
                        visibility_buffer,
                        queries: vec![pending_query.take().expect("pending GPU query missing")],
                    });
                (batch_id, true)
            }
        };

        if is_leader {
            self.lead_coalesced_submission(key, batch_id);
        }
        receiver.recv().map_err(|error| {
            OperationError::service_error(format!(
                "GPU exact coalesced response channel closed: {error}",
            ))
        })?
    }

    pub fn clear_resident_candidates(&self) {
        self.resident_candidates.lock().clear();
        self.resident_visibility.lock().clear();
    }

    pub fn new(
        device: Arc<gpu::Device>,
        vector_storage: &crate::vector_storage::VectorStorageEnum,
        candidate_capacity: usize,
        context_count: usize,
        batch_max_queries: usize,
        batch_window_us: usize,
        stopped: &std::sync::atomic::AtomicBool,
    ) -> OperationResult<Self> {
        if candidate_capacity == 0
            || context_count == 0
            || batch_max_queries == 0
            || batch_max_queries > GPU_EXACT_SUBMISSION_BATCH_LIMIT
        {
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
        let visibility_descriptor_set_layout = gpu::DescriptorSetLayout::builder()
            .add_storage_buffer(0)
            .add_storage_buffer(1)
            .add_storage_buffer(2)
            .add_storage_buffer(3)
            .build(device.clone())?;
        let score_shader = ShaderBuilder::new(device.clone())
            .with_shader_code(include_str!("shaders/run_exact_filtered_score.comp"))
            .with_parameters(vector_storage.as_ref())
            .build("run_exact_filtered_score.comp")?;
        let score_pipeline = gpu::Pipeline::builder()
            .add_descriptor_set_layout(0, descriptor_set_layout.clone())
            .add_descriptor_set_layout(1, vector_storage.descriptor_set_layout())
            .add_shader(score_shader)
            .build(device.clone())?;
        let block_topk_shader = ShaderBuilder::new(device.clone())
            .with_shader_code(include_str!("shaders/run_exact_filtered_block_topk.comp"))
            .with_parameters(vector_storage.as_ref())
            .build("run_exact_filtered_block_topk.comp")?;
        let block_topk_pipeline = gpu::Pipeline::builder()
            .add_descriptor_set_layout(0, descriptor_set_layout.clone())
            .add_descriptor_set_layout(1, vector_storage.descriptor_set_layout())
            .add_shader(block_topk_shader)
            .build(device.clone())?;
        let visible_score_shader = ShaderBuilder::new(device.clone())
            .with_shader_code(include_str!(
                "shaders/run_exact_filtered_visible_score.comp"
            ))
            .with_parameters(vector_storage.as_ref())
            .build("run_exact_filtered_visible_score.comp")?;
        let visible_score_pipeline = gpu::Pipeline::builder()
            .add_descriptor_set_layout(0, visibility_descriptor_set_layout.clone())
            .add_descriptor_set_layout(1, vector_storage.descriptor_set_layout())
            .add_shader(visible_score_shader)
            .build(device.clone())?;
        let visible_block_topk_shader = ShaderBuilder::new(device.clone())
            .with_shader_code(include_str!(
                "shaders/run_exact_filtered_visible_block_topk.comp"
            ))
            .with_parameters(vector_storage.as_ref())
            .build("run_exact_filtered_visible_block_topk.comp")?;
        let visible_block_topk_pipeline = gpu::Pipeline::builder()
            .add_descriptor_set_layout(0, visibility_descriptor_set_layout.clone())
            .add_descriptor_set_layout(1, vector_storage.descriptor_set_layout())
            .add_shader(visible_block_topk_shader)
            .build(device)?;

        let stats = Arc::new(GpuExactSearchStats::default());
        let contexts = (0..context_count)
            .map(|queue_index| {
                GpuExactSearchContext::new(
                    vector_storage.clone(),
                    score_pipeline.clone(),
                    block_topk_pipeline.clone(),
                    visible_score_pipeline.clone(),
                    visible_block_topk_pipeline.clone(),
                    descriptor_set_layout.clone(),
                    visibility_descriptor_set_layout.clone(),
                    candidate_capacity,
                    queue_index,
                    stats.clone(),
                )
            })
            .collect::<OperationResult<Vec<_>>>()?;

        Ok(Self {
            contexts: Mutex::new(contexts),
            resident_candidates: Mutex::new(GpuResidentCandidateCache::default()),
            resident_visibility: Mutex::new(GpuResidentVisibilityCache::default()),
            submission_batches: Mutex::new(GpuSubmissionBatchState::default()),
            submission_ready: Condvar::new(),
            candidate_capacity,
            context_count,
            batch_max_queries,
            batch_window: std::time::Duration::from_micros(batch_window_us as u64),
            stats,
        })
    }

    pub fn search(
        &self,
        query: &[f32],
        candidates: Arc<Vec<PointOffsetType>>,
        top: usize,
        reuse_candidate_buffer: bool,
        visibility: Option<GpuVisibilitySnapshot<'_>>,
    ) -> OperationResult<Option<Vec<ScoredPointOffset>>> {
        let queries = [query];
        self.search_batch(
            &queries,
            candidates,
            top,
            reuse_candidate_buffer,
            visibility,
        )
        .map(|batch| batch.map(|mut results| results.pop().unwrap_or_default()))
    }

    pub fn search_batch(
        &self,
        queries: &[&[f32]],
        candidates: Arc<Vec<PointOffsetType>>,
        top: usize,
        reuse_candidate_buffer: bool,
        visibility: Option<GpuVisibilitySnapshot<'_>>,
    ) -> OperationResult<Option<Vec<Vec<ScoredPointOffset>>>> {
        GpuExactSearchStats::add(&self.stats.context_attempts, 1);
        let Some(mut context) = self.contexts.lock().pop() else {
            GpuExactSearchStats::add(&self.stats.context_unavailable, 1);
            return Ok(None);
        };
        let result = (|| {
            let resident_candidate_started = std::time::Instant::now();
            let (resident_candidate_buffer, resident_candidate_hit) = if reuse_candidate_buffer {
                let mut resident_candidates = self.resident_candidates.lock();
                if let Some(buffer) = resident_candidates.get(&candidates) {
                    (Some(buffer), Some(true))
                } else {
                    let buffer = context.upload_resident_candidates(candidates.as_slice())?;
                    resident_candidates.insert(candidates.clone(), buffer.clone());
                    (Some(buffer), Some(false))
                }
            } else {
                (None, None)
            };
            let resident_candidate_ns = resident_candidate_started.elapsed().as_nanos() as u64;
            let resident_visibility_started = std::time::Instant::now();
            let (visibility_buffer, resident_visibility_hit) = if let Some(visibility) = visibility
            {
                if visibility.generation == 0
                    || visibility.point_count == 0
                    || visibility.deleted.len() > visibility.point_count
                {
                    return Err(OperationError::service_error(
                        "GPU visibility snapshot identity is invalid",
                    ));
                }
                let mut resident_visibility = self.resident_visibility.lock();
                if let Some(buffer) =
                    resident_visibility.get(visibility.generation, visibility.point_count)
                {
                    (Some(buffer), Some(true))
                } else {
                    let words = pack_deleted_words(visibility.deleted, visibility.point_count);
                    let buffer = context.upload_resident_visibility(words.as_slice())?;
                    resident_visibility.insert(
                        visibility.generation,
                        visibility.point_count,
                        buffer.clone(),
                    );
                    (Some(buffer), Some(false))
                }
            } else {
                (None, None)
            };
            let resident_visibility_ns = resident_visibility_started.elapsed().as_nanos() as u64;
            context.search_batch(
                queries,
                candidates.as_slice(),
                top,
                resident_candidate_buffer,
                resident_candidate_hit,
                visibility_buffer,
                resident_visibility_hit,
                resident_candidate_ns,
                resident_visibility_ns,
            )
        })();
        self.contexts.lock().push(context);
        result.map(Some)
    }

    pub fn record_outer_breakdown(
        &self,
        query_count: usize,
        predicate_ns: u64,
        visibility_ns: u64,
        cache_ns: u64,
        postprocess_ns: u64,
        filter_cache_hit: Option<bool>,
    ) {
        self.stats.record_outer(
            query_count,
            predicate_ns,
            visibility_ns,
            cache_ns,
            postprocess_ns,
            filter_cache_hit,
        );
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicBool;

    use common::bitvec::BitVec;
    use common::counter::hardware_counter::HardwareCounterCell;

    use super::*;
    use crate::types::Distance;
    use crate::vector_storage::VectorStorage;
    use crate::vector_storage::dense::volatile_dense_vector_storage::new_volatile_dense_vector_storage;

    #[test]
    fn filter_candidate_cache_is_epoch_scoped() {
        let filter = Filter::new();
        let candidates = Arc::new(vec![1, 3, 5]);
        let mut cache = GpuFilterCandidateCache::default();

        assert!(cache.get(&filter, 7).is_none());
        cache.insert(filter.clone(), candidates.clone(), 7);
        assert_eq!(cache.get(&filter, 7).as_deref(), Some(candidates.as_ref()));
        assert!(cache.get(&filter, 8).is_none());
    }

    #[test]
    fn deleted_words_are_lsb_first_and_hide_padding() {
        let mut deleted = BitVec::repeat(false, 35);
        deleted.set(0, true);
        deleted.set(31, true);
        deleted.set(33, true);
        let words = pack_deleted_words(&deleted, deleted.len());
        assert_eq!(words.len(), 2);
        assert_eq!(words[0], 0x8000_0001);
        assert_eq!(words[1] & 0b111, 0b010);
        assert_eq!(words[1] >> 3, (1u32 << 29) - 1);
    }

    #[test]
    fn deleted_words_zero_fill_a_short_proxy_mask() {
        let mut deleted = BitVec::repeat(false, 5);
        deleted.set(1, true);
        deleted.set(4, true);

        let words = pack_deleted_words(&deleted, 65);

        assert_eq!(words.len(), 3);
        assert_eq!(words[0], 0b1_0010);
        assert_eq!(words[1], 0);
        assert_eq!(words[2], u32::MAX << 1);
    }

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
                        ((row * 17 + column * 13 + row * column * 7) % 65_521) as f32 / 65_521.0
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
        let cache =
            Arc::new(GpuExactSearchCache::new(device, &storage, COUNT, 8, 1, 0, &stopped).unwrap());
        let query = Arc::new(query);
        let candidates = Arc::new((0..COUNT as PointOffsetType).collect::<Vec<_>>());
        let mut observed = Vec::new();
        for reuse_candidate_buffer in [false, true] {
            let threads = (0..8)
                .map(|_| {
                    let cache = cache.clone();
                    let query = query.clone();
                    let candidates = candidates.clone();
                    std::thread::spawn(move || {
                        cache
                            .search(
                                query.as_slice(),
                                candidates.clone(),
                                TOP,
                                reuse_candidate_buffer,
                                None,
                            )
                            .unwrap()
                            .unwrap()
                    })
                })
                .collect::<Vec<_>>();
            observed.extend(threads.into_iter().map(|thread| thread.join().unwrap()));
        }

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

        let mut deleted = BitVec::repeat(false, COUNT);
        deleted.set(TARGET, true);
        deleted.set(expected[1].0 as usize, true);
        let visible_expected = expected
            .iter()
            .copied()
            .filter(|(idx, _)| !deleted[*idx as usize])
            .collect::<Vec<_>>();
        for _ in 0..2 {
            let result = cache
                .search(
                    query.as_slice(),
                    candidates.clone(),
                    TOP,
                    true,
                    Some(GpuVisibilitySnapshot {
                        generation: 7,
                        deleted: &deleted,
                        point_count: COUNT,
                    }),
                )
                .unwrap()
                .unwrap();
            for (actual, &(expected_id, expected_score)) in
                result.iter().zip(&visible_expected[..TOP])
            {
                assert_eq!(actual.idx, expected_id);
                assert!((actual.score - expected_score).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn exact_filtered_gpu_block_topk_matches_cpu_l2() {
        const DIM: usize = 128;
        const COUNT: usize = BLOCK_TOPK_MIN_CANDIDATES;
        const TARGET: usize = 12_345;
        const TARGET_TWO: usize = 54_321;
        const TOP: usize = 10;

        let vectors = (0..COUNT)
            .map(|row| {
                (0..DIM)
                    .map(|column| {
                        ((row * 17 + column * 13 + row * column * 7) % 65_521) as f32 / 65_521.0
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
            GpuExactSearchCache::new(device, &storage, COUNT, 1, 4, 50_000, &stopped).unwrap(),
        );
        let candidates = Arc::new((0..COUNT as PointOffsetType).collect::<Vec<_>>());

        let mut expected = vectors
            .iter()
            .enumerate()
            .map(|(idx, vector)| {
                let score = -vector
                    .iter()
                    .zip(query.iter())
                    .map(|(left, right)| (left - right) * (left - right))
                    .sum::<f32>();
                ScoredPointOffset {
                    idx: idx as PointOffsetType,
                    score,
                }
            })
            .collect::<Vec<_>>();
        expected.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.idx.cmp(&right.idx))
        });

        for _ in 0..2 {
            let result = cache
                .search(query.as_slice(), candidates.clone(), TOP, true, None)
                .unwrap()
                .unwrap();
            assert_eq!(result[0].idx, TARGET as PointOffsetType);
            for (actual, expected) in result.iter().zip(&expected[..TOP]) {
                assert_eq!(actual.idx, expected.idx);
                assert!((actual.score - expected.score).abs() < 1e-4);
            }
        }

        let query_two = vectors[TARGET_TWO].clone();
        let mut expected_two = vectors
            .iter()
            .enumerate()
            .map(|(idx, vector)| {
                let score = -vector
                    .iter()
                    .zip(query_two.iter())
                    .map(|(left, right)| (left - right) * (left - right))
                    .sum::<f32>();
                ScoredPointOffset {
                    idx: idx as PointOffsetType,
                    score,
                }
            })
            .collect::<Vec<_>>();
        expected_two.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.idx.cmp(&right.idx))
        });
        let batch_queries = [query.as_slice(), query_two.as_slice()];
        let batch_results = cache
            .search_batch(&batch_queries, candidates.clone(), TOP, true, None)
            .unwrap()
            .unwrap();
        assert_eq!(batch_results.len(), 2);
        for (actual, expected) in batch_results[0].iter().zip(&expected[..TOP]) {
            assert_eq!(actual.idx, expected.idx);
            assert!((actual.score - expected.score).abs() < 1e-4);
        }
        for (actual, expected) in batch_results[1].iter().zip(&expected_two[..TOP]) {
            assert_eq!(actual.idx, expected.idx);
            assert!((actual.score - expected.score).abs() < 1e-4);
        }

        let mut deleted = BitVec::repeat(false, COUNT);
        for point in expected.iter().take(7) {
            deleted.set(point.idx as usize, true);
        }
        let visible_expected = expected
            .iter()
            .filter(|point| !deleted[point.idx as usize])
            .collect::<Vec<_>>();
        for _ in 0..2 {
            let result = cache
                .search(
                    query.as_slice(),
                    candidates.clone(),
                    TOP,
                    true,
                    Some(GpuVisibilitySnapshot {
                        generation: 11,
                        deleted: &deleted,
                        point_count: COUNT,
                    }),
                )
                .unwrap()
                .unwrap();
            for (actual, expected) in result.iter().zip(&visible_expected[..TOP]) {
                assert_eq!(actual.idx, expected.idx);
                assert!((actual.score - expected.score).abs() < 1e-4);
            }
        }

        let visible_expected_two = expected_two
            .iter()
            .filter(|point| !deleted[point.idx as usize])
            .collect::<Vec<_>>();
        let visible_batch_results = cache
            .search_batch(
                &batch_queries,
                candidates.clone(),
                TOP,
                true,
                Some(GpuVisibilitySnapshot {
                    generation: 12,
                    deleted: &deleted,
                    point_count: COUNT,
                }),
            )
            .unwrap()
            .unwrap();
        for (actual, expected) in visible_batch_results[0]
            .iter()
            .zip(visible_expected[..TOP].iter().copied())
        {
            assert_eq!(actual.idx, expected.idx);
            assert!((actual.score - expected.score).abs() < 1e-4);
        }
        for (actual, expected) in visible_batch_results[1]
            .iter()
            .zip(visible_expected_two[..TOP].iter().copied())
        {
            assert_eq!(actual.idx, expected.idx);
            assert!((actual.score - expected.score).abs() < 1e-4);
        }

        let deleted = Arc::new(deleted);
        let query = Arc::new(query);
        let query_two = Arc::new(query_two);
        let barrier = Arc::new(std::sync::Barrier::new(
            GPU_EXACT_SUBMISSION_BATCH_LIMIT + 1,
        ));
        let threads = (0..GPU_EXACT_SUBMISSION_BATCH_LIMIT)
            .map(|query_index| {
                let cache = cache.clone();
                let candidates = candidates.clone();
                let deleted = deleted.clone();
                let query = if query_index.is_multiple_of(2) {
                    query.clone()
                } else {
                    query_two.clone()
                };
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    cache
                        .search_coalesced(
                            query.as_slice(),
                            candidates,
                            TOP,
                            true,
                            Some(GpuVisibilitySnapshot {
                                generation: 12,
                                deleted: deleted.as_bitslice(),
                                point_count: COUNT,
                            }),
                        )
                        .unwrap()
                        .unwrap()
                })
            })
            .collect::<Vec<_>>();
        barrier.wait();
        let coalesced_results = threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect::<Vec<_>>();
        for (query_index, result) in coalesced_results.iter().enumerate() {
            let expected = if query_index.is_multiple_of(2) {
                &visible_expected
            } else {
                &visible_expected_two
            };
            for (actual, expected) in result.iter().zip(expected[..TOP].iter().copied()) {
                assert_eq!(actual.idx, expected.idx);
                assert!((actual.score - expected.score).abs() < 1e-4);
            }
        }
        assert_eq!(cache.stats.coalesced_batches.load(Ordering::Relaxed), 1);
        assert_eq!(
            cache.stats.coalesced_queries.load(Ordering::Relaxed),
            GPU_EXACT_SUBMISSION_BATCH_LIMIT as u64,
        );
    }
}
