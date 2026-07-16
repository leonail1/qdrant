//! Candidate-major exact scoring for factorized filtered batches.
//!
//! This module deliberately implements a narrow, fail-closed primitive:
//!
//! - nearest-neighbour queries only;
//! - single dense `f32` vectors;
//! - cosine distance;
//! - at most 64 queries;
//! - original, non-quantized vector storage.
//!
//! Eligibility is supplied as owned `(point_id, query_mask)` candidates. A
//! candidate vector is read once and scored only for the queries selected by
//! its mask. Work is balanced into score-pair tiles. Tiles are independent and
//! can be scheduled on different workers, then merged into final per-query
//! top-k results.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::atomic::AtomicBool;

use common::bitvec::BitSlice;
use common::counter::hardware_counter::HardwareCounterCell;
use common::fixed_length_priority_queue::FixedLengthPriorityQueue;
use common::types::{PointOffsetType, ScoredPointOffset};

use crate::common::operation_error::{OperationError, OperationResult, check_process_stopped};
use crate::data_types::vectors::{QueryVector, VectorElementType, VectorInternal};
use crate::index::query_optimization::factorized_filter::FactorizedFilterPlan;
use crate::spaces::metric::Metric;
use crate::spaces::simple::CosineMetric;
use crate::types::{Distance, VectorStorageDatatype};
use crate::vector_storage::{
    DenseVectorStorage, VectorStorageEnum, VectorStorageRead, check_deleted_condition,
};

pub const DEFAULT_MAX_TILES: usize = 56;
pub const DEFAULT_MIN_SCORE_PAIRS_PER_TILE: usize = 4 * 1024;
pub const DEFAULT_ORDINAL_ORDERING_THRESHOLD: usize = 4 * 1024;

/// Cost and access-order policy for a candidate-major exact plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CandidateMajorExactConfig {
    /// Hard cap on independently schedulable tiles.
    pub max_tiles: usize,
    /// Do not create another tile until total work justifies approximately this
    /// many query-point score pairs per tile.
    pub min_score_pairs_per_tile: usize,
    /// At or above this many unique candidates, order candidates by point
    /// ordinal to enable sequential reads and prefetch in dense storage.
    pub ordinal_ordering_threshold: usize,
}

impl Default for CandidateMajorExactConfig {
    fn default() -> Self {
        Self {
            max_tiles: DEFAULT_MAX_TILES,
            min_score_pairs_per_tile: DEFAULT_MIN_SCORE_PAIRS_PER_TILE,
            ordinal_ordering_threshold: DEFAULT_ORDINAL_ORDERING_THRESHOLD,
        }
    }
}

/// One candidate and the queries for which it is eligible.
///
/// Query `i` is selected when bit `i` is set in [`Self::query_mask`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FactorizedExactCandidate {
    pub point_id: PointOffsetType,
    pub query_mask: u64,
}

impl FactorizedExactCandidate {
    pub const fn new(point_id: PointOffsetType, query_mask: u64) -> Self {
        Self {
            point_id,
            query_mask,
        }
    }
}

/// Generate owned candidates from a point-ordinal mask array.
///
/// Small candidate sets follow `touched_points` and avoid scanning the full
/// mask array. Once `touched_points.len() >= ordinal_scan_threshold`, the full
/// mask array is scanned in point-ordinal order so downstream dense reads are
/// sequential. A threshold of zero always selects the ordinal scan.
pub fn collect_factorized_exact_candidates(
    point_query_masks: &[u64],
    touched_points: &[PointOffsetType],
    ordinal_scan_threshold: usize,
) -> OperationResult<Vec<FactorizedExactCandidate>> {
    if point_query_masks.len() as u64 > PointOffsetType::MAX as u64 + 1 {
        return Err(OperationError::service_error(
            "point-query mask array exceeds the point ordinal type",
        ));
    }

    if touched_points.len() >= ordinal_scan_threshold {
        return Ok(point_query_masks
            .iter()
            .enumerate()
            .filter_map(|(point_id, &query_mask)| {
                (query_mask != 0).then_some(FactorizedExactCandidate::new(
                    point_id as PointOffsetType,
                    query_mask,
                ))
            })
            .collect());
    }

    touched_points
        .iter()
        .filter_map(|&point_id| {
            let query_mask = point_query_masks.get(point_id as usize).copied();
            match query_mask {
                Some(0) => None,
                Some(query_mask) => Some(Ok(FactorizedExactCandidate::new(point_id, query_mask))),
                None => Some(Err(OperationError::service_error(
                    "touched point ordinal is outside the point-query mask array",
                ))),
            }
        })
        .collect()
}

/// Immutable, owned work plan. Each tile can be scored independently.
#[derive(Clone, Debug)]
pub struct CandidateMajorExactPlan {
    queries: Vec<Vec<VectorElementType>>,
    candidates: Vec<FactorizedExactCandidate>,
    tiles: Vec<Range<usize>>,
    top: usize,
    vector_dim: usize,
    input_candidates: usize,
    duplicate_candidates_merged: usize,
    planned_score_pairs: usize,
    ordinal_ordered: bool,
    ordinal_sort_applied: bool,
    prepare_stats: CandidateMajorExactPrepareStats,
}

impl CandidateMajorExactPlan {
    pub fn queries(&self) -> &[Vec<VectorElementType>] {
        &self.queries
    }

    pub fn candidates(&self) -> &[FactorizedExactCandidate] {
        &self.candidates
    }

    pub fn tiles(&self) -> &[Range<usize>] {
        &self.tiles
    }

    pub fn tile_count(&self) -> usize {
        self.tiles.len()
    }

    pub fn top(&self) -> usize {
        self.top
    }

    pub fn planned_score_pairs(&self) -> usize {
        self.planned_score_pairs
    }

    pub fn ordinal_ordered(&self) -> bool {
        self.ordinal_ordered
    }

    pub fn ordinal_sort_applied(&self) -> bool {
        self.ordinal_sort_applied
    }

    pub fn prepare_stats(&self) -> CandidateMajorExactPrepareStats {
        self.prepare_stats
    }

    pub(crate) fn set_prepare_stats(&mut self, prepare_stats: CandidateMajorExactPrepareStats) {
        self.prepare_stats = prepare_stats;
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CandidateMajorExactPrepareStats {
    pub posting_atoms: usize,
    pub posting_points: usize,
    pub single_valued: bool,
    pub atom_build_us: u64,
    pub posting_fetch_us: u64,
    pub candidate_collect_us: u64,
    pub plan_build_us: u64,
}

#[derive(Debug, PartialEq)]
pub struct CandidateMajorExactTileOutput {
    pub tile_ordinal: usize,
    pub results: Vec<Vec<ScoredPointOffset>>,
    pub vector_reads: usize,
    pub scored_pairs: usize,
}

/// Breakdown emitted after all tile outputs are merged.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CandidateMajorExactStats {
    pub input_candidates: usize,
    pub duplicate_candidates_merged: usize,
    pub vector_reads: usize,
    pub scored_pairs: usize,
    pub tiles: usize,
    pub ordinal_ordered: bool,
    pub ordinal_sort_applied: bool,
}

#[derive(Debug, PartialEq)]
pub struct CandidateMajorExactOutput {
    pub results: Vec<Vec<ScoredPointOffset>>,
    pub stats: CandidateMajorExactStats,
}

/// Validate inputs, preprocess queries, consolidate candidates and construct
/// balanced independent score-pair tiles.
#[allow(clippy::too_many_arguments)]
pub fn build_candidate_major_exact_plan(
    storage: &VectorStorageEnum,
    queries: &[&QueryVector],
    factorized_plan: &FactorizedFilterPlan,
    candidates: Vec<FactorizedExactCandidate>,
    point_deleted: &BitSlice,
    top: usize,
    config: CandidateMajorExactConfig,
    is_stopped: &AtomicBool,
) -> OperationResult<CandidateMajorExactPlan> {
    validate_batch_shape(storage, queries, factorized_plan, config)?;

    let query_count = queries.len();
    let valid_query_mask = if query_count == u64::BITS as usize {
        u64::MAX
    } else {
        (1_u64 << query_count) - 1
    };
    let vector_dim = dense_f32_dim(storage)?;
    let preprocessed_queries = preprocess_queries(queries, vector_dim)?;

    let input_candidates = candidates.len();
    let input_is_ordinal_ordered = candidates
        .windows(2)
        .all(|pair| pair[0].point_id <= pair[1].point_id);
    // Production candidates are emitted from the point-ordinal mask scan and
    // are therefore already sorted. Merge adjacent duplicates directly in
    // that common path instead of allocating and probing a large HashMap.
    // Keep the HashMap fallback for the public primitive's arbitrary input
    // contract and for tests which intentionally supply unsorted candidates.
    let mut candidate_positions = (!input_is_ordinal_ordered)
        .then(|| HashMap::<PointOffsetType, usize>::with_capacity(candidates.len()));
    let mut live_candidates: Vec<FactorizedExactCandidate> = Vec::with_capacity(candidates.len());
    let mut duplicate_candidates_merged = 0;
    let total_vector_count = storage.total_vector_count();
    let deleted_vectors = storage.deleted_vector_bitslice();

    for (candidate_offset, candidate) in candidates.into_iter().enumerate() {
        if candidate_offset % 4096 == 0 {
            check_process_stopped(is_stopped)?;
        }
        if candidate.query_mask & !valid_query_mask != 0 {
            return Err(OperationError::service_error(
                "candidate query mask contains an out-of-range query bit",
            ));
        }
        if candidate.query_mask == 0
            || candidate.point_id as usize >= total_vector_count
            || !check_deleted_condition(candidate.point_id, deleted_vectors, point_deleted)
        {
            continue;
        }

        if input_is_ordinal_ordered {
            if let Some(last) = live_candidates.last_mut()
                && last.point_id == candidate.point_id
            {
                last.query_mask |= candidate.query_mask;
                duplicate_candidates_merged += 1;
            } else {
                live_candidates.push(candidate);
            }
        } else {
            let candidate_positions = candidate_positions
                .as_mut()
                .expect("unordered input always allocates candidate positions");
            if let Some(&position) = candidate_positions.get(&candidate.point_id) {
                live_candidates[position].query_mask |= candidate.query_mask;
                duplicate_candidates_merged += 1;
            } else {
                candidate_positions.insert(candidate.point_id, live_candidates.len());
                live_candidates.push(candidate);
            }
        }
    }

    let already_ordinal_ordered = input_is_ordinal_ordered
        || live_candidates
            .windows(2)
            .all(|pair| pair[0].point_id <= pair[1].point_id);
    let should_order_by_ordinal = live_candidates.len() >= config.ordinal_ordering_threshold;
    let ordinal_sort_applied = should_order_by_ordinal && !already_ordinal_ordered;
    if ordinal_sort_applied {
        live_candidates.sort_unstable_by_key(|candidate| candidate.point_id);
    }
    let ordinal_ordered = already_ordinal_ordered || ordinal_sort_applied;

    let planned_score_pairs = live_candidates.iter().try_fold(0_usize, |sum, candidate| {
        sum.checked_add(candidate.query_mask.count_ones() as usize)
            .ok_or_else(|| OperationError::service_error("candidate score-pair count overflow"))
    })?;
    let tiles = if top == 0 {
        Vec::new()
    } else {
        build_score_pair_tiles(
            &live_candidates,
            planned_score_pairs,
            config.max_tiles,
            config.min_score_pairs_per_tile,
        )
    };

    Ok(CandidateMajorExactPlan {
        queries: preprocessed_queries,
        candidates: live_candidates,
        tiles,
        top,
        vector_dim,
        input_candidates,
        duplicate_candidates_merged,
        planned_score_pairs,
        ordinal_ordered,
        ordinal_sort_applied,
        prepare_stats: CandidateMajorExactPrepareStats::default(),
    })
}

/// Score one tile. This is the worker-level entry point for parallel execution.
pub fn score_candidate_major_exact_tile(
    plan: &CandidateMajorExactPlan,
    tile_ordinal: usize,
    storage: &VectorStorageEnum,
    mut hardware_counter: HardwareCounterCell,
    is_stopped: &AtomicBool,
) -> OperationResult<CandidateMajorExactTileOutput> {
    check_process_stopped(is_stopped)?;
    validate_storage_shape(storage)?;
    if dense_f32_dim(storage)? != plan.vector_dim {
        return Err(OperationError::service_error(
            "candidate-major exact plan vector dimension differs from storage",
        ));
    }
    let tile_range = plan.tiles.get(tile_ordinal).ok_or_else(|| {
        OperationError::service_error("candidate-major exact tile ordinal is out of range")
    })?;
    let tile = &plan.candidates[tile_range.clone()];
    let scored_pairs = tile
        .iter()
        .map(|candidate| candidate.query_mask.count_ones() as usize)
        .sum();

    hardware_counter.set_cpu_multiplier(plan.vector_dim * size_of::<VectorElementType>());
    hardware_counter.set_vector_io_read_multiplier(if storage.is_on_disk() {
        plan.vector_dim * size_of::<VectorElementType>()
    } else {
        0
    });
    hardware_counter.cpu_counter().incr_delta(scored_pairs);
    hardware_counter.vector_io_read().incr_delta(tile.len());

    Ok(CandidateMajorExactTileOutput {
        tile_ordinal,
        results: score_tile_inner(storage, &plan.queries, tile, plan.top)?,
        vector_reads: tile.len(),
        scored_pairs,
    })
}

/// Merge independently scored tiles. Tile outputs may arrive in any order.
pub fn merge_candidate_major_exact_tiles(
    plan: &CandidateMajorExactPlan,
    tile_outputs: Vec<CandidateMajorExactTileOutput>,
) -> OperationResult<CandidateMajorExactOutput> {
    if tile_outputs.len() != plan.tiles.len() {
        return Err(OperationError::service_error(
            "candidate-major exact tile output count differs from plan",
        ));
    }

    let mut ordered_outputs: Vec<Option<CandidateMajorExactTileOutput>> =
        (0..plan.tiles.len()).map(|_| None).collect();
    for output in tile_outputs {
        let output_slot = ordered_outputs
            .get_mut(output.tile_ordinal)
            .ok_or_else(|| {
                OperationError::service_error(
                    "candidate-major exact tile output ordinal is out of range",
                )
            })?;
        if output_slot.is_some() {
            return Err(OperationError::service_error(
                "candidate-major exact tile output ordinal is duplicated",
            ));
        }
        *output_slot = Some(output);
    }

    let mut merged_queues = (0..plan.queries.len())
        .map(|_| FixedLengthPriorityQueue::new(plan.top.max(1)))
        .collect::<Vec<_>>();
    let mut vector_reads = 0_usize;
    let mut scored_pairs = 0_usize;

    for output in ordered_outputs {
        let output = output.ok_or_else(|| {
            OperationError::service_error("candidate-major exact tile output is missing")
        })?;
        if output.results.len() != plan.queries.len() {
            return Err(OperationError::service_error(
                "candidate-major exact tile query result count differs from plan",
            ));
        }
        vector_reads = vector_reads
            .checked_add(output.vector_reads)
            .ok_or_else(|| OperationError::service_error("merged vector-read count overflow"))?;
        scored_pairs = scored_pairs
            .checked_add(output.scored_pairs)
            .ok_or_else(|| OperationError::service_error("merged score-pair count overflow"))?;
        for (merged_queue, query_tile_results) in merged_queues.iter_mut().zip(output.results) {
            for scored_point in query_tile_results {
                merged_queue.push(scored_point);
            }
        }
    }
    if plan.top != 0
        && (vector_reads != plan.candidates.len() || scored_pairs != plan.planned_score_pairs)
    {
        return Err(OperationError::service_error(
            "candidate-major exact tile outputs do not cover the complete plan",
        ));
    }

    let results = if plan.top == 0 {
        vec![Vec::new(); plan.queries.len()]
    } else {
        merged_queues
            .into_iter()
            .map(FixedLengthPriorityQueue::into_sorted_vec)
            .collect()
    };

    Ok(CandidateMajorExactOutput {
        results,
        stats: CandidateMajorExactStats {
            input_candidates: plan.input_candidates,
            duplicate_candidates_merged: plan.duplicate_candidates_merged,
            vector_reads,
            scored_pairs,
            tiles: plan.tiles.len(),
            ordinal_ordered: plan.ordinal_ordered,
            ordinal_sort_applied: plan.ordinal_sort_applied,
        },
    })
}

/// Monolithic convenience wrapper. Production collection scheduling can instead
/// build a plan, score tiles on separate workers, and call the merge function.
#[allow(clippy::too_many_arguments)]
pub fn search_candidate_major_exact_cosine(
    storage: &VectorStorageEnum,
    queries: &[&QueryVector],
    factorized_plan: &FactorizedFilterPlan,
    candidates: Vec<FactorizedExactCandidate>,
    point_deleted: &BitSlice,
    top: usize,
    config: CandidateMajorExactConfig,
    hardware_counter: HardwareCounterCell,
    is_stopped: &AtomicBool,
) -> OperationResult<CandidateMajorExactOutput> {
    let plan = build_candidate_major_exact_plan(
        storage,
        queries,
        factorized_plan,
        candidates,
        point_deleted,
        top,
        config,
        is_stopped,
    )?;
    let tile_outputs = (0..plan.tile_count())
        .map(|tile_ordinal| {
            score_candidate_major_exact_tile(
                &plan,
                tile_ordinal,
                storage,
                hardware_counter.fork(),
                is_stopped,
            )
        })
        .collect::<OperationResult<Vec<_>>>()?;
    merge_candidate_major_exact_tiles(&plan, tile_outputs)
}

fn validate_batch_shape(
    storage: &VectorStorageEnum,
    queries: &[&QueryVector],
    factorized_plan: &FactorizedFilterPlan,
    config: CandidateMajorExactConfig,
) -> OperationResult<()> {
    if queries.is_empty() {
        return Err(OperationError::service_error(
            "candidate-major exact search requires at least one query",
        ));
    }
    if queries.len() > u64::BITS as usize {
        return Err(OperationError::service_error(
            "candidate-major exact search supports at most 64 queries",
        ));
    }
    if factorized_plan.query_count() != queries.len() {
        return Err(OperationError::service_error(
            "factorized plan query count differs from vector query count",
        ));
    }
    if config.max_tiles == 0 {
        return Err(OperationError::service_error(
            "candidate-major exact max tile count must be greater than zero",
        ));
    }
    if config.min_score_pairs_per_tile == 0 {
        return Err(OperationError::service_error(
            "minimum score pairs per tile must be greater than zero",
        ));
    }
    validate_storage_shape(storage)
}

fn validate_storage_shape(storage: &VectorStorageEnum) -> OperationResult<()> {
    if storage.distance() != Distance::Cosine {
        return Err(OperationError::service_error(
            "candidate-major exact search currently supports cosine distance only",
        ));
    }
    if storage.datatype() != VectorStorageDatatype::Float32 {
        return Err(OperationError::service_error(
            "candidate-major exact search currently supports f32 storage only",
        ));
    }
    dense_f32_dim(storage).map(|_| ())
}

fn preprocess_queries(
    queries: &[&QueryVector],
    expected_dim: usize,
) -> OperationResult<Vec<Vec<VectorElementType>>> {
    queries
        .iter()
        .enumerate()
        .map(|(query_index, &query)| {
            let QueryVector::Nearest(VectorInternal::Dense(vector)) = query else {
                return Err(OperationError::service_error(format!(
                    "query {query_index} is not a nearest dense query",
                )));
            };
            if vector.len() != expected_dim {
                return Err(OperationError::service_error(format!(
                    "query {query_index} dimension {} differs from storage dimension {expected_dim}",
                    vector.len(),
                )));
            }
            Ok(<CosineMetric as Metric<VectorElementType>>::preprocess(
                vector.clone(),
            ))
        })
        .collect()
}

fn build_score_pair_tiles(
    candidates: &[FactorizedExactCandidate],
    total_score_pairs: usize,
    max_tiles: usize,
    min_score_pairs_per_tile: usize,
) -> Vec<Range<usize>> {
    if candidates.is_empty() || total_score_pairs == 0 {
        return Vec::new();
    }

    let tile_count = total_score_pairs
        .div_ceil(min_score_pairs_per_tile)
        .clamp(1, max_tiles)
        .min(candidates.len());
    let target_score_pairs = total_score_pairs.div_ceil(tile_count);
    let mut tiles = Vec::with_capacity(tile_count);
    let mut start = 0;

    for tile_ordinal in 0..tile_count {
        if tile_ordinal + 1 == tile_count {
            tiles.push(start..candidates.len());
            break;
        }

        let future_tiles = tile_count - tile_ordinal - 1;
        let latest_end = candidates.len() - future_tiles;
        let mut end = start;
        let mut score_pairs = 0;
        while end < latest_end {
            let next_pairs = candidates[end].query_mask.count_ones() as usize;
            if end > start && score_pairs >= target_score_pairs {
                break;
            }
            if end > start && score_pairs.saturating_add(next_pairs) > target_score_pairs {
                let undershoot = target_score_pairs.saturating_sub(score_pairs);
                let overshoot = score_pairs
                    .saturating_add(next_pairs)
                    .saturating_sub(target_score_pairs);
                if undershoot <= overshoot {
                    break;
                }
            }
            score_pairs = score_pairs.saturating_add(next_pairs);
            end += 1;
        }
        if end == start {
            end += 1;
        }
        tiles.push(start..end);
        start = end;
    }
    tiles
}

fn score_tile_inner(
    storage: &VectorStorageEnum,
    queries: &[Vec<VectorElementType>],
    candidates: &[FactorizedExactCandidate],
    top: usize,
) -> OperationResult<Vec<Vec<ScoredPointOffset>>> {
    if top == 0 {
        return Ok(vec![Vec::new(); queries.len()]);
    }

    let point_ids = candidates
        .iter()
        .map(|candidate| candidate.point_id)
        .collect::<Vec<_>>();
    let mut queues = (0..queries.len())
        .map(|_| FixedLengthPriorityQueue::new(top))
        .collect::<Vec<_>>();

    for_each_dense_f32_batch(storage, &point_ids, |candidate_index, vector| {
        let candidate = candidates[candidate_index];
        let mut query_mask = candidate.query_mask;
        while query_mask != 0 {
            let query_index = query_mask.trailing_zeros() as usize;
            query_mask &= query_mask - 1;
            queues[query_index].push(ScoredPointOffset {
                idx: candidate.point_id,
                score: <CosineMetric as Metric<VectorElementType>>::similarity(
                    &queries[query_index],
                    vector,
                ),
            });
        }
    })?;

    Ok(queues
        .into_iter()
        .map(FixedLengthPriorityQueue::into_sorted_vec)
        .collect())
}

fn dense_f32_dim(storage: &VectorStorageEnum) -> OperationResult<usize> {
    let dim = match storage {
        VectorStorageEnum::DenseVolatile(storage) => storage.vector_dim(),
        VectorStorageEnum::DenseMemmap(storage) => storage.vector_dim(),
        #[cfg(target_os = "linux")]
        VectorStorageEnum::DenseUring(storage) => storage.vector_dim(),
        VectorStorageEnum::DenseAppendableMemmap(storage) => storage.vector_dim(),
        VectorStorageEnum::EmptyDense(storage) => storage.vector_dim(),
        _ => {
            return Err(OperationError::service_error(
                "candidate-major exact search requires single dense f32 storage",
            ));
        }
    };
    Ok(dim)
}

fn for_each_dense_f32_batch(
    storage: &VectorStorageEnum,
    point_ids: &[PointOffsetType],
    mut callback: impl FnMut(usize, &[VectorElementType]),
) -> OperationResult<()> {
    match storage {
        VectorStorageEnum::DenseVolatile(storage) => {
            storage.for_each_in_dense_batch(point_ids, &mut callback)
        }
        VectorStorageEnum::DenseMemmap(storage) => {
            storage.for_each_in_dense_batch(point_ids, &mut callback)
        }
        #[cfg(target_os = "linux")]
        VectorStorageEnum::DenseUring(storage) => {
            storage.for_each_in_dense_batch(point_ids, &mut callback)
        }
        VectorStorageEnum::DenseAppendableMemmap(storage) => {
            storage.for_each_in_dense_batch(point_ids, &mut callback)
        }
        VectorStorageEnum::EmptyDense(storage) => {
            storage.for_each_in_dense_batch(point_ids, &mut callback)
        }
        _ => {
            return Err(OperationError::service_error(
                "candidate-major exact search requires single dense f32 storage",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::atomic::AtomicBool;

    use common::bitvec::{BitSlice, BitVec};
    use common::counter::hardware_counter::HardwareCounterCell;
    use common::fixed_length_priority_queue::FixedLengthPriorityQueue;
    use common::types::{PointOffsetType, ScoredPointOffset};

    use super::{
        CandidateMajorExactConfig, FactorizedExactCandidate, build_candidate_major_exact_plan,
        collect_factorized_exact_candidates, merge_candidate_major_exact_tiles,
        score_candidate_major_exact_tile, search_candidate_major_exact_cosine,
    };
    use crate::data_types::vectors::{QueryVector, VectorElementType};
    use crate::index::query_optimization::factorized_filter::FactorizedFilterPlan;
    use crate::json_path::JsonPath;
    use crate::types::{Condition, Distance, FieldCondition, Filter};
    use crate::vector_storage::dense::volatile_dense_vector_storage::new_volatile_dense_vector_storage;
    use crate::vector_storage::{
        VectorStorage, VectorStorageEnum, VectorStorageRead, check_deleted_condition,
        new_raw_scorer,
    };

    static NOT_STOPPED: AtomicBool = AtomicBool::new(false);

    fn config(
        max_tiles: usize,
        min_score_pairs_per_tile: usize,
        ordinal_ordering_threshold: usize,
    ) -> CandidateMajorExactConfig {
        CandidateMajorExactConfig {
            max_tiles,
            min_score_pairs_per_tile,
            ordinal_ordering_threshold,
        }
    }

    fn filter(values: Vec<i64>) -> Filter {
        Filter::new_must(Condition::Field(FieldCondition::new_match(
            JsonPath::new("group_id"),
            values.into(),
        )))
    }

    fn plan(query_count: usize) -> FactorizedFilterPlan {
        let filters = (0..query_count)
            .map(|query_index| {
                filter(vec![
                    query_index as i64,
                    query_index.saturating_add(1) as i64,
                ])
            })
            .collect::<Vec<_>>();
        FactorizedFilterPlan::try_from_filters(&filters).unwrap()
    }

    fn storage(points: &[Vec<VectorElementType>]) -> VectorStorageEnum {
        let mut storage = new_volatile_dense_vector_storage(points[0].len(), Distance::Cosine);
        for (point_id, point) in points.iter().enumerate() {
            let point = Distance::Cosine.preprocess_vector::<VectorElementType>(point.clone());
            storage
                .insert_vector(
                    point_id as PointOffsetType,
                    point.as_slice().into(),
                    &HardwareCounterCell::disposable(),
                )
                .unwrap();
        }
        storage
    }

    fn query_refs(queries: &[QueryVector]) -> Vec<&QueryVector> {
        queries.iter().collect()
    }

    fn independent_results(
        storage: &VectorStorageEnum,
        queries: &[QueryVector],
        candidates: &[FactorizedExactCandidate],
        point_deleted: &BitSlice,
        top: usize,
    ) -> Vec<Vec<ScoredPointOffset>> {
        let mut consolidated = BTreeMap::<PointOffsetType, u64>::new();
        for candidate in candidates {
            *consolidated.entry(candidate.point_id).or_default() |= candidate.query_mask;
        }

        queries
            .iter()
            .enumerate()
            .map(|(query_index, query)| {
                let point_ids = consolidated
                    .iter()
                    .filter_map(|(&point_id, &query_mask)| {
                        ((query_mask & (1_u64 << query_index)) != 0
                            && (point_id as usize) < storage.total_vector_count()
                            && check_deleted_condition(
                                point_id,
                                storage.deleted_vector_bitslice(),
                                point_deleted,
                            ))
                        .then_some(point_id)
                    })
                    .collect::<Vec<_>>();
                let scorer =
                    new_raw_scorer(query.clone(), storage, HardwareCounterCell::disposable())
                        .unwrap();
                let mut scores = vec![0.0; point_ids.len()];
                scorer.score_points(&point_ids, &mut scores);
                let mut queue = FixedLengthPriorityQueue::new(top);
                for (&idx, &score) in point_ids.iter().zip(&scores) {
                    queue.push(ScoredPointOffset { idx, score });
                }
                queue.into_sorted_vec()
            })
            .collect()
    }

    fn assert_score_equivalent(left: &[Vec<ScoredPointOffset>], right: &[Vec<ScoredPointOffset>]) {
        assert_eq!(left.len(), right.len());
        for (left_query, right_query) in left.iter().zip(right) {
            assert_eq!(left_query.len(), right_query.len());
            assert_eq!(
                left_query
                    .iter()
                    .map(|point| point.score)
                    .collect::<Vec<_>>(),
                right_query
                    .iter()
                    .map(|point| point.score)
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn overlap_matches_independent_scoring_and_merges_duplicate_candidates() {
        let storage = storage(&[
            vec![1.0, 0.1, 0.0],
            vec![0.7, 0.8, 0.1],
            vec![0.1, 0.9, 0.5],
            vec![0.2, 0.1, 1.0],
        ]);
        let queries: Vec<QueryVector> = vec![
            [1.0, 0.0, 0.0].into(),
            [0.0, 1.0, 0.0].into(),
            [0.0, 0.0, 1.0].into(),
        ];
        let candidates = vec![
            FactorizedExactCandidate::new(0, 0b001),
            FactorizedExactCandidate::new(1, 0b001),
            FactorizedExactCandidate::new(1, 0b010),
            FactorizedExactCandidate::new(2, 0b110),
            FactorizedExactCandidate::new(3, 0b100),
        ];
        let deleted = BitVec::repeat(false, 4);
        let expected = independent_results(&storage, &queries, &candidates, &deleted, 2);

        let output = search_candidate_major_exact_cosine(
            &storage,
            &query_refs(&queries),
            &plan(queries.len()),
            candidates,
            &deleted,
            2,
            config(4, 2, usize::MAX),
            HardwareCounterCell::disposable(),
            &NOT_STOPPED,
        )
        .unwrap();

        assert_eq!(output.results, expected);
        assert_eq!(output.stats.input_candidates, 5);
        assert_eq!(output.stats.duplicate_candidates_merged, 1);
        assert_eq!(output.stats.vector_reads, 4);
        assert_eq!(output.stats.scored_pairs, 6);
        assert!(output.stats.tiles > 1);
    }

    #[test]
    fn deleted_points_and_vectors_are_not_scored() {
        let mut storage = storage(&[
            vec![1.0, 0.0, 0.0],
            vec![0.9, 0.2, 0.0],
            vec![0.8, 0.4, 0.0],
            vec![0.1, 1.0, 0.0],
        ]);
        storage.delete_vector(1).unwrap();
        let mut deleted = BitVec::repeat(false, 4);
        deleted.set(2, true);
        let queries: Vec<QueryVector> = vec![[1.0, 0.0, 0.0].into(), [0.0, 1.0, 0.0].into()];
        let candidates = (0..4)
            .map(|point_id| FactorizedExactCandidate::new(point_id, 0b11))
            .collect::<Vec<_>>();
        let expected = independent_results(&storage, &queries, &candidates, &deleted, 3);

        let output = search_candidate_major_exact_cosine(
            &storage,
            &query_refs(&queries),
            &plan(queries.len()),
            candidates,
            &deleted,
            3,
            config(4, 1, usize::MAX),
            HardwareCounterCell::disposable(),
            &NOT_STOPPED,
        )
        .unwrap();

        assert_eq!(output.results, expected);
        assert_eq!(output.stats.vector_reads, 2);
        assert_eq!(output.stats.scored_pairs, 4);
        assert!(
            output
                .results
                .iter()
                .flatten()
                .all(|point| point.idx != 1 && point.idx != 2)
        );
    }

    #[test]
    fn independently_scored_tiles_merge_in_any_completion_order() {
        let storage = storage(&[
            vec![1.0, 0.1, 0.0],
            vec![0.9, 0.3, 0.1],
            vec![0.7, 0.6, 0.2],
            vec![0.5, 0.8, 0.3],
            vec![0.3, 0.9, 0.4],
            vec![0.2, 0.7, 0.8],
            vec![0.1, 0.4, 1.0],
            vec![0.4, 0.2, 0.9],
        ]);
        let queries: Vec<QueryVector> = vec![
            [1.0, 0.0, 0.0].into(),
            [0.0, 1.0, 0.0].into(),
            [0.0, 0.0, 1.0].into(),
        ];
        let masks = [0b001, 0b011, 0b010, 0b111, 0b110, 0b100, 0b101, 0b111];
        let candidates = masks
            .into_iter()
            .enumerate()
            .map(|(point_id, query_mask)| {
                FactorizedExactCandidate::new(point_id as PointOffsetType, query_mask)
            })
            .collect::<Vec<_>>();
        let deleted = BitVec::repeat(false, candidates.len());
        let expected = independent_results(&storage, &queries, &candidates, &deleted, 3);
        let exact_plan = build_candidate_major_exact_plan(
            &storage,
            &query_refs(&queries),
            &plan(queries.len()),
            candidates,
            &deleted,
            3,
            config(4, 1, usize::MAX),
            &NOT_STOPPED,
        )
        .unwrap();
        assert!(exact_plan.tile_count() > 1);

        let tile_outputs = (0..exact_plan.tile_count())
            .rev()
            .map(|tile_ordinal| {
                score_candidate_major_exact_tile(
                    &exact_plan,
                    tile_ordinal,
                    &storage,
                    HardwareCounterCell::disposable(),
                    &NOT_STOPPED,
                )
                .unwrap()
            })
            .collect();
        let merged = merge_candidate_major_exact_tiles(&exact_plan, tile_outputs).unwrap();

        assert_eq!(merged.results, expected);
        assert_eq!(merged.stats.vector_reads, exact_plan.candidates().len());
        assert_eq!(merged.stats.scored_pairs, exact_plan.planned_score_pairs());
    }

    #[test]
    fn ordinal_threshold_switches_from_touched_order_to_sequential_order() {
        let point_query_masks = vec![1, 1, 1, 1, 1, 1];
        let touched_points = vec![5, 4, 3, 2, 1, 0];
        let touched_candidates = collect_factorized_exact_candidates(
            &point_query_masks,
            &touched_points,
            touched_points.len() + 1,
        )
        .unwrap();
        let ordinal_candidates =
            collect_factorized_exact_candidates(&point_query_masks, &touched_points, 3).unwrap();
        assert_eq!(
            touched_candidates
                .iter()
                .map(|candidate| candidate.point_id)
                .collect::<Vec<_>>(),
            touched_points
        );
        assert_eq!(
            ordinal_candidates
                .iter()
                .map(|candidate| candidate.point_id)
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4, 5]
        );

        let storage = storage(&[
            vec![1.0, 0.0],
            vec![1.0, 0.0],
            vec![1.0, 0.0],
            vec![1.0, 0.0],
            vec![1.0, 0.0],
            vec![1.0, 0.0],
        ]);
        let queries: Vec<QueryVector> = vec![[1.0, 0.0].into()];
        let deleted = BitVec::repeat(false, point_query_masks.len());
        let touched_result = search_candidate_major_exact_cosine(
            &storage,
            &query_refs(&queries),
            &plan(1),
            touched_candidates.clone(),
            &deleted,
            2,
            config(1, 1, usize::MAX),
            HardwareCounterCell::disposable(),
            &NOT_STOPPED,
        )
        .unwrap();
        let ordinal_result = search_candidate_major_exact_cosine(
            &storage,
            &query_refs(&queries),
            &plan(1),
            ordinal_candidates,
            &deleted,
            2,
            config(1, 1, usize::MAX),
            HardwareCounterCell::disposable(),
            &NOT_STOPPED,
        )
        .unwrap();
        assert_score_equivalent(&touched_result.results, &ordinal_result.results);

        let sorted_plan = build_candidate_major_exact_plan(
            &storage,
            &query_refs(&queries),
            &plan(1),
            touched_candidates,
            &deleted,
            2,
            config(1, 1, 3),
            &NOT_STOPPED,
        )
        .unwrap();
        assert!(sorted_plan.ordinal_ordered());
        assert!(sorted_plan.ordinal_sort_applied());
        assert_eq!(
            sorted_plan
                .candidates()
                .iter()
                .map(|candidate| candidate.point_id)
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4, 5]
        );
    }
}
