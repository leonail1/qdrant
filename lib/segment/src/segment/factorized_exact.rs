use std::time::{Duration, Instant};

use common::types::{PointOffsetType, ScoredPointOffset};

use super::Segment;
use crate::common::operation_error::{OperationResult, check_process_stopped};
use crate::data_types::query_context::SegmentQueryContext;
use crate::data_types::vectors::QueryVector;
use crate::id_tracker::IdTrackerRead;
use crate::index::factorized_exact::{
    CandidateMajorExactConfig, CandidateMajorExactPlan, CandidateMajorExactPrepareStats,
    CandidateMajorExactTileOutput, build_candidate_major_exact_plan,
    collect_factorized_exact_candidates, score_candidate_major_exact_tile,
};
use crate::index::field_index::{IntegerPostingAtom, IntegerPostingBatch};
use crate::index::query_optimization::factorized_filter::FactorizedFilterPlan;
use crate::types::{ScoredPoint, VectorName, WithPayload, WithVector};

impl Segment {
    /// Build an owned candidate-major exact plan from the segment's native
    /// payload postings.
    ///
    /// `Ok(None)` is a fail-closed eligibility result: the caller must use the
    /// stock search path when this segment is appendable, quantized, lacks the
    /// requested dense storage, or cannot expose every atomic posting.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_candidate_major_exact(
        &self,
        vector_name: &VectorName,
        queries: &[&QueryVector],
        factorized_plan: &FactorizedFilterPlan,
        top: usize,
        config: CandidateMajorExactConfig,
        query_context: &SegmentQueryContext,
    ) -> OperationResult<Option<CandidateMajorExactPlan>> {
        if self.appendable_flag || queries.is_empty() || queries.len() > u64::BITS as usize {
            return Ok(None);
        }

        let Some(vector_data) = self.vector_data.get(vector_name) else {
            return Ok(None);
        };
        if vector_data.quantized_vectors.borrow().is_some() {
            return Ok(None);
        }

        let vector_context = query_context.get_vector_context(vector_name);
        let hardware_counter = vector_context.hardware_counter();
        let is_stopped = vector_context.is_stopped();
        let id_tracker = self.id_tracker.borrow();
        let total_point_count = id_tracker.total_point_count();

        let atom_build_started = Instant::now();
        let posting_atoms = factorized_plan
            .atoms()
            .iter()
            .map(|atom| {
                atom.query_mask()
                    .words()
                    .first()
                    .copied()
                    .map(|query_mask| IntegerPostingAtom::new(atom.value(), query_mask))
            })
            .collect::<Option<Vec<_>>>();
        let Some(posting_atoms) = posting_atoms else {
            return Ok(None);
        };
        let atom_build_elapsed = atom_build_started.elapsed();

        let posting_fetch_started = Instant::now();
        let payload_index = self.payload_index.borrow();
        let posting_batch = payload_index.with_view(|view| {
            view.query_integer_postings_batch(
                factorized_plan.key(),
                &posting_atoms,
                &hardware_counter,
            )
        })?;
        let Some(posting_batch) = posting_batch else {
            return Ok(None);
        };
        let posting_fetch_elapsed = posting_fetch_started.elapsed();
        let posting_points = posting_batch.postings.iter().fold(0_usize, |sum, posting| {
            sum.saturating_add(posting.point_ids.len())
        });
        let single_valued = posting_batch.single_valued;

        let candidate_collect_started = Instant::now();
        let Some(candidates) = candidates_from_integer_postings(
            posting_batch,
            total_point_count,
            config.ordinal_ordering_threshold,
            &is_stopped,
        )?
        else {
            return Ok(None);
        };
        let candidate_collect_elapsed = candidate_collect_started.elapsed();
        let vector_storage = vector_data.vector_storage.borrow();
        let deleted_points = vector_context
            .deleted_points()
            .unwrap_or_else(|| id_tracker.deleted_point_bitslice());
        let plan_build_started = Instant::now();
        let mut plan = build_candidate_major_exact_plan(
            &vector_storage,
            queries,
            factorized_plan,
            candidates,
            deleted_points,
            top,
            config,
            &is_stopped,
        )?;
        let plan_build_elapsed = plan_build_started.elapsed();
        plan.set_prepare_stats(CandidateMajorExactPrepareStats {
            posting_atoms: posting_atoms.len(),
            posting_points,
            single_valued,
            atom_build_us: duration_micros_u64(atom_build_elapsed),
            posting_fetch_us: duration_micros_u64(posting_fetch_elapsed),
            candidate_collect_us: duration_micros_u64(candidate_collect_elapsed),
            plan_build_us: duration_micros_u64(plan_build_elapsed),
        });
        Ok(Some(plan))
    }

    /// Score one independently schedulable tile while holding only a segment
    /// read lock. Collection-level scheduling may run different tiles on
    /// different workers.
    pub fn score_candidate_major_exact_tile(
        &self,
        vector_name: &VectorName,
        plan: &CandidateMajorExactPlan,
        tile_ordinal: usize,
        query_context: &SegmentQueryContext,
    ) -> OperationResult<Option<CandidateMajorExactTileOutput>> {
        let Some(vector_data) = self.vector_data.get(vector_name) else {
            return Ok(None);
        };
        if vector_data.quantized_vectors.borrow().is_some() {
            return Ok(None);
        }

        let vector_context = query_context.get_vector_context(vector_name);
        let vector_storage = vector_data.vector_storage.borrow();
        score_candidate_major_exact_tile(
            plan,
            tile_ordinal,
            &vector_storage,
            vector_context.hardware_counter(),
            &vector_context.is_stopped(),
        )
        .map(Some)
    }

    /// Convert internal candidate-major results to normal segment results
    /// using Qdrant's native ID, payload and vector materialization.
    pub fn materialize_candidate_major_exact(
        &self,
        vector_name: &VectorName,
        internal_results: Vec<Vec<ScoredPointOffset>>,
        with_payload: &WithPayload,
        with_vector: &WithVector,
        query_context: &SegmentQueryContext,
    ) -> OperationResult<Vec<Vec<ScoredPoint>>> {
        let vector_context = query_context.get_vector_context(vector_name);
        let hardware_counter = vector_context.hardware_counter();
        let is_stopped = vector_context.is_stopped();
        self.with_view(|view| {
            internal_results
                .into_iter()
                .map(|internal_result| {
                    view.process_search_result(
                        internal_result,
                        with_payload,
                        with_vector,
                        &hardware_counter,
                        &is_stopped,
                    )
                })
                .collect()
        })
    }
}

fn duration_micros_u64(duration: Duration) -> u64 {
    u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
}

fn candidates_from_integer_postings(
    posting_batch: IntegerPostingBatch,
    total_point_count: usize,
    ordinal_ordering_threshold: usize,
    is_stopped: &std::sync::atomic::AtomicBool,
) -> OperationResult<Option<Vec<crate::index::factorized_exact::FactorizedExactCandidate>>> {
    let IntegerPostingBatch {
        postings,
        single_valued,
    } = posting_batch;
    let posting_points = if single_valued {
        postings
            .iter()
            .try_fold(0_usize, |sum, posting| {
                sum.checked_add(posting.point_ids.len())
            })
            .ok_or_else(|| {
                crate::common::operation_error::OperationError::service_error(
                    "integer posting point count overflow",
                )
            })?
    } else {
        0
    };
    let direct_single_valued = single_valued
        && should_collect_single_valued_directly(
            posting_points,
            total_point_count,
            ordinal_ordering_threshold,
        );

    if direct_single_valued {
        let mut candidates = Vec::with_capacity(posting_points);
        let mut posting_offset = 0_usize;
        for posting in postings {
            check_process_stopped(is_stopped)?;
            for point_id in posting.point_ids {
                if posting_offset % 4096 == 0 {
                    check_process_stopped(is_stopped)?;
                }
                posting_offset += 1;
                if point_id as usize >= total_point_count {
                    return Ok(None);
                }
                candidates.push(
                    crate::index::factorized_exact::FactorizedExactCandidate::new(
                        point_id,
                        posting.query_mask,
                    ),
                );
            }
        }
        if candidates.len() >= ordinal_ordering_threshold {
            candidates.sort_unstable_by_key(|candidate| candidate.point_id);
        }
        return Ok(Some(candidates));
    }

    let mut point_query_masks = vec![0_u64; total_point_count];
    let mut touched_points = Vec::<PointOffsetType>::new();
    let mut posting_offset = 0_usize;
    for posting in postings {
        check_process_stopped(is_stopped)?;
        for point_id in posting.point_ids {
            if posting_offset % 4096 == 0 {
                check_process_stopped(is_stopped)?;
            }
            posting_offset += 1;
            let Some(mask) = point_query_masks.get_mut(point_id as usize) else {
                return Ok(None);
            };
            // Once the ordinal-scan threshold is reached, the collector no
            // longer needs every touched point: its length alone selects the
            // full-mask scan. Cap this auxiliary vector to avoid recording
            // hundreds of thousands of ordinals that will never be consumed.
            if *mask == 0 && touched_points.len() < ordinal_ordering_threshold {
                touched_points.push(point_id);
            }
            *mask |= posting.query_mask;
        }
    }

    collect_factorized_exact_candidates(
        &point_query_masks,
        &touched_points,
        ordinal_ordering_threshold,
    )
    .map(Some)
}

fn should_collect_single_valued_directly(
    posting_points: usize,
    total_point_count: usize,
    ordinal_ordering_threshold: usize,
) -> bool {
    const MAX_DENSITY_DENOMINATOR: usize = 4;

    posting_points < ordinal_ordering_threshold
        || posting_points <= total_point_count / MAX_DENSITY_DENOMINATOR
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicBool;

    use super::{candidates_from_integer_postings, should_collect_single_valued_directly};
    use crate::index::factorized_exact::FactorizedExactCandidate;
    use crate::index::field_index::{IntegerPosting, IntegerPostingBatch};

    #[test]
    fn single_valued_postings_build_candidates_without_dense_mask() {
        let posting_batch = IntegerPostingBatch {
            postings: vec![
                IntegerPosting {
                    query_mask: 0b0011,
                    point_ids: vec![7, 2],
                },
                IntegerPosting {
                    query_mask: 0b0100,
                    point_ids: vec![5],
                },
            ],
            single_valued: true,
        };

        let candidates =
            candidates_from_integer_postings(posting_batch, 8, 2, &AtomicBool::new(false))
                .unwrap()
                .unwrap();
        assert_eq!(
            candidates,
            vec![
                FactorizedExactCandidate::new(2, 0b0011),
                FactorizedExactCandidate::new(5, 0b0100),
                FactorizedExactCandidate::new(7, 0b0011),
            ],
        );
    }

    #[test]
    fn multi_valued_postings_still_merge_query_masks() {
        let posting_batch = IntegerPostingBatch {
            postings: vec![
                IntegerPosting {
                    query_mask: 0b0001,
                    point_ids: vec![2, 5],
                },
                IntegerPosting {
                    query_mask: 0b0110,
                    point_ids: vec![5, 7],
                },
            ],
            single_valued: false,
        };

        let candidates =
            candidates_from_integer_postings(posting_batch, 8, 2, &AtomicBool::new(false))
                .unwrap()
                .unwrap();
        assert_eq!(
            candidates,
            vec![
                FactorizedExactCandidate::new(2, 0b0001),
                FactorizedExactCandidate::new(5, 0b0111),
                FactorizedExactCandidate::new(7, 0b0110),
            ],
        );
    }

    #[test]
    fn single_valued_collection_switches_to_dense_scan_when_postings_are_dense() {
        assert!(should_collect_single_valued_directly(249, 1000, 128));
        assert!(!should_collect_single_valued_directly(251, 1000, 128));
        assert!(should_collect_single_valued_directly(64, 128, 128));
    }
}
