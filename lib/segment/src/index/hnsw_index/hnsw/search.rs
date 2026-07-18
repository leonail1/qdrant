use std::sync::Arc;

use common::bitvec::BitSlice;
use common::counter::hardware_counter::HardwareCounterCell;
use common::cow::BoxCow;
use common::types::{DeferredBehavior, PointOffsetType, ScoredPointOffset};

use super::HNSWIndex;
use crate::common::operation_error::OperationResult;
use crate::data_types::query_context::VectorQueryContext;
use crate::data_types::vectors::{QueryVector, VectorInternal};
use crate::id_tracker::IdTrackerRead;
use crate::index::PayloadIndexRead;
use crate::index::hnsw_index::graph_layers::{GraphLayersWithVectors, SearchAlgorithm};
use crate::index::hnsw_index::point_scorer::{BatchFilteredSearcher, FilteredScorer};
use crate::index::query_estimator::adjust_to_available_vectors;
use crate::index::vector_index_search_common::{
    get_oversampled_top, is_quantized_search, postprocess_search_result,
};
use crate::payload_storage::FilterContext;
#[cfg(feature = "gpu")]
use crate::types::Condition;
use crate::types::{ACORN_MAX_SELECTIVITY_DEFAULT, Filter, SearchParams};
use crate::vector_storage::quantized::quantized_vectors::QuantizedVectors;
use crate::vector_storage::query::DiscoverQuery;
use crate::vector_storage::{VectorStorageEnum, VectorStorageRead, new_raw_scorer};

#[cfg(feature = "gpu")]
pub(super) fn gpu_filter_cacheable(filter: &Filter) -> bool {
    filter.iter_conditions().all(|condition| match condition {
        Condition::Field(_) | Condition::IsEmpty(_) | Condition::IsNull(_) => true,
        Condition::Nested(nested) => gpu_filter_cacheable(nested.filter()),
        Condition::Filter(filter) => gpu_filter_cacheable(filter),
        Condition::HasId(_) | Condition::HasVector(_) | Condition::CustomIdChecker(_) => false,
    })
}
#[cfg(feature = "gpu")]
use crate::index::hnsw_index::gpu::gpu_exact_search::{
    GPU_EXACT_SUBMISSION_BATCH_LIMIT, GpuExactSearchCache, GpuVisibilitySnapshot,
};
#[cfg(feature = "gpu")]
use crate::index::hnsw_index::gpu::gpu_filtered_graph_search::GpuFilteredGraphSearchCache;
#[cfg(feature = "gpu")]
use crate::index::hnsw_index::gpu::gpu_vector_storage::GpuVectorStorage;
#[cfg(feature = "gpu")]
use crate::index::hnsw_index::gpu::{
    GPU_DEVICES_MANAGER, GPU_FILTERED_GRAPH_MIN_CANDIDATES, get_gpu_search_config,
};
#[cfg(feature = "gpu")]
use crate::vector_storage::check_deleted_condition;

impl HNSWIndex {
    #[cfg(feature = "gpu")]
    fn gpu_vector_storage_cache(
        &self,
        vector_storage: &VectorStorageEnum,
        is_stopped: &std::sync::atomic::AtomicBool,
    ) -> Option<&std::sync::Arc<GpuVectorStorage>> {
        self.gpu_vector_storage
            .get_or_init(|| {
                let create = || -> OperationResult<Option<std::sync::Arc<GpuVectorStorage>>> {
                    let manager = GPU_DEVICES_MANAGER.read();
                    let Some(manager) = manager.as_ref() else {
                        return Ok(None);
                    };
                    let Some(device) = manager.lock_device(is_stopped)? else {
                        return Ok(None);
                    };
                    let gpu_vectors = std::sync::Arc::new(GpuVectorStorage::new(
                        device.device(),
                        vector_storage,
                        None,
                        false,
                        is_stopped,
                    )?);
                    log::info!(
                        "Initialized shared GPU vector storage: points={}, aligned_dim={}, resident_vector_bytes={}",
                        gpu_vectors.num_vectors(),
                        gpu_vectors.dim(),
                        gpu_vectors.resident_vector_bytes(),
                    );
                    Ok(Some(gpu_vectors))
                };

                match create() {
                    Ok(storage) => storage,
                    Err(error) => {
                        log::warn!("Failed to initialize shared GPU vector storage: {error}");
                        None
                    }
                }
            })
            .as_ref()
    }

    #[cfg(feature = "gpu")]
    fn gpu_filtered_graph_search_cache(
        &self,
        vector_storage: &VectorStorageEnum,
        ef: usize,
        is_stopped: &std::sync::atomic::AtomicBool,
    ) -> Option<&std::sync::Arc<GpuFilteredGraphSearchCache>> {
        let config = get_gpu_search_config();
        if !config.enabled || !config.router_enabled {
            return None;
        }

        self.gpu_filtered_graph_search
            .get_or_init(|| {
                let create =
                    || -> OperationResult<Option<std::sync::Arc<GpuFilteredGraphSearchCache>>> {
                        let Some(gpu_vectors) =
                            self.gpu_vector_storage_cache(vector_storage, is_stopped)
                        else {
                            return Ok(None);
                        };
                        let cache = GpuFilteredGraphSearchCache::new(
                            gpu_vectors.device(),
                            gpu_vectors.clone(),
                            &self.graph,
                            vector_storage.total_vector_count(),
                            ef,
                            config.contexts,
                            is_stopped,
                        )?;
                        log::info!(
                            "Initialized GPU filtered graph search: points={}, ef={}, contexts={}",
                            vector_storage.total_vector_count(),
                            ef,
                            config.contexts,
                        );
                        Ok(Some(std::sync::Arc::new(cache)))
                    };

                match create() {
                    Ok(cache) => cache,
                    Err(error) => {
                        log::warn!("Failed to initialize GPU filtered graph cache: {error}");
                        None
                    }
                }
            })
            .as_ref()
    }

    #[cfg(feature = "gpu")]
    fn gpu_exact_search_cache(
        &self,
        vector_storage: &VectorStorageEnum,
        is_stopped: &std::sync::atomic::AtomicBool,
    ) -> Option<&std::sync::Arc<GpuExactSearchCache>> {
        let config = get_gpu_search_config();
        if !config.enabled {
            return None;
        }

        self.gpu_exact_search
            .get_or_init(|| {
                let create = || -> OperationResult<Option<std::sync::Arc<GpuExactSearchCache>>> {
                    let Some(gpu_vectors) =
                        self.gpu_vector_storage_cache(vector_storage, is_stopped)
                    else {
                        return Ok(None);
                    };
                    let cache = GpuExactSearchCache::new_with_vector_storage(
                        gpu_vectors.clone(),
                        config.max_candidates,
                        config.contexts,
                        config.batch_max_queries,
                        config.batch_window_us,
                    )?;
                    Ok(Some(std::sync::Arc::new(cache)))
                };

                match create() {
                    Ok(cache) => cache,
                    Err(error) => {
                        log::warn!("Failed to initialize GPU exact search cache: {error}");
                        None
                    }
                }
            })
            .as_ref()
    }

    #[cfg(feature = "gpu")]
    fn search_plain_gpu_exact(
        &self,
        query_vectors: &[&QueryVector],
        filtered_points: &Arc<Vec<PointOffsetType>>,
        top: usize,
        params: Option<&SearchParams>,
        vector_query_context: &VectorQueryContext,
        predicate_ns: u64,
        filter_cache_hit: Option<bool>,
    ) -> OperationResult<Option<Vec<Vec<ScoredPointOffset>>>> {
        let config = get_gpu_search_config();
        if !config.enabled || !params.is_some_and(|params| params.exact) {
            return Ok(None);
        }
        let dense_queries = query_vectors
            .iter()
            .map(|query| match query {
                QueryVector::Nearest(VectorInternal::Dense(vector)) => Some(vector.as_slice()),
                _ => None,
            })
            .collect::<Option<Vec<_>>>();
        let Some(dense_queries) = dense_queries else {
            return Ok(None);
        };

        let id_tracker = self.id_tracker.borrow();
        let vector_storage = self.vector_storage.borrow();
        let deleted_points = vector_query_context
            .deleted_points()
            .unwrap_or_else(|| id_tracker.deleted_point_bitslice());
        let deleted_vectors = vector_storage.deleted_vector_bitslice();
        let point_count = vector_storage.total_vector_count();
        let visibility_started = std::time::Instant::now();
        let visibility_snapshot = vector_query_context
            .deleted_points()
            .zip(vector_query_context.deleted_points_generation())
            .filter(|(deleted, generation)| {
                // A generation is only attached by ProxySegment after it has
                // frozen the wrapped segment and copied its deletion state
                // into this authoritative read-view mask. Wrapped point and
                // vector tombstones may therefore be non-zero, but are
                // already covered by `deleted`; requiring zero here would
                // reject every dynamic optimizer transition.
                *generation != 0 && deleted.len() <= point_count
            })
            .map(|(deleted, generation)| GpuVisibilitySnapshot {
                generation,
                deleted,
                point_count,
            });
        let no_deletions = vector_query_context.deleted_points().is_none()
            && id_tracker.deleted_point_count() == 0
            && vector_storage.deleted_vector_count() == 0;
        let reuse_candidate_buffer =
            filter_cache_hit.is_some() && (no_deletions || visibility_snapshot.is_some());
        let candidates = if reuse_candidate_buffer {
            filtered_points.clone()
        } else {
            Arc::new(
                filtered_points
                    .iter()
                    .copied()
                    .filter(|&point_id| {
                        check_deleted_condition(point_id, deleted_vectors, deleted_points)
                    })
                    .collect::<Vec<_>>(),
            )
        };
        let visibility_ns = visibility_started.elapsed().as_nanos() as u64;
        if candidates.len() < config.min_candidates || candidates.len() > config.max_candidates {
            return Ok(None);
        }

        let cache_started = std::time::Instant::now();
        let cache =
            self.gpu_exact_search_cache(&vector_storage, &vector_query_context.is_stopped());
        let cache_ns = cache_started.elapsed().as_nanos() as u64;
        let Some(cache) = cache else {
            return Ok(None);
        };
        let mut results = Vec::with_capacity(dense_queries.len());
        if let [query] = dense_queries.as_slice() {
            let Some(result) = cache.search_coalesced(
                query,
                candidates.clone(),
                top,
                reuse_candidate_buffer,
                visibility_snapshot,
            )?
            else {
                return Ok(None);
            };
            results.push(result);
        } else {
            let submission_batch_size = if GpuExactSearchCache::supports_submission_batch(
                candidates.len(),
                top,
                reuse_candidate_buffer,
            ) {
                GPU_EXACT_SUBMISSION_BATCH_LIMIT
            } else {
                1
            };
            for queries in dense_queries.chunks(submission_batch_size) {
                let Some(mut batch_results) = cache.search_batch(
                    queries,
                    candidates.clone(),
                    top,
                    reuse_candidate_buffer,
                    visibility_snapshot,
                )?
                else {
                    return Ok(None);
                };
                results.append(&mut batch_results);
            }
        }

        let postprocess_started = std::time::Instant::now();
        let quantized_vectors = self.quantized_vectors.borrow();
        for (search_result, query_vector) in results.iter_mut().zip(query_vectors) {
            *search_result = postprocess_search_result(
                std::mem::take(search_result),
                deleted_points,
                &vector_storage,
                quantized_vectors.as_ref(),
                query_vector,
                params,
                top,
                vector_query_context.hardware_counter(),
            )?;
        }
        let postprocess_ns = postprocess_started.elapsed().as_nanos() as u64;
        cache.record_outer_breakdown(
            dense_queries.len(),
            predicate_ns,
            visibility_ns,
            cache_ns,
            postprocess_ns,
            filter_cache_hit,
        );
        Ok(Some(results))
    }

    pub(super) fn search_with_graph(
        &self,
        vector: &QueryVector,
        filter: Option<&Filter>,
        top: usize,
        params: Option<&SearchParams>,
        custom_entry_points: Option<&[PointOffsetType]>,
        vector_query_context: &VectorQueryContext,
    ) -> OperationResult<Vec<ScoredPointOffset>> {
        let ef = params
            .and_then(|params| params.hnsw_ef)
            .unwrap_or(self.config.ef);
        let acorn_enabled = params
            .and_then(|params| params.acorn)
            .is_some_and(|acorn| acorn.enable);
        let acorn_max_selectivity = params
            .and_then(|params| params.acorn)
            .and_then(|acorn| acorn.max_selectivity)
            .map_or(ACORN_MAX_SELECTIVITY_DEFAULT, |v| *v);

        let is_stopped = vector_query_context.is_stopped();

        let id_tracker = self.id_tracker.borrow();
        let payload_index = self.payload_index.borrow();
        let vector_storage = self.vector_storage.borrow();
        let quantized_vectors = self.quantized_vectors.borrow();

        let deleted_points = vector_query_context
            .deleted_points()
            .unwrap_or_else(|| id_tracker.deleted_point_bitslice());

        let hw_counter = vector_query_context.hardware_counter();
        let oversampled_top = get_oversampled_top(quantized_vectors.as_ref(), params, top);

        let mut algorithm = SearchAlgorithm::Hnsw;
        if acorn_enabled
            && self.config.m0 != 0
            && let Some(filter) = filter
        {
            // NOTE: technically we also might want to use ACORN for unfiltered
            // searches for segments with a lot of deleted points. But in
            // practice, such segments most likely to be picked by an optimizer
            // soon.

            let available_vector_count = vector_storage.available_vector_count();
            let selectivity = if available_vector_count == 0 {
                1.0
            } else {
                let query_point_cardinality =
                    payload_index.with_view(|v| v.estimate_cardinality(filter, &hw_counter))?;
                let query_cardinality = adjust_to_available_vectors(
                    query_point_cardinality,
                    available_vector_count,
                    id_tracker.available_point_count(),
                );
                query_cardinality.exp as f64 / available_vector_count as f64
            };
            if selectivity <= acorn_max_selectivity {
                algorithm = SearchAlgorithm::Acorn;
            }
        }

        #[cfg(feature = "gpu")]
        if algorithm == SearchAlgorithm::Hnsw
            && custom_entry_points.is_none()
            && let Some(filter) = filter
            && let QueryVector::Nearest(VectorInternal::Dense(query)) = vector
        {
            let config = get_gpu_search_config();
            let query_point_cardinality =
                payload_index.with_view(|v| v.estimate_cardinality(filter, &hw_counter))?;
            let query_cardinality = adjust_to_available_vectors(
                query_point_cardinality,
                vector_storage.available_vector_count(),
                id_tracker.available_point_count(),
            );
            let cacheable = config.enabled
                && config.router_enabled
                && query_cardinality.max >= GPU_FILTERED_GRAPH_MIN_CANDIDATES
                && query_cardinality.max <= config.max_candidates
                && !payload_index.is_appendable()
                && gpu_filter_cacheable(filter);
            let visibility_snapshot = vector_query_context
                .deleted_points()
                .zip(vector_query_context.deleted_points_generation())
                .filter(|(deleted, generation)| {
                    *generation != 0 && deleted.len() <= vector_storage.total_vector_count()
                })
                .map(|(deleted, generation)| GpuVisibilitySnapshot {
                    generation,
                    deleted,
                    point_count: vector_storage.total_vector_count(),
                });
            let static_deletions =
                id_tracker.deleted_point_count() != 0 || vector_storage.deleted_vector_count() != 0;
            if cacheable && (!static_deletions || visibility_snapshot.is_some()) {
                let payload_epoch = payload_index.mutation_epoch();
                let cached = self.gpu_filter_candidates.lock().get(filter, payload_epoch);
                let candidates = if let Some(candidates) = cached {
                    candidates
                } else {
                    let candidates = Arc::new(payload_index.with_view(|view| {
                        view.iter_filtered_points(
                            filter,
                            &query_cardinality,
                            &hw_counter,
                            &is_stopped,
                            DeferredBehavior::IncludeAll,
                        )
                        .map(|points| points.collect::<Vec<_>>())
                    })?);
                    self.gpu_filter_candidates.lock().insert(
                        filter.clone(),
                        candidates.clone(),
                        payload_epoch,
                    );
                    candidates
                };
                let gpu_search = payload_index.with_view(|payload_index_view| {
                    let filter_context = payload_index_view.filter_context(filter, &hw_counter)?;
                    let mut points_scorer = construct_search_scorer(
                        vector,
                        &vector_storage,
                        quantized_vectors.as_ref(),
                        deleted_points,
                        params,
                        vector_query_context.hardware_counter(),
                        Some(filter_context),
                    )?;
                    let Some(entry) = self.graph.search_level_zero_entry(
                        &mut points_scorer,
                        None,
                        &is_stopped,
                    )?
                    else {
                        return Ok(Some(Vec::new()));
                    };
                    let Some(cache) =
                        self.gpu_filtered_graph_search_cache(&vector_storage, ef, &is_stopped)
                    else {
                        return Ok(None);
                    };
                    cache.search(
                        query,
                        entry.idx,
                        candidates,
                        visibility_snapshot,
                        oversampled_top,
                        ef,
                    )
                });
                match gpu_search {
                    Ok(Some(search_result)) => {
                        return postprocess_search_result(
                            search_result,
                            id_tracker.deleted_point_bitslice(),
                            &vector_storage,
                            quantized_vectors.as_ref(),
                            vector,
                            params,
                            top,
                            vector_query_context.hardware_counter(),
                        );
                    }
                    Ok(None) => {}
                    Err(error) => {
                        log::warn!(
                            "GPU filtered graph search failed; falling back to CPU: {error}"
                        );
                    }
                }
            }
        }

        let search_with_vectors = || -> OperationResult<Option<Vec<ScoredPointOffset>>> {
            match algorithm {
                SearchAlgorithm::Hnsw => (),
                // ACORN is not implemented for graph with vectors yet (but possible)
                SearchAlgorithm::Acorn => return Ok(None),
            }
            if !self.graph.has_inline_vectors()
                || !is_quantized_search(quantized_vectors.as_ref(), params)
            {
                return Ok(None);
            }
            let Some(quantized_vectors) = quantized_vectors.as_ref() else {
                return Ok(None);
            };

            payload_index.with_view(|payload_index_view| {
                // Quantized vectors are "link vectors"
                let link_scorer_filtered = FilteredScorer::new(
                    vector.to_owned(),
                    &vector_storage,
                    Some(quantized_vectors),
                    filter
                        .map(|f| {
                            payload_index_view
                                .filter_context(f, &hw_counter)
                                .map(BoxCow::Owned)
                        })
                        .transpose()?,
                    deleted_points,
                    vector_query_context.hardware_counter(),
                )?;
                let Some(link_scorer_filtered_bytes) = link_scorer_filtered.scorer_bytes() else {
                    return Ok(None);
                };

                // Full vectors are "base vectors"
                let base_scorer = new_raw_scorer(
                    vector.to_owned(),
                    &vector_storage,
                    vector_query_context.hardware_counter(),
                )?;
                let Some(base_scorer_bytes) = base_scorer.scorer_bytes() else {
                    return Ok(None);
                };

                Ok(Some(self.graph.search_with_vectors(
                    top,
                    std::cmp::max(ef, oversampled_top),
                    &link_scorer_filtered,
                    &link_scorer_filtered_bytes,
                    base_scorer_bytes,
                    custom_entry_points,
                    &vector_query_context.is_stopped(),
                )?))
            })
        };

        let regular_search = || -> OperationResult<Vec<ScoredPointOffset>> {
            payload_index.with_view(|payload_index_view| {
                let filter_context = filter
                    .map(|f| payload_index_view.filter_context(f, &hw_counter))
                    .transpose()?;
                let points_scorer = construct_search_scorer(
                    vector,
                    &vector_storage,
                    quantized_vectors.as_ref(),
                    deleted_points,
                    params,
                    vector_query_context.hardware_counter(),
                    filter_context,
                )?;

                let search_result = self.graph.search(
                    oversampled_top,
                    ef,
                    algorithm,
                    points_scorer,
                    custom_entry_points,
                    &is_stopped,
                )?;

                postprocess_search_result(
                    search_result,
                    id_tracker.deleted_point_bitslice(),
                    &vector_storage,
                    quantized_vectors.as_ref(),
                    vector,
                    params,
                    top,
                    vector_query_context.hardware_counter(),
                )
            })
        };

        // Try to use graph with vectors first.
        if let Some(search_result) = search_with_vectors()? {
            Ok(search_result)
        } else {
            // Graph with vectors is not available, fallback to regular graph search.
            regular_search()
        }
    }

    pub(super) fn search_vectors_with_graph(
        &self,
        vectors: &[&QueryVector],
        filter: Option<&Filter>,
        top: usize,
        params: Option<&SearchParams>,
        vector_query_context: &VectorQueryContext,
    ) -> OperationResult<Vec<Vec<ScoredPointOffset>>> {
        vectors
            .iter()
            .map(|&vector| match vector {
                QueryVector::Discover(discover_query) => self.discover_search_with_graph(
                    discover_query.clone(),
                    filter,
                    top,
                    params,
                    vector_query_context,
                ),
                QueryVector::Nearest(_)
                | QueryVector::RecommendBestScore(_)
                | QueryVector::RecommendSumScores(_)
                | QueryVector::Context(_)
                | QueryVector::FeedbackNaive(_) => {
                    self.search_with_graph(vector, filter, top, params, None, vector_query_context)
                }
            })
            .collect()
    }

    fn search_plain_iterator_batched(
        &self,
        query_vectors: &[&QueryVector],
        points: impl Iterator<Item = PointOffsetType>,
        top: usize,
        params: Option<&SearchParams>,
        vector_query_context: &VectorQueryContext,
    ) -> OperationResult<Vec<Vec<ScoredPointOffset>>> {
        let id_tracker = self.id_tracker.borrow();
        let vector_storage = self.vector_storage.borrow();
        let quantized_vectors = self.quantized_vectors.borrow();

        let deleted_points = vector_query_context
            .deleted_points()
            .unwrap_or_else(|| id_tracker.deleted_point_bitslice());

        let is_stopped = vector_query_context.is_stopped();
        let oversampled_top = get_oversampled_top(quantized_vectors.as_ref(), params, top);

        let batch_filtered_searcher = construct_batch_searcher(
            query_vectors,
            &vector_storage,
            quantized_vectors.as_ref(),
            oversampled_top,
            deleted_points,
            params,
            vector_query_context.hardware_counter(),
            None,
        )?;
        let mut search_results = batch_filtered_searcher.peek_top_iter(points, &is_stopped)?;
        for (search_result, query_vector) in search_results.iter_mut().zip(query_vectors) {
            *search_result = postprocess_search_result(
                std::mem::take(search_result),
                id_tracker.deleted_point_bitslice(),
                &vector_storage,
                quantized_vectors.as_ref(),
                query_vector,
                params,
                top,
                vector_query_context.hardware_counter(),
            )?;
        }
        Ok(search_results)
    }

    pub(super) fn search_plain_batched(
        &self,
        vectors: &[&QueryVector],
        filtered_points: impl Iterator<Item = PointOffsetType>,
        top: usize,
        params: Option<&SearchParams>,
        vector_query_context: &VectorQueryContext,
    ) -> OperationResult<Vec<Vec<ScoredPointOffset>>> {
        self.search_plain_iterator_batched(
            vectors,
            filtered_points,
            top,
            params,
            vector_query_context,
        )
    }

    pub(super) fn search_plain_unfiltered_batched(
        &self,
        vectors: &[&QueryVector],
        top: usize,
        params: Option<&SearchParams>,
        vector_query_context: &VectorQueryContext,
    ) -> OperationResult<Vec<Vec<ScoredPointOffset>>> {
        let id_tracker = self.id_tracker.borrow();
        let ids_iterator = id_tracker.point_mappings().iter_internal();
        self.search_plain_iterator_batched(vectors, ids_iterator, top, params, vector_query_context)
    }

    pub(super) fn search_vectors_plain(
        &self,
        vectors: &[&QueryVector],
        filter: &Filter,
        top: usize,
        params: Option<&SearchParams>,
        vector_query_context: &VectorQueryContext,
    ) -> OperationResult<Vec<Vec<ScoredPointOffset>>> {
        let hw_counter = &vector_query_context.hardware_counter();
        let is_stopped = &vector_query_context.is_stopped();

        let payload_index = self.payload_index.borrow();

        #[cfg(feature = "gpu")]
        let (filtered_points, predicate_ns, filter_cache_hit) = {
            let predicate_started = std::time::Instant::now();
            let query_cardinality =
                payload_index.with_view(|v| v.estimate_cardinality(filter, hw_counter))?;
            // The predicate projection belongs to the database execution
            // layer, not to the GPU exact kernel. Reuse it for stock-compatible
            // CPU plain scoring as well, so a GPU miss or cost-model fallback
            // does not re-run the payload iterator on every query. Keep tiny
            // predicates on Qdrant's native bitmap iterator: caching thousands
            // of distinct, cheap projections costs more than recomputing them.
            let config = get_gpu_search_config();
            let cacheable = config.enabled
                && query_cardinality.max >= config.min_candidates
                && !payload_index.is_appendable()
                && gpu_filter_cacheable(filter);
            let payload_epoch = payload_index.mutation_epoch();
            let cached = cacheable
                .then(|| self.gpu_filter_candidates.lock().get(filter, payload_epoch))
                .flatten();
            let cache_hit = cacheable.then_some(cached.is_some());
            let candidates = if let Some(candidates) = cached {
                candidates
            } else {
                // Assume query is already estimated to be small enough so we can iterate over all matched ids.
                let candidates = Arc::new(payload_index.with_view(|v| {
                    v.iter_filtered_points(
                        filter,
                        &query_cardinality,
                        hw_counter,
                        is_stopped,
                        // No deferred filtering here since it is an HNSW index.
                        DeferredBehavior::IncludeAll,
                    )
                    .map(|it| it.collect::<Vec<_>>())
                })?);
                if cacheable {
                    self.gpu_filter_candidates.lock().insert(
                        filter.clone(),
                        candidates.clone(),
                        payload_epoch,
                    );
                }
                candidates
            };
            (
                candidates,
                predicate_started.elapsed().as_nanos() as u64,
                cache_hit,
            )
        };

        #[cfg(not(feature = "gpu"))]
        let filtered_points = Arc::new(payload_index.with_view(|v| {
            let query_cardinality = v.estimate_cardinality(filter, hw_counter)?;
            v.iter_filtered_points(
                filter,
                &query_cardinality,
                hw_counter,
                is_stopped,
                DeferredBehavior::IncludeAll,
            )
            .map(|it| it.collect::<Vec<_>>())
        })?);

        #[cfg(feature = "gpu")]
        match self.search_plain_gpu_exact(
            vectors,
            &filtered_points,
            top,
            params,
            vector_query_context,
            predicate_ns,
            filter_cache_hit,
        ) {
            Ok(Some(results)) => return Ok(results),
            Ok(None) => {}
            Err(error) => {
                log::warn!("GPU exact filtered search failed; falling back to CPU: {error}");
            }
        }

        self.search_plain_batched(
            vectors,
            filtered_points.iter().copied(),
            top,
            params,
            vector_query_context,
        )
    }

    fn discover_search_with_graph(
        &self,
        discover_query: DiscoverQuery<VectorInternal>,
        filter: Option<&Filter>,
        top: usize,
        params: Option<&SearchParams>,
        vector_query_context: &VectorQueryContext,
    ) -> OperationResult<Vec<ScoredPointOffset>> {
        // Stage 1: Find best entry points using Context search
        let query_vector = QueryVector::Context(discover_query.pairs.clone().into());

        const DISCOVERY_ENTRY_POINT_COUNT: usize = 10;

        let custom_entry_points: Vec<_> = self
            .search_with_graph(
                &query_vector,
                filter,
                DISCOVERY_ENTRY_POINT_COUNT,
                params,
                None,
                vector_query_context,
            )
            .map(|search_result| search_result.iter().map(|x| x.idx).collect())?;

        // Stage 2: Discover search with entry points
        let query_vector = QueryVector::Discover(discover_query);

        self.search_with_graph(
            &query_vector,
            filter,
            top,
            params,
            Some(&custom_entry_points),
            vector_query_context,
        )
    }
}

fn construct_search_scorer<'a>(
    vector: &QueryVector,
    vector_storage: &'a VectorStorageEnum,
    quantized_storage: Option<&'a QuantizedVectors>,
    deleted_points: &'a BitSlice,
    params: Option<&SearchParams>,
    hardware_counter: HardwareCounterCell,
    filter_context: Option<Box<dyn FilterContext + 'a>>,
) -> OperationResult<FilteredScorer<'a>> {
    let quantization_enabled = is_quantized_search(quantized_storage, params);
    FilteredScorer::new(
        vector.to_owned(),
        vector_storage,
        quantization_enabled.then_some(quantized_storage).flatten(),
        filter_context.map(BoxCow::Owned),
        deleted_points,
        hardware_counter,
    )
}

#[allow(clippy::too_many_arguments)]
fn construct_batch_searcher<'a>(
    vectors: &[&QueryVector],
    vector_storage: &'a VectorStorageEnum,
    quantized_storage: Option<&'a QuantizedVectors>,
    top: usize,
    deleted_points: &'a BitSlice,
    params: Option<&SearchParams>,
    hardware_counter: HardwareCounterCell,
    filter_context: Option<Box<dyn FilterContext + 'a>>,
) -> OperationResult<BatchFilteredSearcher<'a>> {
    let quantization_enabled = is_quantized_search(quantized_storage, params);
    BatchFilteredSearcher::new(
        vectors,
        vector_storage,
        quantization_enabled.then_some(quantized_storage).flatten(),
        filter_context.map(BoxCow::Owned),
        top,
        deleted_points,
        hardware_counter,
    )
}
