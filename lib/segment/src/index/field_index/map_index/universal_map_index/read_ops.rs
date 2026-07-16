use std::borrow::Cow;
use std::iter;

use common::bitvec::BitSliceExt;
use common::counter::conditioned_counter::ConditionedCounter;
use common::counter::hardware_counter::HardwareCounterCell;
use common::counter::iterator_hw_measurement::HwMeasurementIteratorExt;
use common::persisted_hashmap::{Key, READ_ENTRY_OVERHEAD};
use common::types::PointOffsetType;
use common::universal_io::UniversalRead;
use itertools::Itertools;

use super::super::read_ops::MapIndexRead;
use super::super::{IdIter, MapIndexKey};
use super::UniversalMapIndex;
use crate::common::operation_error::OperationResult;
use crate::index::field_index::stored_point_to_values::ValuesIter;
use crate::index::field_index::{IntegerPosting, IntegerPostingAtom, IntegerPostingBatch};
use crate::index::payload_config::StorageType;

const PREFETCHED_OFFSETS_MIN_ATOMS: usize = 64;
const PREFETCHED_OFFSETS_MAX_TABLE_BYTES: usize = 256 * 1024;

impl<N: MapIndexKey + Key + ?Sized, S: UniversalRead> MapIndexRead<N> for UniversalMapIndex<N, S> {
    fn check_values_any(
        &self,
        idx: PointOffsetType,
        hw_counter: &HardwareCounterCell,
        check_fn: impl Fn(&N) -> bool,
    ) -> bool {
        let hw_counter = self.make_conditioned_counter(hw_counter);

        // Measure self.deleted access.
        hw_counter
            .payload_index_io_read_counter()
            .incr_delta(size_of::<bool>());

        let is_deleted = self
            .storage
            .deleted
            .get_bit(idx as usize)
            .is_some_and(|b| b);

        if is_deleted {
            return false;
        }

        // FIXME: don't silently ignore errors. Log error? Update ConditionCheckerFn?
        self.storage
            .point_to_values
            .check_values_any(idx, |v| check_fn(v), &hw_counter)
            .unwrap_or(false)
    }

    fn get_values<'a>(
        &'a self,
        idx: PointOffsetType,
        hw_counter: &HardwareCounterCell,
    ) -> Option<impl Iterator<Item = Cow<'a, N>> + 'a>
    where
        N: 'a,
    {
        let hw_counter = self.make_conditioned_counter(hw_counter);

        // We can account cost of reading `bool`, but it will likely be more expensive, than
        // actually reading bool itself.

        if self.storage.deleted.get_bit(idx as usize) == Some(false) {
            self.storage
                .point_to_values
                .values_iter(idx, hw_counter)
                .ok()?
                .map(|iter| Box::new(iter) as Box<dyn Iterator<Item = Cow<'_, N>>>)
        } else {
            None
        }
    }

    fn values_count(&self, idx: PointOffsetType) -> Option<usize> {
        if self.storage.deleted.get_bit(idx as usize) == Some(false) {
            self.storage.point_to_values.get_values_count(idx).ok()?
        } else {
            None
        }
    }

    fn get_indexed_points(&self) -> usize {
        self.storage
            .point_to_values
            .len()
            .saturating_sub(self.deleted_count)
    }

    /// Returns the number of key-value pairs in the index.
    /// Note that is doesn't count deleted pairs.
    fn get_values_count(&self) -> usize {
        self.total_key_value_pairs
    }

    fn get_unique_values_count(&self) -> usize {
        self.storage.value_to_points.keys_count()
    }

    fn get_count_for_value(&self, value: &N, hw_counter: &HardwareCounterCell) -> Option<usize> {
        let hw_counter = self.make_conditioned_counter(hw_counter);

        // Since `value_to_points.get` doesn't actually force read from disk for all values
        // we need to only account for the overhead of hashmap lookup
        hw_counter
            .payload_index_io_read_counter()
            .incr_delta(READ_ENTRY_OVERHEAD);

        match self
            .storage
            .value_to_points
            .unbatched_get_values_count(value)
        {
            Ok(Some(count)) => Some(count),
            Ok(None) => None,
            Err(err) => {
                debug_assert!(
                    false,
                    "Error while getting count for value {value:?}: {err:?}",
                );
                log::error!("Error while getting count for value {value:?}: {err:?}");
                None
            }
        }
    }

    fn get_iterator(&self, value: &N, hw_counter: &HardwareCounterCell) -> IdIter<'_> {
        let hw_counter = self.make_conditioned_counter(hw_counter);

        match self.storage.value_to_points.unbatched_get(value) {
            Ok(Some(values)) => {
                // We're iterating over the whole (mmapped) slice
                hw_counter
                    .payload_index_io_read_counter()
                    .incr_delta(size_of_val(values.as_slice()) + READ_ENTRY_OVERHEAD);

                Box::new(
                    values.into_iter().filter(|idx| {
                        !self.storage.deleted.get_bit(*idx as usize).unwrap_or(false)
                    }),
                )
            }
            Ok(None) => {
                hw_counter
                    .payload_index_io_read_counter()
                    .incr_delta(READ_ENTRY_OVERHEAD);

                Box::new(iter::empty())
            }
            Err(err) => {
                debug_assert!(
                    false,
                    "Error while getting iterator for value {value:?}: {err:?}",
                );
                log::error!("Error while getting iterator for value {value:?}: {err:?}");
                Box::new(iter::empty())
            }
        }
    }

    fn for_each_value(&self, f: impl FnMut(&N) -> OperationResult<()>) -> OperationResult<()> {
        self.storage.value_to_points.for_each_key(f)
    }

    fn for_each_count_per_value(
        &self,
        deferred_internal_id: Option<PointOffsetType>,
        mut f: impl FnMut(&N, usize) -> OperationResult<()>,
    ) -> OperationResult<()> {
        self.storage.value_to_points.for_each_entry(|k, v| {
            let count = v
                .iter()
                .filter(|&&idx| {
                    !self.storage.deleted.get_bit(idx as usize).unwrap_or(true)

                    // TODO(deferred): Maybe we can improve this filter and use take_while instead. For this we
                    // need to make sure that `v` is always sorted which we _can_ enforce when finalizing the index.
                    && deferred_internal_id.is_none_or(|deferred| idx < deferred)
                })
                .unique()
                .count();
            f(k, count)
        })
    }

    fn for_each_value_map(
        &self,
        hw_counter: &HardwareCounterCell,
        mut f: impl FnMut(&N, &mut dyn Iterator<Item = PointOffsetType>) -> OperationResult<()>,
    ) -> OperationResult<()> {
        let hw_counter = self.make_conditioned_counter(hw_counter);
        let deleted = &self.storage.deleted;

        self.storage.value_to_points.for_each_entry(|k, v| {
            hw_counter
                .payload_index_io_read_counter()
                .incr_delta(k.write_bytes());

            let mut iter = v
                .iter()
                .copied()
                .filter(|idx| !deleted.get_bit(*idx as usize).unwrap_or(true))
                .measure_hw_with_acc(
                    hw_counter.new_accumulator(),
                    size_of::<PointOffsetType>(),
                    |i| i.payload_index_io_read_counter(),
                );

            f(k, &mut iter)
        })
    }

    fn storage_type(&self) -> StorageType {
        StorageType::Mmap {
            is_on_disk: self.is_on_disk,
        }
    }

    fn ram_usage_bytes(&self) -> usize {
        self.storage.ram_usage_bytes()
    }

    fn telemetry_index_type(&self) -> &'static str {
        "mmap_map"
    }
}

impl<N: MapIndexKey + Key + ?Sized, S: UniversalRead> UniversalMapIndex<N, S> {
    pub fn for_points_values(
        &self,
        mut points: impl Iterator<Item = PointOffsetType>,
        hw_counter: &HardwareCounterCell,
        mut f: impl FnMut(PointOffsetType, ValuesIter<'_, N>),
    ) -> OperationResult<()> {
        let hw_counter = self.make_conditioned_counter(hw_counter);

        points.try_for_each(|idx| {
            if self.storage.deleted.get_bit(idx as usize) != Some(false) {
                return Ok(());
            }
            if let Some(iter) = self.storage.point_to_values.values_iter(idx, hw_counter)? {
                f(idx, iter);
            }
            Ok(())
        })
    }

    pub(super) fn make_conditioned_counter<'a>(
        &self,
        hw_counter: &'a HardwareCounterCell,
    ) -> ConditionedCounter<'a> {
        ConditionedCounter::new(self.is_on_disk, hw_counter)
    }

    pub fn is_on_disk(&self) -> bool {
        self.is_on_disk
    }
}

impl<S: UniversalRead> UniversalMapIndex<crate::types::IntPayloadType, S> {
    /// Read all requested integer postings through one persisted-hashmap
    /// pipeline. Pipeline callbacks may arrive out of input order, so each
    /// result is written back by atom ordinal and exposed in original order.
    pub(in crate::index::field_index::map_index) fn batched_integer_postings(
        &self,
        atoms: &[IntegerPostingAtom],
        hw_counter: &HardwareCounterCell,
        single_valued: bool,
    ) -> OperationResult<IntegerPostingBatch> {
        let hw_counter = self.make_conditioned_counter(hw_counter);
        let deleted = &self.storage.deleted;
        let mut point_ids = vec![Vec::new(); atoms.len()];

        let requests = || {
            atoms
                .iter()
                .enumerate()
                .map(|(atom_ordinal, atom)| (atom_ordinal, &atom.value))
        };
        let mut collect_posting = |atom_ordinal: usize, values: Option<&[PointOffsetType]>| {
            hw_counter
                .payload_index_io_read_counter()
                .incr_delta(READ_ENTRY_OVERHEAD + values.map_or(0, size_of_val));

            if let Some(values) = values {
                point_ids[atom_ordinal].reserve(values.len());
                point_ids[atom_ordinal].extend(
                    values
                        .iter()
                        .copied()
                        .filter(|point_id| !deleted.get_bit(*point_id as usize).unwrap_or(false)),
                );
            }
            Ok::<(), crate::common::operation_error::OperationError>(())
        };

        let bucket_table_bytes = self
            .storage
            .value_to_points
            .keys_count()
            .saturating_mul(size_of::<u64>());
        if should_prefetch_offsets(atoms.len(), bucket_table_bytes) {
            hw_counter
                .payload_index_io_read_counter()
                .incr_delta(bucket_table_bytes);
            self.storage
                .value_to_points
                .for_each_entry_in_iter_prefetched_offsets(requests(), &mut collect_posting)?;
        } else {
            self.storage
                .value_to_points
                .for_each_entry_in_iter(requests(), &mut collect_posting)?;
        }

        let postings = atoms
            .iter()
            .zip(point_ids)
            .map(|(atom, point_ids)| IntegerPosting {
                query_mask: atom.query_mask,
                point_ids,
            })
            .collect();
        Ok(IntegerPostingBatch {
            postings,
            single_valued,
        })
    }
}

fn should_prefetch_offsets(atom_count: usize, bucket_table_bytes: usize) -> bool {
    atom_count >= PREFETCHED_OFFSETS_MIN_ATOMS
        && bucket_table_bytes <= PREFETCHED_OFFSETS_MAX_TABLE_BYTES
}

#[cfg(test)]
mod tests {
    use super::{PREFETCHED_OFFSETS_MAX_TABLE_BYTES, should_prefetch_offsets};

    #[test]
    fn prefetched_offset_gate_covers_atom_and_table_boundaries() {
        assert!(!should_prefetch_offsets(
            63,
            PREFETCHED_OFFSETS_MAX_TABLE_BYTES
        ));
        assert!(should_prefetch_offsets(
            64,
            PREFETCHED_OFFSETS_MAX_TABLE_BYTES
        ));
        assert!(!should_prefetch_offsets(
            64,
            PREFETCHED_OFFSETS_MAX_TABLE_BYTES + 1
        ));
    }
}
