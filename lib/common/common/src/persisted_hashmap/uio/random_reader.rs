use std::ops::Range;

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use super::super::read_err;
use super::{BucketOffset, Key, MaybeIncompleteEntry, MaybeIncompleteEntryKind, UniversalHashMap};
use crate::aligned_buf::AlignedBuf;
use crate::generic_consts::Random;
use crate::persisted_hashmap::uio::parse_bucket_offset;
use crate::universal_io::{
    BorrowedReadPipeline, Result, UniversalIoError, UniversalRead, UserData,
};

pub(super) enum Request<'a, K: Key + ?Sized> {
    /// Request an entry by the given offset with unknown key.
    Offset(u64),
    /// Request an entry by the given key. Will resolve the offset first.
    Key(&'a K),
    /// Request an entry by a key whose exact entry range was resolved from a
    /// prefetched bucket-offset table.
    KeyAtOffset {
        requested_key: &'a K,
        offset: u64,
        length: u64,
    },
}

/// State machine driver for [`UniversalHashMap::for_each_sparse`].
struct PipelineDriver<'map, 'key, U, K, V, S>
where
    U: UserData,
    K: Key + ?Sized + 'key,
    V: Sized + Copy + FromBytes + Immutable + IntoBytes + KnownLayout,
    S: UniversalRead,
{
    map: &'map UniversalHashMap<K, V, S>,
    entry_kind: MaybeIncompleteEntryKind,
    queue: Vec<Entry<'key, U, K>>,
    entry_read_size_est: u64,
    file_len: u64,
}

struct Entry<'a, U: UserData, K: Key + ?Sized> {
    user_data: U,
    state: State<'a, K>,
}

/// Lifecycle:
///
/// - [`Request::Offset`]             →             [`State::ReadingEntry`]
/// - [`Request::Key`] → [`State::ReadingOffset`] → [`State::ReadingEntry`]
/// - [`State::ReadingEntry`] → [`State::ReadingEntry`]
///   (repeat with larger size if our entry size estimate was too small)
/// - [`State::ReadingEntry`] → done
enum State<'a, K: Key + ?Sized> {
    ReadingOffset {
        requested_key: &'a K,
    },
    ReadingEntry {
        byte_offset: u64,
        buf: AlignedBuf,
        expected_len: u64,
        /// Exact maximum entry length when the caller resolved the next
        /// persisted-hashmap offset. Unknown-length sparse reads leave it unset.
        max_len: Option<u64>,
        requested_key: Option<&'a K>,
    },
}

impl<'key, K, V, S> UniversalHashMap<K, V, S>
where
    K: Key + ?Sized + 'key,
    V: Sized + Copy + FromBytes + Immutable + IntoBytes + KnownLayout,
    S: UniversalRead,
{
    /// Read entries for the requested keys and call the provided closure on
    /// each of them.
    ///
    /// Implementation detail: unlike [`UniversalHashMap::for_each_entry`], it
    /// will try to read only the requested entries.
    pub(super) fn for_each_sparse<U, I, F, E>(
        &self,
        entry_kind: MaybeIncompleteEntryKind,
        requests: I,
        mut f: F,
    ) -> Result<(), E>
    where
        U: UserData,
        I: Iterator<Item = (U, Request<'key, K>)>,
        F: FnMut(U, Option<MaybeIncompleteEntry<'_, K, V>>) -> Result<(), E>,
        E: From<UniversalIoError>,
    {
        let mut sparse = PipelineDriver::new(self, entry_kind)?;
        let mut pipeline = S::BorrowedReadPipeline::new()?;
        let mut requests = requests.into_iter();
        loop {
            while pipeline.can_schedule() {
                let Some((entry, range)) = sparse.schedule_next_entry(&mut requests, &mut f)?
                else {
                    break;
                };
                pipeline.schedule::<Random>(
                    entry,
                    &self.storage.inner,
                    range,
                    align_of::<u128>(),
                )?;
            }
            let Some((entry, data)) = pipeline.wait()? else {
                break;
            };
            sparse.process(entry, &data, &mut f)?;
        }
        Ok(())
    }
}

type ScheduledEntry<'key, U, K> = (Entry<'key, U, K>, Range<u64>);

impl<'map, 'key, U, K, V, S> PipelineDriver<'map, 'key, U, K, V, S>
where
    U: UserData,
    K: Key + ?Sized + 'key,
    V: Copy + FromBytes + Immutable + IntoBytes + KnownLayout,
    S: UniversalRead,
{
    fn new(
        map: &'map UniversalHashMap<K, V, S>,
        entry_kind: MaybeIncompleteEntryKind,
    ) -> Result<Self> {
        Ok(Self {
            map,
            entry_kind,
            queue: Vec::new(),
            entry_read_size_est: entry_kind.estimated_size::<K, V>() as u64,
            file_len: map.storage.len()?,
        })
    }

    /// Produce the next entry to schedule.
    fn schedule_next_entry<E, F>(
        &mut self,
        requests: &mut impl Iterator<Item = (U, Request<'key, K>)>,
        f: &mut F,
    ) -> Result<Option<ScheduledEntry<'key, U, K>>, E>
    where
        E: From<UniversalIoError>,
        F: FnMut(U, Option<MaybeIncompleteEntry<'_, K, V>>) -> Result<(), E>,
    {
        if let Some(entry) = self.queue.pop() {
            match &entry.state {
                State::ReadingOffset { .. } => unreachable!("we don't schedule ReadingOffset"),
                State::ReadingEntry {
                    byte_offset,
                    buf,
                    expected_len,
                    max_len,
                    requested_key: _,
                } => {
                    let start = byte_offset.checked_add(buf.len() as u64).ok_or_else(|| {
                        E::from(UniversalIoError::Io(read_err(
                            "persisted-hashmap entry read offset overflow",
                        )))
                    })?;
                    let mut end = byte_offset.checked_add(*expected_len).ok_or_else(|| {
                        E::from(UniversalIoError::Io(read_err(
                            "persisted-hashmap entry read length overflow",
                        )))
                    })?;
                    if let Some(max_len) = max_len {
                        let max_end = byte_offset.checked_add(*max_len).ok_or_else(|| {
                            E::from(UniversalIoError::Io(read_err(
                                "persisted-hashmap exact entry range overflow",
                            )))
                        })?;
                        end = end.min(max_end);
                    }
                    let end = end.min(self.file_len);
                    if end <= start {
                        let message = if max_len.is_some() {
                            "persisted-hashmap entry is truncated within its exact byte range"
                        } else {
                            "unexpected eof"
                        };
                        return Err(E::from(UniversalIoError::Io(read_err(message))));
                    }
                    return Ok(Some((entry, start..end)));
                }
            };
        }

        for (user_data, request) in requests.by_ref() {
            let (state, range);
            match request {
                Request::Offset(offset) => {
                    let byte_offset =
                        self.map.entries_start.checked_add(offset).ok_or_else(|| {
                            E::from(UniversalIoError::Io(read_err(
                                "persisted-hashmap entry offset overflow",
                            )))
                        })?;
                    state = State::ReadingEntry {
                        byte_offset,
                        buf: AlignedBuf::new_for_offset(byte_offset, align_of::<u128>()),
                        expected_len: self.entry_read_size_est,
                        max_len: None,
                        requested_key: None,
                    };
                    let range_end = byte_offset
                        .checked_add(self.entry_read_size_est)
                        .ok_or_else(|| {
                            E::from(UniversalIoError::Io(read_err(
                                "persisted-hashmap entry read length overflow",
                            )))
                        })?;
                    range = byte_offset..range_end;
                }
                Request::Key(requested_key) => {
                    // PHF miss: no stored entry; report immediately and continue.
                    let Some(hash) = self.map.phf.get(requested_key) else {
                        f(user_data, None)?;
                        continue;
                    };
                    // PHF hit: schedule the bucket-offset read; transitions to
                    // Loading once the offset arrives.
                    let bucket_delta = hash
                        .checked_mul(size_of::<BucketOffset>() as u64)
                        .ok_or_else(|| {
                            E::from(UniversalIoError::Io(read_err(
                                "persisted-hashmap bucket offset overflow",
                            )))
                        })?;
                    let bucket_byte_offset = self
                        .map
                        .header
                        .buckets_pos
                        .checked_add(bucket_delta)
                        .ok_or_else(|| {
                            E::from(UniversalIoError::Io(read_err(
                                "persisted-hashmap bucket position overflow",
                            )))
                        })?;
                    state = State::ReadingOffset { requested_key };
                    let range_end = bucket_byte_offset
                        .checked_add(size_of::<BucketOffset>() as u64)
                        .ok_or_else(|| {
                            E::from(UniversalIoError::Io(read_err(
                                "persisted-hashmap bucket read overflow",
                            )))
                        })?;
                    range = bucket_byte_offset..range_end;
                }
                Request::KeyAtOffset {
                    requested_key,
                    offset,
                    length,
                } => {
                    let byte_offset =
                        self.map.entries_start.checked_add(offset).ok_or_else(|| {
                            E::from(UniversalIoError::Io(read_err(
                                "persisted-hashmap exact entry offset overflow",
                            )))
                        })?;
                    state = State::ReadingEntry {
                        byte_offset,
                        buf: AlignedBuf::new_for_offset(byte_offset, align_of::<u128>()),
                        expected_len: length,
                        max_len: Some(length),
                        requested_key: Some(requested_key),
                    };
                    let range_end = byte_offset.checked_add(length).ok_or_else(|| {
                        E::from(UniversalIoError::Io(read_err(
                            "persisted-hashmap exact entry range overflow",
                        )))
                    })?;
                    range = byte_offset..range_end;
                }
            }
            let entry = Entry { user_data, state };
            let range = range.start.min(self.file_len)..range.end.min(self.file_len);
            return Ok(Some((entry, range)));
        }

        Ok(None)
    }

    /// Process a completed read result.
    fn process<E, F>(&mut self, entry: Entry<'key, U, K>, data: &[u8], f: &mut F) -> Result<(), E>
    where
        E: From<UniversalIoError>,
        F: FnMut(U, Option<MaybeIncompleteEntry<'_, K, V>>) -> Result<(), E>,
    {
        match entry.state {
            State::ReadingOffset { requested_key } => {
                let entry_offset = parse_bucket_offset(data)?;
                let byte_offset = self
                    .map
                    .entries_start
                    .checked_add(entry_offset)
                    .ok_or_else(|| {
                        E::from(UniversalIoError::Io(read_err(
                            "persisted-hashmap entry offset overflow",
                        )))
                    })?;
                self.queue.push(Entry {
                    user_data: entry.user_data,
                    state: State::ReadingEntry {
                        byte_offset,
                        buf: AlignedBuf::new_for_offset(byte_offset, align_of::<u128>()),
                        expected_len: self.entry_read_size_est,
                        max_len: None,
                        requested_key: Some(requested_key),
                    },
                });
            }

            State::ReadingEntry {
                byte_offset,
                mut buf,
                expected_len: _,
                max_len,
                mut requested_key,
            } => {
                buf.extend_from_slice(data);
                let parsed =
                    MaybeIncompleteEntry::partial_parse(&buf).map_err(UniversalIoError::from)?;

                if let Some(key) = requested_key
                    && let Some(stored_key) = parsed.key()
                {
                    if key != stored_key {
                        f(entry.user_data, None)?;
                        return Ok(());
                    }
                    requested_key = None;
                }

                if parsed.satisfies_kind(self.entry_kind) {
                    f(entry.user_data, Some(parsed))?;
                } else {
                    if max_len.is_some_and(|max_len| buf.len() as u64 >= max_len) {
                        return Err(E::from(UniversalIoError::Io(read_err(
                            "persisted-hashmap entry is truncated within its exact byte range",
                        ))));
                    }
                    let next_expected_len = (buf.len() as u64 + 1)
                        .next_power_of_two()
                        .max(K::KEY_SIZE_EST as u64);
                    self.queue.push(Entry {
                        user_data: entry.user_data,
                        state: State::ReadingEntry {
                            byte_offset,
                            // `+ 1` so the size strictly grows when `buf.len()` is already a
                            // power of two; otherwise the next refill reads 0 bytes and loops.
                            expected_len: max_len.map_or(next_expected_len, |max_len| {
                                next_expected_len.min(max_len)
                            }),
                            max_len,
                            buf,
                            requested_key,
                        },
                    })
                }
            }
        }

        Ok(())
    }
}
