//! A durable shard: an HNSW index backed by a WAL and periodic snapshots, with
//! crash recovery on open.
//!
//! Write path (the ordering that makes recovery sound): for every mutation we
//! **append to the WAL first** (fsync by default), then apply it to the in-memory
//! index. A write is only acknowledged after its WAL append returns durable, so
//! a crash either has the record on disk (it will replay) or never acked it.
//!
//! Recovery on open: load the newest snapshot (if any) to rebuild the index and
//! learn the covered WAL offset, run WAL recovery (which truncates any torn
//! tail), then replay the WAL records that lie beyond the snapshot offset.

use crate::hlc::Hlc;
use crate::snapshot::Snapshot;
use crate::wal::{Wal, WalOp, WalRecord};
use qv_hnsw::{HnswIndex, Metric, VectorIndex};
use std::path::{Path, PathBuf};

const WAL_FILE: &str = "shard.wal";
const SNAPSHOT_FILE: &str = "shard.snap";

#[derive(Debug, thiserror::Error)]
pub enum ShardError {
    #[error("wal: {0}")]
    Wal(#[from] crate::wal::WalError),
    #[error("snapshot: {0}")]
    Snapshot(#[from] crate::snapshot::SnapshotError),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("index: {0}")]
    Index(String),
    #[error("dimension mismatch: shard is {expected}-d, got {got}-d")]
    DimMismatch { expected: usize, got: usize },
}

/// A single durable shard.
pub struct Shard {
    dir: PathBuf,
    dim: usize,
    metric: Metric,
    index: HnswIndex,
    wal: Wal,
    /// Snapshot when the WAL since the last snapshot exceeds this many bytes.
    snapshot_threshold_bytes: u64,
    /// WAL offset covered by the last snapshot taken (or recovered).
    last_snapshot_offset: u64,
}

impl Shard {
    /// Open (or create) a shard at `dir`, recovering any prior state.
    ///
    /// `snapshot_wal_mb` mirrors the node config; `batch_ms` is the WAL fsync
    /// batching window (0 = fsync per record).
    pub fn open(
        dir: impl AsRef<Path>,
        dim: usize,
        metric: Metric,
        snapshot_wal_mb: u64,
        batch_ms: u64,
    ) -> Result<Self, ShardError> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir)?;
        let wal_path = dir.join(WAL_FILE);
        let snap_path = dir.join(SNAPSHOT_FILE);

        // 1. Load snapshot (rebuilds index + tells us the covered WAL offset).
        let (mut index, snap_offset) = match Snapshot::load(&snap_path)? {
            Some(s) => (s.index, s.wal_offset),
            None => (HnswIndex::new(dim, metric), 0),
        };

        // 2. WAL recovery: truncates any torn tail, returns surviving records.
        let (records, wal_end) = Wal::recover(&wal_path)?;

        // 3. Replay only records beyond the snapshot offset. We replay by record
        //    index rather than byte offset: the snapshot offset is a byte offset,
        //    so we re-walk and apply records whose cumulative end exceeds it.
        //    Simpler and equally correct: rebuild from snapshot, then apply every
        //    surviving record whose effect is not already in the snapshot. Since
        //    a snapshot's index already includes all records up to snap_offset,
        //    we must skip those. We track byte position to decide.
        let mut pos: u64 = 0;
        for rec in &records {
            let rec_len = framed_len(rec);
            let rec_end = pos + rec_len;
            if rec_end > snap_offset {
                apply_to_index(&mut index, rec).map_err(ShardError::Index)?;
            }
            pos = rec_end;
        }

        // 4. Open the WAL for appending at the recovered (possibly truncated) end.
        let wal = Wal::open(&wal_path, batch_ms)?;
        debug_assert_eq!(wal.offset(), wal_end);

        Ok(Self {
            dir,
            dim,
            metric,
            index,
            wal,
            snapshot_threshold_bytes: snapshot_wal_mb.saturating_mul(1024 * 1024),
            last_snapshot_offset: snap_offset,
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
    pub fn metric(&self) -> Metric {
        self.metric
    }
    pub fn len(&self) -> usize {
        self.index.len()
    }
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }
    pub fn wal_offset(&self) -> u64 {
        self.wal.offset()
    }

    /// Durable upsert: WAL first (fsync by default), then index.
    pub fn upsert(
        &mut self,
        hlc: Hlc,
        id: u64,
        vector: &[f32],
        payload: Vec<u8>,
    ) -> Result<(), ShardError> {
        if vector.len() != self.dim {
            return Err(ShardError::DimMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }
        let rec = WalRecord::upsert(hlc, id, vector.to_vec(), payload);
        self.wal.append(&rec)?;
        self.index
            .insert(id, vector)
            .map_err(|e| ShardError::Index(e.to_string()))?;
        self.maybe_snapshot()?;
        Ok(())
    }

    /// Durable delete (tombstone): WAL first, then index.
    pub fn delete(&mut self, hlc: Hlc, id: u64) -> Result<bool, ShardError> {
        let rec = WalRecord::delete(hlc, id);
        self.wal.append(&rec)?;
        let removed = self.index.delete(id);
        self.maybe_snapshot()?;
        Ok(removed)
    }

    /// k-NN search over the live index.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> Result<Vec<(u64, f32)>, ShardError> {
        self.index
            .search(query, k, ef_search)
            .map_err(|e| ShardError::Index(e.to_string()))
    }

    /// Stored vector for an id, if live.
    pub fn get(&self, id: u64) -> Option<Vec<f32>> {
        self.index.get_vector(id)
    }

    /// Flush the WAL to stable storage (relevant in batch mode).
    pub fn flush(&mut self) -> Result<(), ShardError> {
        self.wal.flush()?;
        Ok(())
    }

    /// Force a snapshot now (used by tests and explicit checkpoints).
    pub fn snapshot_now(&mut self) -> Result<(), ShardError> {
        self.wal.flush()?;
        let offset = self.wal.offset();
        let snap_path = self.dir.join(SNAPSHOT_FILE);
        Snapshot::write(&snap_path, offset, &self.index)?;
        self.last_snapshot_offset = offset;
        Ok(())
    }

    /// Snapshot if the WAL has grown past the threshold since the last snapshot.
    fn maybe_snapshot(&mut self) -> Result<(), ShardError> {
        if self.snapshot_threshold_bytes == 0 {
            return Ok(());
        }
        let grown = self.wal.offset().saturating_sub(self.last_snapshot_offset);
        if grown >= self.snapshot_threshold_bytes {
            self.snapshot_now()?;
        }
        Ok(())
    }
}

/// Apply a recovered WAL record to the in-memory index.
fn apply_to_index(index: &mut HnswIndex, rec: &WalRecord) -> Result<(), String> {
    match rec.op {
        WalOp::Upsert => index.insert(rec.id, &rec.vector).map_err(|e| e.to_string()),
        WalOp::Delete => {
            index.delete(rec.id);
            Ok(())
        }
    }
}

/// The total framed byte length a record occupies on disk: the 4-byte frame
/// length prefix + 4-byte CRC + body. Must match the WAL encoder exactly so the
/// replay byte accounting lines up with the snapshot offset.
fn framed_len(rec: &WalRecord) -> u64 {
    // body = op(1) + hlc(8) + id(8) + dim(4) + vector(dim*4) + plen(4) + payload
    let body = 1 + 8 + 8 + 4 + rec.vector.len() * 4 + 4 + rec.payload.len();
    // frame prefix(4) + crc(4) + body
    (4 + 4 + body) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn h(n: u64) -> Hlc {
        Hlc::new(n, 0)
    }

    #[test]
    fn write_then_reopen_recovers() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("shard0");

        {
            let mut shard = Shard::open(&p, 3, Metric::L2, 128, 0).unwrap();
            for i in 0..50u64 {
                shard
                    .upsert(h(i + 1), i, &[i as f32, 0.0, 1.0], vec![])
                    .unwrap();
            }
            shard.delete(h(1000), 0).unwrap();
            assert_eq!(shard.len(), 49);
        } // drop: flushes

        // Reopen: must recover 49 live points from the WAL (no snapshot taken).
        let shard = Shard::open(&p, 3, Metric::L2, 128, 0).unwrap();
        assert_eq!(shard.len(), 49);
        assert!(shard.get(0).is_none()); // deleted
        assert!(shard.get(1).is_some());
    }

    #[test]
    fn snapshot_plus_tail_replay() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("shard1");

        {
            let mut shard = Shard::open(&p, 2, Metric::L2, 128, 0).unwrap();
            for i in 0..30u64 {
                shard.upsert(h(i + 1), i, &[i as f32, 1.0], vec![]).unwrap();
            }
            shard.snapshot_now().unwrap(); // snapshot covers first 30
            for i in 30..60u64 {
                shard.upsert(h(i + 1), i, &[i as f32, 1.0], vec![]).unwrap();
            }
            assert_eq!(shard.len(), 60);
        }

        // Reopen: snapshot (30) + WAL tail replay (30) = 60.
        let shard = Shard::open(&p, 2, Metric::L2, 128, 0).unwrap();
        assert_eq!(shard.len(), 60);
        for i in 0..60u64 {
            assert!(shard.get(i).is_some(), "missing id {i} after recovery");
        }
    }

    #[test]
    fn recovery_matches_replay_reference() {
        // The recovered index must match an index rebuilt purely by replaying
        // every WAL record into a fresh HnswIndex.
        let dir = tempdir().unwrap();
        let p = dir.path().join("shard2");

        {
            let mut shard = Shard::open(&p, 2, Metric::L2, 128, 0).unwrap();
            for i in 0..40u64 {
                shard.upsert(h(i + 1), i, &[i as f32, 2.0], vec![]).unwrap();
            }
            shard.snapshot_now().unwrap();
            for i in 40..80u64 {
                shard.upsert(h(i + 1), i, &[i as f32, 2.0], vec![]).unwrap();
            }
            shard.delete(h(9999), 5).unwrap();
        }

        let recovered = Shard::open(&p, 2, Metric::L2, 128, 0).unwrap();

        // Reference: replay the whole WAL from scratch.
        let (records, _) = Wal::recover(p.join("shard.wal")).unwrap();
        let mut reference = HnswIndex::new(2, Metric::L2);
        for rec in &records {
            apply_to_index(&mut reference, rec).unwrap();
        }

        assert_eq!(recovered.len(), reference.len());
        // Spot-check a set of queries return identical id sets.
        for qi in 0..20u64 {
            let q = [qi as f32, 2.0];
            let a: Vec<u64> = recovered
                .search(&q, 5, 64)
                .unwrap()
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            let b: Vec<u64> = reference
                .search(&q, 5, 64)
                .unwrap()
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            assert_eq!(a, b, "recovered vs replay-reference mismatch at q={qi}");
        }
    }
}
