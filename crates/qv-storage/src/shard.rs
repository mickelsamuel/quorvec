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
use std::collections::HashMap;
use std::path::{Path, PathBuf};

const WAL_FILE: &str = "shard.wal";
const SNAPSHOT_FILE: &str = "shard.snap";

/// The per-point version a shard tracks for last-writer-wins (M4).
///
/// Every id the shard has ever seen carries its winning [`Hlc`] and whether the
/// winner was a delete (tombstone). Incoming writes with an HLC `<=` the stored
/// one are rejected as stale; ties break deterministically by HLC (which already
/// embeds the node-id-free counter — the coordinator's stamp is globally
/// ordered). A tombstone keeps its HLC so a later-but-equal resurrection cannot
/// silently win.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PointVersion {
    pub hlc: Hlc,
    pub tombstone: bool,
}

/// The outcome of applying a versioned write to a shard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteOutcome {
    /// The write won LWW and was applied (durable).
    Applied,
    /// The write lost LWW (stored HLC was `>=` incoming) and was ignored.
    Stale,
}

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
    /// Per-id winning version (HLC + tombstone) for last-writer-wins (M4).
    versions: HashMap<u64, PointVersion>,
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

        // 1. Load snapshot (rebuilds index + versions + the covered WAL offset).
        let (mut index, mut versions, snap_offset) = match Snapshot::load(&snap_path)? {
            Some(s) => {
                let versions: HashMap<u64, PointVersion> = s
                    .versions
                    .into_iter()
                    .map(|(id, v)| {
                        (
                            id,
                            PointVersion {
                                hlc: v.hlc,
                                tombstone: v.tombstone,
                            },
                        )
                    })
                    .collect();
                (s.index, versions, s.wal_offset)
            }
            None => (HnswIndex::new(dim, metric), HashMap::new(), 0),
        };

        // 2. WAL recovery: truncates any torn tail, returns surviving records.
        let (records, wal_end) = Wal::recover(&wal_path)?;

        // 3. Replay only records beyond the snapshot offset. The snapshot already
        //    reflects all writes up to snap_offset (index AND versions), so we
        //    re-walk and apply only records whose cumulative end exceeds it. The
        //    WAL is the LWW source of truth: replaying it reproduces the same
        //    winner set the live path produced (records are appended in the order
        //    the shard accepted them, so a stale write was never appended).
        let mut pos: u64 = 0;
        for rec in &records {
            let rec_len = framed_len(rec);
            let rec_end = pos + rec_len;
            if rec_end > snap_offset {
                apply_record(&mut index, &mut versions, rec).map_err(ShardError::Index)?;
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
            versions,
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
    ///
    /// Last-writer-wins by HLC: if a strictly-greater HLC is already stored for
    /// `id`, the write is ignored as stale (returns [`WriteOutcome::Stale`]) and
    /// nothing is appended. Otherwise it wins, is appended, applied to the index,
    /// and recorded as the id's version. (Backward-compatible callers that don't
    /// care about LWW can use the legacy `upsert` wrapper below.)
    pub fn upsert_versioned(
        &mut self,
        hlc: Hlc,
        id: u64,
        vector: &[f32],
        payload: Vec<u8>,
    ) -> Result<WriteOutcome, ShardError> {
        if vector.len() != self.dim {
            return Err(ShardError::DimMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }
        if self.is_stale(id, hlc) {
            return Ok(WriteOutcome::Stale);
        }
        let rec = WalRecord::upsert(hlc, id, vector.to_vec(), payload);
        self.wal.append(&rec)?;
        self.index
            .insert(id, vector)
            .map_err(|e| ShardError::Index(e.to_string()))?;
        self.versions.insert(
            id,
            PointVersion {
                hlc,
                tombstone: false,
            },
        );
        self.maybe_snapshot()?;
        Ok(WriteOutcome::Applied)
    }

    /// Durable delete (tombstone) with LWW: a stale delete is ignored. A winning
    /// delete tombstones the id (kept as a versioned tombstone so a later-equal
    /// resurrection cannot silently win).
    pub fn delete_versioned(&mut self, hlc: Hlc, id: u64) -> Result<WriteOutcome, ShardError> {
        if self.is_stale(id, hlc) {
            return Ok(WriteOutcome::Stale);
        }
        let rec = WalRecord::delete(hlc, id);
        self.wal.append(&rec)?;
        self.index.delete(id);
        self.versions.insert(
            id,
            PointVersion {
                hlc,
                tombstone: true,
            },
        );
        self.maybe_snapshot()?;
        Ok(WriteOutcome::Applied)
    }

    /// Legacy unconditional upsert (pre-M4). Stamps the version with `hlc` but
    /// does not enforce LWW; retained for single-node/test callers. New
    /// data-plane code uses [`upsert_versioned`].
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
        self.versions.insert(
            id,
            PointVersion {
                hlc,
                tombstone: false,
            },
        );
        self.maybe_snapshot()?;
        Ok(())
    }

    /// Legacy unconditional delete (pre-M4). New code uses [`delete_versioned`].
    pub fn delete(&mut self, hlc: Hlc, id: u64) -> Result<bool, ShardError> {
        let rec = WalRecord::delete(hlc, id);
        self.wal.append(&rec)?;
        let removed = self.index.delete(id);
        self.versions.insert(
            id,
            PointVersion {
                hlc,
                tombstone: true,
            },
        );
        self.maybe_snapshot()?;
        Ok(removed)
    }

    /// Whether an incoming `hlc` for `id` loses LWW against the stored version
    /// (stored HLC `>=` incoming). A first write for an id is never stale.
    fn is_stale(&self, id: u64, hlc: Hlc) -> bool {
        match self.versions.get(&id) {
            Some(v) => hlc <= v.hlc,
            None => false,
        }
    }

    /// The stored version (HLC + tombstone) for an id, if the shard has seen it.
    pub fn version(&self, id: u64) -> Option<PointVersion> {
        self.versions.get(&id).copied()
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
        let versions = self
            .versions
            .iter()
            .map(|(id, v)| {
                (
                    *id,
                    crate::snapshot::VersionEntry {
                        hlc: v.hlc,
                        tombstone: v.tombstone,
                    },
                )
            })
            .collect();
        Snapshot::write(&snap_path, offset, &self.index, &versions)?;
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

/// Apply a recovered WAL record to the in-memory index AND the version map.
///
/// The WAL is the LWW source of truth: records were only appended when they won
/// (the live path never appends a stale write), so replaying in WAL order
/// reproduces the same winner set. We still take the max HLC per id defensively,
/// so a hand-constructed/out-of-order log still converges to the highest HLC.
fn apply_record(
    index: &mut HnswIndex,
    versions: &mut HashMap<u64, PointVersion>,
    rec: &WalRecord,
) -> Result<(), String> {
    // Skip if a strictly-greater version is already present (defensive LWW).
    if let Some(v) = versions.get(&rec.id) {
        if rec.hlc < v.hlc {
            return Ok(());
        }
    }
    match rec.op {
        WalOp::Upsert => {
            index
                .insert(rec.id, &rec.vector)
                .map_err(|e| e.to_string())?;
            versions.insert(
                rec.id,
                PointVersion {
                    hlc: rec.hlc,
                    tombstone: false,
                },
            );
            Ok(())
        }
        WalOp::Delete => {
            index.delete(rec.id);
            versions.insert(
                rec.id,
                PointVersion {
                    hlc: rec.hlc,
                    tombstone: true,
                },
            );
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
    fn lww_rejects_stale_and_keeps_winner() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("lww");
        let mut shard = Shard::open(&p, 2, Metric::L2, 128, 0).unwrap();

        // Winner at hlc=10.
        assert_eq!(
            shard
                .upsert_versioned(h(10), 1, &[1.0, 1.0], vec![])
                .unwrap(),
            WriteOutcome::Applied
        );
        // A lower-hlc write loses LWW and is ignored.
        assert_eq!(
            shard
                .upsert_versioned(h(5), 1, &[9.0, 9.0], vec![])
                .unwrap(),
            WriteOutcome::Stale
        );
        // The stored vector is still the hlc=10 winner.
        assert_eq!(shard.get(1), Some(vec![1.0, 1.0]));
        assert_eq!(shard.version(1).unwrap().hlc, h(10));

        // A higher-hlc write wins and replaces.
        assert_eq!(
            shard
                .upsert_versioned(h(20), 1, &[2.0, 2.0], vec![])
                .unwrap(),
            WriteOutcome::Applied
        );
        assert_eq!(shard.get(1), Some(vec![2.0, 2.0]));
        assert_eq!(shard.version(1).unwrap().hlc, h(20));

        // An equal-hlc write is NOT newer -> stale (deterministic, no flapping).
        assert_eq!(
            shard
                .upsert_versioned(h(20), 1, &[3.0, 3.0], vec![])
                .unwrap(),
            WriteOutcome::Stale
        );
        assert_eq!(shard.get(1), Some(vec![2.0, 2.0]));
    }

    #[test]
    fn lww_tombstone_versioned() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("lww_del");
        let mut shard = Shard::open(&p, 2, Metric::L2, 128, 0).unwrap();

        shard
            .upsert_versioned(h(10), 1, &[1.0, 1.0], vec![])
            .unwrap();
        // Delete at a higher hlc wins -> tombstone.
        assert_eq!(
            shard.delete_versioned(h(20), 1).unwrap(),
            WriteOutcome::Applied
        );
        assert_eq!(shard.get(1), None);
        assert!(shard.version(1).unwrap().tombstone);

        // A resurrect at a LOWER hlc than the tombstone loses (stays deleted).
        assert_eq!(
            shard
                .upsert_versioned(h(15), 1, &[7.0, 7.0], vec![])
                .unwrap(),
            WriteOutcome::Stale
        );
        assert_eq!(shard.get(1), None);

        // A resurrect at a HIGHER hlc wins (comes back).
        assert_eq!(
            shard
                .upsert_versioned(h(30), 1, &[7.0, 7.0], vec![])
                .unwrap(),
            WriteOutcome::Applied
        );
        assert_eq!(shard.get(1), Some(vec![7.0, 7.0]));
    }

    #[test]
    fn versions_survive_snapshot_and_recovery() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("ver_recover");
        {
            let mut shard = Shard::open(&p, 2, Metric::L2, 128, 0).unwrap();
            shard
                .upsert_versioned(h(10), 1, &[1.0, 1.0], vec![])
                .unwrap();
            shard
                .upsert_versioned(h(20), 2, &[2.0, 2.0], vec![])
                .unwrap();
            shard.delete_versioned(h(30), 2).unwrap();
            shard.snapshot_now().unwrap();
            shard
                .upsert_versioned(h(40), 3, &[3.0, 3.0], vec![])
                .unwrap();
        }
        // Reopen: versions restored from snapshot + WAL tail; LWW still enforced.
        let mut shard = Shard::open(&p, 2, Metric::L2, 128, 0).unwrap();
        assert_eq!(shard.version(1).unwrap().hlc, h(10));
        assert!(shard.version(2).unwrap().tombstone);
        assert_eq!(shard.version(2).unwrap().hlc, h(30));
        assert_eq!(shard.version(3).unwrap().hlc, h(40));
        // A stale write against the recovered version is still rejected.
        assert_eq!(
            shard
                .upsert_versioned(h(5), 1, &[9.0, 9.0], vec![])
                .unwrap(),
            WriteOutcome::Stale
        );
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
        let mut ref_versions = HashMap::new();
        for rec in &records {
            apply_record(&mut reference, &mut ref_versions, rec).unwrap();
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
