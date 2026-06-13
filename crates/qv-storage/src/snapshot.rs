//! Shard snapshots: a serialized point-in-time image of a shard's index plus
//! the WAL offset it covers.
//!
//! On restart the shard loads the newest snapshot, rebuilds the index from it,
//! and replays only the WAL records written after the snapshot's offset. The
//! HNSW graph already serializes its full state (graph + id maps + tombstones)
//! via serde, so the snapshot is that plus the covered WAL offset.
//!
//! Snapshots are written atomically: serialize to a temp file in the same
//! directory, fsync, then rename over the destination (rename is atomic on a
//! single filesystem), so a crash never leaves a half-written snapshot in place.

use crate::hlc::Hlc;
use qv_hnsw::HnswIndex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

/// Serializable per-point version stored in the snapshot (mirrors
/// `shard::PointVersion`, kept here to avoid a circular type dependency).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct VersionEntry {
    pub hlc: Hlc,
    pub tombstone: bool,
}

/// On-disk snapshot: the index image, the per-point LWW versions, and the WAL
/// offset it already includes.
#[derive(Debug, Serialize, Deserialize)]
pub struct Snapshot {
    /// WAL byte offset up to and including which this image reflects all writes.
    pub wal_offset: u64,
    /// The serialized index.
    pub index: HnswIndex,
    /// Per-id winning version for last-writer-wins (M4). Older snapshots without
    /// this field deserialize it as empty (serde default) and the shard rebuilds
    /// it from the WAL tail on recovery.
    #[serde(default)]
    pub versions: HashMap<u64, VersionEntry>,
}

#[derive(Debug, thiserror::Error)]
pub enum SnapshotError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialize: {0}")]
    Serialize(#[from] bincode::Error),
}

impl Snapshot {
    /// Write atomically to `path` (temp + fsync + rename).
    pub fn write(
        path: impl AsRef<Path>,
        wal_offset: u64,
        index: &HnswIndex,
        versions: &HashMap<u64, VersionEntry>,
    ) -> Result<(), SnapshotError> {
        let path = path.as_ref();
        let tmp: PathBuf = path.with_extension("snap.tmp");

        let snap = SnapshotRef {
            wal_offset,
            index,
            versions,
        };
        let bytes = bincode::serialize(&snap)?;

        {
            let mut f = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&tmp)?;
            f.write_all(&bytes)?;
            f.flush()?;
            f.sync_all()?;
        }
        // Atomic replace.
        fs::rename(&tmp, path)?;
        // fsync the directory entry so the rename is durable.
        if let Some(parent) = path.parent() {
            if let Ok(dir) = File::open(parent) {
                let _ = dir.sync_all();
            }
        }
        Ok(())
    }

    /// Load a snapshot from `path`, if it exists.
    pub fn load(path: impl AsRef<Path>) -> Result<Option<Snapshot>, SnapshotError> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(None);
        }
        let bytes = fs::read(path)?;
        let snap: Snapshot = bincode::deserialize(&bytes)?;
        Ok(Some(snap))
    }
}

/// Borrowing form used only for serialization (avoids cloning the index).
#[derive(Serialize)]
struct SnapshotRef<'a> {
    wal_offset: u64,
    index: &'a HnswIndex,
    versions: &'a HashMap<u64, VersionEntry>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use qv_hnsw::{Metric, VectorIndex};
    use tempfile::tempdir;

    #[test]
    fn write_load_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("shard.snap");

        let mut idx = HnswIndex::new(4, Metric::L2);
        for i in 0..100u64 {
            idx.insert(i, &[i as f32, 0.0, 1.0, 2.0]).unwrap();
        }

        let versions = HashMap::new();
        Snapshot::write(&path, 4096, &idx, &versions).unwrap();
        let loaded = Snapshot::load(&path).unwrap().unwrap();
        assert_eq!(loaded.wal_offset, 4096);
        assert_eq!(loaded.index.len(), 100);

        // Search results identical.
        let q = [5.0, 0.0, 1.0, 2.0];
        assert_eq!(
            idx.search(&q, 5, 64).unwrap(),
            loaded.index.search(&q, 5, 64).unwrap()
        );
    }
}
