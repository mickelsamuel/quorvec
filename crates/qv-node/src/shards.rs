//! This node's durable shard storage (the E2 rewire).
//!
//! A collection is `shard_count` shards (locked spec). Each shard this node is a
//! replica for is a durable [`Shard`] (WAL + snapshots + crash recovery, from
//! `qv-storage`) living under `data_dir/<collection>/shard_<idx>/`. A node only
//! materializes the shards it is assigned by placement — not every shard of
//! every collection.
//!
//! This layer is purely local: it opens/creates shards on demand and applies
//! reads/writes to them. Routing (which node owns a shard, forwarding to it) is
//! the coordinator's job in the gRPC layer; here we only ever touch shards this
//! node physically holds.
//!
//! Concurrency: a `Mutex` per shard (writes mutate the WAL + index; the plan's
//! per-shard exclusive-write choice). The map of shards is behind an `RwLock`.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};

use qv_cluster::Metric as ClusterMetric;
use qv_hnsw::Metric;
use qv_storage::{Hlc, Shard, ShardError};

/// Key identifying one shard replica on this node.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ShardKey {
    pub collection: String,
    pub shard_idx: u32,
}

/// A fully-qualified reference to one shard on this node: which shard, plus the
/// schema facts (dim/metric) needed to open it. Bundled so the data ops take one
/// argument instead of four.
#[derive(Debug, Clone)]
pub struct ShardRef<'a> {
    pub collection: &'a str,
    pub shard_idx: u32,
    pub dim: usize,
    pub metric: Metric,
}

/// A point as stored/returned by the data plane: id + version + value/tombstone.
#[derive(Debug, Clone)]
pub struct StoredPoint {
    pub id: u64,
    pub hlc: Hlc,
    pub tombstone: bool,
    pub vector: Vec<f32>,
    pub payload: Vec<u8>,
}

/// Errors from the shard store.
#[derive(Debug, thiserror::Error)]
pub enum ShardStoreError {
    #[error("shard: {0}")]
    Shard(#[from] ShardError),
    #[error("dimension mismatch: collection is {expected}-d, got {got}-d")]
    DimMismatch { expected: usize, got: usize },
}

/// Convert a cluster-domain metric to the hnsw metric.
pub fn metric_from_cluster(m: ClusterMetric) -> Metric {
    match m {
        ClusterMetric::L2 => Metric::L2,
        ClusterMetric::Cosine => Metric::Cosine,
    }
}

/// This node's durable shards, keyed by (collection, shard_idx).
pub struct ShardStore {
    data_dir: PathBuf,
    snapshot_wal_mb: u64,
    wal_batch_ms: u64,
    shards: RwLock<HashMap<ShardKey, Arc<Mutex<Shard>>>>,
}

impl ShardStore {
    pub fn new(data_dir: PathBuf, snapshot_wal_mb: u64, wal_batch_ms: u64) -> Self {
        Self {
            data_dir,
            snapshot_wal_mb,
            wal_batch_ms,
            shards: RwLock::new(HashMap::new()),
        }
    }

    fn shard_dir(&self, collection: &str, shard_idx: u32) -> PathBuf {
        self.data_dir
            .join(collection)
            .join(format!("shard_{shard_idx}"))
    }

    /// Get (opening/creating if needed) the durable shard for
    /// (collection, shard_idx) with the given dim/metric. Idempotent: opening an
    /// existing shard recovers it from disk.
    pub fn get_or_open(
        &self,
        collection: &str,
        shard_idx: u32,
        dim: usize,
        metric: Metric,
    ) -> Result<Arc<Mutex<Shard>>, ShardStoreError> {
        let key = ShardKey {
            collection: collection.to_string(),
            shard_idx,
        };
        if let Some(s) = self.shards.read().unwrap().get(&key) {
            return Ok(s.clone());
        }
        let mut map = self.shards.write().unwrap();
        // Re-check under the write lock (another thread may have opened it).
        if let Some(s) = map.get(&key) {
            return Ok(s.clone());
        }
        let dir = self.shard_dir(collection, shard_idx);
        let shard = Shard::open(&dir, dim, metric, self.snapshot_wal_mb, self.wal_batch_ms)?;
        let arc = Arc::new(Mutex::new(shard));
        map.insert(key, arc.clone());
        Ok(arc)
    }

    /// Whether this node currently holds a materialized shard.
    pub fn has_shard(&self, collection: &str, shard_idx: u32) -> bool {
        let key = ShardKey {
            collection: collection.to_string(),
            shard_idx,
        };
        self.shards.read().unwrap().contains_key(&key)
    }

    /// Apply a durable upsert to a shard this node holds.
    pub fn upsert(
        &self,
        sref: &ShardRef<'_>,
        hlc: Hlc,
        id: u64,
        vector: &[f32],
        payload: Vec<u8>,
    ) -> Result<(), ShardStoreError> {
        if vector.len() != sref.dim {
            return Err(ShardStoreError::DimMismatch {
                expected: sref.dim,
                got: vector.len(),
            });
        }
        let shard = self.get_or_open(sref.collection, sref.shard_idx, sref.dim, sref.metric)?;
        let mut guard = shard.lock().unwrap();
        guard.upsert(hlc, id, vector, payload)?;
        Ok(())
    }

    /// Apply a durable delete (tombstone) to a shard this node holds.
    pub fn delete(&self, sref: &ShardRef<'_>, hlc: Hlc, id: u64) -> Result<bool, ShardStoreError> {
        let shard = self.get_or_open(sref.collection, sref.shard_idx, sref.dim, sref.metric)?;
        let mut guard = shard.lock().unwrap();
        Ok(guard.delete(hlc, id)?)
    }

    /// Get a stored vector by id from a shard this node holds (live only).
    pub fn get(&self, sref: &ShardRef<'_>, id: u64) -> Result<Option<Vec<f32>>, ShardStoreError> {
        let shard = self.get_or_open(sref.collection, sref.shard_idx, sref.dim, sref.metric)?;
        let guard = shard.lock().unwrap();
        Ok(guard.get(id))
    }

    /// Search a single shard this node holds.
    pub fn search_shard(
        &self,
        sref: &ShardRef<'_>,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> Result<Vec<(u64, f32)>, ShardStoreError> {
        let shard = self.get_or_open(sref.collection, sref.shard_idx, sref.dim, sref.metric)?;
        let guard = shard.lock().unwrap();
        Ok(guard.search(query, k, ef_search)?)
    }

    /// Drop all materialized shards for a collection (used on DropCollection).
    /// Removes them from the in-memory map; on-disk files are left for the
    /// background reaper / are overwritten if the collection is recreated. (v1:
    /// we leave the directory; physical deletion is a labeled follow-up.)
    pub fn drop_collection(&self, collection: &str) {
        let mut map = self.shards.write().unwrap();
        map.retain(|k, _| k.collection != collection);
    }
}
