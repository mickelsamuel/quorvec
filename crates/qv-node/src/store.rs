//! In-memory collection store for the single-node M0 baseline.
//!
//! Each collection is one [`VectorIndex`] guarded by an `RwLock` (the plan's
//! per-shard concurrency choice: shared reads, exclusive writes — the simple,
//! correct option; finer locking is a labeled future item). Sharding,
//! replication, and durability are added in later milestones; here a collection
//! is a single logical shard living entirely in this node's memory.

use qv_hnsw::{BruteForceIndex, Metric, VectorIndex};
use std::collections::HashMap;
use std::sync::RwLock;

/// Schema-level facts about a collection, fixed at creation.
//
// shard_count / replication_n are recorded at creation but only consumed once
// sharding and replication land (M3/M4); allow dead_code until then.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct CollectionSchema {
    pub dim: u32,
    pub metric: Metric,
    pub shard_count: u32,
    pub replication_n: u32,
}

/// One collection: its schema plus the live index behind a lock.
pub struct Collection {
    pub schema: CollectionSchema,
    pub index: RwLock<Box<dyn VectorIndex>>,
}

impl Collection {
    fn new(schema: CollectionSchema) -> Self {
        // M0 uses the brute-force exact index. M1 swaps this for HNSW behind the
        // same `VectorIndex` trait with no change to the service layer.
        let index: Box<dyn VectorIndex> =
            Box::new(BruteForceIndex::new(schema.dim as usize, schema.metric));
        Self {
            schema,
            index: RwLock::new(index),
        }
    }
}

/// Errors from store operations, mapped to gRPC status by the service layer.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("collection '{0}' already exists")]
    AlreadyExists(String),
    #[error("collection '{0}' not found")]
    NotFound(String),
    #[error("invalid argument: {0}")]
    Invalid(String),
}

/// The node's whole logical dataset for M0: a name -> collection map.
#[derive(Default)]
pub struct Store {
    collections: RwLock<HashMap<String, Collection>>,
}

impl Store {
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a collection. Applies server-side defaults for zero-valued
    /// shard_count / replication_n.
    pub fn create_collection(
        &self,
        name: &str,
        dim: u32,
        metric: Metric,
        shard_count: u32,
        replication_n: u32,
    ) -> Result<(), StoreError> {
        if name.is_empty() {
            return Err(StoreError::Invalid("collection name is empty".into()));
        }
        if dim == 0 {
            return Err(StoreError::Invalid("dim must be > 0".into()));
        }
        let schema = CollectionSchema {
            dim,
            metric,
            shard_count: if shard_count == 0 { 8 } else { shard_count },
            replication_n: if replication_n == 0 { 3 } else { replication_n },
        };

        let mut map = self.collections.write().unwrap();
        if map.contains_key(name) {
            return Err(StoreError::AlreadyExists(name.to_string()));
        }
        map.insert(name.to_string(), Collection::new(schema));
        Ok(())
    }

    /// Drop a collection.
    pub fn drop_collection(&self, name: &str) -> Result<(), StoreError> {
        let mut map = self.collections.write().unwrap();
        if map.remove(name).is_none() {
            return Err(StoreError::NotFound(name.to_string()));
        }
        Ok(())
    }

    /// The schema for a collection, if it exists. Used by tests and by
    /// ClusterInfo/routing from M3; allow dead_code in the M0 binary build.
    #[allow(dead_code)]
    pub fn schema(&self, name: &str) -> Result<CollectionSchema, StoreError> {
        let map = self.collections.read().unwrap();
        map.get(name)
            .map(|c| c.schema.clone())
            .ok_or_else(|| StoreError::NotFound(name.to_string()))
    }

    /// Upsert points into a collection. Returns the number accepted.
    pub fn upsert(&self, name: &str, points: &[(u64, Vec<f32>)]) -> Result<u64, StoreError> {
        let map = self.collections.read().unwrap();
        let coll = map
            .get(name)
            .ok_or_else(|| StoreError::NotFound(name.to_string()))?;
        let dim = coll.schema.dim as usize;
        let mut index = coll.index.write().unwrap();
        let mut count = 0u64;
        for (id, vector) in points {
            if vector.len() != dim {
                return Err(StoreError::Invalid(format!(
                    "point {id}: expected dim {dim}, got {}",
                    vector.len()
                )));
            }
            index
                .insert(*id, vector)
                .map_err(|e| StoreError::Invalid(e.to_string()))?;
            count += 1;
        }
        Ok(count)
    }

    /// Delete points by id. Returns the number actually removed.
    pub fn delete(&self, name: &str, ids: &[u64]) -> Result<u64, StoreError> {
        let map = self.collections.read().unwrap();
        let coll = map
            .get(name)
            .ok_or_else(|| StoreError::NotFound(name.to_string()))?;
        let mut index = coll.index.write().unwrap();
        let mut count = 0u64;
        for id in ids {
            if index.delete(*id) {
                count += 1;
            }
        }
        Ok(count)
    }

    /// k-NN search.
    pub fn search(
        &self,
        name: &str,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> Result<Vec<(u64, f32)>, StoreError> {
        let map = self.collections.read().unwrap();
        let coll = map
            .get(name)
            .ok_or_else(|| StoreError::NotFound(name.to_string()))?;
        let dim = coll.schema.dim as usize;
        if query.len() != dim {
            return Err(StoreError::Invalid(format!(
                "query dim {} != collection dim {dim}",
                query.len()
            )));
        }
        let index = coll.index.read().unwrap();
        index
            .search(query, k, ef_search)
            .map_err(|e| StoreError::Invalid(e.to_string()))
    }

    /// Get stored vectors by id. Returns `(id, vector)` for each id that is
    /// live; missing/deleted ids are omitted. Backs the gRPC `Get` surface.
    pub fn get(&self, name: &str, ids: &[u64]) -> Result<Vec<(u64, Vec<f32>)>, StoreError> {
        let map = self.collections.read().unwrap();
        let coll = map
            .get(name)
            .ok_or_else(|| StoreError::NotFound(name.to_string()))?;
        let index = coll.index.read().unwrap();
        let mut out = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(v) = index.get_vector(*id) {
                out.push((*id, v));
            }
        }
        Ok(out)
    }

    /// Number of collections (used by tests / ClusterInfo).
    #[allow(dead_code)]
    pub fn collection_count(&self) -> usize {
        self.collections.read().unwrap().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_upsert_search_flow() {
        let store = Store::new();
        store.create_collection("c", 3, Metric::L2, 0, 0).unwrap();

        // defaults applied
        let schema = store.schema("c").unwrap();
        assert_eq!(schema.shard_count, 8);
        assert_eq!(schema.replication_n, 3);

        let pts = vec![
            (1u64, vec![0.0, 0.0, 0.0]),
            (2u64, vec![1.0, 0.0, 0.0]),
            (3u64, vec![9.0, 9.0, 9.0]),
        ];
        assert_eq!(store.upsert("c", &pts).unwrap(), 3);

        let res = store.search("c", &[0.0, 0.0, 0.0], 2, 64).unwrap();
        assert_eq!(res[0].0, 1);
        assert_eq!(res[1].0, 2);

        assert_eq!(store.delete("c", &[1]).unwrap(), 1);
        let res = store.search("c", &[0.0, 0.0, 0.0], 1, 64).unwrap();
        assert_eq!(res[0].0, 2);
    }

    #[test]
    fn duplicate_create_rejected() {
        let store = Store::new();
        store.create_collection("c", 2, Metric::L2, 1, 1).unwrap();
        assert!(matches!(
            store.create_collection("c", 2, Metric::L2, 1, 1),
            Err(StoreError::AlreadyExists(_))
        ));
    }

    #[test]
    fn ops_on_missing_collection() {
        let store = Store::new();
        assert!(matches!(
            store.search("nope", &[0.0], 1, 64),
            Err(StoreError::NotFound(_))
        ));
    }
}
