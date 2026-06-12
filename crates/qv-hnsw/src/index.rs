//! The [`VectorIndex`] trait: the single abstraction the storage/node layers
//! depend on. The M0 brute-force [`BruteForceIndex`] and the M1 HNSW graph both
//! implement it, so the node binary can swap implementations behind the trait
//! and the brute-force one stays in tests as the recall ground-truth oracle.

use crate::distance::Metric;

/// A nearest-neighbor index over `dim`-dimensional `f32` vectors keyed by `u64`.
///
/// Smaller score = closer (it is the metric distance, not a similarity).
/// Implementations must:
///   - treat `insert` of an existing id as a full replace,
///   - make `delete` idempotent (deleting an absent id is a no-op returning false),
///   - never return deleted/tombstoned ids from `search`.
pub trait VectorIndex: Send + Sync {
    /// Vector dimensionality this index was built for.
    fn dim(&self) -> usize;

    /// Distance metric this index scores with.
    fn metric(&self) -> Metric;

    /// Number of live (non-deleted) points.
    fn len(&self) -> usize;

    /// True when there are no live points.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Insert or replace the vector for `id`.
    ///
    /// # Errors
    /// Returns [`IndexError::DimMismatch`] if `vector.len() != self.dim()`.
    fn insert(&mut self, id: u64, vector: &[f32]) -> Result<(), IndexError>;

    /// Remove `id` if present. Returns whether a live point was removed.
    fn delete(&mut self, id: u64) -> bool;

    /// The stored vector for `id`, if it is live. Required by the gRPC `Get`
    /// surface (id lookup), distinct from similarity search.
    fn get_vector(&self, id: u64) -> Option<Vec<f32>>;

    /// `k` nearest neighbors of `query`, nearest first, as `(id, score)`.
    ///
    /// `ef_search` tunes the search-time candidate breadth for graph indexes;
    /// exact indexes ignore it. Returns at most `k` results.
    ///
    /// # Errors
    /// Returns [`IndexError::DimMismatch`] if `query.len() != self.dim()`.
    fn search(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> Result<Vec<(u64, f32)>, IndexError>;
}

/// Errors surfaced by index operations.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum IndexError {
    /// A supplied vector did not match the index dimensionality.
    #[error("dimension mismatch: index is {expected}-d, got {got}-d")]
    DimMismatch { expected: usize, got: usize },
}

/// Exact brute-force index: scans every live vector per query.
///
/// This is the correctness oracle for the whole project — the HNSW recall
/// harness measures itself against this. It is O(n·dim) per query and only
/// suitable for small collections and tests, never the hot path at scale.
#[derive(Clone)]
pub struct BruteForceIndex {
    dim: usize,
    metric: Metric,
    // Parallel arrays keyed by slot; `ids[slot]` is None for a free slot.
    ids: Vec<Option<u64>>,
    vectors: Vec<f32>, // row-major: slot s occupies [s*dim, (s+1)*dim)
    // id -> slot for O(1) replace/delete.
    id_to_slot: std::collections::HashMap<u64, usize>,
    free_slots: Vec<usize>,
    live: usize,
}

impl BruteForceIndex {
    /// New empty index for `dim`-dimensional vectors under `metric`.
    pub fn new(dim: usize, metric: Metric) -> Self {
        assert!(dim > 0, "dim must be positive");
        Self {
            dim,
            metric,
            ids: Vec::new(),
            vectors: Vec::new(),
            id_to_slot: std::collections::HashMap::new(),
            free_slots: Vec::new(),
            live: 0,
        }
    }

    #[inline]
    fn row(&self, slot: usize) -> &[f32] {
        &self.vectors[slot * self.dim..(slot + 1) * self.dim]
    }
}

impl VectorIndex for BruteForceIndex {
    fn dim(&self) -> usize {
        self.dim
    }

    fn metric(&self) -> Metric {
        self.metric
    }

    fn len(&self) -> usize {
        self.live
    }

    fn insert(&mut self, id: u64, vector: &[f32]) -> Result<(), IndexError> {
        if vector.len() != self.dim {
            return Err(IndexError::DimMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }

        if let Some(&slot) = self.id_to_slot.get(&id) {
            // Replace in place.
            self.vectors[slot * self.dim..(slot + 1) * self.dim].copy_from_slice(vector);
            return Ok(());
        }

        let slot = if let Some(s) = self.free_slots.pop() {
            self.ids[s] = Some(id);
            self.vectors[s * self.dim..(s + 1) * self.dim].copy_from_slice(vector);
            s
        } else {
            let s = self.ids.len();
            self.ids.push(Some(id));
            self.vectors.extend_from_slice(vector);
            s
        };
        self.id_to_slot.insert(id, slot);
        self.live += 1;
        Ok(())
    }

    fn delete(&mut self, id: u64) -> bool {
        if let Some(slot) = self.id_to_slot.remove(&id) {
            self.ids[slot] = None;
            self.free_slots.push(slot);
            self.live -= 1;
            true
        } else {
            false
        }
    }

    fn get_vector(&self, id: u64) -> Option<Vec<f32>> {
        self.id_to_slot
            .get(&id)
            .map(|&slot| self.row(slot).to_vec())
    }

    fn search(
        &self,
        query: &[f32],
        k: usize,
        _ef_search: usize,
    ) -> Result<Vec<(u64, f32)>, IndexError> {
        if query.len() != self.dim {
            return Err(IndexError::DimMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        if k == 0 || self.live == 0 {
            return Ok(Vec::new());
        }

        let mut scored: Vec<(u64, f32)> = Vec::with_capacity(self.live);
        for (slot, maybe_id) in self.ids.iter().enumerate() {
            if let Some(id) = maybe_id {
                let d = self.metric.distance(query, self.row(slot));
                scored.push((*id, d));
            }
        }
        // Partial sort: nearest k by score, then deterministic id tiebreak.
        scored.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        scored.truncate(k);
        Ok(scored)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_search_delete_roundtrip() {
        let mut idx = BruteForceIndex::new(3, Metric::L2);
        idx.insert(1, &[0.0, 0.0, 0.0]).unwrap();
        idx.insert(2, &[1.0, 0.0, 0.0]).unwrap();
        idx.insert(3, &[5.0, 5.0, 5.0]).unwrap();
        assert_eq!(idx.len(), 3);

        let res = idx.search(&[0.0, 0.0, 0.0], 2, 64).unwrap();
        assert_eq!(res.len(), 2);
        assert_eq!(res[0].0, 1); // closest is itself
        assert_eq!(res[1].0, 2);

        assert!(idx.delete(1));
        assert!(!idx.delete(1)); // idempotent
        assert_eq!(idx.len(), 2);

        let res = idx.search(&[0.0, 0.0, 0.0], 2, 64).unwrap();
        assert_eq!(res[0].0, 2); // 1 is gone
        assert!(res.iter().all(|(id, _)| *id != 1));
    }

    #[test]
    fn replace_existing_id() {
        let mut idx = BruteForceIndex::new(2, Metric::L2);
        idx.insert(7, &[0.0, 0.0]).unwrap();
        idx.insert(7, &[9.0, 9.0]).unwrap(); // replace
        assert_eq!(idx.len(), 1);
        let res = idx.search(&[9.0, 9.0], 1, 64).unwrap();
        assert_eq!(res[0].0, 7);
        assert!(res[0].1.abs() < 1e-6);
    }

    #[test]
    fn dim_mismatch_errors() {
        let mut idx = BruteForceIndex::new(3, Metric::L2);
        assert_eq!(
            idx.insert(1, &[0.0, 0.0]),
            Err(IndexError::DimMismatch {
                expected: 3,
                got: 2
            })
        );
        assert_eq!(
            idx.search(&[0.0, 0.0], 1, 64).unwrap_err(),
            IndexError::DimMismatch {
                expected: 3,
                got: 2
            }
        );
    }

    #[test]
    fn free_slot_reuse() {
        let mut idx = BruteForceIndex::new(2, Metric::L2);
        idx.insert(1, &[1.0, 1.0]).unwrap();
        idx.insert(2, &[2.0, 2.0]).unwrap();
        idx.delete(1);
        idx.insert(3, &[3.0, 3.0]).unwrap(); // should reuse slot 0
        assert_eq!(idx.len(), 2);
        let res = idx.search(&[3.0, 3.0], 2, 64).unwrap();
        assert_eq!(res[0].0, 3);
    }
}
