//! Two-layer placement, exactly per the locked spec.
//!
//! **Layer 1 — point → shard.** A point's shard within a collection is
//! `XXH64(id) % shard_count`. `shard_count` is fixed at collection creation, so a
//! point's shard never moves.
//!
//! **Layer 2 — shard → nodes.** Physical nodes are placed on a consistent-hash
//! ring with **64 vnodes per node** (configurable via `vnodes`, default 64). A
//! shard's *home* is the ring successor (first vnode clockwise) of
//! `XXH64(collection ‖ shard_idx)`; its **N replicas are the next N distinct
//! physical nodes clockwise** from the home. The whole shard is the unit that
//! moves; node join/leave changes the ring and only affected shards relocate
//! (M5).
//!
//! This module is pure compute — no I/O, no async, no openraft — so it is cheap
//! to recompute from the membership set whenever it changes and is exercised by
//! ordinary unit tests.

use std::collections::BTreeSet;

use xxhash_rust::xxh64::xxh64;

use crate::raft::NodeId;

/// Default virtual nodes per physical node on the ring.
pub const DEFAULT_VNODES: u32 = 64;

/// Seed for the placement hashes. Fixed so placement is deterministic and
/// reproducible across nodes and restarts (every node computes the identical map
/// from the same membership set).
const RING_SEED: u64 = 0;

/// Layer 1: the shard index a point id lands in for a collection of
/// `shard_count` shards. `shard_count` must be > 0 (enforced at create time).
#[inline]
pub fn shard_for_id(id: u64, shard_count: u32) -> u32 {
    debug_assert!(shard_count > 0, "shard_count must be > 0");
    (xxh64(&id.to_le_bytes(), RING_SEED) % shard_count as u64) as u32
}

/// A consistent-hash ring of physical nodes, each placed at `vnodes` positions.
///
/// Built from the current membership set; recomputed whenever membership
/// changes. Replica selection walks the ring clockwise from a shard's home
/// position, collecting distinct physical nodes.
#[derive(Debug, Clone)]
pub struct Ring {
    /// (ring position, physical node id), sorted by position. Multiple entries
    /// per node (the vnodes). Ties on position break by node id for determinism.
    vnodes: Vec<(u64, NodeId)>,
    /// Distinct physical node count (replica selection can never exceed this).
    node_count: usize,
}

impl Ring {
    /// Build a ring from a set of physical node ids with `vnodes` virtual nodes
    /// each. An empty membership yields an empty ring (no placements possible).
    pub fn new(nodes: &BTreeSet<NodeId>, vnodes: u32) -> Self {
        let mut v: Vec<(u64, NodeId)> = Vec::with_capacity(nodes.len() * vnodes as usize);
        for &node in nodes {
            for vi in 0..vnodes {
                // Position = hash(node_id ‖ vnode_index). Distinct per (node,vi).
                let mut key = [0u8; 12];
                key[..8].copy_from_slice(&node.to_le_bytes());
                key[8..].copy_from_slice(&vi.to_le_bytes());
                let pos = xxh64(&key, RING_SEED);
                v.push((pos, node));
            }
        }
        // Deterministic order: by position, then node id to break exact ties.
        v.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        Self {
            vnodes: v,
            node_count: nodes.len(),
        }
    }

    /// True when the ring has no nodes.
    pub fn is_empty(&self) -> bool {
        self.vnodes.is_empty()
    }

    /// Distinct physical node count.
    pub fn node_count(&self) -> usize {
        self.node_count
    }

    /// The home ring position for a shard: `XXH64(collection ‖ shard_idx)`.
    fn home_position(collection: &str, shard_idx: u32) -> u64 {
        let mut key = Vec::with_capacity(collection.len() + 4);
        key.extend_from_slice(collection.as_bytes());
        key.extend_from_slice(&shard_idx.to_le_bytes());
        xxh64(&key, RING_SEED)
    }

    /// The replica node set for `collection`/`shard_idx`: the home node (ring
    /// successor of the shard's home position) plus the next distinct physical
    /// nodes clockwise, up to `replication_n` total. If the cluster has fewer
    /// than `replication_n` physical nodes, every node is a replica (the natural
    /// cap — you cannot place more replicas than there are nodes).
    ///
    /// Returns replicas in clockwise order starting at the home; the first entry
    /// is the home/primary used for single-replica routing.
    pub fn replicas_for_shard(
        &self,
        collection: &str,
        shard_idx: u32,
        replication_n: u32,
    ) -> Vec<NodeId> {
        if self.vnodes.is_empty() {
            return Vec::new();
        }
        let want = (replication_n as usize).min(self.node_count);
        let home = Self::home_position(collection, shard_idx);

        // First vnode clockwise at-or-after `home`; wrap to the ring start.
        let start = self.vnodes.partition_point(|(pos, _)| *pos < home) % self.vnodes.len();

        let mut replicas: Vec<NodeId> = Vec::with_capacity(want);
        let n = self.vnodes.len();
        for step in 0..n {
            let (_, node) = self.vnodes[(start + step) % n];
            if !replicas.contains(&node) {
                replicas.push(node);
                if replicas.len() == want {
                    break;
                }
            }
        }
        replicas
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nodes(ids: &[NodeId]) -> BTreeSet<NodeId> {
        ids.iter().copied().collect()
    }

    #[test]
    fn shard_for_id_is_stable_and_bounded() {
        for id in 0..1000u64 {
            let s = shard_for_id(id, 8);
            assert!(s < 8);
            // Deterministic.
            assert_eq!(s, shard_for_id(id, 8));
        }
    }

    #[test]
    fn shard_distribution_is_reasonable() {
        // 8 shards over 100k ids — every shard should get a healthy share.
        let mut counts = [0usize; 8];
        for id in 0..100_000u64 {
            counts[shard_for_id(id, 8) as usize] += 1;
        }
        let min = *counts.iter().min().unwrap();
        let max = *counts.iter().max().unwrap();
        // Within 20% of perfectly even (12500) — XXH64 spreads well.
        assert!(min > 10_000, "min shard underfilled: {min}");
        assert!(max < 15_000, "max shard overfilled: {max}");
    }

    #[test]
    fn replicas_distinct_and_count_capped() {
        let ring = Ring::new(&nodes(&[1, 2, 3, 4, 5]), DEFAULT_VNODES);
        let r = ring.replicas_for_shard("vectors", 0, 3);
        assert_eq!(r.len(), 3, "N=3 with 5 nodes gives 3 replicas");
        // All distinct.
        let set: BTreeSet<_> = r.iter().copied().collect();
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn replicas_capped_by_cluster_size() {
        let ring = Ring::new(&nodes(&[1, 2]), DEFAULT_VNODES);
        let r = ring.replicas_for_shard("vectors", 0, 3);
        // Only 2 nodes exist, so at most 2 replicas.
        assert_eq!(r.len(), 2);
    }

    #[test]
    fn placement_is_deterministic_across_rebuilds() {
        let a = Ring::new(&nodes(&[10, 20, 30]), DEFAULT_VNODES);
        let b = Ring::new(&nodes(&[30, 10, 20]), DEFAULT_VNODES); // different insert order
        for shard in 0..8u32 {
            assert_eq!(
                a.replicas_for_shard("c", shard, 2),
                b.replicas_for_shard("c", shard, 2),
                "placement must not depend on membership insertion order"
            );
        }
    }

    #[test]
    fn empty_ring_yields_no_replicas() {
        let ring = Ring::new(&BTreeSet::new(), DEFAULT_VNODES);
        assert!(ring.is_empty());
        assert!(ring.replicas_for_shard("c", 0, 3).is_empty());
    }

    #[test]
    fn adding_a_node_moves_only_some_shards() {
        // Consistent hashing property: going 3 -> 4 nodes should relocate only a
        // minority of shard homes, not reshuffle everything.
        let before = Ring::new(&nodes(&[1, 2, 3]), DEFAULT_VNODES);
        let after = Ring::new(&nodes(&[1, 2, 3, 4]), DEFAULT_VNODES);
        let shard_count = 64u32;
        let mut moved = 0;
        for s in 0..shard_count {
            let b = before.replicas_for_shard("c", s, 1)[0];
            let a = after.replicas_for_shard("c", s, 1)[0];
            if a != b {
                moved += 1;
            }
        }
        // Far fewer than all shards move; with 4 nodes ~1/4 is the expectation.
        assert!(
            moved < shard_count as usize / 2,
            "too many shards moved on node add: {moved}/{shard_count}"
        );
    }
}
