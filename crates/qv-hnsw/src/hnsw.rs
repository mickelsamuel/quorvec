//! An owned HNSW (Hierarchical Navigable Small World) graph implementing
//! [`VectorIndex`].
//!
//! This follows Malkov & Yashunin with the parameters and choices locked by the
//! quorvec plan:
//!   - `M = 16` (max neighbors per node on layers >= 1),
//!   - `M0 = 32` (max neighbors on the base layer 0),
//!   - `ef_construction = 200`,
//!   - level assignment `l = floor(-ln(U(0,1]) / ln(M))`,
//!   - **neighbor selection by the pruning heuristic** (Algorithm 4 in the
//!     paper): a candidate is kept only if it is closer to the new element than
//!     to any already-selected neighbor — NOT "keep the M closest". This is the
//!     single most common correctness trap (findings Q2 pitfall #1).
//!
//! Deletes are logical: a tombstone bitmap marks removed ids; tombstoned nodes
//! are skipped when collecting results but remain in the graph as routing
//! waypoints. There is no compaction in v1 (a documented limitation).
//!
//! Concurrency: this type is not internally synchronized. quorvec wraps a whole
//! shard's index in an `RwLock` (shared reads, exclusive writes) — the simple,
//! correct choice; finer-grained locking is a labeled future item.

use crate::distance::Metric;
use crate::index::{IndexError, VectorIndex};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

/// Default max neighbors per node on layers >= 1.
pub const DEFAULT_M: usize = 16;
/// Default max neighbors on the base layer (layer 0).
pub const DEFAULT_M0: usize = 32;
/// Default construction-time candidate breadth.
pub const DEFAULT_EF_CONSTRUCTION: usize = 200;

/// Tunable construction parameters.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct HnswParams {
    /// Max neighbors per node on layers >= 1.
    pub m: usize,
    /// Max neighbors on layer 0.
    pub m0: usize,
    /// Candidate list breadth during insertion.
    pub ef_construction: usize,
}

impl Default for HnswParams {
    fn default() -> Self {
        Self {
            m: DEFAULT_M,
            m0: DEFAULT_M0,
            ef_construction: DEFAULT_EF_CONSTRUCTION,
        }
    }
}

// ---- Deterministic level RNG ----------------------------------------------

/// A tiny SplitMix64 PRNG. Owning the RNG (rather than pulling `rand` into this
/// no-I/O crate's runtime surface) keeps level assignment deterministic and
/// serializable: the same seed + insert order reproduces the same graph, which
/// the serialization round-trip test relies on.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform f64 in (0, 1].
    fn next_unit(&mut self) -> f64 {
        // 53-bit mantissa; map [0, 2^53) -> (0, 1].
        let bits = self.next_u64() >> 11;
        let u = (bits as f64) / ((1u64 << 53) as f64); // [0, 1)
        1.0 - u // (0, 1]
    }
}

// ---- Graph node ------------------------------------------------------------

/// One stored element. Neighbors are per-layer adjacency lists of node indices.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Node {
    id: u64,
    vector: Vec<f32>,
    /// `neighbors[layer]` is the adjacency list at that layer. `len()-1` is the
    /// node's top layer.
    neighbors: Vec<Vec<u32>>,
}

impl Node {
    fn top_layer(&self) -> usize {
        self.neighbors.len() - 1
    }
}

// ---- Candidate ordering helpers -------------------------------------------

/// Min-heap entry by distance (nearer = higher priority). `BinaryHeap` is a
/// max-heap, so we invert the comparison.
#[derive(Copy, Clone, Debug)]
struct MinCandidate {
    dist: f32,
    node: u32,
}
impl PartialEq for MinCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist && self.node == other.node
    }
}
impl Eq for MinCandidate {}
impl Ord for MinCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse so the smallest distance is "greatest" (pops first).
        other
            .dist
            .partial_cmp(&self.dist)
            .unwrap_or(Ordering::Equal)
            .then(other.node.cmp(&self.node))
    }
}
impl PartialOrd for MinCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Max-heap entry by distance (farther = higher priority). Used for the
/// bounded result set during search/construction.
#[derive(Copy, Clone, Debug)]
struct MaxCandidate {
    dist: f32,
    node: u32,
}
impl PartialEq for MaxCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist && self.node == other.node
    }
}
impl Eq for MaxCandidate {}
impl Ord for MaxCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(Ordering::Equal)
            .then(self.node.cmp(&other.node))
    }
}
impl PartialOrd for MaxCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// ---- The graph -------------------------------------------------------------

/// The HNSW index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswIndex {
    dim: usize,
    metric: Metric,
    params: HnswParams,
    /// `1 / ln(M)`, the level-generation normalization factor.
    level_mult: f64,
    nodes: Vec<Node>,
    /// id -> node index, for replace/delete/get.
    id_to_node: HashMap<u64, u32>,
    /// Tombstoned node indices (logical delete). Skipped in results, kept as
    /// routing waypoints.
    deleted: Vec<bool>,
    /// Number of live (non-tombstoned) nodes.
    live: usize,
    /// Entry-point node index and its layer; `None` until the first insert.
    entry_point: Option<u32>,
    rng: SplitMix64,
}

impl HnswIndex {
    /// New empty index with default parameters and a fixed seed (deterministic
    /// builds).
    pub fn new(dim: usize, metric: Metric) -> Self {
        Self::with_params(dim, metric, HnswParams::default(), 0x5EED_1234_ABCD_0001)
    }

    /// New empty index with explicit parameters and level-RNG seed.
    pub fn with_params(dim: usize, metric: Metric, params: HnswParams, seed: u64) -> Self {
        assert!(dim > 0, "dim must be positive");
        assert!(params.m >= 2, "M must be >= 2");
        assert!(params.m0 >= params.m, "M0 should be >= M");
        assert!(params.ef_construction >= 1, "ef_construction must be >= 1");
        Self {
            dim,
            metric,
            params,
            level_mult: 1.0 / (params.m as f64).ln(),
            nodes: Vec::new(),
            id_to_node: HashMap::new(),
            deleted: Vec::new(),
            live: 0,
            entry_point: None,
            rng: SplitMix64::new(seed),
        }
    }

    /// Max neighbor count permitted on a given layer.
    #[inline]
    fn m_max(&self, layer: usize) -> usize {
        if layer == 0 {
            self.params.m0
        } else {
            self.params.m
        }
    }

    /// Draw a random top layer for a new node: `floor(-ln(U(0,1]) * level_mult)`.
    fn random_level(&mut self) -> usize {
        let u = self.rng.next_unit();
        (-(u.ln()) * self.level_mult).floor() as usize
    }

    #[inline]
    fn dist(&self, a: u32, query: &[f32]) -> f32 {
        self.metric.distance(&self.nodes[a as usize].vector, query)
    }

    /// Greedy descent on a single upper layer: from `entry`, hop to ever-closer
    /// neighbors until no neighbor improves. Returns the closest node found.
    fn greedy_search_layer(&self, query: &[f32], entry: u32, layer: usize) -> u32 {
        let mut current = entry;
        let mut current_dist = self.dist(current, query);
        loop {
            let mut improved = false;
            let node = &self.nodes[current as usize];
            if layer < node.neighbors.len() {
                for &nbr in &node.neighbors[layer] {
                    let d = self.dist(nbr, query);
                    if d < current_dist {
                        current_dist = d;
                        current = nbr;
                        improved = true;
                    }
                }
            }
            if !improved {
                return current;
            }
        }
    }

    /// Layer search returning up to `ef` nearest nodes to `query` reachable from
    /// `entry` at `layer`. Returns `(dist, node)` pairs, unsorted. Includes
    /// tombstoned nodes (they are valid routing waypoints); result filtering
    /// happens at the public boundary.
    fn search_layer(&self, query: &[f32], entry: u32, ef: usize, layer: usize) -> Vec<(f32, u32)> {
        let mut visited: HashMap<u32, ()> = HashMap::new();
        visited.insert(entry, ());

        let entry_dist = self.dist(entry, query);
        // Candidates to expand: a min-heap on distance.
        let mut candidates: BinaryHeap<MinCandidate> = BinaryHeap::new();
        candidates.push(MinCandidate {
            dist: entry_dist,
            node: entry,
        });
        // Best results so far: a max-heap bounded to `ef` (farthest at the top).
        let mut results: BinaryHeap<MaxCandidate> = BinaryHeap::new();
        results.push(MaxCandidate {
            dist: entry_dist,
            node: entry,
        });

        while let Some(MinCandidate {
            dist: c_dist,
            node: c,
        }) = candidates.pop()
        {
            // If the nearest unexpanded candidate is farther than the current
            // worst result and we already have ef results, stop.
            if let Some(worst) = results.peek() {
                if c_dist > worst.dist && results.len() >= ef {
                    break;
                }
            }
            let node = &self.nodes[c as usize];
            if layer < node.neighbors.len() {
                for &nbr in &node.neighbors[layer] {
                    if visited.insert(nbr, ()).is_some() {
                        continue;
                    }
                    let d = self.dist(nbr, query);
                    let worst = results.peek().map(|w| w.dist).unwrap_or(f32::INFINITY);
                    if d < worst || results.len() < ef {
                        candidates.push(MinCandidate { dist: d, node: nbr });
                        results.push(MaxCandidate { dist: d, node: nbr });
                        if results.len() > ef {
                            results.pop(); // drop the farthest
                        }
                    }
                }
            }
        }

        results.into_iter().map(|m| (m.dist, m.node)).collect()
    }

    /// The pruning heuristic (paper Algorithm 4): from `candidates` (any order),
    /// select at most `m` neighbors such that each kept candidate is closer to
    /// the query/base element than to every already-kept neighbor. This yields a
    /// navigable, diverse neighborhood — the key difference from naive
    /// "keep the m closest".
    ///
    /// `candidates` carries each candidate's distance to the base element in
    /// `.0`, so the base vector itself is not needed here: the keep test
    /// compares `distance(cand, selected)` against that precomputed
    /// `distance(cand, base)`.
    fn select_neighbors_heuristic(&self, candidates: Vec<(f32, u32)>, m: usize) -> Vec<u32> {
        // Work from nearest to farthest.
        let mut cands = candidates;
        cands.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

        let mut selected: Vec<u32> = Vec::with_capacity(m);
        for (cand_dist, cand) in cands {
            if selected.len() >= m {
                break;
            }
            // Keep `cand` only if it is closer to `base` than to any already
            // selected neighbor.
            let mut keep = true;
            for &s in &selected {
                let d_to_selected = self.metric.distance(
                    &self.nodes[cand as usize].vector,
                    &self.nodes[s as usize].vector,
                );
                if d_to_selected < cand_dist {
                    keep = false;
                    break;
                }
            }
            if keep {
                selected.push(cand);
            }
        }
        selected
    }

    /// Insert a brand-new node (id not already present).
    fn insert_new(&mut self, id: u64, vector: &[f32]) {
        let node_level = self.random_level();
        let new_idx = self.nodes.len() as u32;

        // Allocate the node with empty per-layer adjacency.
        let neighbors = vec![Vec::new(); node_level + 1];
        self.nodes.push(Node {
            id,
            vector: vector.to_vec(),
            neighbors,
        });
        self.deleted.push(false);
        self.id_to_node.insert(id, new_idx);
        self.live += 1;

        // First node becomes the entry point.
        let entry = match self.entry_point {
            None => {
                self.entry_point = Some(new_idx);
                return;
            }
            Some(e) => e,
        };

        let entry_level = self.nodes[entry as usize].top_layer();

        // Phase 1: greedy descent from the top down to node_level+1, refining the
        // entry point one layer at a time.
        let mut curr = entry;
        let mut layer = entry_level;
        while layer > node_level {
            curr = self.greedy_search_layer(vector, curr, layer);
            if layer == 0 {
                break;
            }
            layer -= 1;
        }

        // Phase 2: from min(node_level, entry_level) down to 0, find ef_c
        // candidates, select neighbors by the heuristic, and wire bidirectional
        // edges, pruning over-full neighborhoods.
        let start_layer = node_level.min(entry_level);
        let mut entry_for_layer = curr;
        for l in (0..=start_layer).rev() {
            let found = self.search_layer(vector, entry_for_layer, self.params.ef_construction, l);
            // Carry the nearest found node down as the entry for the next layer.
            if let Some(nearest) = found
                .iter()
                .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal))
            {
                entry_for_layer = nearest.1;
            }

            let m = self.m_max(l);
            let selected = self.select_neighbors_heuristic(found, m);

            // Wire new -> selected.
            self.nodes[new_idx as usize].neighbors[l] = selected.clone();

            // Wire selected -> new, pruning each neighbor's list if it overflows.
            for &nbr in &selected {
                self.nodes[nbr as usize].neighbors[l].push(new_idx);
                let m_nbr = self.m_max(l);
                if self.nodes[nbr as usize].neighbors[l].len() > m_nbr {
                    self.prune_neighbors(nbr, l, m_nbr);
                }
            }
        }

        // If the new node rises above the current entry point, it becomes the
        // new entry point.
        if node_level > entry_level {
            self.entry_point = Some(new_idx);
        }
    }

    /// Re-select a node's neighbor list at `layer` down to `m` using the
    /// heuristic, after an edge push overflowed it.
    fn prune_neighbors(&mut self, node: u32, layer: usize, m: usize) {
        let base = self.nodes[node as usize].vector.clone();
        let current = self.nodes[node as usize].neighbors[layer].clone();
        let cands: Vec<(f32, u32)> = current
            .into_iter()
            .map(|n| {
                let d = self.metric.distance(&base, &self.nodes[n as usize].vector);
                (d, n)
            })
            .collect();
        let kept = self.select_neighbors_heuristic(cands, m);
        self.nodes[node as usize].neighbors[layer] = kept;
    }

    /// Live node count (excludes tombstones).
    pub fn live_len(&self) -> usize {
        self.live
    }

    /// Total node count including tombstones (for diagnostics/tests).
    pub fn total_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Construction parameters.
    pub fn params(&self) -> HnswParams {
        self.params
    }

    /// (min, max, mean) degree across nodes at layer 0. Diagnostics only.
    pub fn layer0_degree_stats(&self) -> (usize, usize, f64) {
        if self.nodes.is_empty() {
            return (0, 0, 0.0);
        }
        let mut min = usize::MAX;
        let mut max = 0usize;
        let mut sum = 0usize;
        for node in &self.nodes {
            let d = node.neighbors[0].len();
            min = min.min(d);
            max = max.max(d);
            sum += d;
        }
        (min, max, sum as f64 / self.nodes.len() as f64)
    }

    /// Check structural invariants, returning an error describing the first
    /// violation found. Used by property tests and available for debugging.
    ///
    /// Invariants checked:
    ///   1. every neighbor index references an existing node (no dangling edges),
    ///   2. each layer's adjacency respects its `m_max` degree bound,
    ///   3. no node lists itself as a neighbor,
    ///   4. `id_to_node` is consistent with stored node ids,
    ///   5. the live count equals the number of non-tombstoned nodes,
    ///   6. the entry point, if set, is a valid node index.
    pub fn validate_invariants(&self) -> Result<(), String> {
        let n = self.nodes.len() as u32;

        if let Some(ep) = self.entry_point {
            if ep >= n {
                return Err(format!("entry_point {ep} out of range ({n} nodes)"));
            }
        } else if !self.nodes.is_empty() {
            return Err("entry_point is None but nodes exist".into());
        }

        let mut counted_live = 0usize;
        for (i, node) in self.nodes.iter().enumerate() {
            let i = i as u32;
            if !self.deleted[i as usize] {
                counted_live += 1;
            }
            // id_to_node consistency.
            match self.id_to_node.get(&node.id) {
                Some(&mapped) if mapped == i => {}
                _ => {
                    return Err(format!(
                        "id_to_node inconsistent for node {i} (id {})",
                        node.id
                    ))
                }
            }
            for (layer, adj) in node.neighbors.iter().enumerate() {
                let bound = self.m_max(layer);
                if adj.len() > bound {
                    return Err(format!(
                        "node {i} layer {layer} degree {} exceeds m_max {bound}",
                        adj.len()
                    ));
                }
                for &nbr in adj {
                    if nbr >= n {
                        return Err(format!(
                            "node {i} layer {layer} has dangling neighbor {nbr}"
                        ));
                    }
                    if nbr == i {
                        return Err(format!("node {i} layer {layer} is its own neighbor"));
                    }
                }
            }
        }

        if counted_live != self.live {
            return Err(format!(
                "live count {} != non-tombstoned nodes {counted_live}",
                self.live
            ));
        }
        Ok(())
    }
}

impl VectorIndex for HnswIndex {
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

        // Replace: if the id exists (live or tombstoned), update its vector in
        // place. The simplest correct behavior in a no-compaction graph is to
        // overwrite the vector and revive a tombstone; the existing edges remain
        // valid routing structure. (Re-linking on replace is a labeled future
        // refinement — v1 documents replace as overwrite + revive.)
        if let Some(&idx) = self.id_to_node.get(&id) {
            self.nodes[idx as usize].vector.copy_from_slice(vector);
            if self.deleted[idx as usize] {
                self.deleted[idx as usize] = false;
                self.live += 1;
            }
            return Ok(());
        }

        self.insert_new(id, vector);
        Ok(())
    }

    fn delete(&mut self, id: u64) -> bool {
        if let Some(&idx) = self.id_to_node.get(&id) {
            if !self.deleted[idx as usize] {
                self.deleted[idx as usize] = true;
                self.live -= 1;
                return true;
            }
        }
        false
    }

    fn get_vector(&self, id: u64) -> Option<Vec<f32>> {
        self.id_to_node.get(&id).and_then(|&idx| {
            if self.deleted[idx as usize] {
                None
            } else {
                Some(self.nodes[idx as usize].vector.clone())
            }
        })
    }

    fn search(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
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

        let entry = match self.entry_point {
            Some(e) => e,
            None => return Ok(Vec::new()),
        };

        // Descend the upper layers greedily to find a good layer-0 entry.
        let entry_level = self.nodes[entry as usize].top_layer();
        let mut curr = entry;
        let mut layer = entry_level;
        while layer > 0 {
            curr = self.greedy_search_layer(query, curr, layer);
            layer -= 1;
        }

        // Layer-0 search with breadth max(ef_search, k).
        let ef = ef_search.max(k);
        let mut found = self.search_layer(query, curr, ef, 0);
        // Nearest first.
        found.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

        // Filter tombstones, map node -> id, take k.
        let mut out = Vec::with_capacity(k);
        for (dist, node) in found {
            if self.deleted[node as usize] {
                continue;
            }
            out.push((self.nodes[node as usize].id, dist));
            if out.len() == k {
                break;
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::BruteForceIndex;

    fn deterministic_vecs(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = SplitMix64::new(seed);
        (0..n)
            .map(|_| {
                (0..dim)
                    .map(|_| (rng.next_unit() as f32) * 2.0 - 1.0)
                    .collect()
            })
            .collect()
    }

    #[test]
    fn basic_insert_search() {
        let mut idx = HnswIndex::new(3, Metric::L2);
        idx.insert(1, &[0.0, 0.0, 0.0]).unwrap();
        idx.insert(2, &[1.0, 0.0, 0.0]).unwrap();
        idx.insert(3, &[9.0, 9.0, 9.0]).unwrap();
        assert_eq!(idx.len(), 3);

        let res = idx.search(&[0.0, 0.0, 0.0], 2, 64).unwrap();
        assert_eq!(res[0].0, 1);
        assert_eq!(res[1].0, 2);
    }

    #[test]
    fn delete_skips_in_results_and_get() {
        let mut idx = HnswIndex::new(2, Metric::L2);
        for i in 0..20u64 {
            idx.insert(i, &[i as f32, 0.0]).unwrap();
        }
        assert!(idx.delete(0));
        assert!(!idx.delete(0)); // idempotent
        assert_eq!(idx.len(), 19);

        let res = idx.search(&[0.0, 0.0], 5, 64).unwrap();
        assert!(res.iter().all(|(id, _)| *id != 0), "tombstone returned");
        assert!(idx.get_vector(0).is_none());
    }

    #[test]
    fn replace_revives_and_overwrites() {
        let mut idx = HnswIndex::new(2, Metric::L2);
        idx.insert(5, &[0.0, 0.0]).unwrap();
        idx.delete(5);
        assert_eq!(idx.len(), 0);
        idx.insert(5, &[3.0, 4.0]).unwrap(); // revive + overwrite
        assert_eq!(idx.len(), 1);
        assert_eq!(idx.get_vector(5).unwrap(), vec![3.0, 4.0]);
    }

    #[test]
    fn recall_vs_bruteforce_small() {
        // A modest correctness check that lives in unit tests; the full 10k/100
        // recall harness is an integration test.
        let dim = 16;
        let n = 2000;
        let vecs = deterministic_vecs(n, dim, 0xABCDEF);

        let mut hnsw = HnswIndex::new(dim, Metric::L2);
        let mut oracle = BruteForceIndex::new(dim, Metric::L2);
        for (i, v) in vecs.iter().enumerate() {
            hnsw.insert(i as u64, v).unwrap();
            oracle.insert(i as u64, v).unwrap();
        }

        let queries = deterministic_vecs(50, dim, 0x123456);
        let mut hits = 0usize;
        let mut total = 0usize;
        for q in &queries {
            let h: Vec<u64> = hnsw
                .search(q, 10, 64)
                .unwrap()
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            let o: Vec<u64> = oracle
                .search(q, 10, 64)
                .unwrap()
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            for id in &h {
                if o.contains(id) {
                    hits += 1;
                }
            }
            total += o.len();
        }
        let recall = hits as f64 / total as f64;
        assert!(recall >= 0.95, "unit recall@10 = {recall} (< 0.95)");
    }
}
