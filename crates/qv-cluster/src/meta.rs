//! The metadata-plane domain: what the Raft log carries and what the state
//! machine holds.
//!
//! Per the locked spec, the openraft state machine holds
//! `{membership, collection schemas, shard map + replica assignments, shard
//! states}`. openraft owns *membership* natively (it is part of every Raft state
//! machine), so [`MetaState`] holds the quorvec-specific remainder — collection
//! schemas and the derived shard map — plus the node-address directory needed to
//! route data-plane RPCs. Join/Leave/Create/Drop and shard-state transitions are
//! all [`MetaRequest`]s applied through Raft.
//!
//! The shard map is *derived*: it is recomputed from the live membership set and
//! the per-collection schema via [`crate::placement`] whenever either changes, so
//! it is never an independent source of truth that can drift. Shard *states*
//! (Active/Syncing/Dead) are explicit, since they are operational facts the ring
//! cannot infer (M5 sets Syncing during transfer; M3 leaves new shards Active).

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::placement::{Ring, DEFAULT_VNODES};
use crate::raft::NodeId;

/// Distance metric for a collection (mirrors the proto enum; kept independent of
/// the proto crate so qv-cluster has no gRPC dependency).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Metric {
    L2,
    Cosine,
}

/// Operational state of one shard replica.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShardState {
    /// Live and serving reads/writes.
    Active,
    /// Receiving a shard transfer; not yet authoritative (M5).
    Syncing,
    /// Known dead; excluded from routing until recovered (M5/M6).
    Dead,
}

/// Schema-level facts about a collection, fixed at creation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CollectionSchema {
    pub dim: u32,
    pub metric: Metric,
    pub shard_count: u32,
    pub replication_n: u32,
}

/// One replica assignment in the derived shard map: which node holds a given
/// shard of a given collection, and that replica's operational state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardAssignment {
    pub collection: String,
    pub shard_idx: u32,
    pub node_id: NodeId,
    pub state: ShardState,
}

/// A mutation applied through the Raft log. This is the openraft `D` type.
///
/// Every entry that changes cluster metadata is one of these. Data-plane point
/// operations are NOT here — they never go through Raft (locked spec).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetaRequest {
    /// Register a node's advertise address in the directory. Membership proper
    /// (voter/learner set) is changed via the Raft membership API; this records
    /// the address so peers can route data-plane RPCs to it.
    RegisterNode {
        node_id: NodeId,
        advertise_addr: String,
    },
    /// Remove a node from the directory (paired with a membership change).
    DeregisterNode { node_id: NodeId },
    /// Create a collection with a fixed schema.
    CreateCollection {
        name: String,
        schema: CollectionSchema,
    },
    /// Drop a collection and all its shard assignments.
    DropCollection { name: String },
    /// Set a single shard replica's operational state (M5 transfer lifecycle).
    SetShardState {
        collection: String,
        shard_idx: u32,
        node_id: NodeId,
        state: ShardState,
    },
    /// A no-op, used to commit a barrier through the log (e.g. to confirm
    /// leadership / linearize a read). Applying it changes nothing.
    Noop,
}

/// The response returned from applying a [`MetaRequest`]. This is the openraft
/// `R` type. Kept small and serializable; callers mostly care that the entry
/// committed, not about a rich return value.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct MetaResponse {
    /// Human-readable note for logs/tests (e.g. "created", "exists", "dropped").
    pub note: String,
}

impl MetaResponse {
    pub fn new(note: impl Into<String>) -> Self {
        Self { note: note.into() }
    }
}

/// The quorvec-specific metadata the state machine maintains and serializes into
/// every Raft snapshot. openraft tracks the voter/learner membership separately;
/// this is the application state layered on top.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MetaState {
    /// node_id -> advertise address. The data-plane routing directory.
    pub nodes: BTreeMap<NodeId, String>,
    /// collection name -> schema.
    pub collections: BTreeMap<String, CollectionSchema>,
    /// Per-replica operational state overrides, keyed by
    /// (collection, shard_idx, node_id). Absent = Active. M3 keeps everything
    /// Active; M5 writes Syncing/Dead here.
    pub shard_states: BTreeMap<(String, u32, NodeId), ShardState>,
}

impl MetaState {
    /// Apply a committed [`MetaRequest`], mutating the metadata. Returns a short
    /// note describing what happened (surfaced in [`MetaResponse`]).
    pub fn apply(&mut self, req: &MetaRequest) -> MetaResponse {
        match req {
            MetaRequest::RegisterNode {
                node_id,
                advertise_addr,
            } => {
                self.nodes.insert(*node_id, advertise_addr.clone());
                MetaResponse::new("registered")
            }
            MetaRequest::DeregisterNode { node_id } => {
                self.nodes.remove(node_id);
                // Drop any shard-state overrides referencing the gone node.
                self.shard_states.retain(|(_, _, n), _| n != node_id);
                MetaResponse::new("deregistered")
            }
            MetaRequest::CreateCollection { name, schema } => {
                if self.collections.contains_key(name) {
                    MetaResponse::new("exists")
                } else {
                    self.collections.insert(name.clone(), schema.clone());
                    MetaResponse::new("created")
                }
            }
            MetaRequest::DropCollection { name } => {
                let existed = self.collections.remove(name).is_some();
                self.shard_states.retain(|(c, _, _), _| c != name);
                MetaResponse::new(if existed { "dropped" } else { "absent" })
            }
            MetaRequest::SetShardState {
                collection,
                shard_idx,
                node_id,
                state,
            } => {
                let key = (collection.clone(), *shard_idx, *node_id);
                if *state == ShardState::Active {
                    self.shard_states.remove(&key); // Active is the default
                } else {
                    self.shard_states.insert(key, *state);
                }
                MetaResponse::new("shard-state-set")
            }
            MetaRequest::Noop => MetaResponse::new("noop"),
        }
    }

    /// The set of physical node ids currently in the directory.
    pub fn node_set(&self) -> BTreeSet<NodeId> {
        self.nodes.keys().copied().collect()
    }

    /// Build the consistent-hash ring from the current membership.
    pub fn ring(&self) -> Ring {
        Ring::new(&self.node_set(), DEFAULT_VNODES)
    }

    /// The operational state of a specific replica (Active unless overridden).
    pub fn shard_state(&self, collection: &str, shard_idx: u32, node_id: NodeId) -> ShardState {
        self.shard_states
            .get(&(collection.to_string(), shard_idx, node_id))
            .copied()
            .unwrap_or(ShardState::Active)
    }

    /// The replica node ids for a shard, in clockwise order from the home
    /// (first = primary used for single-replica routing). Empty if the
    /// collection is unknown or the cluster is empty.
    pub fn replicas_for_shard(&self, collection: &str, shard_idx: u32) -> Vec<NodeId> {
        let Some(schema) = self.collections.get(collection) else {
            return Vec::new();
        };
        self.ring()
            .replicas_for_shard(collection, shard_idx, schema.replication_n)
    }

    /// The full derived shard map: every (collection, shard, replica-node) with
    /// its current operational state. Recomputed from membership + schemas; this
    /// is what `ClusterInfo` surfaces.
    pub fn shard_map(&self) -> Vec<ShardAssignment> {
        let mut out = Vec::new();
        let ring = self.ring();
        for (name, schema) in &self.collections {
            for shard_idx in 0..schema.shard_count {
                let replicas = ring.replicas_for_shard(name, shard_idx, schema.replication_n);
                for node_id in replicas {
                    let state = self.shard_state(name, shard_idx, node_id);
                    out.push(ShardAssignment {
                        collection: name.clone(),
                        shard_idx,
                        node_id,
                        state,
                    });
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schema(dim: u32, n: u32, shards: u32) -> CollectionSchema {
        CollectionSchema {
            dim,
            metric: Metric::L2,
            shard_count: shards,
            replication_n: n,
        }
    }

    #[test]
    fn create_and_drop_collection() {
        let mut s = MetaState::default();
        assert_eq!(
            s.apply(&MetaRequest::CreateCollection {
                name: "c".into(),
                schema: schema(64, 3, 8),
            })
            .note,
            "created"
        );
        // Duplicate is a no-op (idempotent through the log).
        assert_eq!(
            s.apply(&MetaRequest::CreateCollection {
                name: "c".into(),
                schema: schema(64, 3, 8),
            })
            .note,
            "exists"
        );
        assert!(s.collections.contains_key("c"));
        assert_eq!(
            s.apply(&MetaRequest::DropCollection { name: "c".into() })
                .note,
            "dropped"
        );
        assert!(!s.collections.contains_key("c"));
    }

    #[test]
    fn register_nodes_then_shard_map_populates() {
        let mut s = MetaState::default();
        for n in 1..=5u64 {
            s.apply(&MetaRequest::RegisterNode {
                node_id: n,
                advertise_addr: format!("127.0.0.1:70{n:02}"),
            });
        }
        s.apply(&MetaRequest::CreateCollection {
            name: "vectors".into(),
            schema: schema(64, 3, 8),
        });
        let map = s.shard_map();
        // 8 shards x 3 replicas = 24 assignments.
        assert_eq!(map.len(), 24);
        // Every assignment is Active by default.
        assert!(map.iter().all(|a| a.state == ShardState::Active));
        // Each shard has exactly 3 distinct replica nodes.
        for shard in 0..8u32 {
            let r = s.replicas_for_shard("vectors", shard);
            assert_eq!(r.len(), 3);
            let distinct: BTreeSet<_> = r.iter().copied().collect();
            assert_eq!(distinct.len(), 3);
        }
    }

    #[test]
    fn shard_state_override_and_clear() {
        let mut s = MetaState::default();
        s.apply(&MetaRequest::RegisterNode {
            node_id: 1,
            advertise_addr: "a".into(),
        });
        s.apply(&MetaRequest::SetShardState {
            collection: "c".into(),
            shard_idx: 0,
            node_id: 1,
            state: ShardState::Syncing,
        });
        assert_eq!(s.shard_state("c", 0, 1), ShardState::Syncing);
        // Setting back to Active removes the override.
        s.apply(&MetaRequest::SetShardState {
            collection: "c".into(),
            shard_idx: 0,
            node_id: 1,
            state: ShardState::Active,
        });
        assert_eq!(s.shard_state("c", 0, 1), ShardState::Active);
        assert!(s.shard_states.is_empty());
    }

    #[test]
    fn deregister_node_drops_its_shard_states() {
        let mut s = MetaState::default();
        s.apply(&MetaRequest::RegisterNode {
            node_id: 7,
            advertise_addr: "a".into(),
        });
        s.apply(&MetaRequest::SetShardState {
            collection: "c".into(),
            shard_idx: 2,
            node_id: 7,
            state: ShardState::Dead,
        });
        s.apply(&MetaRequest::DeregisterNode { node_id: 7 });
        assert!(!s.nodes.contains_key(&7));
        assert!(s.shard_states.is_empty());
    }
}
