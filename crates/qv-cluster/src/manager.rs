//! The cluster manager: the node-facing handle over the metadata-plane Raft.
//!
//! It owns the `Raft` instance, the state-machine store (for consistent metadata
//! reads), and the node's own identity. It exposes exactly the operations the
//! gRPC node layer needs:
//!
//! - [`ClusterManager::bootstrap`] — initialize a brand-new single-node cluster
//!   (this node becomes the founding voter + leader).
//! - [`ClusterManager::join_node`] — admit a node: add it as a learner, replicate
//!   the log to it, then promote it to a voter via a membership change. This is
//!   the "Join through Raft" path (locked spec).
//! - [`ClusterManager::leave_node`] — remove a node from the voter set and the
//!   directory (the "Leave through Raft" path).
//! - [`ClusterManager::create_collection`] / [`drop_collection`] — schema changes
//!   committed through the Raft log.
//! - [`ClusterManager::handle_raft_rpc`] — feed an incoming serialized Raft RPC
//!   (from a peer) into the local Raft and serialize the reply.
//! - metadata reads ([`metadata`], [`cluster_info`], routing helpers).
//!
//! Errors are deliberately coarse: callers (the gRPC layer) translate them into
//! `tonic::Status`. Anything that is a genuine openraft API surprise is surfaced
//! verbatim, never silently swallowed.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Arc;

use openraft::Config;
use qv_proto::internal::{RaftEnvelope, RaftReply, RaftRpcKind};
use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::meta::{CollectionSchema, MetaRequest, MetaState, ShardAssignment};
use crate::raft::{LogStore, Node, NodeId, Raft, RaftGrpcNetwork, StateMachineStore};

/// Coarse cluster-manager error. The gRPC layer maps these to tonic statuses.
#[derive(Debug, thiserror::Error)]
pub enum ManagerError {
    #[error("raft: {0}")]
    Raft(String),
    #[error("not the leader{0}")]
    NotLeader(String),
    #[error("serialization: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("invalid request: {0}")]
    Invalid(String),
}

/// A node's handle on the metadata-plane cluster.
#[derive(Clone)]
pub struct ClusterManager {
    node_id: NodeId,
    advertise_addr: String,
    raft: Raft,
    sm: StateMachineStore,
}

impl ClusterManager {
    /// Construct the manager and start the local Raft instance with **durable**
    /// storage under `<data_dir>/raft` (ruling R6). Does NOT form or join a
    /// cluster — call [`bootstrap`] (founding node) or have an existing leader
    /// [`join_node`] this node afterward. On restart, the durable log +
    /// state-machine image are recovered before Raft starts, so a node (or the
    /// whole cluster) comes back with its metadata intact.
    pub async fn start(
        node_id: NodeId,
        advertise_addr: String,
        data_dir: impl AsRef<Path>,
    ) -> Result<Self, ManagerError> {
        // Conservative timers: heartbeat 250ms, election 1000-1500ms. Slow enough
        // to be stable on a shared localhost host, fast enough that a leader-kill
        // re-elects within a couple seconds (the M3 acceptance window).
        let config = Config {
            cluster_name: "quorvec-meta".to_string(),
            heartbeat_interval: 250,
            election_timeout_min: 1000,
            election_timeout_max: 1500,
            // Snapshot the metadata state machine periodically; it is tiny.
            snapshot_policy: openraft::SnapshotPolicy::LogsSinceLast(1024),
            ..Default::default()
        };
        let config = Arc::new(
            config
                .validate()
                .map_err(|e| ManagerError::Raft(format!("invalid raft config: {e}")))?,
        );

        // Durable Raft storage under <data_dir>/raft (ruling R6).
        let raft_dir = data_dir.as_ref().join("raft");
        let log_store = LogStore::open(&raft_dir)
            .map_err(|e| ManagerError::Raft(format!("open durable raft log store: {e}")))?;
        let sm = StateMachineStore::open(&raft_dir)
            .map_err(|e| ManagerError::Raft(format!("open durable raft state machine: {e}")))?;
        let network = RaftGrpcNetwork;

        let raft = Raft::new(node_id, config, network, log_store, sm.clone())
            .await
            .map_err(|e| ManagerError::Raft(format!("raft init: {e}")))?;

        Ok(Self {
            node_id,
            advertise_addr,
            raft,
            sm,
        })
    }

    /// This node's id.
    pub fn node_id(&self) -> NodeId {
        self.node_id
    }

    /// The underlying Raft handle (for advanced callers / tests).
    pub fn raft(&self) -> &Raft {
        &self.raft
    }

    fn node_record(&self) -> Node {
        Node::new(self.advertise_addr.clone())
    }

    /// Initialize a brand-new single-node cluster with this node as the founding
    /// voter, then register its address in the directory.
    ///
    /// **Restart-safe (ruling R6).** With durable Raft storage, a seed node that
    /// restarts already has its membership + log on disk. Re-running `initialize`
    /// would be rejected (`InitializeError::NotAllowed`), so we first check
    /// `is_initialized()` and, when already initialized, simply resume from the
    /// recovered durable state — no re-initialize, no re-register. This is what
    /// lets the founding node survive a full-cluster restart.
    pub async fn bootstrap(&self) -> Result<(), ManagerError> {
        let already = self
            .raft
            .is_initialized()
            .await
            .map_err(|e| ManagerError::Raft(format!("is_initialized: {e}")))?;
        if already {
            // Durable state recovered: the cluster already exists. Resume.
            return Ok(());
        }

        let mut members = BTreeMap::new();
        members.insert(self.node_id, self.node_record());
        self.raft
            .initialize(members)
            .await
            .map_err(|e| ManagerError::Raft(format!("initialize: {e}")))?;

        // Record this node's advertise address in the directory.
        self.write(MetaRequest::RegisterNode {
            node_id: self.node_id,
            advertise_addr: self.advertise_addr.clone(),
        })
        .await?;
        Ok(())
    }

    /// Admit a node to the cluster: register its address, add it as a learner
    /// (replicate the log to it), then promote the full voter set to include it.
    /// This is invoked on the LEADER (a Join request received elsewhere is
    /// forwarded to the leader by the gRPC layer).
    pub async fn join_node(
        &self,
        new_id: NodeId,
        advertise_addr: String,
    ) -> Result<(), ManagerError> {
        let node = Node::new(advertise_addr.clone());

        // 1. Record the address so peers (and the new node) can route.
        self.write(MetaRequest::RegisterNode {
            node_id: new_id,
            advertise_addr,
        })
        .await?;

        // 2. Add as a learner; block until its log catches up.
        self.raft
            .add_learner(new_id, node, true)
            .await
            .map_err(|e| ManagerError::Raft(format!("add_learner: {e}")))?;

        // 3. Promote: new voter set = existing voters ∪ {new_id}. retain=true
        //    keeps any other learners as learners.
        let mut voters = self.current_voters();
        voters.insert(new_id);
        self.raft
            .change_membership(voters, true)
            .await
            .map_err(|e| ManagerError::Raft(format!("change_membership(join): {e}")))?;
        Ok(())
    }

    /// Remove a node from the voter set and the directory. Invoked on the leader.
    pub async fn leave_node(&self, node_id: NodeId) -> Result<(), ManagerError> {
        let mut voters = self.current_voters();
        if !voters.remove(&node_id) {
            return Err(ManagerError::Invalid(format!(
                "node {node_id} is not a current voter"
            )));
        }
        if voters.is_empty() {
            return Err(ManagerError::Invalid(
                "refusing to remove the last voter".into(),
            ));
        }
        // retain=false: the leaving node is fully removed (not demoted to learner).
        self.raft
            .change_membership(voters, false)
            .await
            .map_err(|e| ManagerError::Raft(format!("change_membership(leave): {e}")))?;
        self.write(MetaRequest::DeregisterNode { node_id }).await?;
        Ok(())
    }

    /// Create a collection (schema committed through the Raft log).
    pub async fn create_collection(
        &self,
        name: String,
        schema: CollectionSchema,
    ) -> Result<String, ManagerError> {
        let resp = self
            .write(MetaRequest::CreateCollection { name, schema })
            .await?;
        Ok(resp)
    }

    /// Drop a collection (committed through the Raft log).
    pub async fn drop_collection(&self, name: String) -> Result<String, ManagerError> {
        let resp = self.write(MetaRequest::DropCollection { name }).await?;
        Ok(resp)
    }

    /// Commit a metadata mutation through the Raft log; returns the apply note.
    pub async fn write(&self, req: MetaRequest) -> Result<String, ManagerError> {
        let resp = self.raft.client_write(req).await.map_err(|e| {
            // A non-leader write is the common, recoverable case — surface it as
            // NotLeader so the gRPC layer can redirect rather than treat it fatal.
            let s = e.to_string();
            if s.contains("ForwardToLeader") || s.contains("not the leader") {
                ManagerError::NotLeader(format!(": {s}"))
            } else {
                ManagerError::Raft(format!("client_write: {s}"))
            }
        })?;
        Ok(resp.data.note)
    }

    /// The current voter set from Raft metrics.
    fn current_voters(&self) -> BTreeSet<NodeId> {
        let metrics = self.raft.metrics().borrow().clone();
        metrics.membership_config.membership().voter_ids().collect()
    }

    /// The current Raft leader, if known.
    pub fn current_leader(&self) -> Option<NodeId> {
        self.raft.metrics().borrow().current_leader
    }

    /// A consistent read of the metadata state machine.
    pub async fn metadata(&self) -> MetaState {
        self.sm.read_state().await
    }

    /// The derived shard map (for ClusterInfo).
    pub async fn shard_map(&self) -> Vec<ShardAssignment> {
        self.sm.read_state().await.shard_map()
    }

    /// The node directory (node_id -> advertise addr).
    pub async fn nodes(&self) -> BTreeMap<NodeId, String> {
        self.sm.read_state().await.nodes
    }

    /// Feed an incoming serialized Raft RPC from a peer into the local Raft and
    /// serialize the reply. This is the receiving end of [`crate::raft::network`].
    pub async fn handle_raft_rpc(&self, env: RaftEnvelope) -> Result<RaftReply, ManagerError> {
        let kind = RaftRpcKind::try_from(env.kind).unwrap_or(RaftRpcKind::Unspecified);
        let payload = match kind {
            RaftRpcKind::Vote => {
                let req = de(&env.payload)?;
                let res = self.raft.vote(req).await;
                ser(&res)?
            }
            RaftRpcKind::Append => {
                let req = de(&env.payload)?;
                let res = self.raft.append_entries(req).await;
                ser(&res)?
            }
            RaftRpcKind::Snapshot => {
                let req = de(&env.payload)?;
                let res = self.raft.install_snapshot(req).await;
                ser(&res)?
            }
            RaftRpcKind::Unspecified => {
                return Err(ManagerError::Invalid("unspecified raft rpc kind".into()))
            }
        };
        Ok(RaftReply { payload })
    }
}

fn de<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, ManagerError> {
    serde_json::from_slice(bytes).map_err(ManagerError::Serde)
}

fn ser<T: Serialize>(v: &T) -> Result<Vec<u8>, ManagerError> {
    serde_json::to_vec(v).map_err(ManagerError::Serde)
}

// Re-export the proto request/reply so the node layer can name them without a
// direct qv-proto dependency edge for this purpose.
pub use qv_proto::internal::{RaftEnvelope as RaftRpcRequest, RaftReply as RaftRpcResponse};

/// Build a [`CollectionSchema`] applying the server-side defaults for zero-valued
/// shard_count / replication_n (proto3 zero values).
pub fn schema_with_defaults(
    dim: u32,
    metric: crate::meta::Metric,
    shard_count: u32,
    replication_n: u32,
) -> CollectionSchema {
    CollectionSchema {
        dim,
        metric,
        shard_count: if shard_count == 0 { 8 } else { shard_count },
        replication_n: if replication_n == 0 { 3 } else { replication_n },
    }
}
