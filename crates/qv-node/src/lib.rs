//! `qv-node` library surface: the building blocks the binary wires together and
//! that integration tests drive directly.
//!
//! M3 makes the node cluster-aware: the metadata plane is the openraft-backed
//! [`qv_cluster::ClusterManager`], the data plane is durable shards
//! ([`shards::ShardStore`]) routed by [`router`], and two gRPC services (the
//! locked v1 client surface [`service`] and the internal node-to-node surface
//! [`internal_service`]) share one listen port.

pub mod config;
pub mod internal_service;
pub mod node_state;
pub mod rebalance;
pub mod router;
pub mod service;
pub mod shards;

pub use config::NodeConfig;
pub use internal_service::InternalService;
pub use node_state::NodeState;
pub use service::QuorvecService;
pub use shards::ShardStore;

use std::sync::Arc;

use qv_cluster::ClusterManager;

/// Build and start a node's shared runtime state from a config.
///
/// Starts the metadata-plane Raft (does not yet form/join a cluster) and opens
/// the durable shard store. Bootstrapping a new single-node cluster or joining an
/// existing one is an explicit follow-up step (`bootstrap` flag / a Join call),
/// so a node can come up and then be wired into a cluster deterministically.
pub async fn build_node_state(cfg: &NodeConfig) -> anyhow::Result<Arc<NodeState>> {
    let advertise = cfg.advertise().to_string();
    // Durable Raft storage lives under <data_dir>/raft (ruling R6).
    let cluster =
        ClusterManager::start(cfg.node_id, advertise.clone(), cfg.data_dir.clone()).await?;
    let shards = ShardStore::new(cfg.data_dir.clone(), cfg.snapshot_wal_mb, cfg.wal_batch_ms);
    Ok(NodeState::new(
        cfg.node_id,
        advertise,
        cluster,
        shards,
        cfg.data_dir.clone(),
    ))
}
