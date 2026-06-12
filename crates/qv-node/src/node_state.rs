//! The node's shared runtime state: the cluster manager (metadata plane), this
//! node's durable shards, and the HLC clock for stamping writes.
//!
//! This is the single `Arc`-shared struct both gRPC services (the v1 client
//! service and the internal node-to-node service) hold. It is deliberately thin:
//! the metadata authority lives in [`qv_cluster::ClusterManager`], the durable
//! data lives in [`crate::shards::ShardStore`], and routing logic lives in
//! [`crate::router`].

use std::sync::Arc;

use qv_cluster::ClusterManager;
use qv_storage::HlcClock;

use crate::shards::ShardStore;

/// Shared node runtime state.
pub struct NodeState {
    pub node_id: u64,
    pub advertise_addr: String,
    /// Metadata plane (openraft): schemas, membership, shard map.
    pub cluster: ClusterManager,
    /// This node's durable shards.
    pub shards: ShardStore,
    /// Monotonic HLC for coordinator write stamping (M3 local; M4 cross-node).
    pub clock: HlcClock,
}

impl NodeState {
    pub fn new(
        node_id: u64,
        advertise_addr: String,
        cluster: ClusterManager,
        shards: ShardStore,
    ) -> Arc<Self> {
        Arc::new(Self {
            node_id,
            advertise_addr,
            cluster,
            shards,
            clock: HlcClock::new(),
        })
    }
}
