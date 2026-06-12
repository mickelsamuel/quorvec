//! `qv-cluster` — the distributed control and data plane for quorvec.
//!
//! - [`placement`]: the two-layer placement (point→shard, shard→nodes via the
//!   consistent-hash ring with 64 vnodes/node). Pure compute.
//! - [`meta`]: the metadata domain — the Raft request/response types and the
//!   [`meta::MetaState`] the state machine holds (node directory, collection
//!   schemas, shard states), plus the derived shard map.
//! - [`raft`]: the openraft type config + in-memory storage + state machine for
//!   the metadata plane.
//!
//! Quorum replication, hinted handoff, read repair, and scatter-gather search
//! (the data plane, M4) layer on top of the placement + node directory exposed
//! here.

pub mod manager;
pub mod meta;
pub mod placement;
pub mod raft;

pub use manager::{schema_with_defaults, ClusterManager, ManagerError};
pub use meta::{
    CollectionSchema, MetaRequest, MetaResponse, MetaState, Metric, ShardAssignment, ShardState,
};
pub use placement::{shard_for_id, Ring, DEFAULT_VNODES};
