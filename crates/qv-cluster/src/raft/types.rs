//! openraft type configuration for the quorvec metadata plane.
//!
//! The metadata plane is the ONLY place Raft is used (data-plane ops never touch
//! consensus). The Raft log carries [`MetaRequest`]s — membership and schema/shard
//! mutations — and the state machine ([`super::state`]) applies them to the
//! authoritative cluster metadata.
//!
//! Pinned to openraft 0.9.24 (latest stable). `NodeId = u64` and `Node =
//! BasicNode` (advertise address) are the openraft defaults; only `D`/`R` are
//! customized via `declare_raft_types!`, exactly as the canonical
//! `raft-kv-memstore` example does.

// `declare_raft_types!` expands to reference `Cursor` for the default
// `SnapshotData = Cursor<Vec<u8>>`, so it must be in scope here.
use std::io::Cursor;

use openraft::BasicNode;

use crate::meta::{MetaRequest, MetaResponse};

/// Numeric Raft node id (same id space as the quorvec node id).
pub type NodeId = u64;

/// Node address record stored in Raft membership.
pub type Node = BasicNode;

openraft::declare_raft_types!(
    /// quorvec metadata-plane Raft type configuration.
    pub TypeConfig:
        D = MetaRequest,
        R = MetaResponse,
);

/// The concrete Raft handle for the metadata plane.
pub type Raft = openraft::Raft<TypeConfig>;

/// openraft error aliases used across the cluster crate.
pub mod typ {
    use openraft::error::Infallible;

    use super::{NodeId, TypeConfig};

    pub type Entry = openraft::Entry<TypeConfig>;
    pub type RaftError<E = Infallible> = openraft::error::RaftError<NodeId, E>;
    pub type RPCError<E = Infallible> =
        openraft::error::RPCError<NodeId, super::Node, RaftError<E>>;
    pub type ClientWriteError = openraft::error::ClientWriteError<NodeId, super::Node>;
    pub type CheckIsLeaderError = openraft::error::CheckIsLeaderError<NodeId, super::Node>;
    pub type ForwardToLeader = openraft::error::ForwardToLeader<NodeId, super::Node>;
    pub type InitializeError = openraft::error::InitializeError<NodeId, super::Node>;
    pub type ClientWriteResponse = openraft::raft::ClientWriteResponse<TypeConfig>;
}
