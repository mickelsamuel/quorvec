//! The metadata-plane Raft: type config, in-memory storage, and the state
//! machine wrapping [`crate::meta::MetaState`].
//!
//! Raft is used for the metadata plane ONLY (locked spec): membership, collection
//! schemas, and the derived shard map. Data-plane point operations never touch
//! consensus.

pub mod network;
pub mod store;
pub mod types;

pub use network::RaftGrpcNetwork;
pub use store::{LogStore, StateMachineStore, StoredSnapshot};
pub use types::{typ, Node, NodeId, Raft, TypeConfig};
