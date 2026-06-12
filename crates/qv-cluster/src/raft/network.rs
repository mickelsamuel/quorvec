//! The [`RaftNetwork`] implementation for the metadata plane.
//!
//! Raft RPCs (vote, append_entries, install_snapshot) travel between nodes over
//! the **internal** gRPC service (`QuorvecInternal::RaftRpc`), NOT the locked v1
//! client surface. Each openraft request is serialized (serde_json, via
//! openraft's `serde` feature) into a `RaftEnvelope`, sent to the target node's
//! advertise address, and the reply deserialized back into the openraft response
//! type. The envelope's `kind` tells the receiver which type to decode into.
//!
//! Per openraft's `new_client` contract, the factory does NOT connect eagerly;
//! the tonic channel connects lazily on first use and reconnects as needed.

use openraft::error::{InstallSnapshotError, NetworkError, RemoteError, Unreachable};
use openraft::network::{RPCOption, RaftNetwork, RaftNetworkFactory};
use openraft::raft::{
    AppendEntriesRequest, AppendEntriesResponse, InstallSnapshotRequest, InstallSnapshotResponse,
    VoteRequest, VoteResponse,
};
use qv_proto::internal::{RaftEnvelope, RaftRpcKind};
use qv_proto::QuorvecInternalClient;
use tonic::transport::Channel;

use crate::raft::typ::{RPCError, RaftError};
use crate::raft::{Node, NodeId, TypeConfig};

/// The network factory: turns a target (node id + address) into a connection.
#[derive(Clone, Default)]
pub struct RaftGrpcNetwork;

impl RaftNetworkFactory<TypeConfig> for RaftGrpcNetwork {
    type Network = RaftGrpcConnection;

    async fn new_client(&mut self, target: NodeId, node: &Node) -> Self::Network {
        RaftGrpcConnection {
            target,
            addr: node.addr.clone(),
        }
    }
}

/// A lazily-connecting client to one peer's internal Raft endpoint.
pub struct RaftGrpcConnection {
    target: NodeId,
    addr: String,
}

impl RaftGrpcConnection {
    /// Connect (lazily) to the peer's internal service. `addr` is a host:port
    /// advertise address; we prefix the http scheme tonic expects.
    async fn client(&self) -> Result<QuorvecInternalClient<Channel>, Unreachable> {
        let endpoint = if self.addr.starts_with("http://") || self.addr.starts_with("https://") {
            self.addr.clone()
        } else {
            format!("http://{}", self.addr)
        };
        QuorvecInternalClient::connect(endpoint)
            .await
            .map_err(|e| Unreachable::new(&e))
    }

    /// Send one serialized Raft RPC and return the raw reply bytes.
    async fn call(&self, kind: RaftRpcKind, payload: Vec<u8>) -> Result<Vec<u8>, Unreachable> {
        let mut client = self.client().await?;
        let reply = client
            .raft_rpc(RaftEnvelope {
                kind: kind as i32,
                payload,
            })
            .await
            .map_err(|e| Unreachable::new(&e))?;
        Ok(reply.into_inner().payload)
    }
}

// openraft's `RPCError` is a large enum by design; these helpers must return it
// to compose with the trait methods via `?`, so the lint does not apply here.

/// Serialize an openraft request; map serde errors into a network error. The
/// error parameter `E` lets this be used in both the default-error RPCs (vote,
/// append) and the `InstallSnapshotError` RPC.
#[allow(clippy::result_large_err)]
fn encode<T: serde::Serialize, E>(v: &T) -> Result<Vec<u8>, RPCError<E>>
where
    E: std::error::Error,
{
    serde_json::to_vec(v).map_err(|e| RPCError::Network(NetworkError::new(&e)))
}

/// Deserialize an openraft response; map serde errors into a network error.
#[allow(clippy::result_large_err)]
fn decode<T: serde::de::DeserializeOwned, E>(bytes: &[u8]) -> Result<T, RPCError<E>>
where
    E: std::error::Error,
{
    serde_json::from_slice(bytes).map_err(|e| RPCError::Network(NetworkError::new(&e)))
}

impl RaftNetwork<TypeConfig> for RaftGrpcConnection {
    async fn append_entries(
        &mut self,
        rpc: AppendEntriesRequest<TypeConfig>,
        _option: RPCOption,
    ) -> Result<AppendEntriesResponse<NodeId>, RPCError> {
        let payload = encode(&rpc)?;
        let bytes = self
            .call(RaftRpcKind::Append, payload)
            .await
            .map_err(RPCError::Unreachable)?;
        // The reply is `Result<AppendEntriesResponse, RaftError>` serialized.
        let res: Result<AppendEntriesResponse<NodeId>, RaftError> = decode(&bytes)?;
        res.map_err(|e| RPCError::RemoteError(RemoteError::new(self.target, e)))
    }

    async fn install_snapshot(
        &mut self,
        rpc: InstallSnapshotRequest<TypeConfig>,
        _option: RPCOption,
    ) -> Result<InstallSnapshotResponse<NodeId>, RPCError<InstallSnapshotError>> {
        let payload = encode(&rpc)?;
        let bytes = self
            .call(RaftRpcKind::Snapshot, payload)
            .await
            .map_err(RPCError::Unreachable)?;
        let res: Result<InstallSnapshotResponse<NodeId>, RaftError<InstallSnapshotError>> =
            decode(&bytes)?;
        res.map_err(|e| RPCError::RemoteError(RemoteError::new(self.target, e)))
    }

    async fn vote(
        &mut self,
        rpc: VoteRequest<NodeId>,
        _option: RPCOption,
    ) -> Result<VoteResponse<NodeId>, RPCError> {
        let payload = encode(&rpc)?;
        let bytes = self
            .call(RaftRpcKind::Vote, payload)
            .await
            .map_err(RPCError::Unreachable)?;
        let res: Result<VoteResponse<NodeId>, RaftError> = decode(&bytes)?;
        res.map_err(|e| RPCError::RemoteError(RemoteError::new(self.target, e)))
    }
}
