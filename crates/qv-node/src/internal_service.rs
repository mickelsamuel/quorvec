//! The internal node-to-node gRPC service.
//!
//! Two responsibilities:
//! - **Metadata plane:** receive serialized Raft RPCs (vote/append/snapshot) and
//!   feed them into the local Raft via the cluster manager.
//! - **Data plane:** receive replica point operations forwarded by a coordinator
//!   and apply them to this node's durable shards.
//!
//! This service is bound on the same address as the v1 service (one tonic Router
//! serves both), so a node has a single listen port.

use std::sync::Arc;

use qv_proto::internal::quorvec_internal_server::QuorvecInternal;
use qv_proto::internal::{
    RaftEnvelope, RaftReply, ReplicaGetRequest, ReplicaGetResponse, ReplicaSearchRequest,
    ReplicaSearchResponse, ReplicaWriteRequest, ReplicaWriteResponse,
};
use tonic::{Request, Response, Status};

use crate::node_state::NodeState;
use crate::router;

const DEFAULT_K: u32 = 10;
const DEFAULT_EF_SEARCH: u32 = 64;

/// The internal service handler.
pub struct InternalService {
    state: Arc<NodeState>,
}

impl InternalService {
    pub fn new(state: Arc<NodeState>) -> Self {
        Self { state }
    }

    /// Look up a collection schema from the metadata, erroring if unknown.
    async fn schema(&self, collection: &str) -> Result<qv_cluster::CollectionSchema, Status> {
        self.state
            .cluster
            .metadata()
            .await
            .collections
            .get(collection)
            .cloned()
            .ok_or_else(|| Status::not_found(format!("collection '{collection}' not found")))
    }
}

fn router_err(e: router::RouterError) -> Status {
    Status::internal(e.to_string())
}

#[tonic::async_trait]
impl QuorvecInternal for InternalService {
    async fn raft_rpc(
        &self,
        request: Request<RaftEnvelope>,
    ) -> Result<Response<RaftReply>, Status> {
        let env = request.into_inner();
        let reply = self
            .state
            .cluster
            .handle_raft_rpc(env)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;
        Ok(Response::new(reply))
    }

    async fn replica_write(
        &self,
        request: Request<ReplicaWriteRequest>,
    ) -> Result<Response<ReplicaWriteResponse>, Status> {
        let req = request.into_inner();
        let schema = self.schema(&req.collection).await?;
        let applied =
            router::apply_replica_write(&self.state, &req, &schema).map_err(router_err)?;
        Ok(Response::new(ReplicaWriteResponse { applied }))
    }

    async fn replica_get(
        &self,
        request: Request<ReplicaGetRequest>,
    ) -> Result<Response<ReplicaGetResponse>, Status> {
        let req = request.into_inner();
        let schema = self.schema(&req.collection).await?;
        let points = router::apply_replica_get(
            &self.state,
            &req.collection,
            req.shard_idx,
            &req.ids,
            &schema,
        )
        .map_err(router_err)?;
        Ok(Response::new(ReplicaGetResponse { points }))
    }

    async fn replica_search(
        &self,
        request: Request<ReplicaSearchRequest>,
    ) -> Result<Response<ReplicaSearchResponse>, Status> {
        let req = request.into_inner();
        let schema = self.schema(&req.collection).await?;
        let k = if req.k == 0 { DEFAULT_K } else { req.k } as usize;
        let ef = if req.ef_search == 0 {
            DEFAULT_EF_SEARCH
        } else {
            req.ef_search
        } as usize;
        let results = router::apply_replica_search(
            &self.state,
            &req.collection,
            req.shard_idx,
            &req.vector,
            k,
            ef,
            &schema,
        )
        .map_err(router_err)?;
        Ok(Response::new(ReplicaSearchResponse { results }))
    }
}
