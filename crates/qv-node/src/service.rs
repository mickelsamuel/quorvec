//! The v1 client gRPC service: maps the locked `Quorvec` surface onto the
//! cluster manager (metadata plane) and the data router (data plane).
//!
//! M3 wiring:
//! - CreateCollection / DropCollection go through the metadata-plane Raft log
//!   (committed cluster-wide before returning).
//! - Upsert / Delete / Get / Search are data-plane ops routed by the coordinator
//!   to the owning shard's primary replica (single-replica routing; M4 adds
//!   N/W/R quorums).
//! - ClusterInfo reflects the Raft-replicated metadata (nodes, shard map, leader).
//! - Join / Leave are metadata-plane ops: handled on the leader, forwarded there
//!   otherwise.

use std::sync::Arc;

use qv_cluster::{schema_with_defaults, ManagerError, Metric};
use qv_proto::v1;
use qv_proto::v1::quorvec_server::Quorvec;
use tonic::{Request, Response, Status};

use crate::node_state::NodeState;
use crate::router::{self, Consistency, RouterError};

const VERSION: &str = concat!("quorvec ", env!("CARGO_PKG_VERSION"), " (M3 cluster)");

const DEFAULT_K: u32 = 10;
const DEFAULT_EF_SEARCH: u32 = 64;

/// The v1 service handler.
pub struct QuorvecService {
    state: Arc<NodeState>,
}

impl QuorvecService {
    pub fn new(state: Arc<NodeState>) -> Self {
        Self { state }
    }
}

fn metric_from_proto(m: i32) -> Result<Metric, Status> {
    match v1::Metric::try_from(m).unwrap_or(v1::Metric::Unspecified) {
        v1::Metric::L2 => Ok(Metric::L2),
        v1::Metric::Cosine => Ok(Metric::Cosine),
        v1::Metric::Unspecified => Err(Status::invalid_argument("metric must be L2 or COSINE")),
    }
}

fn manager_err_to_status(e: ManagerError) -> Status {
    match e {
        ManagerError::NotLeader(_) => Status::failed_precondition(e.to_string()),
        ManagerError::Invalid(_) => Status::invalid_argument(e.to_string()),
        ManagerError::Serde(_) | ManagerError::Raft(_) => Status::internal(e.to_string()),
    }
}

fn router_err_to_status(e: RouterError) -> Status {
    match e {
        RouterError::NotFound(_) => Status::not_found(e.to_string()),
        RouterError::DimMismatch { .. } => Status::invalid_argument(e.to_string()),
        RouterError::NoReplica { .. }
        | RouterError::WriteQuorum { .. }
        | RouterError::ReadQuorum { .. } => Status::unavailable(e.to_string()),
        RouterError::Shard(_) | RouterError::Hint(_) => Status::internal(e.to_string()),
    }
}

#[tonic::async_trait]
impl Quorvec for QuorvecService {
    async fn create_collection(
        &self,
        request: Request<v1::CreateCollectionRequest>,
    ) -> Result<Response<v1::CreateCollectionResponse>, Status> {
        let req = request.into_inner();
        if req.name.is_empty() {
            return Err(Status::invalid_argument("collection name is empty"));
        }
        if req.dim == 0 {
            return Err(Status::invalid_argument("dim must be > 0"));
        }
        let metric = metric_from_proto(req.metric)?;
        let schema = schema_with_defaults(req.dim, metric, req.shard_count, req.replication_n);

        let note = self
            .state
            .cluster
            .create_collection(req.name.clone(), schema)
            .await
            .map_err(manager_err_to_status)?;
        if note == "exists" {
            return Err(Status::already_exists(format!(
                "collection '{}' already exists",
                req.name
            )));
        }
        Ok(Response::new(v1::CreateCollectionResponse {}))
    }

    async fn drop_collection(
        &self,
        request: Request<v1::DropCollectionRequest>,
    ) -> Result<Response<v1::DropCollectionResponse>, Status> {
        let req = request.into_inner();
        let note = self
            .state
            .cluster
            .drop_collection(req.name.clone())
            .await
            .map_err(manager_err_to_status)?;
        // Drop this node's materialized shards for the collection.
        self.state.shards.drop_collection(&req.name);
        if note == "absent" {
            return Err(Status::not_found(format!(
                "collection '{}' not found",
                req.name
            )));
        }
        Ok(Response::new(v1::DropCollectionResponse {}))
    }

    async fn upsert(
        &self,
        request: Request<v1::UpsertRequest>,
    ) -> Result<Response<v1::UpsertResponse>, Status> {
        let req = request.into_inner();
        let consistency = Consistency::from_proto(req.consistency);
        let points: Vec<(u64, Vec<f32>, Vec<u8>)> = req
            .points
            .into_iter()
            .map(|p| (p.id, p.vector, p.payload.unwrap_or_default()))
            .collect();
        let upserted =
            router::coordinate_upsert(&self.state, &req.collection, &points, consistency)
                .await
                .map_err(router_err_to_status)?;
        Ok(Response::new(v1::UpsertResponse { upserted }))
    }

    async fn delete(
        &self,
        request: Request<v1::DeleteRequest>,
    ) -> Result<Response<v1::DeleteResponse>, Status> {
        let req = request.into_inner();
        let consistency = Consistency::from_proto(req.consistency);
        let deleted =
            router::coordinate_delete(&self.state, &req.collection, &req.ids, consistency)
                .await
                .map_err(router_err_to_status)?;
        Ok(Response::new(v1::DeleteResponse { deleted }))
    }

    async fn get(
        &self,
        request: Request<v1::GetRequest>,
    ) -> Result<Response<v1::GetResponse>, Status> {
        let req = request.into_inner();
        let consistency = Consistency::from_proto(req.read_consistency);
        let points = router::coordinate_get(&self.state, &req.collection, &req.ids, consistency)
            .await
            .map_err(router_err_to_status)?
            .into_iter()
            .map(|(id, vector)| v1::Point {
                id,
                vector,
                payload: None,
            })
            .collect();
        Ok(Response::new(v1::GetResponse { points }))
    }

    async fn search(
        &self,
        request: Request<v1::SearchRequest>,
    ) -> Result<Response<v1::SearchResponse>, Status> {
        let req = request.into_inner();
        let k = if req.k == 0 { DEFAULT_K } else { req.k } as usize;
        let ef = if req.ef_search == 0 {
            DEFAULT_EF_SEARCH
        } else {
            req.ef_search
        } as usize;

        let hits = router::coordinate_search(&self.state, &req.collection, &req.vector, k, ef)
            .await
            .map_err(router_err_to_status)?;
        let results = hits
            .into_iter()
            .map(|(id, score)| v1::ScoredPoint {
                id,
                score,
                payload: None,
            })
            .collect();
        Ok(Response::new(v1::SearchResponse { results }))
    }

    async fn cluster_info(
        &self,
        _request: Request<v1::ClusterInfoRequest>,
    ) -> Result<Response<v1::ClusterInfoResponse>, Status> {
        let meta = self.state.cluster.metadata().await;
        let nodes = meta
            .nodes
            .iter()
            .map(|(id, addr)| v1::NodeInfo {
                node_id: *id,
                advertise_addr: addr.clone(),
            })
            .collect();
        let shard_map = meta
            .shard_map()
            .into_iter()
            .map(|a| v1::ShardAssignment {
                collection: a.collection,
                shard_idx: a.shard_idx,
                node_id: a.node_id,
                state: shard_state_to_proto(a.state) as i32,
            })
            .collect();
        let raft_leader = self.state.cluster.current_leader().unwrap_or(0);
        Ok(Response::new(v1::ClusterInfoResponse {
            nodes,
            shard_map,
            raft_leader,
            version: VERSION.to_string(),
        }))
    }

    async fn join(
        &self,
        request: Request<v1::JoinRequest>,
    ) -> Result<Response<v1::JoinResponse>, Status> {
        let req = request.into_inner();
        // The joining node proposes its own id derived from its advertise addr is
        // not robust; quorvec assigns ids out-of-band (config) and the joiner
        // passes its id in the addr as "id@addr". For M3 the integration harness
        // calls join with a concrete id; accept "id@host:port" or just an addr
        // (then we reject, since we need an explicit id).
        let (new_id, addr) = parse_join_target(&req.advertise_addr)
            .ok_or_else(|| Status::invalid_argument("join target must be 'node_id@host:port'"))?;

        self.state
            .cluster
            .join_node(new_id, addr)
            .await
            .map_err(manager_err_to_status)?;
        Ok(Response::new(v1::JoinResponse { node_id: new_id }))
    }

    async fn leave(
        &self,
        request: Request<v1::LeaveRequest>,
    ) -> Result<Response<v1::LeaveResponse>, Status> {
        let req = request.into_inner();
        self.state
            .cluster
            .leave_node(req.node_id)
            .await
            .map_err(manager_err_to_status)?;
        Ok(Response::new(v1::LeaveResponse {}))
    }

    async fn health(
        &self,
        _request: Request<v1::HealthRequest>,
    ) -> Result<Response<v1::HealthResponse>, Status> {
        Ok(Response::new(v1::HealthResponse {
            status: "ok".to_string(),
        }))
    }
}

fn shard_state_to_proto(s: qv_cluster::ShardState) -> v1::ShardState {
    match s {
        qv_cluster::ShardState::Active => v1::ShardState::Active,
        qv_cluster::ShardState::Syncing => v1::ShardState::Syncing,
        qv_cluster::ShardState::Dead => v1::ShardState::Dead,
    }
}

/// Parse a join target of the form `node_id@host:port`.
fn parse_join_target(s: &str) -> Option<(u64, String)> {
    let (id_str, addr) = s.split_once('@')?;
    let id: u64 = id_str.parse().ok()?;
    if addr.is_empty() {
        return None;
    }
    Some((id, addr.to_string()))
}
