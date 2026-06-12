//! gRPC service: maps the generated `Quorvec` trait onto the in-memory [`Store`].
//!
//! M0 is single-node. Consistency levels are accepted and validated but, with a
//! single replica, every level resolves locally — the field is plumbed through
//! now so the wire contract is stable when replication lands in M4. Join/Leave
//! are metadata-plane operations and return `Unimplemented` until M3.

use crate::store::{Store, StoreError};
use qv_hnsw::Metric;
use qv_proto::v1;
use qv_proto::v1::quorvec_server::Quorvec;
use std::sync::Arc;
use tonic::{Request, Response, Status};

/// Server version string surfaced via ClusterInfo.
const VERSION: &str = concat!("quorvec ", env!("CARGO_PKG_VERSION"), " (M0 single-node)");

/// Default search parameters when the request leaves them at proto3 zero.
const DEFAULT_K: u32 = 10;
const DEFAULT_EF_SEARCH: u32 = 64;

/// The service handler. Holds shared node state.
pub struct QuorvecService {
    store: Arc<Store>,
    node_id: u64,
    advertise_addr: String,
}

impl QuorvecService {
    pub fn new(store: Arc<Store>, node_id: u64, advertise_addr: String) -> Self {
        Self {
            store,
            node_id,
            advertise_addr,
        }
    }
}

// ---- Conversions -----------------------------------------------------------

fn metric_from_proto(m: i32) -> Result<Metric, Status> {
    match v1::Metric::try_from(m).unwrap_or(v1::Metric::Unspecified) {
        v1::Metric::L2 => Ok(Metric::L2),
        v1::Metric::Cosine => Ok(Metric::Cosine),
        v1::Metric::Unspecified => Err(Status::invalid_argument("metric must be L2 or COSINE")),
    }
}

fn store_err_to_status(e: StoreError) -> Status {
    match e {
        StoreError::AlreadyExists(_) => Status::already_exists(e.to_string()),
        StoreError::NotFound(_) => Status::not_found(e.to_string()),
        StoreError::Invalid(_) => Status::invalid_argument(e.to_string()),
    }
}

#[tonic::async_trait]
impl Quorvec for QuorvecService {
    async fn create_collection(
        &self,
        request: Request<v1::CreateCollectionRequest>,
    ) -> Result<Response<v1::CreateCollectionResponse>, Status> {
        let req = request.into_inner();
        let metric = metric_from_proto(req.metric)?;
        self.store
            .create_collection(
                &req.name,
                req.dim,
                metric,
                req.shard_count,
                req.replication_n,
            )
            .map_err(store_err_to_status)?;
        Ok(Response::new(v1::CreateCollectionResponse {}))
    }

    async fn drop_collection(
        &self,
        request: Request<v1::DropCollectionRequest>,
    ) -> Result<Response<v1::DropCollectionResponse>, Status> {
        let req = request.into_inner();
        self.store
            .drop_collection(&req.name)
            .map_err(store_err_to_status)?;
        Ok(Response::new(v1::DropCollectionResponse {}))
    }

    async fn upsert(
        &self,
        request: Request<v1::UpsertRequest>,
    ) -> Result<Response<v1::UpsertResponse>, Status> {
        let req = request.into_inner();
        let points: Vec<(u64, Vec<f32>)> =
            req.points.into_iter().map(|p| (p.id, p.vector)).collect();
        let upserted = self
            .store
            .upsert(&req.collection, &points)
            .map_err(store_err_to_status)?;
        Ok(Response::new(v1::UpsertResponse { upserted }))
    }

    async fn delete(
        &self,
        request: Request<v1::DeleteRequest>,
    ) -> Result<Response<v1::DeleteResponse>, Status> {
        let req = request.into_inner();
        let deleted = self
            .store
            .delete(&req.collection, &req.ids)
            .map_err(store_err_to_status)?;
        Ok(Response::new(v1::DeleteResponse { deleted }))
    }

    async fn get(
        &self,
        request: Request<v1::GetRequest>,
    ) -> Result<Response<v1::GetResponse>, Status> {
        let req = request.into_inner();
        // M0: Get returns the stored vector for each requested id. The
        // brute-force index is search-oriented, so we recover an exact point
        // via a self-query: search for the vector is not possible by id alone,
        // so we expose Get through the store's id lookup.
        let points = self
            .store
            .get(&req.collection, &req.ids)
            .map_err(store_err_to_status)?
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

        let hits = self
            .store
            .search(&req.collection, &req.vector, k, ef)
            .map_err(store_err_to_status)?;
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
        // Single-node view. Shard map / Raft leadership land in M3.
        Ok(Response::new(v1::ClusterInfoResponse {
            nodes: vec![v1::NodeInfo {
                node_id: self.node_id,
                advertise_addr: self.advertise_addr.clone(),
            }],
            shard_map: Vec::new(),
            raft_leader: self.node_id,
            version: VERSION.to_string(),
        }))
    }

    async fn join(
        &self,
        _request: Request<v1::JoinRequest>,
    ) -> Result<Response<v1::JoinResponse>, Status> {
        Err(Status::unimplemented(
            "Join is a metadata-plane operation; available from M3",
        ))
    }

    async fn leave(
        &self,
        _request: Request<v1::LeaveRequest>,
    ) -> Result<Response<v1::LeaveResponse>, Status> {
        Err(Status::unimplemented(
            "Leave is a metadata-plane operation; available from M3",
        ))
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
