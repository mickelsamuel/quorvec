//! `qv-client` — a thin convenience wrapper over the generated gRPC client.
//!
//! It exists so tests, benches, and the failure harness share one ergonomic
//! surface instead of re-deriving request structs everywhere. It adds no
//! behavior beyond connecting and forwarding calls.

use qv_proto::v1;
use qv_proto::QuorvecClient;
use tonic::transport::Channel;

/// Errors from the client wrapper.
#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    #[error("transport: {0}")]
    Transport(#[from] tonic::transport::Error),
    #[error("rpc status: {0}")]
    Status(#[from] tonic::Status),
}

/// A connected quorvec client.
#[derive(Clone)]
pub struct Client {
    inner: QuorvecClient<Channel>,
}

impl Client {
    /// Connect to a quorvec node at `endpoint` (e.g. `http://127.0.0.1:7000`).
    pub async fn connect(endpoint: impl Into<String>) -> Result<Self, ClientError> {
        let channel = Channel::from_shared(endpoint.into())
            .map_err(|e| ClientError::Status(tonic::Status::invalid_argument(e.to_string())))?
            .connect()
            .await?;
        Ok(Self {
            inner: QuorvecClient::new(channel),
        })
    }

    /// Create a collection. `shard_count`/`replication_n` of 0 take server defaults.
    pub async fn create_collection(
        &mut self,
        name: &str,
        dim: u32,
        metric: v1::Metric,
        shard_count: u32,
        replication_n: u32,
    ) -> Result<(), ClientError> {
        self.inner
            .create_collection(v1::CreateCollectionRequest {
                name: name.to_string(),
                dim,
                metric: metric as i32,
                shard_count,
                replication_n,
            })
            .await?;
        Ok(())
    }

    /// Drop a collection.
    pub async fn drop_collection(&mut self, name: &str) -> Result<(), ClientError> {
        self.inner
            .drop_collection(v1::DropCollectionRequest {
                name: name.to_string(),
            })
            .await?;
        Ok(())
    }

    /// Upsert points. Returns the number accepted.
    pub async fn upsert(
        &mut self,
        collection: &str,
        points: Vec<v1::Point>,
        consistency: v1::Consistency,
    ) -> Result<u64, ClientError> {
        let resp = self
            .inner
            .upsert(v1::UpsertRequest {
                collection: collection.to_string(),
                points,
                consistency: consistency as i32,
            })
            .await?;
        Ok(resp.into_inner().upserted)
    }

    /// Delete points by id. Returns the number deleted.
    pub async fn delete(
        &mut self,
        collection: &str,
        ids: Vec<u64>,
        consistency: v1::Consistency,
    ) -> Result<u64, ClientError> {
        let resp = self
            .inner
            .delete(v1::DeleteRequest {
                collection: collection.to_string(),
                ids,
                consistency: consistency as i32,
            })
            .await?;
        Ok(resp.into_inner().deleted)
    }

    /// Get points by id (HLC-merged).
    pub async fn get(
        &mut self,
        collection: &str,
        ids: Vec<u64>,
        read_consistency: v1::Consistency,
    ) -> Result<Vec<v1::Point>, ClientError> {
        let resp = self
            .inner
            .get(v1::GetRequest {
                collection: collection.to_string(),
                ids,
                read_consistency: read_consistency as i32,
            })
            .await?;
        Ok(resp.into_inner().points)
    }

    /// k-NN search. `k`/`ef_search` of 0 take server defaults (10 / 64).
    pub async fn search(
        &mut self,
        collection: &str,
        vector: Vec<f32>,
        k: u32,
        ef_search: u32,
    ) -> Result<Vec<v1::ScoredPoint>, ClientError> {
        let resp = self
            .inner
            .search(v1::SearchRequest {
                collection: collection.to_string(),
                vector,
                k,
                ef_search,
            })
            .await?;
        Ok(resp.into_inner().results)
    }

    /// Cluster topology snapshot.
    pub async fn cluster_info(&mut self) -> Result<v1::ClusterInfoResponse, ClientError> {
        let resp = self.inner.cluster_info(v1::ClusterInfoRequest {}).await?;
        Ok(resp.into_inner())
    }

    /// Health probe.
    pub async fn health(&mut self) -> Result<String, ClientError> {
        let resp = self.inner.health(v1::HealthRequest {}).await?;
        Ok(resp.into_inner().status)
    }
}
