//! Data-plane routing (M3 single-replica; M4 extends this to N/W/R quorum).
//!
//! The coordinator (whichever node received the client request) resolves each
//! point to its shard via [`qv_cluster::shard_for_id`], looks up the shard's
//! replica nodes from the metadata shard map, and — for M3 — routes to the
//! **primary** (the first/home replica). If the primary is this node it applies
//! locally; otherwise it forwards over the internal `ReplicaWrite/ReplicaGet/
//! ReplicaSearch` RPCs. Search scatters to one replica per shard and merges a
//! global top-k.
//!
//! M3 deliberately uses single-replica routing (the plan: "single-replica
//! routing, no quorums yet"). The richer quorum/HLC-LWW/hinted-handoff/read-
//! repair path is M4; this module's shape (resolve shard -> pick replica(s) ->
//! local-or-forward) is what M4 generalizes.

use std::collections::HashMap;

use qv_cluster::{shard_for_id, CollectionSchema, MetaState};
use qv_proto::internal::{
    ReplicaGetRequest, ReplicaPoint, ReplicaScored, ReplicaSearchRequest, ReplicaWriteRequest,
};
use qv_proto::QuorvecInternalClient;
use qv_storage::Hlc;
use tonic::transport::Channel;

use crate::node_state::NodeState;
use crate::shards::{metric_from_cluster, ShardRef};

/// A routing/coordination error.
#[derive(Debug, thiserror::Error)]
pub enum RouterError {
    #[error("collection '{0}' not found")]
    NotFound(String),
    #[error("no replica available for {collection} shard {shard_idx}")]
    NoReplica { collection: String, shard_idx: u32 },
    #[error("dimension mismatch: collection is {expected}-d, got {got}-d")]
    DimMismatch { expected: usize, got: usize },
    #[error("shard store: {0}")]
    Shard(#[from] crate::shards::ShardStoreError),
    #[error("internal transport to {addr}: {source}")]
    Transport {
        addr: String,
        source: tonic::transport::Error,
    },
    #[error("internal rpc to {addr}: {source}")]
    Rpc { addr: String, source: tonic::Status },
}

/// WAL op codes shared with the internal proto.
const OP_UPSERT: u32 = 0;
const OP_DELETE: u32 = 1;

/// Resolve the metadata snapshot + schema for a collection, erroring if unknown.
async fn schema_of(
    state: &NodeState,
    collection: &str,
) -> Result<(MetaState, CollectionSchema), RouterError> {
    let meta = state.cluster.metadata().await;
    let schema = meta
        .collections
        .get(collection)
        .cloned()
        .ok_or_else(|| RouterError::NotFound(collection.to_string()))?;
    Ok((meta, schema))
}

/// The primary (home) replica node id for a shard, plus its advertise address.
fn primary_for(meta: &MetaState, collection: &str, shard_idx: u32) -> Option<(u64, String)> {
    let replicas = meta.replicas_for_shard(collection, shard_idx);
    let primary = *replicas.first()?;
    let addr = meta.nodes.get(&primary).cloned()?;
    Some((primary, addr))
}

/// Connect an internal client to a peer advertise address.
async fn internal_client(addr: &str) -> Result<QuorvecInternalClient<Channel>, RouterError> {
    let endpoint = if addr.starts_with("http://") || addr.starts_with("https://") {
        addr.to_string()
    } else {
        format!("http://{addr}")
    };
    QuorvecInternalClient::connect(endpoint)
        .await
        .map_err(|source| RouterError::Transport {
            addr: addr.to_string(),
            source,
        })
}

/// Coordinate an upsert of `(id, vector, payload)` points into a collection.
/// Returns the number accepted. M3: each point is stamped with a local HLC and
/// routed to its shard's primary replica (single replica).
pub async fn coordinate_upsert(
    state: &NodeState,
    collection: &str,
    points: &[(u64, Vec<f32>, Vec<u8>)],
) -> Result<u64, RouterError> {
    let (meta, schema) = schema_of(state, collection).await?;
    let dim = schema.dim as usize;
    let metric = metric_from_cluster(schema.metric);

    let mut accepted = 0u64;
    for (id, vector, payload) in points {
        if vector.len() != dim {
            return Err(RouterError::DimMismatch {
                expected: dim,
                got: vector.len(),
            });
        }
        let shard_idx = shard_for_id(*id, schema.shard_count);
        let hlc = state.clock.now();

        let (primary, addr) =
            primary_for(&meta, collection, shard_idx).ok_or_else(|| RouterError::NoReplica {
                collection: collection.to_string(),
                shard_idx,
            })?;

        if primary == state.node_id {
            let sref = ShardRef {
                collection,
                shard_idx,
                dim,
                metric,
            };
            state
                .shards
                .upsert(&sref, hlc, *id, vector, payload.clone())?;
        } else {
            let mut client = internal_client(&addr).await?;
            client
                .replica_write(ReplicaWriteRequest {
                    collection: collection.to_string(),
                    shard_idx,
                    op: OP_UPSERT,
                    id: *id,
                    hlc: hlc.pack(),
                    vector: vector.clone(),
                    payload: payload.clone(),
                })
                .await
                .map_err(|source| RouterError::Rpc {
                    addr: addr.clone(),
                    source,
                })?;
        }
        accepted += 1;
    }
    Ok(accepted)
}

/// Coordinate a delete (tombstone) of ids. Returns the count routed.
pub async fn coordinate_delete(
    state: &NodeState,
    collection: &str,
    ids: &[u64],
) -> Result<u64, RouterError> {
    let (meta, schema) = schema_of(state, collection).await?;
    let dim = schema.dim as usize;
    let metric = metric_from_cluster(schema.metric);

    let mut count = 0u64;
    for id in ids {
        let shard_idx = shard_for_id(*id, schema.shard_count);
        let hlc = state.clock.now();
        let (primary, addr) =
            primary_for(&meta, collection, shard_idx).ok_or_else(|| RouterError::NoReplica {
                collection: collection.to_string(),
                shard_idx,
            })?;
        if primary == state.node_id {
            let sref = ShardRef {
                collection,
                shard_idx,
                dim,
                metric,
            };
            state.shards.delete(&sref, hlc, *id)?;
        } else {
            let mut client = internal_client(&addr).await?;
            client
                .replica_write(ReplicaWriteRequest {
                    collection: collection.to_string(),
                    shard_idx,
                    op: OP_DELETE,
                    id: *id,
                    hlc: hlc.pack(),
                    vector: Vec::new(),
                    payload: Vec::new(),
                })
                .await
                .map_err(|source| RouterError::Rpc {
                    addr: addr.clone(),
                    source,
                })?;
        }
        count += 1;
    }
    Ok(count)
}

/// Coordinate a Get of ids: route each to its shard's primary, collect live
/// points. M3 reads from the single primary replica (M4 adds R-quorum + merge).
pub async fn coordinate_get(
    state: &NodeState,
    collection: &str,
    ids: &[u64],
) -> Result<Vec<(u64, Vec<f32>)>, RouterError> {
    let (meta, schema) = schema_of(state, collection).await?;
    let dim = schema.dim as usize;
    let metric = metric_from_cluster(schema.metric);

    // Group ids by their shard's primary so we batch remote Gets per node.
    let mut local_ids: Vec<u64> = Vec::new();
    let mut local_shards: Vec<u32> = Vec::new();
    // (addr, shard_idx) -> ids
    let mut remote: HashMap<(String, u32), Vec<u64>> = HashMap::new();

    for id in ids {
        let shard_idx = shard_for_id(*id, schema.shard_count);
        let (primary, addr) =
            primary_for(&meta, collection, shard_idx).ok_or_else(|| RouterError::NoReplica {
                collection: collection.to_string(),
                shard_idx,
            })?;
        if primary == state.node_id {
            local_ids.push(*id);
            local_shards.push(shard_idx);
        } else {
            remote.entry((addr, shard_idx)).or_default().push(*id);
        }
    }

    let mut out: Vec<(u64, Vec<f32>)> = Vec::new();

    // Local reads.
    for (id, shard_idx) in local_ids.iter().zip(local_shards.iter()) {
        let sref = ShardRef {
            collection,
            shard_idx: *shard_idx,
            dim,
            metric,
        };
        if let Some(v) = state.shards.get(&sref, *id)? {
            out.push((*id, v));
        }
    }

    // Remote reads.
    for ((addr, shard_idx), batch) in remote {
        let mut client = internal_client(&addr).await?;
        let resp = client
            .replica_get(ReplicaGetRequest {
                collection: collection.to_string(),
                shard_idx,
                ids: batch,
            })
            .await
            .map_err(|source| RouterError::Rpc {
                addr: addr.clone(),
                source,
            })?
            .into_inner();
        for p in resp.points {
            if !p.tombstone {
                out.push((p.id, p.vector));
            }
        }
    }

    Ok(out)
}

/// Coordinate a Search: scatter to one replica per shard, gather, merge global
/// top-k by ascending score (smaller = closer).
pub async fn coordinate_search(
    state: &NodeState,
    collection: &str,
    query: &[f32],
    k: usize,
    ef_search: usize,
) -> Result<Vec<(u64, f32)>, RouterError> {
    let (meta, schema) = schema_of(state, collection).await?;
    let dim = schema.dim as usize;
    let metric = metric_from_cluster(schema.metric);
    if query.len() != dim {
        return Err(RouterError::DimMismatch {
            expected: dim,
            got: query.len(),
        });
    }

    let mut merged: Vec<(u64, f32)> = Vec::new();

    for shard_idx in 0..schema.shard_count {
        // M3: one replica per shard = the primary. (M4: one *healthy* replica.)
        let (primary, addr) =
            primary_for(&meta, collection, shard_idx).ok_or_else(|| RouterError::NoReplica {
                collection: collection.to_string(),
                shard_idx,
            })?;

        let hits = if primary == state.node_id {
            let sref = ShardRef {
                collection,
                shard_idx,
                dim,
                metric,
            };
            state.shards.search_shard(&sref, query, k, ef_search)?
        } else {
            let mut client = internal_client(&addr).await?;
            let resp = client
                .replica_search(ReplicaSearchRequest {
                    collection: collection.to_string(),
                    shard_idx,
                    vector: query.to_vec(),
                    k: k as u32,
                    ef_search: ef_search as u32,
                })
                .await
                .map_err(|source| RouterError::Rpc {
                    addr: addr.clone(),
                    source,
                })?
                .into_inner();
            resp.results.into_iter().map(|r| (r.id, r.score)).collect()
        };
        merged.extend(hits);
    }

    // Global top-k: ascending score, id tiebreak (matches single-node ordering).
    merged.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    merged.truncate(k);
    Ok(merged)
}

// ---- Internal-service helpers (the receiving/replica side) -----------------

/// Apply a replica write locally (called by the internal service handler when a
/// peer forwards a point op to this node).
pub fn apply_replica_write(
    state: &NodeState,
    req: &ReplicaWriteRequest,
    schema: &CollectionSchema,
) -> Result<bool, RouterError> {
    let hlc = Hlc::unpack(req.hlc);
    // Advance this node's clock to keep the HLC monotone across nodes (M4 relies
    // on this; harmless in M3).
    let _ = state.clock.update(hlc);
    let sref = ShardRef {
        collection: &req.collection,
        shard_idx: req.shard_idx,
        dim: schema.dim as usize,
        metric: metric_from_cluster(schema.metric),
    };

    match req.op {
        OP_DELETE => {
            let removed = state.shards.delete(&sref, hlc, req.id)?;
            Ok(removed)
        }
        _ => {
            state
                .shards
                .upsert(&sref, hlc, req.id, &req.vector, req.payload.clone())?;
            Ok(true)
        }
    }
}

/// Read points locally for a replica Get.
pub fn apply_replica_get(
    state: &NodeState,
    collection: &str,
    shard_idx: u32,
    ids: &[u64],
    schema: &CollectionSchema,
) -> Result<Vec<ReplicaPoint>, RouterError> {
    let sref = ShardRef {
        collection,
        shard_idx,
        dim: schema.dim as usize,
        metric: metric_from_cluster(schema.metric),
    };
    let mut out = Vec::new();
    for id in ids {
        if let Some(v) = state.shards.get(&sref, *id)? {
            out.push(ReplicaPoint {
                id: *id,
                hlc: 0, // M4 surfaces the stored HLC; M3 Get does not need it
                tombstone: false,
                vector: v,
                payload: Vec::new(),
            });
        }
    }
    Ok(out)
}

/// Search a single shard locally for a replica Search.
pub fn apply_replica_search(
    state: &NodeState,
    collection: &str,
    shard_idx: u32,
    query: &[f32],
    k: usize,
    ef_search: usize,
    schema: &CollectionSchema,
) -> Result<Vec<ReplicaScored>, RouterError> {
    let sref = ShardRef {
        collection,
        shard_idx,
        dim: schema.dim as usize,
        metric: metric_from_cluster(schema.metric),
    };
    let hits = state.shards.search_shard(&sref, query, k, ef_search)?;
    Ok(hits
        .into_iter()
        .map(|(id, score)| ReplicaScored {
            id,
            score,
            payload: Vec::new(),
        })
        .collect())
}
