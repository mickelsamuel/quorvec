//! Data-plane routing and coordination (M4: tunable quorum replication).
//!
//! The coordinator (whichever node received the client request) stamps each
//! write with an HLC, resolves each point to its shard, and replicates to the
//! shard's **N replicas**, acking the client at **W** (ONE / QUORUM / ALL). The
//! locked Dynamo-style semantics:
//!
//! - **N/W/R quorum.** N = the shard's replica count; W/R from the consistency
//!   level (ONE=1, QUORUM=⌊N/2⌋+1, ALL=N).
//! - **HLC LWW.** Per-point version = the coordinator's HLC; replicas reject a
//!   write whose HLC is not newer than what they hold ([`qv_storage`] enforces
//!   this; the coordinator counts an applied-or-superseded replica as a success
//!   because the data has converged either way).
//! - **Hinted handoff.** A replica that is unreachable for a write does not block
//!   the quorum: the coordinator records a durable hint (the op + the target
//!   replica) on a stand-in node (the next distinct ring node), which replays it
//!   when the target returns. Hints are WAL-persisted ([`qv_storage::HintLog`]).
//! - **Get** reads R replicas, LWW-merges by HLC, returns the newest, and async
//!   **read-repairs** any replica that returned a stale (or missing) version.
//! - **Search** scatters to one *healthy* replica per shard (falling forward
//!   through the replica list) and merges a global top-k.
//! - **Delete** = an HLC-stamped tombstone, replicated exactly like an upsert.

use qv_cluster::{shard_for_id, CollectionSchema, MetaState, ShardState};
use qv_proto::internal::{
    ReplicaGetRequest, ReplicaPoint, ReplicaScored, ReplicaSearchRequest, ReplicaWriteRequest,
    ShardRecord, StreamShardRequest,
};
use qv_proto::QuorvecInternalClient;
use qv_storage::{Hint, Hlc, WalOp, WriteOutcome};
use tonic::transport::Channel;

use crate::node_state::NodeState;
use crate::shards::{metric_from_cluster, ShardRef, StoredPoint};

/// A routing/coordination error.
#[derive(Debug, thiserror::Error)]
pub enum RouterError {
    #[error("collection '{0}' not found")]
    NotFound(String),
    #[error("no replica available for {collection} shard {shard_idx}")]
    NoReplica { collection: String, shard_idx: u32 },
    #[error("dimension mismatch: collection is {expected}-d, got {got}-d")]
    DimMismatch { expected: usize, got: usize },
    #[error("write quorum not met for {collection} shard {shard_idx}: needed {needed}, got {got}")]
    WriteQuorum {
        collection: String,
        shard_idx: u32,
        needed: usize,
        got: usize,
    },
    #[error("read quorum not met for {collection} shard {shard_idx}: needed {needed}, got {got}")]
    ReadQuorum {
        collection: String,
        shard_idx: u32,
        needed: usize,
        got: usize,
    },
    #[error("shard store: {0}")]
    Shard(#[from] crate::shards::ShardStoreError),
    #[error("hint store: {0}")]
    Hint(#[from] qv_storage::HintError),
}

const OP_UPSERT: u32 = 0;
const OP_DELETE: u32 = 1;

/// The tunable consistency level (mirrors the proto enum), resolved to a replica
/// count given N.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Consistency {
    One,
    Quorum,
    All,
}

impl Consistency {
    /// Map the proto enum value to a level (unspecified -> QUORUM, the safe
    /// default per the locked proto comment).
    pub fn from_proto(v: i32) -> Self {
        match v {
            1 => Consistency::One,    // CONSISTENCY_ONE
            3 => Consistency::All,    // CONSISTENCY_ALL
            _ => Consistency::Quorum, // QUORUM or UNSPECIFIED
        }
    }

    /// Required ack/read count for a replica set of size `n`.
    pub fn count(self, n: usize) -> usize {
        match self {
            Consistency::One => 1,
            Consistency::Quorum => n / 2 + 1,
            Consistency::All => n,
        }
        .clamp(1, n.max(1))
    }
}

// ---- shared helpers --------------------------------------------------------

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

/// The replica node ids for a shard with their advertise addresses, clockwise
/// from the home (first = primary). Replicas whose address is unknown are
/// dropped (they cannot be routed to).
///
/// This is the **write** view: it includes replicas in every state, because a
/// `Syncing` replica (one receiving an M5 shard transfer) must still receive
/// concurrent live writes so the transfer cannot lose a write that lands
/// mid-flight (LWW reconciles the streamed-older vs live-newer records).
fn replicas_with_addrs(meta: &MetaState, collection: &str, shard_idx: u32) -> Vec<(u64, String)> {
    meta.replicas_for_shard(collection, shard_idx)
        .into_iter()
        .filter_map(|id| meta.nodes.get(&id).map(|a| (id, a.clone())))
        .collect()
}

/// The **read/search** view of a shard's replicas: only those in `Active` state.
/// A `Syncing` replica is excluded because it may not yet hold the shard's data
/// (it is mid-transfer), so routing a read there could miss points. During a
/// transfer the prior holders stay `Active`, so an Active replica with the data
/// always remains — reads route around the syncing one and never fail or go
/// stale because of the rebalance.
fn active_replicas_with_addrs(
    meta: &MetaState,
    collection: &str,
    shard_idx: u32,
) -> Vec<(u64, String)> {
    meta.replicas_for_shard(collection, shard_idx)
        .into_iter()
        .filter(|id| meta.shard_state(collection, shard_idx, *id) == ShardState::Active)
        .filter_map(|id| meta.nodes.get(&id).map(|a| (id, a.clone())))
        .collect()
}

async fn internal_client(addr: &str) -> Result<QuorvecInternalClient<Channel>, ConnError> {
    let endpoint = if addr.starts_with("http://") || addr.starts_with("https://") {
        addr.to_string()
    } else {
        format!("http://{addr}")
    };
    QuorvecInternalClient::connect(endpoint)
        .await
        .map_err(|_| ConnError)
}

/// A marker for "could not reach the peer" — drives hinted handoff and
/// healthy-replica fallback. Distinct from a logical RouterError so the
/// coordinator can route around it instead of failing.
struct ConnError;

/// Apply one write to a replica: locally if it's us, else over the internal RPC.
/// Returns Ok(true) when the data converged at that replica (applied OR a newer
/// version already present), Ok(false)/Err only on unreachable/transport errors.
#[allow(clippy::too_many_arguments)]
async fn write_one_replica(
    state: &NodeState,
    node_id: u64,
    addr: &str,
    collection: &str,
    shard_idx: u32,
    op: u32,
    id: u64,
    hlc: Hlc,
    vector: &[f32],
    payload: &[u8],
    schema: &CollectionSchema,
) -> Result<bool, ConnError> {
    if node_id == state.node_id {
        let sref = ShardRef {
            collection,
            shard_idx,
            dim: schema.dim as usize,
            metric: metric_from_cluster(schema.metric),
        };
        let res = if op == OP_DELETE {
            state.shards.delete(&sref, hlc, id)
        } else {
            state
                .shards
                .upsert(&sref, hlc, id, vector, payload.to_vec())
        };
        // Applied or Stale both mean "converged at this replica".
        match res {
            Ok(_) => Ok(true),
            Err(_) => Ok(false), // local error is not "unreachable"; count as not-acked
        }
    } else {
        let mut client = internal_client(addr).await?;
        client
            .replica_write(ReplicaWriteRequest {
                collection: collection.to_string(),
                shard_idx,
                op,
                id,
                hlc: hlc.pack(),
                vector: vector.to_vec(),
                payload: payload.to_vec(),
            })
            .await
            .map_err(|_| ConnError)?;
        Ok(true)
    }
}

// ---- writes (Upsert / Delete) ---------------------------------------------

/// Coordinate an upsert at the given consistency. Each point is HLC-stamped and
/// replicated to its shard's N replicas; the client is acked once W replicas
/// converge. Unreachable replicas get a hinted handoff (so the quorum is not
/// blocked and the write is not lost).
pub async fn coordinate_upsert(
    state: &NodeState,
    collection: &str,
    points: &[(u64, Vec<f32>, Vec<u8>)],
    consistency: Consistency,
) -> Result<u64, RouterError> {
    let (meta, schema) = schema_of(state, collection).await?;
    let dim = schema.dim as usize;
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
        replicate_write(
            state,
            &meta,
            &schema,
            collection,
            shard_idx,
            OP_UPSERT,
            *id,
            hlc,
            vector,
            payload,
            consistency,
        )
        .await?;
        accepted += 1;
    }
    Ok(accepted)
}

/// Coordinate a delete (HLC-stamped tombstone) at the given consistency.
pub async fn coordinate_delete(
    state: &NodeState,
    collection: &str,
    ids: &[u64],
    consistency: Consistency,
) -> Result<u64, RouterError> {
    let (meta, schema) = schema_of(state, collection).await?;
    let mut count = 0u64;
    let empty: Vec<f32> = Vec::new();
    for id in ids {
        let shard_idx = shard_for_id(*id, schema.shard_count);
        let hlc = state.clock.now();
        replicate_write(
            state,
            &meta,
            &schema,
            collection,
            shard_idx,
            OP_DELETE,
            *id,
            hlc,
            &empty,
            &[],
            consistency,
        )
        .await?;
        count += 1;
    }
    Ok(count)
}

/// Replicate one write to all N replicas, ack at W, hint the unreachable ones.
#[allow(clippy::too_many_arguments)]
async fn replicate_write(
    state: &NodeState,
    meta: &MetaState,
    schema: &CollectionSchema,
    collection: &str,
    shard_idx: u32,
    op: u32,
    id: u64,
    hlc: Hlc,
    vector: &[f32],
    payload: &[u8],
    consistency: Consistency,
) -> Result<(), RouterError> {
    let replicas = replicas_with_addrs(meta, collection, shard_idx);
    if replicas.is_empty() {
        return Err(RouterError::NoReplica {
            collection: collection.to_string(),
            shard_idx,
        });
    }
    let n = replicas.len();
    let w = consistency.count(n);

    let mut acks = 0usize;
    let mut unreachable: Vec<u64> = Vec::new();

    for (node_id, addr) in &replicas {
        match write_one_replica(
            state, *node_id, addr, collection, shard_idx, op, id, hlc, vector, payload, schema,
        )
        .await
        {
            Ok(true) => acks += 1,
            Ok(false) => unreachable.push(*node_id),
            Err(ConnError) => unreachable.push(*node_id),
        }
    }

    // Hinted handoff for every replica we couldn't reach: buffer the write on a
    // stand-in (the next distinct ring node not in the replica set, else this
    // coordinator) so it can be replayed when the target returns.
    if !unreachable.is_empty() {
        let wal_op = if op == OP_DELETE {
            WalOp::Delete
        } else {
            WalOp::Upsert
        };
        for target in &unreachable {
            let hint = Hint {
                target_node: *target,
                collection: collection.to_string(),
                shard_idx,
                op: wal_op,
                hlc,
                id,
                vector: vector.to_vec(),
                payload: payload.to_vec(),
            };
            // The coordinator is the hint holder: it is up (it is running this
            // code) and is the natural stand-in. It replays to the target when
            // the target returns (a "sloppy quorum" — the write is durably
            // buffered, never lost, and the live quorum is not blocked). Placing
            // the hint on a different stand-in ring node is a labeled refinement.
            state.buffer_hint(&hint)?;
        }
    }

    if acks >= w {
        Ok(())
    } else {
        Err(RouterError::WriteQuorum {
            collection: collection.to_string(),
            shard_idx,
            needed: w,
            got: acks,
        })
    }
}

// ---- Get (R-read + LWW merge + async read repair) --------------------------

/// Coordinate a Get at the given read consistency: read R replicas per id,
/// LWW-merge by HLC, return the live winners, and async-repair stale replicas.
pub async fn coordinate_get(
    state: &NodeState,
    collection: &str,
    ids: &[u64],
    consistency: Consistency,
) -> Result<Vec<(u64, Vec<f32>)>, RouterError> {
    let (meta, schema) = schema_of(state, collection).await?;
    let mut out: Vec<(u64, Vec<f32>)> = Vec::new();

    for id in ids {
        let shard_idx = shard_for_id(*id, schema.shard_count);
        // Reads use only Active replicas: a Syncing replica may be mid-transfer
        // and not yet hold the data.
        let replicas = active_replicas_with_addrs(&meta, collection, shard_idx);
        if replicas.is_empty() {
            return Err(RouterError::NoReplica {
                collection: collection.to_string(),
                shard_idx,
            });
        }
        let n = replicas.len();
        let r = consistency.count(n);

        // Read every reachable replica; collect (node, addr, version).
        let mut responses: Vec<(u64, String, Option<StoredPoint>)> = Vec::new();
        for (node_id, addr) in &replicas {
            if let Ok(opt) =
                read_one_replica(state, *node_id, addr, collection, shard_idx, *id, &schema).await
            {
                responses.push((*node_id, addr.clone(), opt));
            }
        }
        if responses.len() < r {
            return Err(RouterError::ReadQuorum {
                collection: collection.to_string(),
                shard_idx,
                needed: r,
                got: responses.len(),
            });
        }

        // LWW-merge: highest HLC wins (tombstone or value).
        let winner = responses
            .iter()
            .filter_map(|(_, _, p)| p.clone())
            .max_by_key(|p| p.hlc);

        if let Some(win) = winner {
            // Async read repair: push the winner to any replica that returned a
            // strictly-older version (or nothing). Fire-and-forget.
            spawn_read_repair(state, collection, shard_idx, &schema, &responses, &win);

            if !win.tombstone {
                out.push((win.id, win.vector));
            }
        }
    }
    Ok(out)
}

/// Read one replica's versioned point for `id`.
#[allow(clippy::too_many_arguments)]
async fn read_one_replica(
    state: &NodeState,
    node_id: u64,
    addr: &str,
    collection: &str,
    shard_idx: u32,
    id: u64,
    schema: &CollectionSchema,
) -> Result<Option<StoredPoint>, ConnError> {
    if node_id == state.node_id {
        let sref = ShardRef {
            collection,
            shard_idx,
            dim: schema.dim as usize,
            metric: metric_from_cluster(schema.metric),
        };
        state.shards.get_versioned(&sref, id).map_err(|_| ConnError)
    } else {
        let mut client = internal_client(addr).await?;
        let resp = client
            .replica_get(ReplicaGetRequest {
                collection: collection.to_string(),
                shard_idx,
                ids: vec![id],
            })
            .await
            .map_err(|_| ConnError)?
            .into_inner();
        Ok(resp.points.into_iter().next().map(|p| StoredPoint {
            id: p.id,
            hlc: Hlc::unpack(p.hlc),
            tombstone: p.tombstone,
            vector: p.vector,
            payload: p.payload,
        }))
    }
}

/// Spawn async read repair: push `winner` to replicas that are behind it.
fn spawn_read_repair(
    state: &NodeState,
    collection: &str,
    shard_idx: u32,
    schema: &CollectionSchema,
    responses: &[(u64, String, Option<StoredPoint>)],
    winner: &StoredPoint,
) {
    // Determine the stale replicas (older HLC or missing) synchronously, then
    // move the work to a task so the read returns promptly.
    let mut stale: Vec<(u64, String)> = Vec::new();
    for (node_id, addr, p) in responses {
        let behind = match p {
            Some(sp) => sp.hlc < winner.hlc,
            None => true,
        };
        if behind {
            stale.push((*node_id, addr.clone()));
        }
    }
    if stale.is_empty() {
        return;
    }

    // We cannot move &NodeState into a task; the data-plane repair only needs the
    // node id (to know "is it me"), the shard store handle, and the internal
    // client. Since NodeState is held in an Arc by callers, repair runs inline
    // for the local replica and spawns remote pushes. To keep this synchronous-
    // safe without threading an Arc here, we apply local repair immediately and
    // spawn the remote pushes with owned data.
    let op = if winner.tombstone {
        OP_DELETE
    } else {
        OP_UPSERT
    };
    let self_id = state.node_id;
    let collection = collection.to_string();
    let dim = schema.dim as usize;
    let metric = metric_from_cluster(schema.metric);
    let winner = winner.clone();

    // Local repair (synchronous, cheap).
    for (node_id, _addr) in stale.iter().filter(|(nid, _)| *nid == self_id) {
        let _ = *node_id;
        let sref = ShardRef {
            collection: &collection,
            shard_idx,
            dim,
            metric,
        };
        let _ = if winner.tombstone {
            state.shards.delete(&sref, winner.hlc, winner.id)
        } else {
            state.shards.upsert(
                &sref,
                winner.hlc,
                winner.id,
                &winner.vector,
                winner.payload.clone(),
            )
        };
    }

    // Remote repair (spawned, owned).
    let remote: Vec<(u64, String)> = stale
        .into_iter()
        .filter(|(nid, _)| *nid != self_id)
        .collect();
    if remote.is_empty() {
        return;
    }
    tokio::spawn(async move {
        for (_node_id, addr) in remote {
            if let Ok(mut client) = internal_client(&addr).await {
                let _ = client
                    .replica_write(ReplicaWriteRequest {
                        collection: collection.clone(),
                        shard_idx,
                        op,
                        id: winner.id,
                        hlc: winner.hlc.pack(),
                        vector: winner.vector.clone(),
                        payload: winner.payload.clone(),
                    })
                    .await;
            }
        }
    });
}

// ---- Search (scatter to one healthy replica per shard) ---------------------

/// Coordinate a Search: for each shard, query one *healthy* replica (the first
/// reachable one, primary-first), then merge a global top-k by ascending score.
pub async fn coordinate_search(
    state: &NodeState,
    collection: &str,
    query: &[f32],
    k: usize,
    ef_search: usize,
) -> Result<Vec<(u64, f32)>, RouterError> {
    let (meta, schema) = schema_of(state, collection).await?;
    let dim = schema.dim as usize;
    if query.len() != dim {
        return Err(RouterError::DimMismatch {
            expected: dim,
            got: query.len(),
        });
    }

    let mut merged: Vec<(u64, f32)> = Vec::new();
    for shard_idx in 0..schema.shard_count {
        // Search uses only Active replicas (a Syncing one may lack the data).
        let replicas = active_replicas_with_addrs(&meta, collection, shard_idx);
        if replicas.is_empty() {
            return Err(RouterError::NoReplica {
                collection: collection.to_string(),
                shard_idx,
            });
        }
        // Try replicas in order until one answers (one healthy replica per shard).
        let mut got = false;
        for (node_id, addr) in &replicas {
            match search_one_replica(
                state, *node_id, addr, collection, shard_idx, query, k, ef_search, &schema,
            )
            .await
            {
                Ok(hits) => {
                    merged.extend(hits);
                    got = true;
                    break;
                }
                Err(ConnError) => continue,
            }
        }
        if !got {
            return Err(RouterError::NoReplica {
                collection: collection.to_string(),
                shard_idx,
            });
        }
    }

    merged.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    merged.truncate(k);
    Ok(merged)
}

#[allow(clippy::too_many_arguments)]
async fn search_one_replica(
    state: &NodeState,
    node_id: u64,
    addr: &str,
    collection: &str,
    shard_idx: u32,
    query: &[f32],
    k: usize,
    ef_search: usize,
    schema: &CollectionSchema,
) -> Result<Vec<(u64, f32)>, ConnError> {
    if node_id == state.node_id {
        let sref = ShardRef {
            collection,
            shard_idx,
            dim: schema.dim as usize,
            metric: metric_from_cluster(schema.metric),
        };
        state
            .shards
            .search_shard(&sref, query, k, ef_search)
            .map_err(|_| ConnError)
    } else {
        let mut client = internal_client(addr).await?;
        let resp = client
            .replica_search(ReplicaSearchRequest {
                collection: collection.to_string(),
                shard_idx,
                vector: query.to_vec(),
                k: k as u32,
                ef_search: ef_search as u32,
            })
            .await
            .map_err(|_| ConnError)?
            .into_inner();
        Ok(resp.results.into_iter().map(|r| (r.id, r.score)).collect())
    }
}

// ---- Internal-service helpers (the replica/receiving side) -----------------

/// Apply a replica write locally (peer forwarded a point op to this node).
/// Returns whether the write won LWW at this replica.
pub fn apply_replica_write(
    state: &NodeState,
    req: &ReplicaWriteRequest,
    schema: &CollectionSchema,
) -> Result<bool, RouterError> {
    let hlc = Hlc::unpack(req.hlc);
    // Advance the node's clock on receiving a remote stamp (HLC receive rule).
    let _ = state.clock.update(hlc);
    let sref = ShardRef {
        collection: &req.collection,
        shard_idx: req.shard_idx,
        dim: schema.dim as usize,
        metric: metric_from_cluster(schema.metric),
    };
    let outcome = if req.op == OP_DELETE {
        state.shards.delete(&sref, hlc, req.id)?
    } else {
        state
            .shards
            .upsert(&sref, hlc, req.id, &req.vector, req.payload.clone())?
    };
    Ok(outcome == WriteOutcome::Applied)
}

/// Read points locally for a replica Get (returns HLC + tombstone per id).
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
        if let Some(sp) = state.shards.get_versioned(&sref, *id)? {
            out.push(ReplicaPoint {
                id: sp.id,
                hlc: sp.hlc.pack(),
                tombstone: sp.tombstone,
                vector: sp.vector,
                payload: sp.payload,
            });
        }
    }
    Ok(out)
}

/// Search a single shard locally for a replica Search.
#[allow(clippy::too_many_arguments)]
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

// ---- Hinted-handoff replay -------------------------------------------------

/// Replay buffered hints to replicas that are now reachable. Called periodically
/// (a background loop) and after membership changes. For each hint whose target
/// is reachable, push the write; drop replayed hints, keep the rest.
pub async fn replay_hints(state: &NodeState) -> Result<usize, RouterError> {
    let hints = state.read_all_hints()?;
    if hints.is_empty() {
        return Ok(0);
    }
    let meta = state.cluster.metadata().await;

    let mut keep: Vec<Hint> = Vec::new();
    let mut replayed = 0usize;

    for hint in hints {
        let addr = meta.nodes.get(&hint.target_node).cloned();
        let Some(addr) = addr else {
            keep.push(hint); // target unknown right now; keep buffering
            continue;
        };
        let op = match hint.op {
            WalOp::Delete => OP_DELETE,
            WalOp::Upsert => OP_UPSERT,
        };
        let pushed = match internal_client(&addr).await {
            Ok(mut client) => client
                .replica_write(ReplicaWriteRequest {
                    collection: hint.collection.clone(),
                    shard_idx: hint.shard_idx,
                    op,
                    id: hint.id,
                    hlc: hint.hlc.pack(),
                    vector: hint.vector.clone(),
                    payload: hint.payload.clone(),
                })
                .await
                .is_ok(),
            Err(ConnError) => false,
        };
        if pushed {
            replayed += 1;
        } else {
            keep.push(hint);
        }
    }

    state.rewrite_hints(&keep)?;
    Ok(replayed)
}

// ---- Shard transfer (M5: stream_records) -----------------------------------

/// Server side of `StreamShard`: enumerate every record (live + tombstone) of a
/// shard this node holds, so a pulling target can rebuild it. Returns an empty
/// list if this node does not materialize the shard (the caller picks another
/// source).
pub fn apply_stream_shard(
    state: &NodeState,
    collection: &str,
    shard_idx: u32,
    schema: &CollectionSchema,
) -> Result<Vec<ShardRecord>, RouterError> {
    let sref = ShardRef {
        collection,
        shard_idx,
        dim: schema.dim as usize,
        metric: metric_from_cluster(schema.metric),
    };
    // Only stream from a shard we actually hold (don't lazily create an empty one
    // just to serve a transfer — that would hand the target nothing).
    if !state.shards.has_shard(collection, shard_idx) {
        return Ok(Vec::new());
    }
    let records = state.shards.records(&sref)?;
    Ok(records
        .into_iter()
        .map(|(id, hlc, tombstone, vector)| ShardRecord {
            id,
            hlc: hlc.pack(),
            tombstone,
            vector,
            payload: Vec::new(),
        })
        .collect())
}

/// Target side of `stream_records`: pull a shard's full record set from `source`
/// and apply it to this node's durable shard under LWW.
///
/// LWW is what makes a transfer concurrent-write-safe: a streamed record only
/// wins on the target if its HLC is newer than whatever the target already holds
/// for that id. So a live write that landed on the target mid-transfer (because a
/// Syncing replica still receives writes) is never clobbered by an older streamed
/// copy, and a streamed tombstone correctly suppresses a stale resurrection.
///
/// Returns the number of records the target accepted (won LWW). A transfer from a
/// source that turns out not to hold the shard yields 0 and the caller tries the
/// next source.
pub async fn pull_shard(
    state: &NodeState,
    source_addr: &str,
    collection: &str,
    shard_idx: u32,
    schema: &CollectionSchema,
) -> Result<usize, RouterError> {
    let mut client = internal_client(source_addr)
        .await
        .map_err(|_| RouterError::NoReplica {
            collection: collection.to_string(),
            shard_idx,
        })?;
    let resp = client
        .stream_shard(StreamShardRequest {
            collection: collection.to_string(),
            shard_idx,
        })
        .await
        .map_err(|_| RouterError::NoReplica {
            collection: collection.to_string(),
            shard_idx,
        })?
        .into_inner();

    let sref = ShardRef {
        collection,
        shard_idx,
        dim: schema.dim as usize,
        metric: metric_from_cluster(schema.metric),
    };
    let mut applied = 0usize;
    for rec in resp.records {
        let hlc = Hlc::unpack(rec.hlc);
        // Keep our clock ahead of any HLC we ingest (HLC receive rule).
        let _ = state.clock.update(hlc);
        let outcome = if rec.tombstone {
            state.shards.delete(&sref, hlc, rec.id)?
        } else {
            state
                .shards
                .upsert(&sref, hlc, rec.id, &rec.vector, rec.payload)?
        };
        if outcome == WriteOutcome::Applied {
            applied += 1;
        }
    }
    Ok(applied)
}
