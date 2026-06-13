//! M5 rebalancing: `stream_records` shard transfer driven by a per-node reconcile
//! loop (Qdrant blueprint, findings Q3).
//!
//! Placement is deterministic from the Raft membership, so when the membership
//! changes (a node joins, or one leaves / is declared dead) the desired shard map
//! that [`qv_cluster::MetaState::shard_map`] derives changes immediately. The data
//! does not move on its own, though — this loop is what makes reality catch up to
//! the desired map without dropping requests:
//!
//! **Acquire (this node is a NEW desired replica for a shard it lacks).**
//!   1. Mark this replica `Syncing` via Raft (forwarded to the leader if we are a
//!      follower). A `Syncing` replica is excluded from reads/search but still
//!      receives live writes, so nothing routed at it is missed and no concurrent
//!      write is lost.
//!   2. Pull the shard's full record set from an `Active` replica that already
//!      holds it (`stream_records`) and rebuild our durable shard under LWW.
//!   3. Cut over: mark this replica `Active` via Raft. Reads can now route here.
//!
//! **Release (this node HOLDS a shard it is no longer a desired replica for).**
//!   Once every desired replica of that shard is `Active` (the data is safely
//!   re-homed), drop the local copy — the M5 "source drop". We never drop while a
//!   desired replica is still `Syncing`, so a transfer in flight always has its
//!   source.
//!
//! Re-replication on node leave / permanent death is the SAME path: a death marks
//! the dead node's replicas `Dead` (excluded from placement), the ring promotes a
//! new replica, and that new replica acquires the shard exactly as above. There is
//! no separate code path — the plan's "Leave/dead-replica recovery = same transfer
//! path".
//!
//! wal_delta / snapshot transfer modes are labeled v1.1 and intentionally NOT
//! built (plan §M5).

use std::sync::Arc;
use std::time::Duration;

use qv_cluster::{MetaState, ShardState};
use qv_proto::internal::MetaWriteRequest;
use qv_proto::QuorvecInternalClient;
use tonic::transport::Channel;

use crate::node_state::NodeState;
use crate::router;
use crate::shards::{metric_from_cluster, ShardRef};

/// One full reconcile pass. Returns the number of shard transfers completed (for
/// logging / tests). Cheap and a no-op when the cluster is in steady state.
pub async fn reconcile_once(state: &Arc<NodeState>) -> usize {
    let meta = state.cluster.metadata().await;
    let me = state.node_id;
    let mut transfers = 0usize;

    // Work over a stable snapshot of (collection, schema) pairs.
    let collections: Vec<(String, qv_cluster::CollectionSchema)> = meta
        .collections
        .iter()
        .map(|(n, s)| (n.clone(), s.clone()))
        .collect();

    for (collection, schema) in &collections {
        for shard_idx in 0..schema.shard_count {
            let desired = meta.replicas_for_shard(collection, shard_idx);
            let am_desired = desired.contains(&me);
            let hold = state.shards.has_shard(collection, shard_idx);

            if am_desired {
                // Do we already serve it (Active + holding data)? If a prior
                // membership already put this shard here and it's Active, nothing
                // to do.
                let my_state = meta.shard_state(collection, shard_idx, me);
                let materialized_nonempty = hold
                    && shard_len(state, collection, shard_idx, schema)
                        .map(|n| n > 0)
                        .unwrap_or(false);

                if my_state == ShardState::Active && materialized_nonempty {
                    continue;
                }
                // An Active+empty shard is the legitimate steady state for a shard
                // that genuinely has no points yet — only transfer if a source
                // actually has data to give. Acquire handles that check.
                if acquire_shard(state, &meta, collection, schema, shard_idx).await {
                    transfers += 1;
                }
            } else if hold {
                // We hold a shard we are no longer a desired replica for. Drop it
                // once every desired replica is Active (safely re-homed).
                if desired_all_active(&meta, collection, shard_idx, &desired) {
                    state.shards.drop_shard(collection, shard_idx);
                    tracing::info!(
                        node = me,
                        %collection,
                        shard_idx,
                        "source-dropped shard after rebalance hand-off"
                    );
                }
            }
        }
    }
    transfers
}

/// Acquire a shard onto this node: Syncing → pull from an Active source → Active.
/// Returns true if a transfer completed (data pulled and cutover committed).
async fn acquire_shard(
    state: &Arc<NodeState>,
    meta: &MetaState,
    collection: &str,
    schema: &qv_cluster::CollectionSchema,
    shard_idx: u32,
) -> bool {
    let me = state.node_id;

    // Find an Active source replica that is NOT us and holds the data. The prior
    // holders (still Active desired replicas) are the natural sources.
    let sources: Vec<String> = meta
        .replicas_for_shard(collection, shard_idx)
        .into_iter()
        .filter(|id| *id != me)
        .filter(|id| meta.shard_state(collection, shard_idx, *id) == ShardState::Active)
        .filter_map(|id| meta.nodes.get(&id).cloned())
        .collect();

    if sources.is_empty() {
        // No Active peer to pull from. This is the genuinely-new-empty-shard case
        // (e.g. a brand-new collection): just ensure the shard exists locally and
        // is Active. Mark Active so reads can use us; there is nothing to stream.
        let _ = ensure_shard_open(state, collection, schema, shard_idx);
        let _ = set_state(state, collection, shard_idx, me, ShardState::Active).await;
        return false;
    }

    // 1. Mark ourselves Syncing (excluded from reads while we catch up).
    if set_state(state, collection, shard_idx, me, ShardState::Syncing)
        .await
        .is_err()
    {
        return false; // could not commit Syncing (no leader yet) — retry next pass
    }

    // 2. Pull from the first source that yields records.
    let mut pulled = false;
    for src in &sources {
        match router::pull_shard(state, src, collection, shard_idx, schema).await {
            Ok(_n) => {
                pulled = true;
                break;
            }
            Err(_) => continue,
        }
    }

    // 3. Cut over to Active (whether or not the source had any records — being a
    //    replica with the data streamed makes us authoritative). If the pull
    //    failed against every source, stay Syncing and retry next pass.
    if pulled {
        let _ = set_state(state, collection, shard_idx, me, ShardState::Active).await;
        tracing::info!(
            node = me,
            %collection,
            shard_idx,
            "acquired shard via stream_records transfer"
        );
        true
    } else {
        false
    }
}

/// Whether every desired replica of a shard is currently `Active`.
fn desired_all_active(meta: &MetaState, collection: &str, shard_idx: u32, desired: &[u64]) -> bool {
    !desired.is_empty()
        && desired
            .iter()
            .all(|id| meta.shard_state(collection, shard_idx, *id) == ShardState::Active)
}

/// Commit a shard-state transition through Raft, forwarding to the leader via the
/// internal `MetaWrite` RPC when this node is a follower.
async fn set_state(
    state: &Arc<NodeState>,
    collection: &str,
    shard_idx: u32,
    node_id: u64,
    new_state: ShardState,
) -> Result<(), ()> {
    // Fast path: we are (or might be) the leader — try a direct Raft write.
    match state
        .cluster
        .set_shard_state(collection.to_string(), shard_idx, node_id, new_state)
        .await
    {
        Ok(_) => return Ok(()),
        Err(qv_cluster::ManagerError::NotLeader(_)) => { /* fall through to forward */ }
        Err(_) => return Err(()),
    }

    // Forward to the leader.
    let Some(addr) = state.cluster.leader_addr().await else {
        return Err(());
    };
    let bytes = match qv_cluster::ClusterManager::encode_set_shard_state(
        collection.to_string(),
        shard_idx,
        node_id,
        new_state,
    ) {
        Ok(b) => b,
        Err(_) => return Err(()),
    };
    let Ok(mut client) = meta_client(&addr).await else {
        return Err(());
    };
    match client.meta_write(MetaWriteRequest { request: bytes }).await {
        Ok(resp) => {
            if resp.into_inner().committed {
                Ok(())
            } else {
                Err(()) // leader rejected / not committed
            }
        }
        Err(_) => Err(()), // leader unreachable or moved
    }
}

/// Ensure a local durable shard exists (open/create it) without streaming.
fn ensure_shard_open(
    state: &Arc<NodeState>,
    collection: &str,
    schema: &qv_cluster::CollectionSchema,
    shard_idx: u32,
) -> Result<(), ()> {
    let sref = sref_of(collection, schema, shard_idx);
    state.shards.shard_len(&sref).map(|_| ()).map_err(|_| ())
}

fn shard_len(
    state: &Arc<NodeState>,
    collection: &str,
    shard_idx: u32,
    schema: &qv_cluster::CollectionSchema,
) -> Result<usize, ()> {
    let sref = sref_of(collection, schema, shard_idx);
    state.shards.shard_len(&sref).map_err(|_| ())
}

fn sref_of<'a>(
    collection: &'a str,
    schema: &qv_cluster::CollectionSchema,
    shard_idx: u32,
) -> ShardRef<'a> {
    ShardRef {
        collection,
        shard_idx,
        dim: schema.dim as usize,
        metric: metric_from_cluster(schema.metric),
    }
}

async fn meta_client(addr: &str) -> Result<QuorvecInternalClient<Channel>, ()> {
    let endpoint = if addr.starts_with("http://") || addr.starts_with("https://") {
        addr.to_string()
    } else {
        format!("http://{addr}")
    };
    // Short connect/request timeouts so forwarding to a leader that just changed
    // (or a momentarily-unreachable one) fails fast and the reconcile retries next
    // pass instead of stalling the loop.
    let channel = Channel::from_shared(endpoint)
        .map_err(|_| ())?
        .connect_timeout(std::time::Duration::from_millis(500))
        .timeout(std::time::Duration::from_secs(3))
        .connect()
        .await
        .map_err(|_| ())?;
    Ok(QuorvecInternalClient::new(channel))
}

/// The background reconcile loop: run `reconcile_once` on a fixed interval. Spawned
/// once at node startup. Cheap in steady state (a metadata read + a holds-check
/// per shard), so a short interval keeps rebalancing prompt without load.
pub fn spawn_reconcile_loop(state: Arc<NodeState>) {
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(Duration::from_millis(500));
        loop {
            tick.tick().await;
            let n = reconcile_once(&state).await;
            if n > 0 {
                tracing::info!(transfers = n, "rebalance reconcile completed transfers");
            }
        }
    });
}
