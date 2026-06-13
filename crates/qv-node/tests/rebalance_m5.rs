//! M5 acceptance: rebalancing on the 5-process localhost cluster (R5 binding).
//!
//! Proves plan §M5 acceptance, under continuous client load:
//!   (1) ADD a 6th node → shards move to it (it becomes a replica and acquires the
//!       data via the `stream_records` transfer) with **zero failed requests**
//!       (brief retries are allowed, counted, and printed).
//!   (2) KILL a node permanently (and Leave it from membership) → the
//!       under-replicated shards re-replicate to healthy nodes; request success
//!       rate is printed.
//!
//! Continuous load runs in a background task throughout each scenario: it keeps
//! upserting and searching, retrying a transiently-unavailable request a bounded
//! number of times (a retry is counted and printed, a final failure is a hard
//! failure of the test). This is the "zero failed requests, brief retries
//! allowed and counted" bar.
//!
//! Same real-process harness as cluster_m3 / quorum_m4 / restart_r6.

use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use qv_client::Client;
use qv_proto::v1;

struct NodeProc {
    id: u64,
    addr: String,
    #[allow(dead_code)] // kept so each node's data dir path is owned for its lifetime
    data_dir: PathBuf,
    child: Option<Child>,
}

impl NodeProc {
    fn endpoint(&self) -> String {
        format!("http://{}", self.addr)
    }
    fn kill(&mut self) {
        if let Some(mut c) = self.child.take() {
            let _ = c.kill();
            let _ = c.wait();
        }
    }
}

impl Drop for NodeProc {
    fn drop(&mut self) {
        self.kill();
    }
}

fn free_port() -> u16 {
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let p = l.local_addr().unwrap().port();
    drop(l);
    p
}

fn qv_node_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_qv-node"))
}

fn spawn_node(id: u64, addr: &str, data_dir: &PathBuf, bootstrap: bool) -> Child {
    let mut cmd = Command::new(qv_node_bin());
    cmd.arg("--node-id")
        .arg(id.to_string())
        .arg("--listen")
        .arg(addr)
        .arg("--advertise")
        .arg(addr)
        .arg("--data-dir")
        .arg(data_dir);
    if bootstrap {
        cmd.arg("--bootstrap");
    }
    cmd.env(
        "RUST_LOG",
        std::env::var("RUST_LOG").unwrap_or_else(|_| "warn".into()),
    );
    cmd.spawn().expect("spawn qv-node")
}

async fn wait_healthy(endpoint: &str) {
    let deadline = Instant::now() + Duration::from_secs(25);
    loop {
        if let Ok(mut c) = Client::connect(endpoint.to_string()).await {
            if c.health().await.is_ok() {
                return;
            }
        }
        if Instant::now() > deadline {
            panic!("node at {endpoint} never became healthy");
        }
        tokio::time::sleep(Duration::from_millis(150)).await;
    }
}

const DIM: usize = 16;
const SHARDS: u32 = 12;
const N: u32 = 3;
const POP: u64 = 600;

fn vec_of(id: u64) -> Vec<f32> {
    (0..DIM)
        .map(|d| (id % 97) as f32 + d as f32 * 0.05)
        .collect()
}

/// Shared request counters for the continuous-load workload.
#[derive(Default)]
struct LoadStats {
    attempts: AtomicU64,
    ok: AtomicU64,
    retries: AtomicU64,
    failures: AtomicU64,
}

impl LoadStats {
    fn rate(&self) -> f64 {
        let a = self.attempts.load(Ordering::Relaxed);
        if a == 0 {
            return 100.0;
        }
        100.0 * self.ok.load(Ordering::Relaxed) as f64 / a as f64
    }
    fn print(&self, label: &str) {
        println!(
            "[M5 load] {label}: attempts={} ok={} retries={} failures={} success_rate={:.3}%",
            self.attempts.load(Ordering::Relaxed),
            self.ok.load(Ordering::Relaxed),
            self.retries.load(Ordering::Relaxed),
            self.failures.load(Ordering::Relaxed),
            self.rate(),
        );
    }
}

/// One logical client operation with bounded retries. Returns Ok on eventual
/// success (counting any retries), Err only if every retry was exhausted.
async fn op_with_retry<F, Fut, T>(stats: &LoadStats, mut f: F) -> Result<T, ()>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T, qv_client::ClientError>>,
{
    // `attempts` is incremented only once the op resolves (success or final
    // failure), so at any print point attempts == ok + failures exactly — the
    // reported success rate is unambiguous.
    const MAX_RETRIES: usize = 8;
    for attempt in 0..=MAX_RETRIES {
        match f().await {
            Ok(v) => {
                stats.attempts.fetch_add(1, Ordering::Relaxed);
                stats.ok.fetch_add(1, Ordering::Relaxed);
                return Ok(v);
            }
            Err(_) if attempt < MAX_RETRIES => {
                stats.retries.fetch_add(1, Ordering::Relaxed);
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
            Err(_) => {
                stats.attempts.fetch_add(1, Ordering::Relaxed);
                stats.failures.fetch_add(1, Ordering::Relaxed);
                return Err(());
            }
        }
    }
    Err(())
}

/// The pool of coordinator endpoints the load currently believes are alive. The
/// test mutates this (removes a node it permanently kills) so the workload routes
/// only to live coordinators — exactly what a real client with a health view does.
/// Routing a *client request* at a node you know is dead is a client bug, not a
/// cluster failure, so the acceptance bar (zero failed requests) is measured
/// against the live coordinators.
type LiveEps = Arc<std::sync::Mutex<Vec<String>>>;

/// Drive continuous upsert+search load against the live coordinator endpoints
/// until `stop` is set. Cycles ids so reads exercise data that already exists.
/// Connections are made fresh each op (cheap on localhost) so a coordinator that
/// was removed from the pool is simply never dialled again.
async fn run_load(live: LiveEps, stats: Arc<LoadStats>, stop: Arc<AtomicBool>) {
    let mut i: u64 = 0;
    while !stop.load(Ordering::Relaxed) {
        let ep = {
            let pool = live.lock().unwrap();
            if pool.is_empty() {
                None
            } else {
                Some(pool[(i as usize) % pool.len()].clone())
            }
        };
        let Some(ep) = ep else {
            tokio::time::sleep(Duration::from_millis(20)).await;
            continue;
        };
        let id = i % POP;

        // Upsert (W=QUORUM tolerates a replica being mid-transfer or down).
        let _ = op_with_retry(&stats, || {
            let ep = ep.clone();
            async move {
                let mut c = Client::connect(ep).await?;
                c.upsert(
                    "vectors",
                    vec![v1::Point {
                        id,
                        vector: vec_of(id),
                        payload: None,
                    }],
                    v1::Consistency::Quorum,
                )
                .await
                .map(|_| ())
            }
        })
        .await;

        // Search (reads route to Active replicas only).
        let _ = op_with_retry(&stats, || {
            let ep = ep.clone();
            async move {
                let mut c = Client::connect(ep).await?;
                c.search("vectors", vec_of(id), 10, 64).await.map(|_| ())
            }
        })
        .await;

        i += 1;
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

async fn cluster_info(ep: &str) -> v1::ClusterInfoResponse {
    let mut c = Client::connect(ep.to_string()).await.unwrap();
    c.cluster_info().await.unwrap()
}

/// All replica node-ids that appear for a collection in the shard map (the set of
/// nodes that hold at least one shard of it).
async fn holder_nodes(ep: &str, collection: &str) -> std::collections::BTreeSet<u64> {
    cluster_info(ep)
        .await
        .shard_map
        .iter()
        .filter(|s| s.collection == collection)
        .map(|s| s.node_id)
        .collect()
}

/// Count shard assignments currently in Active state for a collection.
async fn active_assignment_count(ep: &str, collection: &str) -> usize {
    cluster_info(ep)
        .await
        .shard_map
        .iter()
        .filter(|s| s.collection == collection && s.state == v1::ShardState::Active as i32)
        .count()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn rebalancing_m5_acceptance() {
    let tmp = tempfile::tempdir().unwrap();

    // ---- Bring up a 5-node cluster -----------------------------------------
    let mut nodes: Vec<NodeProc> = Vec::new();
    for id in 1..=5u64 {
        let port = free_port();
        let addr = format!("127.0.0.1:{port}");
        let data_dir = tmp.path().join(format!("node{id}"));
        std::fs::create_dir_all(&data_dir).unwrap();
        let child = spawn_node(id, &addr, &data_dir, id == 1);
        nodes.push(NodeProc {
            id,
            addr,
            data_dir,
            child: Some(child),
        });
    }
    for n in &nodes {
        wait_healthy(&n.endpoint()).await;
    }
    let seed_ep = nodes[0].endpoint();
    let mut admin = Client::connect(seed_ep.clone()).await.unwrap();
    for n in nodes.iter().skip(1) {
        let target = format!("{}@{}", n.id, n.addr);
        for _ in 0..20 {
            if admin.join(&target).await.is_ok() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
    }
    tokio::time::sleep(Duration::from_secs(1)).await;

    // Create + populate the collection.
    admin
        .create_collection("vectors", DIM as u32, v1::Metric::L2, SHARDS, N)
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(700)).await;
    let points: Vec<v1::Point> = (0..POP)
        .map(|id| v1::Point {
            id,
            vector: vec_of(id),
            payload: None,
        })
        .collect();
    admin
        .upsert("vectors", points, v1::Consistency::Quorum)
        .await
        .unwrap();

    let holders_before = holder_nodes(&seed_ep, "vectors").await;
    println!("[M5] holder nodes before join: {holders_before:?}");
    assert!(
        !holders_before.contains(&6),
        "node 6 should not hold shards before it joins"
    );

    // ===== Scenario 1: add a 6th node under continuous load ==================
    let live: LiveEps = Arc::new(std::sync::Mutex::new(
        nodes.iter().take(5).map(|n| n.endpoint()).collect(),
    ));
    let stats = Arc::new(LoadStats::default());
    let stop = Arc::new(AtomicBool::new(false));
    let load_handle = {
        let stats = stats.clone();
        let stop = stop.clone();
        let live = live.clone();
        tokio::spawn(run_load(live, stats, stop))
    };
    // Let load run a moment to establish a steady baseline.
    tokio::time::sleep(Duration::from_millis(800)).await;

    // Spawn + join node 6.
    let port6 = free_port();
    let addr6 = format!("127.0.0.1:{port6}");
    let data_dir6 = tmp.path().join("node6");
    std::fs::create_dir_all(&data_dir6).unwrap();
    let child6 = spawn_node(6, &addr6, &data_dir6, false);
    nodes.push(NodeProc {
        id: 6,
        addr: addr6.clone(),
        data_dir: data_dir6,
        child: Some(child6),
    });
    wait_healthy(&format!("http://{addr6}")).await;
    for _ in 0..20 {
        if admin.join(&format!("6@{addr6}")).await.is_ok() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }

    // Wait for node 6 to actually become a shard holder (the rebalance moved
    // shards to it). The reconcile loop + transfer should make this happen.
    let mut joined_holders = std::collections::BTreeSet::new();
    let deadline = Instant::now() + Duration::from_secs(30);
    while Instant::now() < deadline {
        joined_holders = holder_nodes(&seed_ep, "vectors").await;
        if joined_holders.contains(&6) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(400)).await;
    }
    // Let any in-flight transfers cut over to Active.
    tokio::time::sleep(Duration::from_secs(3)).await;
    stats.print("after add-6th-node");

    assert!(
        joined_holders.contains(&6),
        "node 6 never received any shards after joining (rebalance did not move shards)"
    );
    // Zero failed requests (retries allowed + counted + printed).
    assert_eq!(
        stats.failures.load(Ordering::Relaxed),
        0,
        "join rebalance caused failed client requests (not zero)"
    );
    // Every shard assignment should be back to Active once transfers cut over.
    let mut active_after_join = 0;
    for _ in 0..30 {
        active_after_join = active_assignment_count(&seed_ep, "vectors").await;
        if active_after_join as u32 == SHARDS * N {
            break;
        }
        tokio::time::sleep(Duration::from_millis(400)).await;
    }
    assert_eq!(
        active_after_join as u32,
        SHARDS * N,
        "not all shard replicas returned to Active after the join transfer"
    );

    // Data is correct on the rebalanced cluster: a known id is searchable and a
    // Get returns it (it may now live on node 6 for some shards).
    let mut verify = Client::connect(seed_ep.clone()).await.unwrap();
    let got = verify
        .get("vectors", vec![42, 123, 404], v1::Consistency::Quorum)
        .await
        .unwrap();
    assert!(
        got.iter().any(|p| p.id == 42),
        "id 42 missing after rebalance"
    );

    // ===== Scenario 2: kill a node permanently + Leave it ====================
    // Pick a victim that is NOT node 1 (keep the seed/leader for admin ops) and
    // is a current holder, so its loss genuinely under-replicates shards.
    let victim = *holder_nodes(&seed_ep, "vectors")
        .await
        .iter()
        .find(|id| **id != 1)
        .unwrap();
    println!("[M5] killing victim node {victim} permanently");
    let victim_pos = nodes.iter().position(|n| n.id == victim).unwrap();
    let victim_ep = nodes[victim_pos].endpoint();
    // A real client stops routing requests at a node it knows is gone: remove the
    // victim from the load's coordinator pool BEFORE killing it. Requests already
    // in flight to it are covered by the retry budget against the live nodes.
    live.lock().unwrap().retain(|e| e != &victim_ep);
    nodes[victim_pos].kill();

    // Declare it permanently gone via the admin Leave op (membership change → the
    // ring promotes replacement replicas → reconcile re-replicates).
    for _ in 0..40 {
        if admin.leave(victim).await.is_ok() {
            break;
        }
        // The leader may have been the victim; re-resolve admin to a survivor.
        if let Some(s) = nodes.iter().find(|n| n.id != victim && n.child.is_some()) {
            if let Ok(c) = Client::connect(s.endpoint()).await {
                admin = c;
            }
        }
        tokio::time::sleep(Duration::from_millis(300)).await;
    }

    // After Leave, the victim must be gone from the shard map and every shard must
    // recover to N Active replicas among the surviving nodes.
    let survivor_ep = nodes
        .iter()
        .find(|n| n.id != victim && n.child.is_some())
        .unwrap()
        .endpoint();

    let mut recovered = false;
    let deadline = Instant::now() + Duration::from_secs(40);
    while Instant::now() < deadline {
        let info = cluster_info(&survivor_ep).await;
        let vmap: Vec<_> = info
            .shard_map
            .iter()
            .filter(|s| s.collection == "vectors")
            .collect();
        let victim_gone = !vmap.iter().any(|s| s.node_id == victim);
        let full_active = vmap.len() as u32 == SHARDS * N
            && vmap
                .iter()
                .all(|s| s.state == v1::ShardState::Active as i32);
        if victim_gone && full_active {
            recovered = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }

    // Stop the load and report.
    stop.store(true, Ordering::Relaxed);
    let _ = load_handle.await;
    stats.print("after permanent-death + re-replication");

    assert!(
        recovered,
        "under-replicated shards did not re-replicate to N Active healthy replicas after a permanent node death"
    );
    assert_eq!(
        stats.failures.load(Ordering::Relaxed),
        0,
        "client requests failed during/after the permanent-death recovery (not zero)"
    );

    // Final data correctness: the populated ids are still searchable post-recovery.
    let mut fc = Client::connect(survivor_ep).await.unwrap();
    let hits = fc.search("vectors", vec_of(42), 10, 64).await.unwrap();
    assert!(
        !hits.is_empty(),
        "search returned nothing after re-replication"
    );

    // Teardown via Drop.
}
