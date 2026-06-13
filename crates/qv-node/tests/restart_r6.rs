//! R6 acceptance: durable Raft storage survives a **full-cluster restart**.
//!
//! The architect's M5 entry criterion (ruling R6): with the Raft log + state
//! machine persisted under `data_dir`, killing ALL nodes and restarting them must
//! bring the metadata plane back intact — collections and the shard map — without
//! relying on any peer to re-replicate it (there is no surviving peer when the
//! whole cluster is down).
//!
//! What this proves:
//!   1. Bring up a 5-process cluster, create two collections, populate one.
//!   2. Record the shard map.
//!   3. Kill ALL 5 nodes (taskkill /F via Child::kill on Windows).
//!   4. Restart ALL 5 nodes on their SAME ports/data-dirs.
//!   5. The recovered cluster reports the SAME collections and the SAME shard map,
//!      and the populated collection's data is still searchable.
//!
//! Same harness idiom as `cluster_m3.rs` / `quorum_m4.rs`: real `qv-node`
//! processes driven over gRPC.

use std::collections::BTreeSet;
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use qv_client::Client;
use qv_proto::v1;

struct NodeProc {
    id: u64,
    addr: String,
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
    /// Restart on the SAME port + data-dir. The seed (id 1) restarts with
    /// --bootstrap, which is now restart-safe (resumes from durable state instead
    /// of re-initializing).
    fn respawn(&mut self) {
        let child = spawn_node(self.id, &self.addr, &self.data_dir, self.id == 1);
        self.child = Some(child);
    }
}

impl Drop for NodeProc {
    fn drop(&mut self) {
        self.kill();
    }
}

fn free_port() -> u16 {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
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

/// Start a 5-node cluster: node 1 bootstraps and joins 2..=5 through Raft.
async fn start_cluster(tmp: &tempfile::TempDir) -> Vec<NodeProc> {
    let mut nodes = Vec::new();
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
    let mut admin = Client::connect(nodes[0].endpoint()).await.unwrap();
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
    nodes
}

/// A stable, comparable shard-map fingerprint for a collection: the sorted set of
/// (shard_idx, node_id) replica assignments.
async fn shard_fingerprint(ep: &str, collection: &str) -> BTreeSet<(u32, u64)> {
    let mut c = Client::connect(ep.to_string()).await.unwrap();
    let info = c.cluster_info().await.unwrap();
    info.shard_map
        .iter()
        .filter(|s| s.collection == collection)
        .map(|s| (s.shard_idx, s.node_id))
        .collect()
}

async fn collections_seen(ep: &str) -> BTreeSet<String> {
    let mut c = Client::connect(ep.to_string()).await.unwrap();
    let info = c.cluster_info().await.unwrap();
    info.shard_map
        .iter()
        .map(|s| s.collection.clone())
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn full_cluster_restart_recovers_metadata_r6() {
    let tmp = tempfile::tempdir().unwrap();
    let mut nodes = start_cluster(&tmp).await;
    let seed_ep = nodes[0].endpoint();

    const DIM: usize = 16;
    const SHARDS: u32 = 8;
    const N: u32 = 3;

    // ---- 1. Create two collections; populate one ----------------------------
    let mut a = Client::connect(seed_ep.clone()).await.unwrap();
    a.create_collection("alpha", DIM as u32, v1::Metric::L2, SHARDS, N)
        .await
        .unwrap();
    a.create_collection("beta", DIM as u32, v1::Metric::Cosine, 4, N)
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(700)).await;

    let vec_of = |id: u64| -> Vec<f32> { (0..DIM).map(|d| id as f32 + d as f32 * 0.1).collect() };
    let points: Vec<v1::Point> = (0..200u64)
        .map(|id| v1::Point {
            id,
            vector: vec_of(id),
            payload: None,
        })
        .collect();
    a.upsert("alpha", points, v1::Consistency::Quorum)
        .await
        .unwrap();

    // ---- 2. Record the pre-restart metadata ---------------------------------
    let cols_before = collections_seen(&seed_ep).await;
    let alpha_before = shard_fingerprint(&seed_ep, "alpha").await;
    let beta_before = shard_fingerprint(&seed_ep, "beta").await;
    assert!(cols_before.contains("alpha") && cols_before.contains("beta"));
    assert_eq!(
        alpha_before.len() as u32,
        SHARDS * N,
        "alpha should have 8x3 = 24 replica assignments"
    );

    // ---- 3. Kill ALL nodes (the full-cluster outage) ------------------------
    for n in nodes.iter_mut() {
        n.kill();
    }
    // Confirm the cluster is genuinely down: nothing answers.
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert!(
        Client::connect(seed_ep.clone()).await.is_err()
            || Client::connect(seed_ep.clone())
                .await
                .unwrap()
                .health()
                .await
                .is_err(),
        "cluster should be fully down after killing all nodes"
    );

    // ---- 4. Restart ALL nodes on their SAME ports + data-dirs ---------------
    for n in nodes.iter_mut() {
        n.respawn();
    }
    for n in &nodes {
        wait_healthy(&n.endpoint()).await;
    }
    // Give Raft a moment to re-establish a leader from the durable logs.
    tokio::time::sleep(Duration::from_secs(2)).await;

    // ---- 5. Metadata recovered: same collections + same shard map -----------
    // Poll briefly: a just-restarted cluster needs a beat to elect a leader and
    // for every node to surface the recovered metadata.
    let mut cols_after = BTreeSet::new();
    for _ in 0..30 {
        cols_after = collections_seen(&seed_ep).await;
        if cols_after.contains("alpha") && cols_after.contains("beta") {
            break;
        }
        tokio::time::sleep(Duration::from_millis(300)).await;
    }
    assert_eq!(
        cols_after, cols_before,
        "collections did not survive the full-cluster restart"
    );

    let alpha_after = shard_fingerprint(&seed_ep, "alpha").await;
    let beta_after = shard_fingerprint(&seed_ep, "beta").await;
    assert_eq!(
        alpha_after, alpha_before,
        "alpha shard map changed across restart (placement is deterministic from \
         the recovered membership, so it must be identical)"
    );
    assert_eq!(
        beta_after, beta_before,
        "beta shard map changed across restart"
    );

    // Every node agrees on the recovered collections (metadata is consistent).
    for n in &nodes {
        let cols = collections_seen(&n.endpoint()).await;
        assert_eq!(
            cols, cols_before,
            "node {} disagrees on recovered collections",
            n.id
        );
    }

    // ---- 6. Data plane intact too: alpha is still searchable ----------------
    // (The data plane was already durable pre-R6; this confirms the whole node
    // recovers end-to-end, not just the Raft metadata.)
    let mut c = Client::connect(seed_ep.clone()).await.unwrap();
    let hits = c.search("alpha", vec_of(42), 10, 64).await.unwrap();
    assert!(
        !hits.is_empty(),
        "alpha not searchable after full-cluster restart"
    );
    assert!(
        hits.iter().any(|h| h.id == 42),
        "search did not find the queried id after restart"
    );

    // Teardown via Drop.
}
