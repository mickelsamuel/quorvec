//! Cluster smoke check (used by the CI compose smoke job).
//!
//! Usage: `smoke <node_a> <node_b> <node_c>` where each arg is an endpoint like
//! `http://127.0.0.1:7001`. It performs the M3 cross-node flow against a running
//! cluster:
//!
//! - wait for all three to be healthy,
//! - wait until the cluster has elected a leader and all expected nodes appear,
//! - CreateCollection on A,
//! - confirm the collection propagated to B and C (ClusterInfo shard map),
//! - Upsert via B, Search + Get via C,
//!
//! exiting non-zero on any failure so CI fails loudly.

use std::time::{Duration, Instant};

use qv_client::Client;
use qv_proto::v1;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        eprintln!("usage: smoke <node_a> <node_b> <node_c> [expected_node_count]");
        std::process::exit(2);
    }
    let (a_ep, b_ep, c_ep) = (args[0].clone(), args[1].clone(), args[2].clone());
    let expected_nodes: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(3);

    // 1. Health on all three.
    for ep in [&a_ep, &b_ep, &c_ep] {
        wait_healthy(ep).await?;
    }
    println!("[smoke] all nodes healthy");

    // 2. Wait for a leader and the full node directory on node A.
    wait_cluster_ready(&a_ep, expected_nodes).await?;
    println!("[smoke] cluster has a leader and {expected_nodes} nodes");

    // 3. CreateCollection on A.
    let mut a = Client::connect(a_ep.clone()).await?;
    a.create_collection("smoke", 16, v1::Metric::L2, 8, 3)
        .await?;
    println!("[smoke] created collection on A");

    // 4. Confirm propagation to B and C.
    for ep in [&b_ep, &c_ep] {
        wait_collection_visible(ep, "smoke").await?;
    }
    println!("[smoke] collection visible on B and C");

    // 5. Upsert via B.
    let mut b = Client::connect(b_ep.clone()).await?;
    let points: Vec<v1::Point> = (0..200u64)
        .map(|id| v1::Point {
            id,
            vector: (0..16).map(|d| ((id as f32) + d as f32) * 0.01).collect(),
            payload: None,
        })
        .collect();
    let upserted = b.upsert("smoke", points, v1::Consistency::One).await?;
    assert_eq_or_exit(upserted, 200, "upsert via B count");
    println!("[smoke] upserted 200 points via B");

    // 6. Search + Get via C.
    let mut c = Client::connect(c_ep.clone()).await?;
    let query: Vec<f32> = (0..16).map(|d| (5.0 + d as f32) * 0.01).collect();
    let hits = c.search("smoke", query, 10, 64).await?;
    if hits.len() != 10 {
        eprintln!("[smoke] FAIL: search via C returned {} != 10", hits.len());
        std::process::exit(1);
    }
    let got = c
        .get("smoke", vec![0, 1, 199], v1::Consistency::One)
        .await?;
    if got.len() != 3 {
        eprintln!("[smoke] FAIL: get via C returned {} != 3", got.len());
        std::process::exit(1);
    }
    println!("[smoke] search + get via C OK");

    println!("[smoke] PASS");
    Ok(())
}

async fn wait_healthy(ep: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Generous: on a busy CI runner the container image build + cold start of the
    // node process can take a while before the gRPC port accepts connections; this
    // wait must outlast that, or the smoke job flakes on slow runners (not a bug).
    let deadline = Instant::now() + Duration::from_secs(150);
    loop {
        if let Ok(mut c) = Client::connect(ep.to_string()).await {
            if c.health().await.is_ok() {
                return Ok(());
            }
        }
        if Instant::now() > deadline {
            return Err(format!("node {ep} never became healthy").into());
        }
        tokio::time::sleep(Duration::from_millis(300)).await;
    }
}

async fn wait_cluster_ready(
    ep: &str,
    expected_nodes: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let deadline = Instant::now() + Duration::from_secs(90);
    loop {
        if let Ok(mut c) = Client::connect(ep.to_string()).await {
            if let Ok(info) = c.cluster_info().await {
                if info.raft_leader != 0 && info.nodes.len() >= expected_nodes {
                    return Ok(());
                }
            }
        }
        if Instant::now() > deadline {
            return Err(format!("cluster at {ep} not ready (no leader / nodes)").into());
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

async fn wait_collection_visible(
    ep: &str,
    collection: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if let Ok(mut c) = Client::connect(ep.to_string()).await {
            if let Ok(info) = c.cluster_info().await {
                if info.shard_map.iter().any(|s| s.collection == collection) {
                    return Ok(());
                }
            }
        }
        if Instant::now() > deadline {
            return Err(format!("collection '{collection}' never appeared on {ep}").into());
        }
        tokio::time::sleep(Duration::from_millis(300)).await;
    }
}

fn assert_eq_or_exit(got: u64, want: u64, what: &str) {
    if got != want {
        eprintln!("[smoke] FAIL: {what}: {got} != {want}");
        std::process::exit(1);
    }
}
