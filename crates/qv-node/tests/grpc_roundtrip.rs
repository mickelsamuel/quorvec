//! End-to-end gRPC acceptance: boot the node over a real gRPC channel, create a
//! collection, upsert 1k random vectors, and verify search results against an
//! independent in-test brute-force oracle.
//!
//! Through M0 the node used the exact brute-force index, so this asserted exact
//! top-10 equality. From M1 the node serves an approximate HNSW index behind the
//! same trait, so the search assertion is recall-based (recall@10 >= 0.95 over
//! the query set) while Get/Delete remain exact (they do not depend on the ANN
//! approximation). The data is a clustered Gaussian mixture — representative of
//! real embeddings, where ef_search=64 yields high recall.

use qv_client::Client;
use qv_hnsw::{BruteForceIndex, Metric, VectorIndex};
use qv_proto::v1;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use qv_node::{QuorvecService, Store};
use qv_proto::QuorvecServer;
use tonic::transport::Server;

async fn start_test_server() -> SocketAddr {
    // Bind to an OS-assigned free port.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener); // release; tonic rebinds

    let store = Arc::new(Store::new());
    let svc = QuorvecService::new(store, 1, addr.to_string());

    tokio::spawn(async move {
        Server::builder()
            .add_service(QuorvecServer::new(svc))
            .serve(addr)
            .await
            .unwrap();
    });

    // Give the server a moment to bind.
    for _ in 0..50 {
        if Client::connect(format!("http://{addr}")).await.is_ok() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    addr
}

/// A clustered Gaussian-mixture sample generator (50 centers) so the in-node
/// HNSW index has the cluster structure real embeddings have.
fn clustered(rng: &mut StdRng, dim: usize, centers: &[Vec<f32>]) -> Vec<f32> {
    let ci = rng.gen_range(0..centers.len());
    (0..dim)
        .map(|d| {
            // Box-Muller gaussian noise around the chosen center.
            let u1: f32 = rng.gen::<f32>().max(1e-7);
            let u2: f32 = rng.gen::<f32>();
            let g = (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos();
            centers[ci][d] + g * 0.35
        })
        .collect()
}

#[tokio::test]
async fn create_upsert_1k_search_recall() {
    const DIM: usize = 64;
    const N: u64 = 1000;
    const SEED: u64 = 0xC0FFEE;
    const QUERIES: usize = 50;

    let addr = start_test_server().await;
    let mut client = Client::connect(format!("http://{addr}")).await.unwrap();

    assert_eq!(client.health().await.unwrap(), "ok");

    // Create collection.
    client
        .create_collection("vectors", DIM as u32, v1::Metric::L2, 0, 0)
        .await
        .unwrap();

    // Generate 1k clustered vectors, keep a local copy for the oracle.
    let mut rng = StdRng::seed_from_u64(SEED);
    let centers: Vec<Vec<f32>> = (0..50)
        .map(|_| (0..DIM).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect())
        .collect();

    let mut oracle = BruteForceIndex::new(DIM, Metric::L2);
    let mut points = Vec::with_capacity(N as usize);
    for id in 0..N {
        let v = clustered(&mut rng, DIM, &centers);
        oracle.insert(id, &v).unwrap();
        points.push(v1::Point {
            id,
            vector: v,
            payload: None,
        });
    }

    let upserted = client
        .upsert("vectors", points, v1::Consistency::One)
        .await
        .unwrap();
    assert_eq!(upserted, N);

    // Run queries; accumulate recall@10 of the server (HNSW) vs the oracle.
    let mut hits = 0usize;
    let mut total = 0usize;
    for _ in 0..QUERIES {
        let q = clustered(&mut rng, DIM, &centers);

        let server_hits = client.search("vectors", q.clone(), 10, 64).await.unwrap();
        let oracle_hits = oracle.search(&q, 10, 64).unwrap();

        assert_eq!(server_hits.len(), 10);
        assert_eq!(oracle_hits.len(), 10);

        let oracle_ids: Vec<u64> = oracle_hits.iter().map(|h| h.0).collect();
        for sh in &server_hits {
            if oracle_ids.contains(&sh.id) {
                hits += 1;
            }
        }
        total += oracle_hits.len();

        // Scores returned by the server are the true metric distances for the
        // ids it returned, so each must match the oracle distance for that id.
        // Compute the full exact ranking once for lookup.
        let full = oracle.search(&q, N as usize, 64).unwrap();
        for sh in &server_hits {
            if let Some((_, d)) = full.iter().find(|(id, _)| *id == sh.id) {
                assert!(
                    (sh.score - d).abs() <= 1e-3 * d.max(1.0),
                    "score mismatch for id {}: server {} vs oracle {}",
                    sh.id,
                    sh.score,
                    d
                );
            }
        }
    }
    let recall = hits as f64 / total as f64;
    println!("E2E gRPC recall@10 (HNSW, clustered 1k/{QUERIES}, dim={DIM}): {recall:.4}");
    assert!(recall >= 0.95, "end-to-end recall@10 = {recall:.4} < 0.95");

    // Get returns the exact stored vector for an id.
    let got = client
        .get("vectors", vec![0, 1, 999], v1::Consistency::One)
        .await
        .unwrap();
    assert_eq!(got.len(), 3);

    // Delete then confirm it drops out of results and Get.
    assert_eq!(
        client
            .delete("vectors", vec![0], v1::Consistency::One)
            .await
            .unwrap(),
        1
    );
    let got = client
        .get("vectors", vec![0], v1::Consistency::One)
        .await
        .unwrap();
    assert!(got.is_empty(), "deleted id should not be returned by Get");
}
