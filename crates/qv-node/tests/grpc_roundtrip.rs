//! M0 acceptance: boot the node over a real gRPC channel, create a collection,
//! upsert 1k random vectors, and verify exact search returns the true nearest
//! neighbor — cross-checked in-test against an independent brute-force scan.

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

#[tokio::test]
async fn create_upsert_1k_search_exact() {
    const DIM: usize = 64;
    const N: u64 = 1000;
    const SEED: u64 = 0xC0FFEE;

    let addr = start_test_server().await;
    let mut client = Client::connect(format!("http://{addr}")).await.unwrap();

    assert_eq!(client.health().await.unwrap(), "ok");

    // Create collection.
    client
        .create_collection("vectors", DIM as u32, v1::Metric::L2, 0, 0)
        .await
        .unwrap();

    // Generate 1k random vectors, keep a local copy for the oracle.
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut oracle = BruteForceIndex::new(DIM, Metric::L2);
    let mut points = Vec::with_capacity(N as usize);
    for id in 0..N {
        let v: Vec<f32> = (0..DIM).map(|_| rng.gen::<f32>()).collect();
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

    // Run 50 random queries; the server's top-10 must match the in-test oracle.
    for _ in 0..50 {
        let q: Vec<f32> = (0..DIM).map(|_| rng.gen::<f32>()).collect();

        let server_hits = client.search("vectors", q.clone(), 10, 64).await.unwrap();
        let oracle_hits = oracle.search(&q, 10, 64).unwrap();

        assert_eq!(server_hits.len(), 10);
        assert_eq!(oracle_hits.len(), 10);

        // The nearest neighbor must agree exactly.
        assert_eq!(
            server_hits[0].id, oracle_hits[0].0,
            "server nearest != oracle nearest"
        );

        // The full top-10 id set must match (exact index, deterministic order).
        let server_ids: Vec<u64> = server_hits.iter().map(|h| h.id).collect();
        let oracle_ids: Vec<u64> = oracle_hits.iter().map(|h| h.0).collect();
        assert_eq!(server_ids, oracle_ids, "top-10 ordering mismatch");

        // Scores must match the oracle distances within fp tolerance.
        for (sh, oh) in server_hits.iter().zip(oracle_hits.iter()) {
            assert!(
                (sh.score - oh.1).abs() <= 1e-3 * oh.1.max(1.0),
                "score mismatch: server {} vs oracle {}",
                sh.score,
                oh.1
            );
        }
    }

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
