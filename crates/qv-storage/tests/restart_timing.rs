//! M2 acceptance: snapshot + tail-replay restart time on a 1M-vector shard must
//! be < 30 s (record the actual number).
//!
//! This builds a 1M-vector shard once, snapshots it, appends a realistic WAL
//! tail, then measures the cold reopen time (snapshot load + tail replay). It is
//! `#[ignore]` by default because building 1M HNSW nodes is minutes-long; run it
//! explicitly with:
//!
//!   cargo test -p qv-storage --release --test restart_timing -- --ignored --nocapture
//!
//! The recorded number goes in the M2 log.

use qv_storage::{Hlc, Shard};
use std::time::Instant;

const DIM: usize = 128;
const N: u64 = 1_000_000;
const TAIL: u64 = 10_000; // WAL tail beyond the snapshot

fn vec_for(id: u64, dim: usize) -> Vec<f32> {
    // Cheap deterministic vector; cluster-ish so HNSW stays well-formed.
    let base = (id % 1000) as f32;
    (0..dim)
        .map(|d| base * 0.001 + ((id as usize + d) % 131) as f32 * 0.01)
        .collect()
}

#[test]
#[ignore = "slow: builds a 1M-vector shard; run explicitly to record restart timing"]
fn restart_under_30s_on_1m_shard() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path().join("shard_1m");

    // Build phase (not timed for acceptance — only the restart is).
    let build_start = Instant::now();
    {
        // Large snapshot threshold so we control when the snapshot happens.
        let mut shard = Shard::open(&dir, DIM, qv_hnsw::Metric::L2, 100_000, 0).unwrap();
        for id in 0..N {
            let v = vec_for(id, DIM);
            shard.upsert(Hlc::new(id, 0), id, &v, Vec::new()).unwrap();
        }
        // Snapshot covering all 1M, then append a tail to exercise tail replay.
        shard.snapshot_now().unwrap();
        for id in N..(N + TAIL) {
            let v = vec_for(id, DIM);
            shard.upsert(Hlc::new(id, 0), id, &v, Vec::new()).unwrap();
        }
    }
    let build_secs = build_start.elapsed().as_secs_f64();

    // Timed phase: cold reopen = snapshot load + WAL tail replay.
    let restart_start = Instant::now();
    let shard = Shard::open(&dir, DIM, qv_hnsw::Metric::L2, 100_000, 0).unwrap();
    let restart_secs = restart_start.elapsed().as_secs_f64();

    println!(
        "RESTART TIMING: 1M-vector shard + {TAIL} tail; build={build_secs:.1}s; \
         RESTART (snapshot load + tail replay)={restart_secs:.2}s; live={}",
        shard.len()
    );
    assert_eq!(shard.len() as u64, N + TAIL, "recovered count wrong");
    assert!(
        restart_secs < 30.0,
        "restart took {restart_secs:.2}s (>= 30s target)"
    );
}
