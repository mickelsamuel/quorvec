//! Shared helpers for the cluster integration tests.
//!
//! `tests/common/` is a subdirectory module (cargo does NOT compile it as its own
//! test binary), so the cluster tests can `mod common;` it to share code.

#![allow(dead_code)]

use std::fs::OpenOptions;
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// A cross-process serial guard for the heavy cluster integration tests.
///
/// Each `tests/*.rs` integration test is its own binary and `cargo test` runs the
/// binaries concurrently. The cluster tests each spawn a 5-6 process `qv-node`
/// cluster; running several at once on one host (as `cargo test --workspace`
/// does, and as CI does on a small runner) starves CPU so badly that latencies
/// balloon and the "zero failed requests under rebalance" timing assumptions stop
/// holding — a measurement artifact of contention, not a cluster fault. This
/// guard serializes them via a lockfile so only one cluster runs at a time.
pub struct ClusterGuard {
    path: PathBuf,
}

impl Drop for ClusterGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Acquire the global cluster lock (spin on a `create_new` lockfile). A stale lock
/// left by a crashed test is reclaimed after a timeout so a panic never wedges the
/// suite.
pub fn acquire_cluster_lock() -> ClusterGuard {
    let path = std::env::temp_dir().join("quorvec-cluster-test.lock");
    let start = Instant::now();
    loop {
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(f) => {
                drop(f);
                return ClusterGuard { path };
            }
            Err(_) => {
                if let Ok(meta) = std::fs::metadata(&path) {
                    if let Ok(modified) = meta.modified() {
                        if modified.elapsed().unwrap_or_default() > Duration::from_secs(300) {
                            let _ = std::fs::remove_file(&path);
                            continue;
                        }
                    }
                }
                if start.elapsed() > Duration::from_secs(600) {
                    return ClusterGuard { path };
                }
                std::thread::sleep(Duration::from_millis(200));
            }
        }
    }
}
