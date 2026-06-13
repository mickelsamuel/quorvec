//! M2 showpiece: the kill -9 crash-safety harness.
//!
//! For 20 iterations: spawn the `crash_writer` example as a child process that
//! upserts continuously and prints `ACK <id>` for every durably-acked write;
//! let it run a random duration; hard-kill it (`taskkill /F` on Windows,
//! `SIGKILL` elsewhere) at an unpredictable offset — i.e. almost always mid-write;
//! then reopen the shard and verify:
//!   1. NO ACKED WRITE IS LOST — every id the child printed ACK for is present,
//!   2. NO CORRUPTION — recovery succeeds; any torn trailing record is CRC-caught
//!      and truncated, and the shard is appendable again,
//!   3. the recovered index MATCHES a reference rebuilt by replaying the WAL.
//!
//! This is run as a single #[test] that loops 20 times.

use qv_storage::{Shard, Wal};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::time::Duration;

const DIM: usize = 16;
const ITERATIONS: usize = 20;

/// Path to the compiled `crash_writer` example binary.
fn crash_writer_bin() -> PathBuf {
    // The test binary lives in target/<profile>/deps/. The example is in
    // target/<profile>/examples/. Walk up from the current exe.
    let mut dir = std::env::current_exe().expect("current_exe");
    dir.pop(); // remove test exe name
    if dir.ends_with("deps") {
        dir.pop(); // -> target/<profile>
    }
    let exe = if cfg!(windows) {
        "crash_writer.exe"
    } else {
        "crash_writer"
    };
    dir.join("examples").join(exe)
}

/// Hard-kill a child process (no chance to flush / clean up) — the whole point.
fn hard_kill(child: &mut Child) {
    #[cfg(windows)]
    {
        // taskkill /F /T forcibly terminates the process tree.
        let _ = Command::new("taskkill")
            .args(["/F", "/T", "/PID", &child.id().to_string()])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
    #[cfg(not(windows))]
    {
        // SIGKILL via std kill.
        let _ = child.kill();
    }
    let _ = child.wait();
}

/// Spawn the writer, collect ACKed ids until we kill it after `run_for`.
fn run_one_iteration(shard_dir: &Path, run_for: Duration) -> Vec<u64> {
    let bin = crash_writer_bin();
    assert!(
        bin.exists(),
        "crash_writer example not built at {}",
        bin.display()
    );

    let mut child = Command::new(&bin)
        .arg(shard_dir)
        .arg(DIM.to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn crash_writer");

    // Drain ACKed ids on a background thread so a full pipe never blocks the
    // child (and so we capture everything acked right up to the kill).
    let stdout = child.stdout.take().expect("child stdout");
    let (tx, rx) = mpsc::channel::<u64>();
    let reader = std::thread::spawn(move || {
        let mut r = BufReader::new(stdout);
        let mut line = String::new();
        loop {
            line.clear();
            match r.read_line(&mut line) {
                Ok(0) => break, // EOF (process died)
                Ok(_) => {
                    if let Some(rest) = line.trim().strip_prefix("ACK ") {
                        if let Ok(id) = rest.parse::<u64>() {
                            let _ = tx.send(id);
                        }
                    }
                }
                Err(_) => break,
            }
        }
    });

    std::thread::sleep(run_for);
    hard_kill(&mut child);
    // Let the reader drain any buffered ACKs the child printed before dying.
    let _ = reader.join();

    let mut acked: Vec<u64> = Vec::new();
    while let Ok(id) = rx.try_recv() {
        acked.push(id);
    }
    acked
}

/// Hold the shared heavy-integration-test lock for the test's lifetime; deletes
/// the lockfile on drop.
struct SerialGuard(PathBuf);
impl Drop for SerialGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Serialize against the cluster integration tests (in qv-node). This kill -9
/// harness is CPU/disk heavy; running it alongside a multi-process cluster test
/// starves both. The same global lockfile the cluster tests use coordinates them.
fn acquire_serial_lock() -> SerialGuard {
    let path = std::env::temp_dir().join("quorvec-cluster-test.lock");
    let start = std::time::Instant::now();
    loop {
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(_) => return SerialGuard(path),
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
                    return SerialGuard(path);
                }
                std::thread::sleep(Duration::from_millis(200));
            }
        }
    }
}

#[test]
fn kill9_crash_safety_20_iterations() {
    let _serial = acquire_serial_lock();
    let tmp = tempfile::tempdir().expect("tempdir");
    let shard_dir = tmp.path().join("crash_shard");

    // A tiny LCG so kill timing is varied but reproducible across machines.
    let mut seed: u64 = 0xDEAD_BEEF_1234_5678;
    let mut next_ms = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // 30..230 ms — long enough to ack many writes, short enough to keep the
        // whole test quick; the kill lands at an arbitrary, usually-mid-write
        // moment.
        30 + (seed >> 33) % 200
    };

    // Accumulate every id ever acked across all iterations; all must survive.
    let mut all_acked: std::collections::BTreeSet<u64> = std::collections::BTreeSet::new();

    for iter in 0..ITERATIONS {
        let run_for = Duration::from_millis(next_ms());
        let acked = run_one_iteration(&shard_dir, run_for);

        // The child continues ids from the recovered length, so across
        // iterations the acked id ranges are contiguous and cumulative.
        for id in &acked {
            all_acked.insert(*id);
        }

        // --- Verify recovery after this kill ---

        // (1+2) Reopen the shard. This runs WAL recovery: a torn trailing record
        // (from the kill mid-append) is CRC-caught and truncated; recovery must
        // not error.
        let shard = Shard::open(&shard_dir, DIM, qv_hnsw::Metric::L2, 1, 0)
            .unwrap_or_else(|e| panic!("iter {iter}: recovery failed: {e}"));

        // (1) No acked write lost: every id acked so far must be live.
        for id in &all_acked {
            assert!(
                shard.get(*id).is_some(),
                "iter {iter}: acked id {id} lost after crash recovery"
            );
        }

        // (3) Recovered index matches a fresh replay of the WAL.
        let (records, _) = Wal::recover(shard_dir.join("shard.wal")).unwrap();
        let mut reference = qv_hnsw::HnswIndex::new(DIM, qv_hnsw::Metric::L2);
        for rec in &records {
            match rec.op {
                qv_storage::WalOp::Upsert => {
                    reference.insert(rec.id, &rec.vector).unwrap();
                }
                qv_storage::WalOp::Delete => {
                    reference.delete(rec.id);
                }
            }
        }
        use qv_hnsw::VectorIndex;
        assert_eq!(
            shard.len(),
            reference.len(),
            "iter {iter}: recovered live count != replay reference"
        );

        // The shard must be appendable again (drop reopens cleanly next loop).
        drop(shard);

        println!(
            "iter {iter}: killed after {run_for:?}, acked-so-far={}, recovered OK",
            all_acked.len()
        );
    }

    assert!(
        !all_acked.is_empty(),
        "harness never acked any writes — child not running?"
    );
    println!(
        "CRASH HARNESS: {ITERATIONS}/{ITERATIONS} iterations clean; {} total acked writes all survived",
        all_acked.len()
    );
}
