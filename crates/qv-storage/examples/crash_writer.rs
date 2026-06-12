//! Crash-test writer: opens a shard and upserts vectors continuously, printing
//! the id of each **durably acked** write to stdout (one per line, flushed) so
//! the parent harness knows exactly which writes were acknowledged before a
//! `kill -9` / `taskkill /F`.
//!
//! Usage: `crash_writer <shard_dir> <dim>`
//!
//! The contract the harness relies on: a line `ACK <id>` is printed only AFTER
//! `shard.upsert(...)` returns, and `upsert` only returns after the WAL record
//! is fsynced (batch_ms = 0 here). So every ACKed id must survive recovery.

use qv_storage::{Hlc, Shard};
use std::io::Write;

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args.next().expect("usage: crash_writer <dir> <dim>");
    let dim: usize = args
        .next()
        .expect("usage: crash_writer <dir> <dim>")
        .parse()
        .expect("dim must be a number");

    // fsync per record (batch_ms = 0); snapshot threshold small so snapshots
    // actually happen during the run and the snapshot+tail path is exercised.
    let snapshot_wal_mb = 1; // 1 MB
    let mut shard =
        Shard::open(&dir, dim, qv_hnsw::Metric::L2, snapshot_wal_mb, 0).expect("open shard");

    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    // Continue ids after whatever already recovered, so reruns don't collide.
    let mut id: u64 = shard.len() as u64;
    let mut counter: u16 = 0;

    loop {
        // Deterministic-ish vector from the id so the harness can recompute it.
        let v: Vec<f32> = (0..dim)
            .map(|d| ((id as usize + d) % 97) as f32 * 0.01)
            .collect();
        let wall = id; // monotone stand-in wall clock for the test
        counter = counter.wrapping_add(1);
        let hlc = Hlc::new(wall, counter);

        shard.upsert(hlc, id, &v, Vec::new()).expect("upsert");
        // Only now is the write durable + acked. Tell the parent.
        writeln!(out, "ACK {id}").expect("write ack");
        out.flush().expect("flush ack");

        id += 1;
    }
}
