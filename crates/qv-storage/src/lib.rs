//! `qv-storage` — durability for a single shard: WAL, snapshots, recovery.
//!
//! - [`hlc`]: the hybrid logical clock timestamp used as the per-point version.
//! - [`wal`]: the append-only, CRC-checked, torn-tail-safe write-ahead log.
//! - [`snapshot`]: atomic point-in-time index images + covered WAL offset.
//! - [`shard`]: the durable shard tying index + WAL + snapshots together, with
//!   crash recovery on open.

pub mod hints;
pub mod hlc;
pub mod shard;
pub mod snapshot;
pub mod wal;

pub use hints::{Hint, HintError, HintLog};
pub use hlc::{Hlc, HlcClock};
pub use shard::{PointVersion, Shard, ShardError, WriteOutcome};
pub use snapshot::Snapshot;
pub use wal::{Wal, WalError, WalOp, WalRecord};
