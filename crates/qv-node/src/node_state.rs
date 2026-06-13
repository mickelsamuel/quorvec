//! The node's shared runtime state: the cluster manager (metadata plane), this
//! node's durable shards, and the HLC clock for stamping writes.
//!
//! This is the single `Arc`-shared struct both gRPC services (the v1 client
//! service and the internal node-to-node service) hold. It is deliberately thin:
//! the metadata authority lives in [`qv_cluster::ClusterManager`], the durable
//! data lives in [`crate::shards::ShardStore`], and routing logic lives in
//! [`crate::router`].

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use qv_cluster::ClusterManager;
use qv_storage::{Hint, HintLog, HlcClock};

use crate::shards::ShardStore;

/// Shared node runtime state.
pub struct NodeState {
    pub node_id: u64,
    pub advertise_addr: String,
    /// Metadata plane (openraft): schemas, membership, shard map.
    pub cluster: ClusterManager,
    /// This node's durable shards.
    pub shards: ShardStore,
    /// Monotonic HLC for coordinator write stamping (M3 local; M4 cross-node).
    pub clock: HlcClock,
    /// Path to this node's hinted-handoff log (M4).
    hint_log_path: PathBuf,
    /// Serializes hint-log appends (the log is a single append-only file).
    hint_lock: Mutex<()>,
}

impl NodeState {
    pub fn new(
        node_id: u64,
        advertise_addr: String,
        cluster: ClusterManager,
        shards: ShardStore,
        data_dir: PathBuf,
    ) -> Arc<Self> {
        Arc::new(Self {
            node_id,
            advertise_addr,
            cluster,
            shards,
            clock: HlcClock::new(),
            hint_log_path: data_dir.join("hints").join("hints.log"),
            hint_lock: Mutex::new(()),
        })
    }

    /// Durably buffer a hint for a temporarily-unreachable replica (this node is
    /// the hint holder). fsync'd before returning.
    pub fn buffer_hint(&self, hint: &Hint) -> Result<(), qv_storage::HintError> {
        let _g = self.hint_lock.lock().unwrap();
        let mut log = HintLog::open(&self.hint_log_path)?;
        log.append(hint)
    }

    /// Read all buffered hints destined for `target_node`. Does not clear them.
    pub fn read_hints_for(&self, target_node: u64) -> Result<Vec<Hint>, qv_storage::HintError> {
        let _g = self.hint_lock.lock().unwrap();
        let all = HintLog::read_all(&self.hint_log_path)?;
        Ok(all
            .into_iter()
            .filter(|h| h.target_node == target_node)
            .collect())
    }

    /// All buffered hints (every target).
    pub fn read_all_hints(&self) -> Result<Vec<Hint>, qv_storage::HintError> {
        let _g = self.hint_lock.lock().unwrap();
        HintLog::read_all(&self.hint_log_path)
    }

    /// Rewrite the hint log keeping only `keep` (used after a partial replay).
    /// Atomic-enough for v1: rewrite the whole small log under the lock.
    pub fn rewrite_hints(&self, keep: &[Hint]) -> Result<(), qv_storage::HintError> {
        let _g = self.hint_lock.lock().unwrap();
        HintLog::clear(&self.hint_log_path)?;
        if keep.is_empty() {
            return Ok(());
        }
        let mut log = HintLog::open(&self.hint_log_path)?;
        for h in keep {
            log.append(h)?;
        }
        Ok(())
    }
}
