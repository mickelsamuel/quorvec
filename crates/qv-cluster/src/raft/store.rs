//! openraft 0.9.24 **durable** storage for the metadata plane: a file-backed log
//! store and a file-backed state-machine store wrapping [`MetaState`] (ruling
//! R6).
//!
//! Both stores follow the canonical `databendlabs/openraft` v0.9.24 trait shapes
//! (`storage-v2`: [`RaftLogStorage`] for the log + vote + committed,
//! [`RaftStateMachine`] for the applied state + snapshots). The difference from
//! the earlier in-memory build is that every Raft-durable fact now lands on disk
//! under `<data_dir>/raft/` and is read back on open, so a full-cluster restart
//! recovers the metadata plane (collections + shard map) without relying on a
//! peer to re-replicate it.
//!
//! **Durability contract (openraft requires it for correctness).** All log writes
//! — `append`, `save_vote`, `save_committed` — must be durable *before* the call
//! returns / the `LogFlushed` callback fires. We satisfy this by writing each file
//! atomically (temp file → fsync → rename → fsync the dir) before returning. The
//! metadata-plane log is tiny (topology + schema events, never vector data), so
//! rewriting the whole small log file per append is correct and cheap; a
//! segmented log is an unnecessary complication at this size.
//!
//! API note (escalation guard): every method signature here was taken from the
//! openraft 0.9.24 example source / the previously-compiling in-memory store, not
//! from memory of a newer API. If a future openraft bump changes these signatures
//! (e.g. `IOFlushed`/`truncate_after`/stream-`apply` in the 0.10 line), that is
//! the documented API-drift escalation point — do not improvise around it.

use std::collections::BTreeMap;
use std::fmt::Debug;
use std::io::Cursor;
use std::ops::RangeBounds;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use openraft::storage::{LogFlushed, LogState, RaftLogStorage, RaftStateMachine, Snapshot};
use openraft::{
    EntryPayload, LogId, OptionalSend, RaftLogId, RaftLogReader, RaftSnapshotBuilder, SnapshotMeta,
    StorageError, StorageIOError, StoredMembership, Vote,
};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use crate::meta::{MetaResponse, MetaState};
use crate::raft::{NodeId, TypeConfig};

type Node = crate::raft::Node;

// ---- durable file helpers --------------------------------------------------

/// Write `bytes` to `path` atomically and durably: temp file → fsync → rename →
/// fsync the directory entry. A crash leaves either the old file or the new one,
/// never a torn file. This is the primitive every Raft-durable write goes through.
fn atomic_write(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    let tmp = path.with_extension("tmp");
    {
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&tmp)?;
        f.write_all(bytes)?;
        f.flush()?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path)?;
    if let Some(parent) = path.parent() {
        if let Ok(dir) = std::fs::File::open(parent) {
            let _ = dir.sync_all();
        }
    }
    Ok(())
}

/// Read+deserialize a JSON file, or `None` if it does not exist.
fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> std::io::Result<Option<T>> {
    match std::fs::read(path) {
        Ok(bytes) => {
            let v = serde_json::from_slice(&bytes)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            Ok(Some(v))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e),
    }
}

fn io_err<E: std::error::Error + 'static>(e: E) -> StorageError<NodeId> {
    StorageError::from(StorageIOError::write(&e))
}

// ---- Log store -------------------------------------------------------------

/// On-disk image of the log store: the entries, the purge watermark, the vote,
/// and the committed id. Serialized as one JSON blob (the metadata log is tiny).
#[derive(Default, Serialize, Deserialize)]
struct LogDisk {
    last_purged_log_id: Option<LogId<NodeId>>,
    log: BTreeMap<u64, openraft::Entry<TypeConfig>>,
    committed: Option<LogId<NodeId>>,
    vote: Option<Vote<NodeId>>,
}

#[derive(Debug)]
struct LogStoreInner {
    dir: PathBuf,
    last_purged_log_id: Option<LogId<NodeId>>,
    log: BTreeMap<u64, openraft::Entry<TypeConfig>>,
    committed: Option<LogId<NodeId>>,
    vote: Option<Vote<NodeId>>,
}

impl LogStoreInner {
    fn log_path(&self) -> PathBuf {
        self.dir.join("raft-log.json")
    }

    /// Persist the entire log image atomically and durably. Called on the write
    /// path BEFORE we report completion, satisfying openraft's durability rule.
    // openraft's `StorageError` is a large enum by design; the trait methods that
    // call this must return it, so boxing here would only add an unwrap layer.
    #[allow(clippy::result_large_err)]
    fn persist(&self) -> Result<(), StorageError<NodeId>> {
        let disk = LogDisk {
            last_purged_log_id: self.last_purged_log_id,
            // Clone is fine: the metadata log is tiny.
            log: self.log.clone(),
            committed: self.committed,
            vote: self.vote,
        };
        let bytes = serde_json::to_vec(&disk).map_err(io_err)?;
        atomic_write(&self.log_path(), &bytes).map_err(io_err)
    }
}

/// File-backed Raft log store (durable; ruling R6). Persists log entries, the
/// purge watermark, the vote, and the committed id under `<data_dir>/raft/`.
#[derive(Clone, Debug)]
pub struct LogStore {
    inner: Arc<Mutex<LogStoreInner>>,
}

impl LogStore {
    /// Open (or create) the durable log store rooted at `dir` (the node's
    /// `<data_dir>/raft` directory). Recovers any prior log image on open.
    pub fn open(dir: impl AsRef<Path>) -> std::io::Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir)?;
        let log_path = dir.join("raft-log.json");
        let disk: LogDisk = read_json(&log_path)?.unwrap_or_default();
        Ok(Self {
            inner: Arc::new(Mutex::new(LogStoreInner {
                dir,
                last_purged_log_id: disk.last_purged_log_id,
                log: disk.log,
                committed: disk.committed,
                vote: disk.vote,
            })),
        })
    }
}

impl RaftLogReader<TypeConfig> for LogStore {
    async fn try_get_log_entries<RB: RangeBounds<u64> + Clone + Debug + OptionalSend>(
        &mut self,
        range: RB,
    ) -> Result<Vec<openraft::Entry<TypeConfig>>, StorageError<NodeId>> {
        let inner = self.inner.lock().await;
        let entries = inner.log.range(range).map(|(_, v)| v.clone()).collect();
        Ok(entries)
    }
}

impl RaftLogStorage<TypeConfig> for LogStore {
    type LogReader = Self;

    async fn get_log_state(&mut self) -> Result<LogState<TypeConfig>, StorageError<NodeId>> {
        let inner = self.inner.lock().await;
        let last = inner.log.iter().next_back().map(|(_, e)| *e.get_log_id());
        let last_purged = inner.last_purged_log_id;
        let last = match last {
            None => last_purged,
            Some(x) => Some(x),
        };
        Ok(LogState {
            last_purged_log_id: last_purged,
            last_log_id: last,
        })
    }

    async fn save_vote(&mut self, vote: &Vote<NodeId>) -> Result<(), StorageError<NodeId>> {
        let mut inner = self.inner.lock().await;
        inner.vote = Some(*vote);
        // Vote durability is a hard Raft-correctness requirement: persist before
        // returning so a crash cannot resurrect a forgotten vote.
        inner.persist()
    }

    async fn read_vote(&mut self) -> Result<Option<Vote<NodeId>>, StorageError<NodeId>> {
        let inner = self.inner.lock().await;
        Ok(inner.vote)
    }

    async fn save_committed(
        &mut self,
        committed: Option<LogId<NodeId>>,
    ) -> Result<(), StorageError<NodeId>> {
        let mut inner = self.inner.lock().await;
        inner.committed = committed;
        inner.persist()
    }

    async fn read_committed(&mut self) -> Result<Option<LogId<NodeId>>, StorageError<NodeId>> {
        let inner = self.inner.lock().await;
        Ok(inner.committed)
    }

    async fn append<I>(
        &mut self,
        entries: I,
        callback: LogFlushed<TypeConfig>,
    ) -> Result<(), StorageError<NodeId>>
    where
        I: IntoIterator<Item = openraft::Entry<TypeConfig>> + OptionalSend,
    {
        {
            let mut inner = self.inner.lock().await;
            for entry in entries {
                inner.log.insert(entry.get_log_id().index, entry);
            }
            // Persist BEFORE signalling completion: openraft treats the callback
            // as "this log prefix is durable", and Raft correctness depends on it.
            inner.persist()?;
        }
        callback.log_io_completed(Ok(()));
        Ok(())
    }

    async fn truncate(&mut self, log_id: LogId<NodeId>) -> Result<(), StorageError<NodeId>> {
        let mut inner = self.inner.lock().await;
        let keys: Vec<u64> = inner.log.range(log_id.index..).map(|(k, _)| *k).collect();
        for k in keys {
            inner.log.remove(&k);
        }
        inner.persist()
    }

    async fn purge(&mut self, log_id: LogId<NodeId>) -> Result<(), StorageError<NodeId>> {
        let mut inner = self.inner.lock().await;
        inner.last_purged_log_id = Some(log_id);
        let keys: Vec<u64> = inner.log.range(..=log_id.index).map(|(k, _)| *k).collect();
        for k in keys {
            inner.log.remove(&k);
        }
        inner.persist()
    }

    async fn get_log_reader(&mut self) -> Self::LogReader {
        self.clone()
    }
}

// ---- State machine ---------------------------------------------------------

/// A serialized point-in-time image of the metadata state machine.
#[derive(Debug, Clone)]
pub struct StoredSnapshot {
    pub meta: SnapshotMeta<NodeId, Node>,
    /// Serialized [`MetaState`] (the whole quorvec metadata).
    pub data: Vec<u8>,
}

/// The durable on-disk image of the state machine: the applied [`MetaState`], the
/// last-applied log id, and the last membership. Written atomically after every
/// batch of applies and on snapshot install, so a restart restores the exact
/// applied state without replaying from a peer.
#[derive(Default, Serialize, Deserialize)]
struct SmDisk {
    last_applied: Option<LogId<NodeId>>,
    last_membership: StoredMembership<NodeId, Node>,
    state: MetaState,
}

/// In-memory state-machine state plus the directory it persists to.
#[derive(Debug)]
pub struct StateMachineInner {
    dir: PathBuf,
    pub last_applied: Option<LogId<NodeId>>,
    pub last_membership: StoredMembership<NodeId, Node>,
    /// The authoritative quorvec metadata.
    pub state: MetaState,
}

impl StateMachineInner {
    fn sm_path(&self) -> PathBuf {
        self.dir.join("raft-sm.json")
    }

    /// Persist the applied state atomically and durably.
    #[allow(clippy::result_large_err)]
    fn persist(&self) -> Result<(), StorageError<NodeId>> {
        let disk = SmDisk {
            last_applied: self.last_applied,
            last_membership: self.last_membership.clone(),
            state: self.state.clone(),
        };
        let bytes = serde_json::to_vec(&disk).map_err(io_err)?;
        atomic_write(&self.sm_path(), &bytes).map_err(io_err)
    }
}

/// File-backed state-machine store (durable; ruling R6). Persists the applied
/// `MetaState` + last-applied + membership under `<data_dir>/raft/`, and recovers
/// them on open.
#[derive(Debug, Clone)]
pub struct StateMachineStore {
    inner: Arc<Mutex<StateMachineInner>>,
    snapshot: Arc<Mutex<Option<StoredSnapshot>>>,
    snapshot_idx: Arc<Mutex<u64>>,
}

impl StateMachineStore {
    /// Open (or create) the durable state-machine store rooted at `dir`. Recovers
    /// the applied state and the latest snapshot image on open.
    pub fn open(dir: impl AsRef<Path>) -> std::io::Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir)?;
        let sm_path = dir.join("raft-sm.json");
        let disk: SmDisk = read_json(&sm_path)?.unwrap_or_default();

        // Reconstruct an in-memory snapshot image from the applied state so
        // `get_current_snapshot` can serve it immediately after restart (the
        // state itself is the snapshot payload — same as `build_snapshot`).
        let snapshot = {
            let data = serde_json::to_vec(&disk.state).unwrap_or_default();
            let snapshot_id = match disk.last_applied {
                Some(last) => format!("{}-{}-restored", last.leader_id, last.index),
                None => "--restored".to_string(),
            };
            disk.last_applied.map(|_| StoredSnapshot {
                meta: SnapshotMeta {
                    last_log_id: disk.last_applied,
                    last_membership: disk.last_membership.clone(),
                    snapshot_id,
                },
                data,
            })
        };

        Ok(Self {
            inner: Arc::new(Mutex::new(StateMachineInner {
                dir,
                last_applied: disk.last_applied,
                last_membership: disk.last_membership,
                state: disk.state,
            })),
            snapshot: Arc::new(Mutex::new(snapshot)),
            snapshot_idx: Arc::new(Mutex::new(0)),
        })
    }

    /// A consistent read of the current metadata (used by ClusterInfo/routing).
    pub async fn read_state(&self) -> MetaState {
        self.inner.lock().await.state.clone()
    }
}

impl RaftSnapshotBuilder<TypeConfig> for StateMachineStore {
    async fn build_snapshot(&mut self) -> Result<Snapshot<TypeConfig>, StorageError<NodeId>> {
        let (data, last_applied, last_membership) = {
            let inner = self.inner.lock().await;
            let data = serde_json::to_vec(&inner.state)
                .map_err(|e| StorageError::from(StorageIOError::read_state_machine(&e)))?;
            (data, inner.last_applied, inner.last_membership.clone())
        };

        let snapshot_id = {
            let mut idx = self.snapshot_idx.lock().await;
            *idx += 1;
            if let Some(last) = last_applied {
                format!("{}-{}-{}", last.leader_id, last.index, *idx)
            } else {
                format!("--{}", *idx)
            }
        };

        let meta = SnapshotMeta {
            last_log_id: last_applied,
            last_membership,
            snapshot_id,
        };

        let stored = StoredSnapshot {
            meta: meta.clone(),
            data: data.clone(),
        };
        *self.snapshot.lock().await = Some(stored);

        Ok(Snapshot {
            meta,
            snapshot: Box::new(Cursor::new(data)),
        })
    }
}

impl RaftStateMachine<TypeConfig> for StateMachineStore {
    type SnapshotBuilder = Self;

    async fn applied_state(
        &mut self,
    ) -> Result<(Option<LogId<NodeId>>, StoredMembership<NodeId, Node>), StorageError<NodeId>> {
        let inner = self.inner.lock().await;
        Ok((inner.last_applied, inner.last_membership.clone()))
    }

    async fn apply<I>(&mut self, entries: I) -> Result<Vec<MetaResponse>, StorageError<NodeId>>
    where
        I: IntoIterator<Item = openraft::Entry<TypeConfig>> + OptionalSend,
    {
        let mut inner = self.inner.lock().await;
        let mut responses = Vec::new();
        for entry in entries {
            inner.last_applied = Some(*entry.get_log_id());
            match entry.payload {
                EntryPayload::Blank => responses.push(MetaResponse::new("blank")),
                EntryPayload::Normal(ref req) => {
                    let resp = inner.state.apply(req);
                    responses.push(resp);
                }
                EntryPayload::Membership(ref mem) => {
                    inner.last_membership =
                        StoredMembership::new(Some(*entry.get_log_id()), mem.clone());
                    responses.push(MetaResponse::new("membership"));
                }
            }
        }
        // Persist the applied state durably before returning. This is what makes
        // a full-cluster restart recover collections + membership without a peer.
        inner.persist()?;
        Ok(responses)
    }

    async fn get_snapshot_builder(&mut self) -> Self::SnapshotBuilder {
        self.clone()
    }

    async fn begin_receiving_snapshot(
        &mut self,
    ) -> Result<Box<Cursor<Vec<u8>>>, StorageError<NodeId>> {
        Ok(Box::new(Cursor::new(Vec::new())))
    }

    async fn install_snapshot(
        &mut self,
        meta: &SnapshotMeta<NodeId, Node>,
        snapshot: Box<Cursor<Vec<u8>>>,
    ) -> Result<(), StorageError<NodeId>> {
        let bytes = snapshot.into_inner();
        let new_state: MetaState = serde_json::from_slice(&bytes).map_err(|e| {
            StorageError::from(StorageIOError::read_snapshot(Some(meta.signature()), &e))
        })?;

        {
            let mut inner = self.inner.lock().await;
            inner.state = new_state;
            inner.last_applied = meta.last_log_id;
            inner.last_membership = meta.last_membership.clone();
            // A received snapshot is applied state: persist it durably too.
            inner.persist()?;
        }
        *self.snapshot.lock().await = Some(StoredSnapshot {
            meta: meta.clone(),
            data: bytes,
        });
        Ok(())
    }

    async fn get_current_snapshot(
        &mut self,
    ) -> Result<Option<Snapshot<TypeConfig>>, StorageError<NodeId>> {
        let snap = self.snapshot.lock().await;
        match &*snap {
            Some(s) => Ok(Some(Snapshot {
                meta: s.meta.clone(),
                snapshot: Box::new(Cursor::new(s.data.clone())),
            })),
            None => Ok(None),
        }
    }
}
