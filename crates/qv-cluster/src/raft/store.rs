//! openraft 0.9.24 storage for the metadata plane: an in-memory log store and a
//! state-machine store wrapping [`MetaState`].
//!
//! Both stores follow the canonical `databendlabs/openraft` v0.9.24 example
//! (`examples/memstore` log store + `raft-kv-memstore` state machine). The log
//! store is in-memory: the metadata-plane Raft log is tiny (topology + schema
//! events, not data) and is reconstructed by peers from a snapshot + tail on
//! restart; durable on-disk Raft log persistence is a labeled future item and is
//! out of M3 scope. The state machine applies [`MetaRequest`]s to [`MetaState`]
//! and serializes the whole `MetaState` into each Raft snapshot.
//!
//! API note (escalation guard): every method signature here was taken from the
//! openraft 0.9.24 example source, not from memory. If a future openraft bump
//! changes these signatures, that is the documented API-drift escalation point —
//! do not improvise around it.

use std::collections::BTreeMap;
use std::fmt::Debug;
use std::io::Cursor;
use std::ops::RangeBounds;
use std::sync::Arc;

use openraft::storage::{LogFlushed, LogState, RaftLogStorage, RaftStateMachine, Snapshot};
use openraft::{
    EntryPayload, LogId, OptionalSend, RaftLogId, RaftLogReader, RaftSnapshotBuilder, SnapshotMeta,
    StorageError, StorageIOError, StoredMembership, Vote,
};
use tokio::sync::Mutex;

use crate::meta::{MetaResponse, MetaState};
use crate::raft::{NodeId, TypeConfig};

type Node = crate::raft::Node;

// ---- Log store -------------------------------------------------------------

/// In-memory Raft log store (canonical openraft 0.9.24 memstore shape).
#[derive(Clone, Debug, Default)]
pub struct LogStore {
    inner: Arc<Mutex<LogStoreInner>>,
}

#[derive(Debug, Default)]
struct LogStoreInner {
    last_purged_log_id: Option<LogId<NodeId>>,
    log: BTreeMap<u64, openraft::Entry<TypeConfig>>,
    committed: Option<LogId<NodeId>>,
    vote: Option<Vote<NodeId>>,
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
        Ok(())
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
        Ok(())
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
        }
        // In-memory append is durable the instant it returns; signal completion.
        callback.log_io_completed(Ok(()));
        Ok(())
    }

    async fn truncate(&mut self, log_id: LogId<NodeId>) -> Result<(), StorageError<NodeId>> {
        let mut inner = self.inner.lock().await;
        let keys: Vec<u64> = inner.log.range(log_id.index..).map(|(k, _)| *k).collect();
        for k in keys {
            inner.log.remove(&k);
        }
        Ok(())
    }

    async fn purge(&mut self, log_id: LogId<NodeId>) -> Result<(), StorageError<NodeId>> {
        let mut inner = self.inner.lock().await;
        inner.last_purged_log_id = Some(log_id);
        let keys: Vec<u64> = inner.log.range(..=log_id.index).map(|(k, _)| *k).collect();
        for k in keys {
            inner.log.remove(&k);
        }
        Ok(())
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

/// The metadata state-machine store: the applied [`MetaState`], the last-applied
/// log id, the last membership, and the most recent snapshot.
#[derive(Debug, Default)]
pub struct StateMachineInner {
    pub last_applied: Option<LogId<NodeId>>,
    pub last_membership: StoredMembership<NodeId, Node>,
    /// The authoritative quorvec metadata.
    pub state: MetaState,
}

/// State-machine store handle (clonable; shares the inner state + snapshot).
#[derive(Debug, Clone, Default)]
pub struct StateMachineStore {
    inner: Arc<Mutex<StateMachineInner>>,
    snapshot: Arc<Mutex<Option<StoredSnapshot>>>,
    snapshot_idx: Arc<Mutex<u64>>,
}

impl StateMachineStore {
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
