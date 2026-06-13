# quorvec consistency model

This document states exactly what quorvec's data plane guarantees and what its
test suite proves. It deliberately makes no claim the tests do not demonstrate.

## The two planes

quorvec has two independent consistency domains:

- **Metadata plane (strongly consistent, via Raft).** Cluster membership,
  collection schemas, and the derived shard map live in an openraft state
  machine. Every change (node join/leave, collection create/drop, shard-state
  transition) is committed through the Raft log, so all nodes converge on the
  same metadata in the same order. Raft gives this plane linearizable writes and
  a single leader at a time. Data-plane point operations never touch Raft.

- **Data plane (leaderless, tunable-quorum, eventually consistent).** Vector
  points are replicated Dynamo-style: any node can coordinate any request, writes
  go to a shard's N replicas and ack at W, reads gather R replicas and resolve by
  last-writer-wins. This plane is **not linearizable**. It is eventually
  consistent with last-writer-wins conflict resolution. That is by design and is
  the same family Qdrant and Weaviate ship.

## Data-plane mechanics

- **Placement.** A point's shard is `XXH64(id) % shard_count` (fixed per
  collection). A shard's replicas are the first N distinct physical nodes
  clockwise on a 64-vnode-per-node consistent-hash ring from the shard's home.

- **Versions / last-writer-wins.** The coordinator stamps every write with a
  hybrid logical clock (HLC: 48-bit wall ms + 16-bit counter). Each replica
  stores, per id, the winning HLC and whether the winner was a delete
  (tombstone). A write whose HLC is not strictly newer than the stored one is
  rejected as stale. Ties never flap: equal HLC is "not newer", so it loses.
  Deletes are HLC-stamped tombstones replicated exactly like upserts.

- **N / W / R.** N is the shard's replica count. The consistency level chooses
  the threshold: `ONE = 1`, `QUORUM = floor(N/2)+1`, `ALL = N`. A write acks the
  client once W replicas have converged (applied the write, or already hold a
  newer version). A read gathers R replicas and returns the highest-HLC value.

- **Hinted handoff.** If a replica is unreachable for a write, the coordinator
  does not block the quorum: it records a durable hint (the operation plus the
  target replica) in a WAL-style, CRC-checked, fsync'd hint log, and a background
  loop replays the hint to the target when it returns. A hint is durable before
  the write is acked, so an acked write at the chosen W is never silently lost to
  a transient replica outage.

- **Read repair.** A read that observes a replica behind the winning HLC (or
  missing the id) asynchronously pushes the winner to that replica. Repair is
  best-effort and fire-and-forget; it does not change the value the read returns.

- **Search.** Scatter-gather: one healthy replica per shard (primary first,
  falling forward through the replica list to the first reachable one), then a
  global top-k merge by ascending score. Because ANN search is approximate, a
  per-query R > 1 buys little, so search reads a single replica per shard — the
  point operations (upsert/get/delete) carry the consistency guarantees, not
  search.

- **Rebalancing (shard transfer).** Placement is deterministic from the Raft
  membership, so a node join or leave changes which nodes a shard's N replicas
  resolve to. A node that becomes a new replica for a shard it lacks marks itself
  `Syncing` (through Raft), streams the shard's full record set from an `Active`
  holder (`stream_records`), rebuilds its durable shard under LWW, then cuts over
  to `Active`. A node that is no longer a replica drops its copy only once every
  desired replica is `Active`. **Reads and search route to `Active` replicas
  only**, so a replica that is mid-transfer is never read from; meanwhile it still
  receives live writes, so a write that lands during a transfer is not lost (LWW
  reconciles the streamed-older against the live-newer record). The prior holders
  stay `Active` throughout, so an `Active` replica with the data is always
  available — the rebalance does not drop or stale a single request. Node
  leave / permanent death recovers under-replicated shards through this same
  transfer path.

## What `R = W = QUORUM` does and does not give you

With `W + R > N` (e.g. both QUORUM on N=3: 2 + 2 > 3) a read quorum and a write
quorum always intersect in at least one replica, so a QUORUM read observes the
latest QUORUM-acked write's value and returns it (and repairs the laggards). This
is **read-your-writes for QUORUM/QUORUM under the tested conditions** — it is NOT
general linearizability: concurrent writes are resolved by HLC LWW (the later
stamp wins, the earlier is discarded, with no merge), and ONE-level reads or
writes can observe or create staleness that converges later.

## What the tests actually prove

These are integration tests on a 5-process localhost cluster, driving the real
node binary over gRPC. (a)-(d) live in `crates/qv-node/tests/quorum_m4.rs`; the
rebalancing and durable-restart claims live in `tests/rebalance_m5.rs` and
`tests/restart_r6.rs`.

- **(a) W=QUORUM survives one replica down, and the hint replays.** With one of a
  shard's three replicas killed, a QUORUM upsert still succeeds (2 of 3 ack). When
  the killed replica restarts, the buffered hint is replayed to it, and a direct
  replica read against the recovered node shows the new value. *Proves:* quorum
  writes tolerate a single replica loss; hinted handoff is durable and replays.

- **(b) Concurrent conflicting upserts converge to the HLC winner.** Two
  coordinators write different values for the same id; after the writes settle,
  every replica of that shard holds the same single value — the higher-HLC one —
  with no split. *Proves:* LWW convergence, no divergent replicas.

- **(c) R=QUORUM read repairs a stale replica.** One replica is deliberately left
  behind a newer value held by the others; a QUORUM read returns the newer value
  and, shortly after, a direct read of the stale replica shows it has been
  repaired to the winning version. *Proves:* read returns the LWW winner and
  read-repair propagates it to laggards.

- **(d) Search tolerates one replica down per shard.** With a replica of a shard
  killed, search still returns correct results (including the queried id) by
  falling forward to a healthy replica. *Proves:* scatter-gather routes around a
  single per-shard replica loss.

- **(M5) Rebalancing under continuous load, zero failed requests.** With a client
  workload running throughout: adding a 6th node moves shards to it (it becomes a
  holder via the `stream_records` transfer) and every shard replica returns to
  `Active`, at 100% request success (0 failures, brief retries counted and
  printed); then killing a node permanently and `Leave`-ing it re-replicates the
  under-replicated shards to N `Active` healthy replicas, again at 100% request
  success. *Proves:* shard transfer on join and dead-replica recovery do not drop,
  stale, or fail client requests.

- **(R6) Full-cluster restart recovers the metadata plane.** All five nodes are
  killed and restarted on their data dirs; the recovered cluster reports identical
  collections and an identical shard map, every node agrees, and the populated
  data is still searchable. *Proves:* the durable Raft log + state-machine
  snapshot recover the metadata plane with no surviving peer.

## Honest limitations (do not claim beyond these)

- The data plane is **eventually consistent, LWW** — not linearizable, not
  serializable. There is no read-your-writes guarantee at ONE consistency.
- LWW **discards** the losing write; there is no vector-clock merge or
  sibling reconciliation. A clock skew large enough to invert intended order can
  let a logically-older write win — HLC bounds but does not eliminate this.
- Conflict resolution is **per point id**, not per request or per batch.
- Read repair and hint replay are **asynchronous and best-effort**: convergence
  is eventual, not bounded to a deadline by these tests.
- The metadata-plane Raft log **and** state-machine snapshot are **durable on
  disk** (under `<data_dir>/raft/`, fsync'd before each acknowledged write). A
  single restarted node resumes from its own persisted log, and a **full-cluster
  restart** (every node killed and restarted) recovers all collections and the
  shard map without relying on a surviving peer — proven by
  `crates/qv-node/tests/restart_r6.rs`. The *data* plane is likewise durable on
  disk (WAL + snapshots).
- **Shard transfer is `stream_records` only.** A transfer ships the whole shard's
  record set and rebuilds the target's index. The faster `wal_delta` (ship only
  the WAL diff) and `snapshot` transfer modes are labeled v1.1 and not built. The
  stream is currently a single unary response (fine at the tested scale); chunked
  server-streaming for very large shards is a labeled follow-up.
- These properties are demonstrated on a **single-host 5-process cluster**, not a
  multi-host deployment, and under the specific fault injections above — not an
  exhaustive partition/linearizability suite (that is the M6 failure-testing
  work, which will also document the data plane's expected non-linearizable
  verdicts under a Maelstrom register workload).
