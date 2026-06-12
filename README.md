# quorvec

A distributed vector database in Rust: HNSW shards, a Raft metadata plane, and tunable-quorum replication for the data plane.

**Status: building in public, early. M0 (single-node baseline) is in. Numbers and benchmark curves come later, and only with the raw data behind them committed to this repo.** Nothing here is production-ready, and there are no performance claims yet — when there are, every one will trace back to a test or a CSV you can rerun.

## The problem

Most "I built a vector database" projects stop at a single node: an HNSW index plus some persistence, or a design doc describing how it *would* distribute. The interesting part — running a real multi-node cluster, watching what happens when you kill a node or partition the network, and producing standard recall-vs-QPS curves on recognized datasets — is where almost all of them stop. quorvec is an attempt to actually build and measure that part.

The architecture is the same family that Qdrant and Weaviate ship, and that is deliberate. This is not a new design; it is a credible, well-understood one, built solo to understand it from the inside. Credit to both projects for the pattern.

## Architecture sketch

Two planes, kept separate on purpose:

- **Metadata plane (Raft).** Cluster topology, collection schemas, and the shard map live in a Raft-replicated state machine. Membership changes and schema operations go through consensus. Data operations never touch Raft.
- **Data plane (leaderless, tunable quorum).** A collection is partitioned into shards by hashing the point id. Shards are placed on the cluster with a consistent-hash ring, and each shard is replicated to N nodes. Any node can coordinate a request. Writes are stamped with a hybrid logical clock and acknowledged at a tunable W (one / quorum / all), with last-writer-wins conflict resolution; reads collect R replicas and merge. This is the Dynamo-style approach.

Each shard owns an HNSW graph (the graph is hand-written here, not vendored; the SIMD distance kernels are vendored from simsimd) plus an append-only write-ahead log and periodic snapshots for durability.

### Crates

| Crate | Role |
|---|---|
| `qv-proto` | Generated gRPC types and service stubs (tonic + prost). |
| `qv-hnsw` | The owned HNSW graph, distance kernels, the `VectorIndex` trait, and a brute-force exact oracle. |
| `qv-storage` | Per-shard WAL, snapshots, and crash recovery. |
| `qv-cluster` | Ring placement, the Raft metadata state machine, quorum replication, hinted handoff, read repair, shard transfer. |
| `qv-node` | The node binary: config, gRPC wiring, lifecycle. |
| `qv-client` | A thin async Rust client for tests, benches, and the failure harness. |

The metadata plane, replication, and the failure and benchmark suites are later milestones. What runs today is a single node serving the v1 gRPC surface over an exact index.

## Build and run

Requires a Rust toolchain (the exact version is pinned in `rust-toolchain.toml`; rustup picks it up automatically). No system `protoc` is needed — a vendored one is used at build time.

```sh
# Build and test the whole workspace.
cargo test --workspace

# Run a single node.
cargo run -p qv-node -- --node-id 1 --listen 127.0.0.1:7000 --data-dir ./qv-data
```

The node speaks gRPC. The service definition is in [`proto/quorvec.proto`](proto/quorvec.proto). With [`grpcurl`](https://github.com/fullstorydev/grpcurl) you can create a collection, upsert vectors, and search:

```sh
grpcurl -plaintext -import-path proto -proto quorvec.proto \
  -d '{"name":"vectors","dim":64,"metric":"METRIC_L2"}' \
  127.0.0.1:7000 quorvec.v1.Quorvec/CreateCollection
```

## License

MIT. See [LICENSE](LICENSE).
