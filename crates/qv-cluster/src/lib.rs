//! `qv-cluster` — the distributed control and data plane for quorvec (M3+).
//!
//! This crate holds the consistent-hash ring, the openraft metadata state
//! machine, tunable-quorum replication, hinted handoff, read repair, and shard
//! transfer. None of it is built before M3, which has an explicit architect
//! coordination point (container-runtime contention). Empty placeholder for now
//! so the workspace layout matches the plan.

// M3+ modules land here.
