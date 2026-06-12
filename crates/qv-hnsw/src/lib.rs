//! `qv-hnsw` — the owned vector index layer for quorvec.
//!
//! This crate is pure compute: no I/O, no async. It exposes:
//!   - [`VectorIndex`]: the trait the rest of the system depends on,
//!   - [`BruteForceIndex`]: an exact oracle (M0; also the recall ground truth),
//!   - [`distance`]: L2 / cosine kernels with a SIMD path and scalar fallback.
//!
//! The HNSW graph (M1) implements [`VectorIndex`] alongside the brute-force
//! oracle.

pub mod distance;
pub mod index;

pub use distance::Metric;
pub use index::{BruteForceIndex, IndexError, VectorIndex};
