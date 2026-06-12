//! `qv-node` library surface: the building blocks the binary wires together and
//! that integration tests drive directly (config, the in-memory store, and the
//! gRPC service implementation).
//!
//! The binary (`main.rs`) is a thin shell over these. Keeping them in a library
//! lets integration tests construct a real server in-process without `#[path]`
//! source includes.

pub mod config;
pub mod service;
pub mod store;

pub use config::NodeConfig;
pub use service::QuorvecService;
pub use store::Store;
