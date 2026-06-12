//! Generated quorvec gRPC types and service stubs.
//!
//! The code in [`v1`] is produced at build time from `proto/quorvec.proto` by
//! `tonic-prost-build`. Nothing here is hand-written; do not edit the generated
//! surface. The proto file is the locked v1 contract.

#![allow(clippy::all)]
#![allow(rustdoc::all)]

/// quorvec.v1 package: messages, enums, client, and server stubs.
pub mod v1 {
    tonic::include_proto!("quorvec.v1");
}

/// quorvec.internal package: the private node-to-node surface (metadata-plane
/// Raft transport + data-plane replica RPCs). NOT the locked v1 contract.
pub mod internal {
    tonic::include_proto!("quorvec.internal");
}

// Re-export the most-used items at the crate root for convenience.
pub use v1::quorvec_client::QuorvecClient;
pub use v1::quorvec_server::{Quorvec, QuorvecServer};

pub use internal::quorvec_internal_client::QuorvecInternalClient;
pub use internal::quorvec_internal_server::{QuorvecInternal, QuorvecInternalServer};
