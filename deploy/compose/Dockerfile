# Multi-stage build for the quorvec node binary.
#
# Stage 1 builds a release `qv-node` with the pinned toolchain (from
# rust-toolchain.toml). protoc is supplied by the vendored crate, so no system
# protobuf-compiler is needed. Stage 2 is a slim runtime image.
#
# This image is authored for the CI compose smoke job (it runs on GitHub's
# ubuntu runners, where Docker is free). Local dev/test uses the 5-process
# localhost cluster instead (architect ruling R5: no local Docker / WSL2).

# ---- builder ---------------------------------------------------------------
FROM rust:1.96-slim-bookworm AS builder

WORKDIR /build

# Copy the workspace sources. (Cargo.lock pins the full graph.) tools/ and
# maelstrom/ are added in later milestones (M5–M7); they are not workspace
# members yet, so they are intentionally not copied here.
COPY Cargo.toml Cargo.lock rust-toolchain.toml deny.toml ./
COPY proto ./proto
COPY crates ./crates

# Build only the node binary in release.
RUN cargo build --release --bin qv-node

# ---- runtime ---------------------------------------------------------------
FROM debian:bookworm-slim AS runtime

# A non-root user for the service.
RUN useradd --system --create-home --uid 10001 quorvec

COPY --from=builder /build/target/release/qv-node /usr/local/bin/qv-node
COPY deploy/compose/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

USER quorvec
WORKDIR /home/quorvec

# gRPC listen port (overridden per-node by compose env).
EXPOSE 7000

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
