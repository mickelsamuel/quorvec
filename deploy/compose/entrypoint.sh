#!/usr/bin/env bash
# quorvec node container entrypoint.
#
# Driven entirely by env vars set per-service in the compose file:
#   QV_NODE_ID     numeric node id (required)
#   QV_LISTEN      listen/advertise address, e.g. 0.0.0.0:7000 (required)
#   QV_ADVERTISE   address peers use, e.g. qv1:7000 (defaults to QV_LISTEN)
#   QV_DATA_DIR    shard data dir (default /home/quorvec/data)
#   QV_BOOTSTRAP   "1" to form a new cluster (the seed node)
#   QV_JOIN        seed-only: comma-separated id@host:port peers to admit
set -euo pipefail

: "${QV_NODE_ID:?QV_NODE_ID is required}"
: "${QV_LISTEN:?QV_LISTEN is required}"
QV_DATA_DIR="${QV_DATA_DIR:-/home/quorvec/data}"

args=(--node-id "${QV_NODE_ID}" --listen "${QV_LISTEN}" --data-dir "${QV_DATA_DIR}")

if [[ -n "${QV_ADVERTISE:-}" ]]; then
  args+=(--advertise "${QV_ADVERTISE}")
fi
if [[ "${QV_BOOTSTRAP:-0}" == "1" ]]; then
  args+=(--bootstrap)
fi
if [[ -n "${QV_JOIN:-}" ]]; then
  args+=(--join "${QV_JOIN}")
fi

mkdir -p "${QV_DATA_DIR}"
exec qv-node "${args[@]}"
