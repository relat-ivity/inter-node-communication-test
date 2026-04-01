#!/usr/bin/env bash
# Run the benchmark across two nodes via RDMA (mlx5 / RoCEv2 + GPUDirect).
# TCP fallback is intentionally disabled — if RDMA is unavailable the run aborts.
#
# Usage:
#   bash run.sh <node0_ip> <node1_ip>
#
# Required env (set before calling):
#   NCCL_IB_HCA       mlx5 device to use, e.g. mlx5_0  or  mlx5_0:1
#                     check available devices: ibv_devinfo | grep hca_id
#
# Optional env:
#   BENCH_GPU_ID      GPU device index on each node     (default 0)
#   BENCH_BUF_GB      D2H/H2D buffer size in GB         (default 4)
#   BENCH_RS_BUF_GB   ReduceScatter buffer in GB        (default 2)
#   BENCH_ITERS       iterations per phase              (default 20)
#   BENCH_WARMUP      warmup iterations                 (default 5)
#   NCCL_IB_GID_INDEX GID index for RoCEv2             (default 3, verify with show_gids)
#   NCCL_DEBUG        NCCL log level                    (default WARN)

set -euo pipefail

NODE0=${1:?"Usage: $0 <node0_ip> <node1_ip>"}
NODE1=${2:?"Usage: $0 <node0_ip> <node1_ip>"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="${SCRIPT_DIR}/bench"

if [[ ! -x "$BINARY" ]]; then
    echo "[run.sh] Binary not found. Building first..."
    make -C "$SCRIPT_DIR"
fi

# ── RDMA is mandatory ─────────────────────────────────────────────────────────
if [[ -z "${NCCL_IB_HCA:-}" ]]; then
    echo "ERROR: NCCL_IB_HCA is not set."
    echo "       Run  ibv_devinfo | grep hca_id  to list available mlx5 devices."
    echo "       Then:  NCCL_IB_HCA=mlx5_0 bash run.sh ..."
    exit 1
fi

# Verify the specified HCA exists on this machine
if ! ibv_devinfo -d "${NCCL_IB_HCA%%:*}" &>/dev/null; then
    echo "ERROR: HCA '${NCCL_IB_HCA%%:*}' not found on this node."
    echo "       Run  ibv_devinfo | grep hca_id  to list available devices."
    exit 1
fi

# ── env ───────────────────────────────────────────────────────────────────────
export BENCH_GPU_ID=${BENCH_GPU_ID:-0}
export BENCH_BUF_GB=${BENCH_BUF_GB:-4}
export BENCH_RS_BUF_GB=${BENCH_RS_BUF_GB:-2}
export BENCH_ITERS=${BENCH_ITERS:-20}
export BENCH_WARMUP=${BENCH_WARMUP:-5}

# Force RDMA — no TCP fallback
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5        # GPUDirect RDMA: NIC reads GPU memory directly
export NCCL_P2P_DISABLE=0
export NCCL_IB_HCA                 # already set by caller
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}   # 3 = RoCEv2
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

echo "========================================================"
echo "  Transport : RDMA — mlx5 / RoCEv2 + GPUDirect"
echo "  HCA       : ${NCCL_IB_HCA}"
echo "  GID index : ${NCCL_IB_GID_INDEX}"
echo "  GPU       : device ${BENCH_GPU_ID}"
echo "  Nodes     : ${NODE0}  ${NODE1}"
echo "========================================================"
echo ""

mpirun \
    -np 2 \
    -host "${NODE0},${NODE1}" \
    -x BENCH_GPU_ID \
    -x BENCH_BUF_GB \
    -x BENCH_RS_BUF_GB \
    -x BENCH_ITERS \
    -x BENCH_WARMUP \
    -x NCCL_IB_DISABLE \
    -x NCCL_IB_HCA \
    -x NCCL_IB_GID_INDEX \
    -x NCCL_NET_GDR_LEVEL \
    -x NCCL_P2P_DISABLE \
    -x NCCL_DEBUG \
    --map-by node \
    "$BINARY"
