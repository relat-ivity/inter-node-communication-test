#!/usr/bin/env bash
# Run the benchmark across two nodes via RDMA (mlx5 / RoCEv2 + GPUDirect).
# NCCL is forced to use IB/RDMA only. If RDMA is unavailable, the run aborts.
#
# Usage:
#   bash run.sh
#   bash run.sh <node0_ip> <node1_ip>
#
# Edit defaults here first.

DEFAULT_NODE0=
DEFAULT_NODE1=
DEFAULT_NCCL_IB_HCA=mlx5_4
DEFAULT_BENCH_GPU_ID=4
DEFAULT_BENCH_BUF_GB=4
DEFAULT_BENCH_RS_BUF_GB=2
DEFAULT_BENCH_ITERS=20
DEFAULT_BENCH_WARMUP=5
DEFAULT_NCCL_IB_GID_INDEX=3
DEFAULT_NCCL_DEBUG=WARN
DEFAULT_NCCL_DEBUG_SUBSYS=INIT,NET
DEFAULT_NCCL_NET_GDR_LEVEL=SYS

set -euo pipefail

NODE0=${1:-$DEFAULT_NODE0}
NODE1=${2:-$DEFAULT_NODE1}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="${SCRIPT_DIR}/bench"

export NCCL_IB_HCA=${NCCL_IB_HCA:-$DEFAULT_NCCL_IB_HCA}
export BENCH_GPU_ID=${BENCH_GPU_ID:-$DEFAULT_BENCH_GPU_ID}
export BENCH_BUF_GB=${BENCH_BUF_GB:-$DEFAULT_BENCH_BUF_GB}
export BENCH_RS_BUF_GB=${BENCH_RS_BUF_GB:-$DEFAULT_BENCH_RS_BUF_GB}
export BENCH_ITERS=${BENCH_ITERS:-$DEFAULT_BENCH_ITERS}
export BENCH_WARMUP=${BENCH_WARMUP:-$DEFAULT_BENCH_WARMUP}

# Force NCCL to use RDMA/IB only.
export NCCL_NET=IB
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-$DEFAULT_NCCL_NET_GDR_LEVEL}
export NCCL_P2P_DISABLE=0
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-$DEFAULT_NCCL_IB_GID_INDEX}
export NCCL_DEBUG=${NCCL_DEBUG:-$DEFAULT_NCCL_DEBUG}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-$DEFAULT_NCCL_DEBUG_SUBSYS}

if [[ ! -x "$BINARY" ]]; then
    echo "[run.sh] Binary not found. Building first..."
    make -C "$SCRIPT_DIR"
fi

if [[ -z "$NODE0" || -z "$NODE1" ]]; then
    echo "ERROR: NODE0/NODE1 is empty. Set DEFAULT_NODE0 and DEFAULT_NODE1 at the top of run.sh,"
    echo "       or call: bash run.sh <node0_ip> <node1_ip>"
    exit 1
fi

if ! ibv_devinfo -d "${NCCL_IB_HCA%%:*}" &>/dev/null; then
    echo "ERROR: HCA '${NCCL_IB_HCA%%:*}' not found on this node."
    echo "       Run ibv_devinfo | grep hca_id to list available devices."
    exit 1
fi

echo "========================================================"
echo "  Transport : NCCL NET=IB (RDMA only)"
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
    -x NCCL_NET \
    -x NCCL_IB_DISABLE \
    -x NCCL_IB_HCA \
    -x NCCL_IB_GID_INDEX \
    -x NCCL_NET_GDR_LEVEL \
    -x NCCL_P2P_DISABLE \
    -x NCCL_DEBUG \
    -x NCCL_DEBUG_SUBSYS \
    --map-by node \
    "$BINARY"
