#!/usr/bin/env bash
# Start one benchmark process per node.
#
# Usage:
#   On node0: bash run.sh rank0
#   On node1: bash run.sh rank1
#
# Edit defaults here first.
# To run the GDR benchmark:
#   BENCH_BIN=gdr_bench bash run.sh rank0
#   BENCH_BIN=gdr_bench bash run.sh rank1

DEFAULT_NODE0=
DEFAULT_NODE1=
DEFAULT_BENCH_BIN=cuda_bench
DEFAULT_NCCL_IB_HCA=mlx5_4
DEFAULT_BENCH_GPU_ID=4
DEFAULT_BENCH_BUF_GB=4
DEFAULT_BENCH_P2P_BUF_GB=2
DEFAULT_BENCH_ITERS=20
DEFAULT_BENCH_WARMUP=5
DEFAULT_BENCH_MASTER_PORT=29500
DEFAULT_NCCL_IB_GID_INDEX=3
DEFAULT_NCCL_DEBUG=WARN
DEFAULT_NCCL_DEBUG_SUBSYS=INIT,NET
DEFAULT_NCCL_NET_GDR_LEVEL=SYS
DEFAULT_BENCH_GDR_USE_ODP=0

set -euo pipefail

ROLE=${1:-}
if [[ -z "$ROLE" ]]; then
    echo "Usage: bash run.sh rank0 | rank1"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_BIN=${BENCH_BIN:-$DEFAULT_BENCH_BIN}
case "$BENCH_BIN" in
    cuda_bench|gdr_bench)
        ;;
    *)
        echo "ERROR: BENCH_BIN must be cuda_bench or gdr_bench."
        exit 1
        ;;
esac

BINARY="${SCRIPT_DIR}/${BENCH_BIN}"

NODE0=${NODE0:-$DEFAULT_NODE0}
NODE1=${NODE1:-$DEFAULT_NODE1}

export NCCL_IB_HCA=${NCCL_IB_HCA:-$DEFAULT_NCCL_IB_HCA}
export BENCH_GPU_ID=${BENCH_GPU_ID:-$DEFAULT_BENCH_GPU_ID}
export BENCH_BUF_GB=${BENCH_BUF_GB:-$DEFAULT_BENCH_BUF_GB}
export BENCH_P2P_BUF_GB=${BENCH_P2P_BUF_GB:-${BENCH_AG_BUF_GB:-${BENCH_RS_BUF_GB:-$DEFAULT_BENCH_P2P_BUF_GB}}}
export BENCH_ITERS=${BENCH_ITERS:-$DEFAULT_BENCH_ITERS}
export BENCH_WARMUP=${BENCH_WARMUP:-$DEFAULT_BENCH_WARMUP}
export BENCH_WORLD_SIZE=2
export BENCH_MASTER_ADDR=${BENCH_MASTER_ADDR:-$NODE0}
export BENCH_MASTER_PORT=${BENCH_MASTER_PORT:-$DEFAULT_BENCH_MASTER_PORT}
DEFAULT_GDR_NIC_FROM_HCA=${NCCL_IB_HCA%%,*}
DEFAULT_GDR_NIC_FROM_HCA=${DEFAULT_GDR_NIC_FROM_HCA%%:*}
export BENCH_GDR_NIC=${BENCH_GDR_NIC:-$DEFAULT_GDR_NIC_FROM_HCA}
export BENCH_GDR_USE_ODP=${BENCH_GDR_USE_ODP:-$DEFAULT_BENCH_GDR_USE_ODP}

# Force NCCL to use RDMA/IB only.
export NCCL_NET=IB
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-$DEFAULT_NCCL_NET_GDR_LEVEL}
export NCCL_P2P_DISABLE=0
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-$DEFAULT_NCCL_IB_GID_INDEX}
export NCCL_DEBUG=${NCCL_DEBUG:-$DEFAULT_NCCL_DEBUG}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-$DEFAULT_NCCL_DEBUG_SUBSYS}

case "$ROLE" in
    rank0|0)
        export BENCH_RANK=0
        LOCAL_NODE_IP=$NODE0
        ;;
    rank1|1)
        export BENCH_RANK=1
        LOCAL_NODE_IP=$NODE1
        ;;
    *)
        echo "Usage: bash run.sh rank0 | rank1"
        exit 1
        ;;
esac

if [[ ! -x "$BINARY" ]]; then
    echo "[run.sh] ${BENCH_BIN} not found. Building first..."
    make -C "$SCRIPT_DIR" "$BENCH_BIN"
fi

if [[ -z "$NODE0" || -z "$NODE1" ]]; then
    echo "ERROR: NODE0/NODE1 is empty. Set defaults at the top of run.sh."
    exit 1
fi

if ! ibv_devinfo -d "${NCCL_IB_HCA%%:*}" &>/dev/null; then
    echo "ERROR: HCA '${NCCL_IB_HCA%%:*}' not found on this node."
    echo "       Run ibv_devinfo | grep hca_id to list available devices."
    exit 1
fi

if [[ "$BENCH_BIN" == "gdr_bench" ]] && ! ibv_devinfo -d "${BENCH_GDR_NIC}" &>/dev/null; then
    echo "ERROR: GDR NIC '${BENCH_GDR_NIC}' not found on this node."
    echo "       Set BENCH_GDR_NIC explicitly or check ibv_devinfo output."
    exit 1
fi

echo "========================================================"
echo "  Role      : ${ROLE}"
echo "  Binary    : ${BENCH_BIN}"
echo "  Local IP  : ${LOCAL_NODE_IP}"
echo "  Master    : ${BENCH_MASTER_ADDR}:${BENCH_MASTER_PORT}"
echo "  Transport : NCCL NET=IB (RDMA only)"
echo "  HCA       : ${NCCL_IB_HCA}"
if [[ "$BENCH_BIN" == "gdr_bench" ]]; then
    echo "  GDR NIC   : ${BENCH_GDR_NIC}"
    echo "  GDR ODP   : ${BENCH_GDR_USE_ODP}"
fi
echo "  GID index : ${NCCL_IB_GID_INDEX}"
echo "  GPU       : device ${BENCH_GPU_ID}"
echo "========================================================"
echo ""

exec "$BINARY"
