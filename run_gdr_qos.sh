#!/usr/bin/env bash
# Start one QoS benchmark process per node.
#
# Usage:
#   On node0: bash run_gdr_qos.sh rank0
#   On node1: bash run_gdr_qos.sh rank1

DEFAULT_NODE0=
DEFAULT_NODE1=
DEFAULT_NCCL_IB_HCA=mlx5_0
DEFAULT_BENCH_GPU_ID=0
# Buffer defaults are in KiB. Use $((1 << 20)) for 1 GiB.
DEFAULT_BENCH_BUF_KB=4
DEFAULT_BENCH_P2P_BUF_KB=4
DEFAULT_BENCH_ITERS=10
DEFAULT_BENCH_WARMUP=100
DEFAULT_BENCH_MASTER_PORT=29500
DEFAULT_NCCL_DEBUG=WARN
DEFAULT_NCCL_DEBUG_SUBSYS=INIT,NET
DEFAULT_NCCL_NET_GDR_LEVEL=SYS
DEFAULT_BENCH_GDR_USE_ODP=0
DEFAULT_NCCL_IB_TC=104
DEFAULT_BENCH_GDR_IB_TC=0
DEFAULT_BENCH_QOS_MIN_NCCL_RATIO=0.99

set -euo pipefail

ROLE=${1:-}
if [[ -z "$ROLE" ]]; then
    echo "Usage: bash run_gdr_qos.sh rank0 | rank1"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
BINARY="${BUILD_DIR}/gdr_qos_bench"

NODE0=${NODE0:-$DEFAULT_NODE0}
NODE1=${NODE1:-$DEFAULT_NODE1}

export NCCL_IB_HCA=${NCCL_IB_HCA:-$DEFAULT_NCCL_IB_HCA}
export BENCH_GPU_ID=${BENCH_GPU_ID:-$DEFAULT_BENCH_GPU_ID}
if [[ -n "${BENCH_BUF_KB:-}" ]]; then
    export BENCH_BUF_KB
elif [[ -n "${BENCH_BUF_GB:-}" ]]; then
    export BENCH_BUF_GB
else
    export BENCH_BUF_KB=$DEFAULT_BENCH_BUF_KB
fi
if [[ -n "${BENCH_P2P_BUF_KB:-}" ]]; then
    export BENCH_P2P_BUF_KB
elif [[ -n "${BENCH_AG_BUF_KB:-}" ]]; then
    export BENCH_P2P_BUF_KB=$BENCH_AG_BUF_KB
elif [[ -n "${BENCH_RS_BUF_KB:-}" ]]; then
    export BENCH_P2P_BUF_KB=$BENCH_RS_BUF_KB
elif [[ -n "${BENCH_P2P_BUF_GB:-}" || -n "${BENCH_AG_BUF_GB:-}" || -n "${BENCH_RS_BUF_GB:-}" ]]; then
    export BENCH_P2P_BUF_GB=${BENCH_P2P_BUF_GB:-${BENCH_AG_BUF_GB:-${BENCH_RS_BUF_GB:-}}}
else
    export BENCH_P2P_BUF_KB=$DEFAULT_BENCH_P2P_BUF_KB
fi
export BENCH_ITERS=${BENCH_ITERS:-$DEFAULT_BENCH_ITERS}
export BENCH_WARMUP=${BENCH_WARMUP:-$DEFAULT_BENCH_WARMUP}
export BENCH_WORLD_SIZE=2
export BENCH_MASTER_ADDR=${BENCH_MASTER_ADDR:-$NODE0}
export BENCH_MASTER_PORT=${BENCH_MASTER_PORT:-$DEFAULT_BENCH_MASTER_PORT}
DEFAULT_GDR_NIC_FROM_HCA=${NCCL_IB_HCA%%,*}
DEFAULT_GDR_NIC_FROM_HCA=${DEFAULT_GDR_NIC_FROM_HCA%%:*}
export BENCH_GDR_NIC=${BENCH_GDR_NIC:-$DEFAULT_GDR_NIC_FROM_HCA}
export BENCH_GDR_USE_ODP=${BENCH_GDR_USE_ODP:-$DEFAULT_BENCH_GDR_USE_ODP}
export BENCH_GDR_IB_TC=${BENCH_GDR_IB_TC:-$DEFAULT_BENCH_GDR_IB_TC}
export BENCH_QOS_MIN_NCCL_RATIO=${BENCH_QOS_MIN_NCCL_RATIO:-$DEFAULT_BENCH_QOS_MIN_NCCL_RATIO}

# Force NCCL to use RDMA/IB only.
export NCCL_NET=IB
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-$DEFAULT_NCCL_NET_GDR_LEVEL}
export NCCL_P2P_DISABLE=0
export NCCL_IB_TC=${NCCL_IB_TC:-$DEFAULT_NCCL_IB_TC}
if [[ -n "${NCCL_IB_GID_INDEX:-}" ]]; then
    export NCCL_IB_GID_INDEX
else
    unset NCCL_IB_GID_INDEX || true
fi
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
        echo "Usage: bash run_gdr_qos.sh rank0 | rank1"
        exit 1
        ;;
esac

if [[ ! -x "$BINARY" ]]; then
    echo "[run_gdr_qos.sh] gdr_qos_bench not found. Building first..."
    make -C "$SCRIPT_DIR" gdr_qos_bench
fi

if [[ -z "$NODE0" || -z "$NODE1" ]]; then
    echo "ERROR: NODE0/NODE1 is empty. Set defaults at the top of run_gdr_qos.sh."
    exit 1
fi

if ! ibv_devinfo -d "${NCCL_IB_HCA%%:*}" &>/dev/null; then
    echo "ERROR: HCA '${NCCL_IB_HCA%%:*}' not found on this node."
    echo "       Run ibv_devinfo | grep hca_id to list available devices."
    exit 1
fi

if ! ibv_devinfo -d "${BENCH_GDR_NIC}" &>/dev/null; then
    echo "ERROR: GDR NIC '${BENCH_GDR_NIC}' not found on this node."
    echo "       Set BENCH_GDR_NIC explicitly or check ibv_devinfo output."
    exit 1
fi

if [[ "${NCCL_IB_TC}" == "${BENCH_GDR_IB_TC}" ]]; then
    echo "ERROR: NCCL_IB_TC and BENCH_GDR_IB_TC must differ for RoCE QoS validation."
    exit 1
fi

echo "========================================================"
echo "  Role          : ${ROLE}"
echo "  Binary        : gdr_qos_bench"
echo "  Local IP      : ${LOCAL_NODE_IP}"
echo "  Master        : ${BENCH_MASTER_ADDR}:${BENCH_MASTER_PORT}"
echo "  Transport     : NCCL NET=IB (verbs/RoCE)"
echo "  HCA           : ${NCCL_IB_HCA}"
echo "  GDR NIC       : ${BENCH_GDR_NIC}"
echo "  GDR ODP       : ${BENCH_GDR_USE_ODP}"
echo "  NCCL TC       : ${NCCL_IB_TC} (DSCP=$((NCCL_IB_TC >> 2)))"
echo "  GDR TC        : ${BENCH_GDR_IB_TC} (DSCP=$((BENCH_GDR_IB_TC >> 2)))"
echo "  NCCL target   : ${BENCH_QOS_MIN_NCCL_RATIO} of solo BW"
echo "  GID index     : ${NCCL_IB_GID_INDEX:-auto}"
echo "  GPU           : device ${BENCH_GPU_ID}"
echo "========================================================"
echo ""

exec "$BINARY"
