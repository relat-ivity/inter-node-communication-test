#!/usr/bin/env bash
# 网卡 TC 限流脚本。可通过 `IF=<iface> bash scripts/nic/nic_qos.sh` 覆盖网卡名。
set -euo pipefail

IF=${IF:-enp25s0f0np0}

# 1. 让主机自己管 QoS
sudo mlnx_qos -i "$IF" -d os

# 2. 按 DSCP 做信任
sudo mlnx_qos -i "$IF" --trust=dscp

# 3. DSCP 映射到 priority
# NCCL: DSCP 26 -> prio 6
# GDR : DSCP 0  -> prio 0
sudo mlnx_qos -i "$IF" --dscp2prio='set,26,6'
sudo mlnx_qos -i "$IF" --dscp2prio='set,0,0'

# 4. priority 映射到 TC
# prio0 -> TC0
# prio6 -> TC6
sudo mlnx_qos -i "$IF" -p 0,0,0,0,0,0,6,0

# 所有 TC 设成 ETS，然后给 TC0 1%，TC6 99%
sudo mlnx_qos -i "$IF" -s ets,ets,ets,ets,ets,ets,ets,ets -t 1,0,0,0,0,0,99,0
