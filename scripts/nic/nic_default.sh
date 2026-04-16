#!/usr/bin/env bash
# 还原默认 NIC 配置。可通过 `IF=<iface> bash scripts/nic/nic_default.sh` 覆盖网卡名。
set -euo pipefail

IF=${IF:-enp25s0f0np0}

sudo mlnx_qos -i "$IF" --dscp2prio='del,26,6'
sudo mlnx_qos -i "$IF" --dscp2prio='del,0,0'
sudo mlnx_qos -i "$IF" --trust=pcp
sudo mlnx_qos -i "$IF" -p 0,1,2,3,4,5,6,7
sudo mlnx_qos -i "$IF" -s vendor,vendor,vendor,vendor,vendor,vendor,vendor,vendor
sudo mlnx_qos -i "$IF" -t 0,0,0,0,0,0,0,0
