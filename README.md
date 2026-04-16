# Inter-node Communication Benchmark

这个仓库用于在两台节点之间测试 GPU memcpy、NCCL send/recv、GPUDirect RDMA Copy，以及 RoCE QoS 限流配置对带宽的影响。

## 目录结构

```text
bench/                 benchmark CUDA 源码
gdr/                   GPUDirect RDMA Copy 支撑代码
scripts/nic/           NIC QoS 配置脚本
build/                 make 生成的二进制目录
run.sh                 cuda_bench / gdr_bench 启动脚本
run_gdr_qos.sh         gdr_qos_bench 启动脚本
```

## 编译

默认使用 `/usr/local/cuda` 和 `/usr/local/nccl`，如路径不同可以在编译时覆盖：

```bash
make CUDA_HOME=/usr/local/cuda NCCL_HOME=/usr/local/nccl
```

也可以只编译单个目标：

```bash
make cuda_bench
make gdr_bench
make gdr_qos_bench
```

如果 GPU 架构不是 `sm_90`，覆盖 `CUDA_ARCH`：

```bash
make CUDA_ARCH="-gencode arch=compute_80,code=sm_80"
```

## 通用运行配置

在两台节点上使用同一份代码，分别启动 `rank0` 和 `rank1`。推荐先通过环境变量配置，不必直接改脚本：

```bash
export NODE0=<rank0_ip>
export NODE1=<rank1_ip>
export NCCL_IB_HCA=mlx5_4
export BENCH_GPU_ID=4
export BENCH_BUF_GB=4
export BENCH_P2P_BUF_GB=2
export BENCH_ITERS=20
export BENCH_WARMUP=5
```

`BENCH_BUF_GB` 和 `BENCH_P2P_BUF_GB` 保留原变量名，但程序内部按 GiB 解释：`1` 个变量单位 = `1ULL << 30` bytes。输出带宽也使用 `GiB/s`。

如果 RoCE 环境需要指定 GID index：

```bash
export NCCL_IB_GID_INDEX=<gid_index>
```

## 运行 cuda_bench

`cuda_bench` 测试 rank0 本地 H2D/D2H 和 node0 到 node1 的 NCCL send/recv 并发干扰。

在 node0：

```bash
bash run.sh rank0
```

在 node1：

```bash
bash run.sh rank1
```

## 运行 gdr_bench

`gdr_bench` 使用 GPUDirect RDMA Copy 替代 rank0 本地 memcpy 路径。`BENCH_GDR_NIC` 默认从 `NCCL_IB_HCA` 去掉端口后得到，也可以显式指定。

```bash
export BENCH_GDR_NIC=mlx5_4
export BENCH_GDR_USE_ODP=0
```

在 node0：

```bash
BENCH_BIN=gdr_bench bash run.sh rank0
```

在 node1：

```bash
BENCH_BIN=gdr_bench bash run.sh rank1
```

## 配置 NIC QoS

NIC 配置脚本集中在 `scripts/nic/`。默认网卡名是 `enp25s0f0np0`，可以用 `IF` 覆盖。

启用 QoS 限流配置：

```bash
IF=enp25s0f0np0 bash scripts/nic/nic_qos.sh
```

恢复默认 NIC 配置：

```bash
IF=enp25s0f0np0 bash scripts/nic/nic_default.sh
```

当前 QoS 脚本使用的默认映射：

```text
NCCL: Traffic Class 104 -> DSCP 26 -> prio 6 -> TC6 -> 99%
GDR : Traffic Class 0   -> DSCP 0  -> prio 0 -> TC0 -> 1%
```

这些脚本依赖 Mellanox `mlnx_qos`，需要在对应节点上有 sudo 权限。交换机侧也需要按 DSCP/PCP 做队列映射、ETS/PFC 等配置，否则 benchmark 只能打标和验证，不能单独完成真实限流。

## 运行 gdr_qos_bench

先在需要的节点上应用 NIC QoS 配置，然后设置 QoS 相关变量：

```bash
export NCCL_IB_TC=104
export BENCH_GDR_IB_TC=0
export BENCH_QOS_MIN_NCCL_RATIO=0.99
```

在 node0：

```bash
bash run_gdr_qos.sh rank0
```

在 node1：

```bash
bash run_gdr_qos.sh rank1
```

`gdr_qos_bench` 会比较 solo NCCL 带宽和 concurrent GDR + NCCL 带宽。如果 `concurrent / solo` 达到 `BENCH_QOS_MIN_NCCL_RATIO`，结果显示 `PASS`；否则显示 `FAIL`，通常需要继续检查 NIC 和交换机 QoS 策略。

## 常用排查

确认 RDMA 设备名：

```bash
ibv_devinfo | grep hca_id
```

确认链路是 RoCE/Ethernet：

```bash
ibstat
```

清理构建产物：

```bash
make clean
```
