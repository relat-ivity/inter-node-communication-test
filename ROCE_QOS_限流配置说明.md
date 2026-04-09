# RoCE 网卡限流配置说明

## 1. 目的

本文说明如何在 `RoCE` 环境下，通过 `Traffic Class / DSCP` 区分 `NCCL` 和 `GDR` 流量，并在交换机侧使用 `DSCP/PCP -> queue`、`ETS`、`PFC` 做真正的带宽控制。

本仓库里的代码现在只做两件事：

- 给 `NCCL` 设置 `NCCL_IB_TC`
- 给 `GDR` 的 verbs QP 设置 `GRH traffic_class`

真正的限流仍然在网络设备上完成，不在 benchmark 代码里完成。

## 2. 适用前提

- `ibstat` 显示 `Link layer: Ethernet`
- 传输类型为 `RoCE` / `RoCEv2`
- 交换机支持按 `DSCP/PCP` 做 QoS

如果你的链路是原生 `InfiniBand`，则应该走 `SL/VL/OpenSM QoS`，不适用本文。

## 3. 基本思路

把两类流量区分开：

- `NCCL`：高优先级流量
- `GDR`：低优先级流量

实现路径如下：

1. 代码里给两类流量打不同的 `Traffic Class`
2. `Traffic Class` 对应不同的 `DSCP`
3. 交换机根据 `DSCP/PCP` 把流量放入不同队列
4. 通过 `ETS` 控制各队列带宽
5. 如有需要，通过 `PFC` 保护高优先级无损流量

## 4. 代码侧约定

当前仓库建议使用：

- `NCCL_IB_TC=104`
- `BENCH_GDR_IB_TC=0`

其中：

- `104 >> 2 = 26`，对应 `DSCP 26`
- `0 >> 2 = 0`，对应 `DSCP 0`

也就是说：

- `NCCL` 走较高优先级 DSCP
- `GDR` 走默认或较低优先级 DSCP

## 5. Traffic Class 和 DSCP 的关系

RoCE 的 `Traffic Class` 是 8 bit：

- 高 6 bit 通常作为 `DSCP`
- 低 2 bit 通常作为 `ECN`

常用换算方式：

```text
Traffic Class = DSCP << 2
DSCP = Traffic Class >> 2
```

例如：

- `DSCP 26 -> TC 104`
- `DSCP 0 -> TC 0`

## 6. 本仓库中的环境变量

QoS benchmark 相关变量：

- `NCCL_IB_TC`
- `BENCH_GDR_IB_TC`
- `BENCH_QOS_MIN_NCCL_RATIO`

说明：

- `NCCL_IB_TC` 用于 `NCCL`
- `BENCH_GDR_IB_TC` 用于 `GDR`
- `BENCH_QOS_MIN_NCCL_RATIO` 只是验证阈值，不负责实际限速

## 7. benchmark 作用

`gdr_qos_bench` 的作用是验证 QoS 是否生效，而不是执行限速。

它会检查：

1. `solo ncclSend` 带宽
2. `concurrent GDR + ncclSend` 带宽
3. `concurrent / solo` 是否达到目标比例

如果交换机侧 QoS 没配置好，程序仍能跑，但结果可能显示 `FAIL`。

## 8. 交换机侧真正要配什么

需要网络侧完成以下内容：

### 8.1 DSCP 分类

把不同 DSCP 映射到不同队列，例如：

- `DSCP 26` -> 高优先级队列
- `DSCP 0` -> 普通/低优先级队列

### 8.2 ETS

对不同队列设置带宽比例，例如：

- 高优先级队列：99%
- 低优先级队列：1%

这一步才是接近“99% / 1%”带宽控制的核心。

### 8.3 PFC

如果网络希望为高优先级 RoCE 队列提供无损保护，可以对对应优先级开启 `PFC`。

是否开启取决于你的网络设计，不是所有流量都必须开。

## 9. 推荐落地方式

建议按下面顺序做：

1. 在 benchmark 里固定两类流量的 `TC`
   - `NCCL_IB_TC=104`
   - `BENCH_GDR_IB_TC=0`
2. 在交换机上把 `DSCP 26` 和 `DSCP 0` 分别映射到不同队列
3. 给队列配置 `ETS`
   - 高优先级队列更大带宽
   - 低优先级队列较小带宽
4. 如有需要，再结合 `PFC`
5. 用 benchmark 验证 `NCCL` 带宽是否达到目标

## 10. 如何估算带宽比例

如果链路速率是 `400 Gbps`，希望给 `GDR` 只留 `1%`：

- 交换机侧应让低优先级队列大约只占 `1%`
- 高优先级队列大约占 `99%`

注意：

- 交换机上的百分比是链路侧带宽份额
- benchmark 看到的是应用层 `GB/s`
- 两者不会完全相等，通常需要实测后微调

## 11. 验证建议

建议同时检查：

- `NCCL_IB_TC` 是否正确导出
- `BENCH_GDR_IB_TC` 是否正确导出
- 交换机上 `DSCP -> queue` 是否正确
- `ETS` 是否真的按预期分配带宽
- benchmark 中 `concurrent / solo` 的比值是否接近目标

## 12. 注意事项

### 12.1 代码不会主动限速

当前代码不会：

- sleep
- token bucket
- 改 chunk 大小
- 改 `post_send` 速率

它只做：

- 打 `TC`
- 打印配置
- 输出 `PASS/FAIL`

### 12.2 交换机配置才是关键

如果交换机没有按 `DSCP` 正确分类和调度，那么代码里设置再多 `TC` 也不会带来实际带宽隔离效果。

### 12.3 优先确认 NIC 和交换机 DSCP/PCP 策略

不同厂商交换机、不同网卡驱动对：

- `TC`
- `DSCP`
- `PCP`
- `ETS`
- `PFC`

的默认映射可能不同，实际部署前要先核实现网策略。
