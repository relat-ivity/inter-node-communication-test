# IB 网卡限流配置说明

## 1. 目的

本文说明如何在原生 InfiniBand 网络中，基于 `SL -> VL -> QoS` 的机制对不同业务流量做带宽控制。

本仓库里的代码目前只负责给不同流量打不同的 `SL`：

- `NCCL` 使用高优先级 `SL`
- `GDR` 使用低优先级 `SL`

真正的限流不在 benchmark 代码里完成，而是在子网管理器 `OpenSM / UFM` 的 QoS 配置里完成。

## 2. 适用前提

- 网络类型必须是原生 InfiniBand
- 交换机和 HCA 支持 QoS / Enhanced QoS
- 子网管理器使用 OpenSM 或 UFM

如果链路是 `RoCE`，则不能按 `VL` 做这一套，需要改成交换机上的 `DSCP/PCP + PFC/ETS` 方案。

## 3. 基本思路

目标是把两类流量分开：

- `NCCL`：高优先级，尽量拿到主要带宽
- `GDR`：低优先级，被限制在较小带宽范围内

实现链路如下：

1. 代码里给两类流量设置不同的 `SL`
2. 在 `OpenSM` 里把不同 `SL` 映射到不同 `VL`
3. 用 `VL Arbitration` 设置不同 `VL` 的调度优先级
4. 如需接近硬性的百分比限制，再用 `Enhanced QoS` 对某个 `SL` 设置带宽上限

## 4. 代码侧配置

建议约定如下：

- `NCCL_IB_SL=0`
- `BENCH_GDR_IB_SL=1`

也就是：

- `NCCL` 走 `SL0`
- `GDR` 走 `SL1`

这样后续在 `OpenSM` 中就可以把：

- `SL0 -> VL0`
- `SL1 -> VL1`

## 5. OpenSM 中的 SL/VL 配置

下面是一个最小可用的思路示例，可放在 `opensm.opts` 或对应的 OpenSM/UFM QoS 配置里。

```conf
qos_ca_max_vls 2
qos_ca_high_limit 0
qos_ca_vlarb_high 0:64
qos_ca_vlarb_low 1:64
qos_ca_sl2vl 0,1,15,15,15,15,15,15,15,15,15,15,15,15,15,15

qos_swe_max_vls 2
qos_swe_high_limit 0
qos_swe_vlarb_high 0:64
qos_swe_vlarb_low 1:64
qos_swe_sl2vl 0,1,15,15,15,15,15,15,15,15,15,15,15,15,15,15
```

含义如下：

- `SL0 -> VL0`
- `SL1 -> VL1`
- `VL0` 走 high arbitration table
- `VL1` 走 low arbitration table

这一步能做的是“优先级隔离”和“倾斜调度”，但还不是严格的带宽上限。

## 6. 真正的带宽限速：Enhanced QoS

如果目标是类似：

- `NCCL` 占 99%
- `GDR` 占 1%

那么只靠 `VL Arbitration` 一般不够，需要再启用 `Enhanced QoS`，给低优先级 `SL` 设置速率上限。

### 6.1 配置文件示例

`Enhanced QoS` 的规则单位通常按 `1 Mbps` 配置。

例如链路是 `400 Gbps`，希望把 `GDR` 所在的 `SL1` 先限制在约 `1%`，可以从 `4000 Mbps` 起步：

```conf
BW_NAMES
gdr_cap = 4000

BW_RULES
default = 1:gdr_cap
```

含义：

- `SL1` 的带宽上限设置为 `4000 Mbps`
- 其他未声明的 `SL` 默认不限制

### 6.2 限制某个指定端口

如果不想全网统一限制，而是只限制某个 HCA 端口，可以写成：

```conf
BW_NAMES
gdr_cap = 4000

BW_RULES
0x<port_guid> = 1:gdr_cap
```

其中：

- `0x<port_guid>` 是目标物理端口的 GUID
- `1:gdr_cap` 表示对 `SL1` 应用该带宽上限

## 7. 如何估算限速值

建议公式：

```text
GDR 上限 Mbps = 链路速率(Gbps) × 1000 × GDR保留比例
```

例如：

- 200G 链路，给 GDR 留 1%，先设 `2000`
- 400G 链路，给 GDR 留 1%，先设 `4000`
- 400G 链路，给 GDR 留 5%，先设 `20000`

注意：

- 这是线速侧的限制值
- benchmark 里看到的是应用层 `GB/s`
- 两者不会完全一一对应，通常需要实测后微调

## 8. 启用方式

需要确保 OpenSM 启用了 QoS。常见方式包括：

- 使用 `-Q` / `--qos`
- 在 UFM/OpenSM 配置中开启 QoS 开关
- 指定 `qos_policy_file`
- 指定 `enhanced_qos_policy_file`

实际生效方式取决于你当前的 OpenSM/UFM 部署方式。

## 9. 验证方法

### 9.1 检查 QoS 是否下发

可使用：

```bash
ibdiagnet --qos
```

重点关注：

- `SL -> VL` 映射是否正确
- `VL arbitration` 是否正确
- 每个 `SL` 的 rate limit 是否已经写入 HCA

### 9.2 跑 benchmark 验证

建议检查：

1. `solo ncclSend` 带宽
2. `concurrent GDR + ncclSend` 带宽
3. `concurrent / solo` 的比例

如果 `NCCL` 侧带宽没有明显改善，优先排查：

- `NCCL` 和 `GDR` 是否真的打到了不同 `SL`
- `SL2VL` 是否已经生效
- `Enhanced QoS` 是否真正下发到了端口

## 10. 重要注意事项

### 10.1 只配 VLArb 不等于严格限速

只配置：

- `SL -> VL`
- `VL Arbitration`

通常只能做到：

- 高优先级流量更容易先发
- 低优先级流量更容易被压制

但它更像“调度优先级”，不是绝对的硬带宽上限。

如果你的目标非常接近：

- `99% / 1%`

那就应该配 `Enhanced QoS`。

### 10.2 本仓库里的 benchmark 只做验证

当前代码里：

- `SL` 是真实设置进去的
- `QoS 比例` 只是验证阈值和打印结果

它不会在应用层：

- `sleep`
- `token bucket`
- 改 chunk 大小
- 改 `post_send` 节奏

所以真正的“限流动作”仍然依赖网络侧 QoS。

### 10.3 关于 GDR 的特殊性

当前 `gdr_copy` 使用的是 loopback RC QP。

这意味着要特别关注一个现实问题：

- 如果这条 GDR 流量没有真正经过物理端口调度器
- 而是在 HCA 内部完成

那么 fabric 级的 `SL/VL/QoS` 对它的影响可能不如预期明显。

如果出现“QoS 已配置但效果不明显”，要重点排查这条路径是否真的经过了物理出端口。

## 11. 推荐落地步骤

建议按下面顺序实施：

1. 先固定业务约定
   - `NCCL_IB_SL=0`
   - `BENCH_GDR_IB_SL=1`
2. 配 `SL2VL`
   - `SL0 -> VL0`
   - `SL1 -> VL1`
3. 配 `VL Arbitration`
   - `VL0` 高优先级
   - `VL1` 低优先级
4. 跑一次 benchmark 看是否已有改善
5. 如果目标是接近 `99% / 1%`
   - 再加 `Enhanced QoS`
   - 给 `SL1` 设置 rate limit
6. 用 `ibdiagnet --qos` 和 benchmark 反复验证、微调

## 12. 参考文档

- OpenSM QoS / `SL2VL` / `VLArb`
  - <https://docs.nvidia.com/networking/display/OFEDv502180/OpenSM>
- UFM Enhanced QoS
  - <https://docs.nvidia.com/networking/display/ufmenterpriseumv6194/Appendix-Enhanced-Quality-of-Service-%28QoS%29>
- `ibdiagnet --qos` 速率限制验证
  - <https://docs.nvidia.com/networking/display/ibdiagnetusermanualv221/qos%2Bconfiguration%2Brate%2Blimit%2Bper%2Bservice%2Blevel%2B%28sl%29>
