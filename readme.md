# Adaptive Dual-Stream Spiking Graph Neural Networks for Event-Driven Tactile Object Recognition

> 用于事件驱动触觉物体识别的自适应双流脉冲图神经网络

本仓库是论文 **“Adaptive Dual-Stream Spiking Graph Neural Networks for Event-Driven Tactile Object Recognition”** 的官方代码实现。  
模型面向事件驱动触觉传感器数据，通过空间图与时间图的双流脉冲图神经网络进行触觉物体识别，并使用可学习的自适应融合模块动态平衡两路信息。

- 论文作者：Kaiqiang Huang, Yusen Wu, Jinde Zhu
- 通讯作者：2241308065@smail.fjtu.edu.cn

---

## 项目简介

事件驱动触觉传感器通过异步脉冲序列报告环境变化，具有高时间分辨率、低延迟、低功耗和良好可扩展性。然而，事件驱动触觉数据通常具有稀疏、异步、不规则结构等特点，传统 CNN 方法难以直接处理。

本工作提出一种自适应双流脉冲图神经网络，主要特点包括：

1. **双流图建模**
   - 空间流：基于触觉单元几何结构构建空间触觉图，并在每个时间步处理。
   - 时间流：基于事件序列构建时间触觉图，并为每个触觉单元处理其时间演化。
2. **自适应融合**
   - 使用可学习标量参数 `alpha = sigmoid(theta)` 动态融合空间流与时间流输出。
   - 模型可自动学习更依赖空间结构还是时间动态。
3. **基于事件的批归一化 EBN**
   - 在 TAGConv 后、LIF/LLIF 激活前加入 EBN，提高训练稳定性与梯度传播效果。
4. **高能效**
   - 相比等价 ANN，SNN 模型可实现约 `50x` 的能耗降低。

---

## 主要结果

| 数据集             |     准确率 |
| ------------------ | ---------: |
| EvTouch-Objects    | **95.83%** |
| EvTouch-Containers | **81.67%** |

### 消融实验

| 模型变体                   | EvTouch-Objects (%) | EvTouch-Containers (%) |
| -------------------------- | ------------------: | ---------------------: |
| 仅空间流                   |               88.75 |                  63.33 |
| 仅时间流                   |               86.11 |                  80.00 |
| 自适应融合，无 EBN         |               95.14 |                  80.00 |
| 自适应融合 + EBN，完整模型 |           **95.83** |              **81.67** |

### 替代梯度函数对比

| 替代梯度函数          | EvTouch-Objects (%) | EvTouch-Containers (%) |
| --------------------- | ------------------: | ---------------------: |
| Rectangular           |           **95.83** |              **81.67** |
| Piecewise Leaky ReLU  |               84.03 |                  73.33 |
| S2NN                  |               85.42 |                  63.33 |
| Piecewise Exponential |               79.86 |                  68.33 |
| Piecewise Quadratic   |               90.97 |                  70.00 |

### 能耗分析

| 模型 | 数据集             |       能量 |   比率 |
| ---- | ------------------ | ---------: | -----: |
| ANN  | EvTouch-Objects    |  865.89 μJ |   1.0x |
| SNN  | EvTouch-Objects    |   17.39 μJ | ~49.8x |
| ANN  | EvTouch-Containers | 1116.55 μJ |   1.0x |
| SNN  | EvTouch-Containers |   22.50 μJ | ~49.6x |

---

## 环境依赖
建议使用 Python 3.8+、PyTorch 和 PyTorch Geometric。

## 数据准备

请先下载两个公开事件驱动触觉数据集：

- **EvTouch-Objects**
- **EvTouch-Containers**

数据基本信息：

- 传感器：NeuTouch
- 触觉单元数：`N = 39`
- 极性通道：`C = 2`，即 ON/OFF 事件
- EvTouch-Objects：36 类，约 720 个样本，`T ≈ 250`
- EvTouch-Containers：20 类，约 300 个样本，`T ≈ 325`

---

## 图构建

### 空间触觉图

- 每个时间步构建一个空间图 `G_s = (V_s, E_s)`。
- 节点：39 个触觉单元。
- 节点特征：该时间步的 ON/OFF 脉冲活动。
- 边：基于触觉单元物理坐标的欧氏距离，采用 Minimum Spanning Tree 构建。

### 时间触觉图

- 每个触觉单元构建一个时间图 `G_t = (V_t, E_t)`。
- 节点：`T` 个时间步。
- 节点特征：该触觉单元在对应时间步的双通道脉冲特征。
- 边：连接相邻时间步，即 `(t, t+1)`。

---

## 模型结构

整体架构由三部分组成：

1. **空间流**
   - 输入：每个时间步的空间图。
   - 结构：TAGConv + EBN + LIF + FC。
   - 输出：对 `T` 个时间步输出取平均，得到空间类别得分 `O_s`。

2. **时间流**
   - 输入：每个触觉单元的时间图。
   - 结构：TAGConv + EBN + LLIF + FC。
   - 输出：对 `N` 个触觉单元输出取平均，得到时间类别得分 `O_t`。

3. **自适应融合模块**

```text
alpha = sigmoid(theta)
O = alpha * O_s + (1 - alpha) * O_t
```

其中 `theta` 为可学习标量，初始化为 0，因此初始时 `alpha ≈ 0.5`。


---

## 联系方式

- 通讯作者：Jinde Zhu
- 邮箱：gnamic00@gmail.com
- 单位：School of Computer Science and Mathematics, Fujian University of Technology, Fuzhou, 350118, China

---

## 致谢

感谢 EvTouch-Objects、EvTouch-Containers 数据集及相关开源工作为本研究提供的数据与基线支持。  
