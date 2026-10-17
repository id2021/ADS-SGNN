- 论文作者：Kaiqiang Huang, Yusen Wu, Jinde Zhu
- 通讯作者：gnamic00@gmail.com

---


## 主要结果

| 数据集             |     准确率 |
| ------------------ | ---------: |
| EvTouch-Objects    | **95.83%** |
| EvTouch-Containers | **81.67%** |

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
