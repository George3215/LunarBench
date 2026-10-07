# 07 — Method

> 状态：**架构骨架 + 已实现的后果模型**（E4-3 已接入 planner 并在多步协议上评价）。
> **刻意不定稿**：完整后果模型（稳定性/边界）尚未解决——E4-5 显示学习到的安全项
> 在每一个权重下都**降低**性能。在这一缺口解决之前，方法章不写成"最终方案"。

---

## 7.1 What has to be predicted

三条失败（H1 / H2 / H3）共同指出的量是：

$$
f_\theta\bigl(O_t, M_t, a\bigr) \;\longrightarrow\; P\bigl(V_{t+H} = 1 \mid O_t, M_t, a\bigr)
\tag{7.1}
$$

其中 $V$ 是**动作真正执行之后**测到的可见性——不是几何预测的可见性（H3），
不是当前可见性（H1），也不是目标方向（H2）。

**这个选择是被实验逼出来的，不是设计偏好的结果。** E4-1 给出直接对照：

| 输入 | ROC-AUC | 相对 $O{+}M$ |
| --- | ---: | ---: |
| $O$ | 0.759 | — |
| $M$ | 0.852 | — |
| $O{+}M$ | 0.854 | — |
| **$O{+}M{+}a$** | **0.934** | **+0.080** |

而 $M \to O{+}M$ 只增加 **+0.002**。
**信息在动作里，不在"看得更多"里**——所以 (7.1) 的输入必须含 $a$。

决策规则：

$$
\hat a_t \;=\; \arg\max_{a \in \mathcal{A}}\ f_\theta\bigl(O_t, M_t, a\bigr)
\tag{7.2}
$$

---

## 7.2 Architecture

```
        O_t (pose, proprioception, short action history)
        M_t (target memory: body-frame position, range, bearing, age)
                     │
                     ▼
          candidate actions  a ∈ A   (10 frozen primitives)
                     │
                     ▼
     ┌───────────────────────────────────┐
     │  action-conditioned consequence   │
     │  model  f_θ(O, M, a)              │
     └───────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
    Δ pose S      P(V=1)      P(unsafe)
   (fwd, lat,               (fall / OOB /
    height, yaw)             margin)
        │            │            │
        └────────────┼────────────┘
                     ▼
              action selection          ← (7.2), plus a safety term
                     │
                     ▼
                  execute
                     │
                     ▼
             new observation  ──►  loop (closed)
```

**为什么预测 $\Delta$pose 而不是直接预测 RGB 未来帧。** H3 已经把缺口定位得很准：
把 oracle 放到**实际执行到的位姿**上，它与物理结果一致 **97.9 %**。
所以真正需要预测的不是可见性本身，而是 $a \to S_{future}$。
把已经验证过的部分（位姿→可见性）继续用 oracle 算，**比端到端学整条链在科学上更干净**，
也更容易定位剩余误差。

---

## 7.3 Current implementation (E4-3)

### Data

24 地形 × 10 状态 × 10 动作 = **2240** 个 $(s,a)$，**全部物理执行**，正例率 0.143。
特征：位姿 + 本体感觉 + 短动作历史；记忆为**体坐标系**下的目标向量 + 存在标志
（缺失用**单独的标志位**表示，不是零向量——"没有记忆"和"记忆在原点"必须可区分）。

标签：**体坐标系下的执行位姿增量** $(\Delta\text{fwd}, \Delta\text{lat}, \Delta\text{height}, \Delta\text{yaw})$
与是否重新看到目标。两者都取自 `rollout.curve[-1]`，因为 `rollout()` 退出时会恢复仿真状态。

### Model and split

梯度提升树（`HistGradientBoosting`, depth 3, 250 iters），
划分一律**按地形留一**——同族地形不能同时出现在两边。绝不随机划分。

### Consequence accuracy

| 分量 | analytic（标称位移） | learned | |
| --- | ---: | ---: | --- |
| forward | 0.829 | **0.152** | 5.5× |
| lateral | 0.418 | **0.145** | 2.9× |
| **height** | **0.090** | 0.149 | **analytic 更好** |
| yaw | 0.538 | **0.147** | 3.7× |
| **平均** | **0.469** | **0.148** | **3.2×** |

**必须一起写限定**：学习模型在**决定成败的分量**上大幅胜出，
但**高度上输给解析模型**（标称位移预测"不变"，而高度变化本来就小）。
不是"学习全面更好"。

### Closed-loop planning

| 方法 | $RSR@1$ | $RSR@3$ | $RSR@5$ | $RSR@10$ |
| --- | ---: | ---: | ---: | ---: |
| 六个经典 baseline | 0.167 | 0.250 | 0.250 | **0.250** |
| **LearnedConsequence (E4-3)** | **0.708** | **0.708** | **0.792** | **0.792** |
| `Oracle`（特权） | 0.833 | 0.875 | 0.875 | 0.875 |
| `OracleSearch`（特权） | 0.750 | 1.000 | 1.000 | 1.000 |

并且它在 **8/8** 对成对世界上**对两个世界选择不同动作**——这是 (7.1) 含有 $a$
的直接后果，也是任何无记忆规则做不到的事。

---

## 7.4 What is deliberately not fixed yet

### 7.4.1 The safety term is implemented but not usable

E4-5 把稳定性纳入后果头（`out_of_bounds` + `upright_min`，因为二值摔倒标签在
720 次单动作 rollout 里 **0 正例**）。安全头本身**可学**：AUC **0.900**，ECE 0.042。

但它**不能当决策项用**：

| 规划器 | 单步成功率 |
| --- | ---: |
| `Learned-V`（无安全项） | **0.3661** |
| `Learned-V+S`（硬否决） | 0.3125 |
| 软惩罚 $\lambda = 0.5/1/2/5$ | 0.3571 / 0.3438 / 0.3482 / 0.3438 |
| `Oracle-V`（真可见性） | 0.3884 |
| `Oracle-S`（真可见性 + 真安全） | **0.4107** |

**安全信息在原理上有价值**（oracle 0.388 → 0.411），
**但学习到的安全头在每一个权重下都更差**。

> 所以在 (7.2) 里怎么写安全项，**现在没有答案**。这决定了方法章不能定稿。

### 7.4.2 Why this is the right place to stop writing

现在能确定的：**输入必须含 $a$**，**要预测的是 $a \to S$ 而不是 $V$ 本身**，
**闭环优于开环规则**。
现在**不能**确定的：安全/终止后果怎么进决策规则。这是下一轮实验的问题，
不是写作问题。

---

## 7.5 Writing discipline for this chapter

- 可以写：$O{+}M{+}a$ 优于 $O{+}M$（**+0.080**）、学习式后果模型优于解析（**3.2×** 平均，
  但是**高度上更差**）、闭环 $RSR@10$ **0.250 → 0.792**。
- **不能**写："World Model 全面优于解析模型"。
- **不能**写："安全头可以用了"——E4-5 的数字反过来。
- **不能**写："我们提出了一个 World Model"作为主要贡献（§2.4 已说明组合不是创新）。
- `Oracle` / `OracleSearch` 是**特权参考**，不是方法，必须标注。
- 跨机制泛化**尚未验证**，不能声称。

---

## 7.6 Open items for the final version

- [ ] 安全/终止后果如何进入决策规则（校准 / 选择性预测 / conformal / 不确定性感知）
- [ ] 从二值 $P(V=1)$ 升级为轨迹 $\hat V^a_{t:t+H}$，顺带给出 $T_{reobs}$ 的预测
- [ ] 跨机制泛化（Setting A 同族不同实例 / B 混合训练留出机制 / C 留一机制）
- [ ] 与 `Oracle` 的**代价对比**：oracle 每个决策点要跑 $|\mathcal A|$ 次物理 rollout，
      学习式只需一次前向
- [ ] 动作空间离散化的 sanity check（10 个固定原语是否影响 H1/H2）
