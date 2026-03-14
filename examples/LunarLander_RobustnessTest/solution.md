# Zero-Shot Robustness Test — 强化学习策略的鲁棒性与泛化能力

## 1. 实验目标

在标准环境中训练得到最优策略后，**不更新任何权重（Zero-Shot）**，直接将策略部署到物理参数发生变化的未知环境中进行测试，以此评估各算法策略的**鲁棒性**和**泛化能力**。

## 2. 实验设置

### 2.1 被测算法（6种）

| 算法 | 网络结构 | 隐藏层 | 模型权重路径 |
|------|----------|--------|-------------|
| **DQN** | MLP Q-Network | 128×2, ReLU | `results/train_dqn/dqn_final.pth` |
| **Double DQN** | MLP Q-Network | 128×2, ReLU | `results/train_double_dqn/double_dqn_final.pth` |
| **Dueling DDQN** | Dueling Q-Network + PER | 256→128 (V/A streams) | `results/train_dueling_double_dqn/dueling_ddqn_final.pth` |
| **PPO** | Actor-Critic (仅用Actor) | 64×2, Tanh | `results/train_ppo/ppo_final.pth` |
| **REINFORCE** | Policy Network | 128×2, ReLU | `results/train_reinforce_normalization/reinforce_final.pth` |
| **A2C-V** | MLPBase + Categorical | 64×2, Tanh + obs归一化 | `results/train_A2C-V(Q)/v/seed_1/models/best_model.pt` |

### 2.2 测试环境（3种）

| 环境 | gravity | enable_wind | wind_power | turbulence_power | 说明 |
|------|---------|-------------|------------|------------------|------|
| **Standard** | -10.0 | False | 15.0 | 1.5 | 与训练环境一致的标准条件 |
| **High-Wind** | -10.0 | True | 20.0 | 2.0 | 强侧风 + 强湍流，测试抗扰动能力 |
| **Low-Gravity** | -5.0 | False | 15.0 | 1.5 | 重力减半，测试动力学泛化能力 |

### 2.3 测试协议

- 每组 (算法, 环境) 进行 **50 个 episode** 的评估
- 总计 6 × 3 × 50 = **900 个 episode**
- **DQN / Double DQN / Dueling DDQN**: ε = 0，纯贪心策略 (argmax Q)
- **PPO**: 确定性动作 (argmax π)
- **REINFORCE**: 确定性动作 (argmax logits)
- **A2C-V**: deterministic=True (mode of categorical distribution)
- 随机种子: SEED = 2024 + episode_index，保证可复现

## 3. 实验结果

### 3.1 总览表

| 算法 | Standard | High-Wind | Low-Gravity |
|------|----------|-----------|-------------|
| **DQN** | 253.1 ± 39.1 | 16.3 ± 235.8 | 230.8 ± 66.9 |
| **Double DQN** | 228.3 ± 64.7 | -51.8 ± 275.6 | 230.8 ± 61.7 |
| **Dueling DDQN** | 248.4 ± 49.3 | **90.4 ± 164.3** | 222.1 ± 67.0 |
| **PPO** | 195.4 ± 85.0 | 9.5 ± 131.3 | 164.9 ± 101.7 |
| **REINFORCE** | 133.7 ± 94.4 | -12.4 ± 97.6 | 20.7 ± 88.8 |
| **A2C-V** | 240.1 ± 56.1 | 26.6 ± 240.0 | **237.1 ± 61.1** |

### 3.2 相对性能变化（对比Standard基线）

| 算法 | High-Wind | Low-Gravity |
|------|-----------|-------------|
| DQN | -93.6% | -8.8% |
| Double DQN | -122.7% | +1.1% |
| **Dueling DDQN** | **-63.6%** | -10.6% |
| PPO | -95.1% | -15.6% |
| REINFORCE | -109.3% | -84.5% |
| A2C-V | -88.9% | **-1.3%** |

### 3.3 详细统计

| 算法 | 环境 | Mean | Std | Min | Max | Median |
|------|------|------|-----|-----|-----|--------|
| DQN | Standard | 253.1 | 39.1 | 139.6 | 301.3 | 261.8 |
| DQN | High-Wind | 16.3 | 235.8 | -759.4 | 282.4 | 5.9 |
| DQN | Low-Gravity | 230.8 | 66.9 | 69.7 | 310.8 | 260.9 |
| Double DQN | Standard | 228.3 | 64.7 | 40.3 | 292.4 | 252.2 |
| Double DQN | High-Wind | -51.8 | 275.6 | -477.1 | 299.9 | -155.9 |
| Double DQN | Low-Gravity | 230.8 | 61.7 | 62.1 | 293.6 | 254.1 |
| Dueling DDQN | Standard | 248.4 | 49.3 | 68.1 | 311.8 | 258.9 |
| Dueling DDQN | High-Wind | 90.4 | 164.3 | -277.6 | 303.1 | 144.3 |
| Dueling DDQN | Low-Gravity | 222.1 | 67.0 | 64.3 | 321.6 | 248.1 |
| PPO | Standard | 195.4 | 85.0 | -123.2 | 272.2 | 225.5 |
| PPO | High-Wind | 9.5 | 131.3 | -147.5 | 292.9 | -36.1 |
| PPO | Low-Gravity | 164.9 | 101.7 | -80.9 | 271.9 | 194.3 |
| REINFORCE | Standard | 133.7 | 94.4 | -15.6 | 273.7 | 154.3 |
| REINFORCE | High-Wind | -12.4 | 97.6 | -241.7 | 244.5 | -25.6 |
| REINFORCE | Low-Gravity | 20.7 | 88.8 | -153.1 | 230.7 | 15.3 |
| A2C-V | Standard | 240.1 | 56.1 | 94.2 | 301.9 | 262.6 |
| A2C-V | High-Wind | 26.6 | 240.0 | -400.0 | 310.9 | 37.2 |
| A2C-V | Low-Gravity | 237.1 | 61.1 | -7.2 | 303.8 | 261.2 |

## 4. 可视化

所有可视化结果保存在 `results/RobustnessTest/` 目录下：

| 文件名 | 说明 |
|--------|------|
| `boxplot_comparison.png` | 每个环境下6种算法的奖励分布箱线图 |
| `grouped_bar_chart.png` | 分组柱状图（均值 ± 标准差） |
| `performance_heatmap.png` | 相对性能变化热力图（以Standard为基准） |
| `radar_chart.png` | 多维度鲁棒性雷达图 |
| `robustness_results.json` | 原始测试数据（JSON格式） |

## 5. 分析与结论

### 5.1 标准环境性能排名

在标准训练条件下，各算法性能排序为：

**DQN (253.1) > Dueling DDQN (248.4) > A2C-V (240.1) > Double DQN (228.3) > PPO (195.4) > REINFORCE (133.7)**

DQN系列（包含三种变体）在标准环境中表现最佳，均超过了"解决"阈值（200分）。

### 5.2 High-Wind 环境鲁棒性分析

高风力环境是最具挑战性的扰动条件，所有算法均出现显著性能下降：

- **Dueling DDQN 最鲁棒**：均值90.4，仅下降63.6%，是唯一在高风力中保持正均值且标准差相对可控的DQN变体。Dueling架构将状态值V(s)与动作优势A(s,a)分离估计的设计，使其在风力干扰下仍能较好区分"好状态"与"差状态"，展现出更强的环境适应能力。
- **A2C-V 次之**：均值26.6，下降88.9%，但方差极大（240.0），表现不稳定。
- **DQN**：均值16.3，下降93.6%，标准差235.8非常大。
- **Double DQN 最差**：均值-51.8，下降122.7%，性能完全崩溃。Double DQN通过降低过估计改进了训练稳定性，但学到的策略对分布外状态的鲁棒性反而更差。
- **PPO 和 REINFORCE**：均值分别为9.5和-12.4，策略梯度类方法在面对剧烈物理变化时也难以应对。

**核心发现**：高风力环境引入了训练期间从未见过的强随机侧向力和湍流，所有方法的方差都急剧增大。Dueling架构的结构优势在此条件下得以体现——它的V/A分流设计提供了更好的状态表征鲁棒性。

### 5.3 Low-Gravity 环境泛化能力分析

低重力环境仅改变了单一物理参数（重力减半），是温和的分布偏移：

- **A2C-V 泛化最佳**：均值237.1，仅下降1.3%，几乎无损迁移。观测归一化（obs_rms）使其对输入尺度变化不敏感。
- **DQN 和 Double DQN**：均值230.8，分别下降8.8%和+1.1%，泛化良好。Q-learning方法对重力变化的适应性强。
- **Dueling DDQN**：均值222.1，下降10.6%，略低于DQN但仍在解决阈值之上。
- **PPO**：均值164.9，下降15.6%，表现中等。
- **REINFORCE 泛化最差**：均值20.7，下降84.5%，策略几乎完全失效。作为无基线的纯策略梯度方法，其学到的策略高度依赖训练分布。

**核心发现**：基于价值函数的方法（DQN系列、A2C-V）对重力变化的泛化能力显著优于纯策略梯度方法（REINFORCE）。观测归一化进一步增强了A2C-V的迁移能力。

### 5.4 综合鲁棒性排名

综合三个环境的表现，算法鲁棒性排名为：

| 排名 | 算法 | 综合评价 |
|------|------|----------|
| 1 | **Dueling DDQN** | 高风力环境中最佳（90.4），低重力中仍保持222+，综合抗扰动能力最强 |
| 2 | **A2C-V** | 低重力泛化最佳（-1.3%），标准环境优秀，但高风力方差极大 |
| 3 | **DQN** | 标准环境最佳（253.1），低重力泛化良好，高风力中表现一般 |
| 4 | **Double DQN** | 低重力泛化好（+1.1%），但高风力中完全崩溃（-122.7%） |
| 5 | **PPO** | 各环境表现中等，方差较大但无灾难性失败 |
| 6 | **REINFORCE** | 标准环境即欠佳，非标准环境中全面崩溃 |

### 5.5 关键洞察

1. **架构设计影响鲁棒性**：Dueling架构通过分离V(s)和A(s,a)的估计，天然具有更好的状态表征泛化能力，在分布外环境中表现突出。
2. **观测归一化是强大的泛化手段**：A2C-V使用的VecNormalize（obs_rms）使策略对观测尺度不敏感，在低重力环境中几乎无损迁移。
3. **减少过估计不等于增强鲁棒性**：Double DQN解决了Q值过估计问题，但在零样本迁移中反而不如vanilla DQN稳定，可能因为更精确的Q值估计更依赖训练分布。
4. **策略梯度方法的脆弱性**：纯策略梯度方法（REINFORCE）在分布偏移下表现最差，缺乏价值函数提供的稳定锚点。
5. **高风力是最难的泛化场景**：引入了训练中完全不存在的随机侧向力，所有算法均大幅衰退，说明零样本泛化的根本局限在于无法应对全新的环境动力学。

## 6. 代码结构

```
examples/LunarLander_RobustnessTest/
├── robustness_test.py         # 主测试脚本：加载6个模型，3种环境各50轮测试
├── visualize_robustness.py    # 可视化脚本：生成4种图表
└── solution.md                # 本文档

results/RobustnessTest/
├── robustness_results.json    # 原始测试数据
├── boxplot_comparison.png     # 箱线图
├── grouped_bar_chart.png      # 分组柱状图
├── performance_heatmap.png    # 热力图
└── radar_chart.png            # 雷达图
```

## 7. 运行方式

```bash
# 激活环境
conda activate robustgymnasium

# 1. 运行零样本鲁棒性测试（约2分钟）
python examples/LunarLander_RobustnessTest/robustness_test.py

# 2. 生成可视化（约5秒）
python examples/LunarLander_RobustnessTest/visualize_robustness.py
```

## 8. 依赖

- Python 3.11+
- PyTorch 2.x
- robust_gymnasium (本项目)
- stable-baselines3 (A2C-V模型的obs_rms反序列化)
- matplotlib, numpy
