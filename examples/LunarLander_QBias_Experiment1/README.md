# Q-Learning 缺陷与进化：实验 1A + 1B

本目录包含两个互补实验：

- 实验 1A：量化 DQN 的高估偏差，并与 Double DQN 对照。
- 实验 1B：可视化 Dueling 架构在特定场景下的状态价值 V(s) 与优势函数 A(s,a) 解耦。

## 1. 实验 1A 目的

- 证明普通 DQN 在训练中会出现预测 Q 值偏高（与真实回报量级脱离）。
- 对比 Double DQN 在同等条件下的预测 Q 值与真实回报的贴合程度。

## 2. 实验 1A 设计（与你提出的 1A 对齐）

1. 分别训练 DQN 与 Double DQN。
2. 每隔固定 Episode（默认 50），运行贪心策略评估回合。
3. 在评估回合中真实访问到的状态上计算：

   `Q_predict = mean_s [ max_a Q(s, a) ]`

  `G_true = mean_s [ sum_k gamma^k r_{t+k} ]`

4. 计算高估差：`overestimation_gap = Q_predict - G_true`。
5. 绘制同轴折线图（同量纲）：预测 Q 与真实折扣回报。

## 3. 实验 1A 理论解释（论文分析点可直接用）

- DQN 使用 `max_a Q_target(s', a)` 进行目标估计。若各动作价值估计含噪声，`max` 操作会系统性偏向较大的噪声项，导致正偏差累积。
- Double DQN 将“选动作”和“评动作”分离：
  - 在线网络选择动作：`a* = argmax_a Q_online(s', a)`
  - 目标网络评估该动作：`Q_target(s', a*)`
- 因为不再用同一个带噪估计器同时做 `argmax` 与取值，正向偏差被显著抑制。
- 注意：若用“未打折的回合总分”去对比 Q 值会量纲不一致。实验 1A 现已改为与 Q 同量纲的真实折扣回报 `G_true`。

## 4. 实验 1A 文件说明

- `experiment_1A_overestimation_bias.py`：实验 1A 主脚本。
- 运行后输出目录（默认 `outputs/`）包含：
  - `metrics.csv`：Episode 级别评估数据
  - `overestimation_aligned.png`：同轴对齐图（Q_predict vs G_true）
  - `eval_return_curve.png`：评估回合总分曲线（辅助）
  - `summary.txt`：偏差统计摘要

## 5. 实验 1A 运行方式

在项目根目录执行：

```bash
python examples/LunarLander_QBias_Study/experiment_1A_overestimation_bias.py
```

可选参数：

```bash
python examples/LunarLander_QBias_Study/experiment_1A_overestimation_bias.py \
  --episodes 600 \
  --eval-interval 50 \
  --eval-episodes 5 \
  --seed 42 \
  --output-dir outputs
```

## 6. 实验 1A 结果解读建议

- 若 `DQN Predicted Q` 曲线明显高于 `DQN True Discounted Return`，说明高估偏差显著。
- 若 `Double DQN Predicted Q` 更接近 `Double DQN True Discounted Return`，说明解耦策略有效压制高估。
- 可在 `summary.txt` 中直接比较 `average gap = Q_predict - G_true`。

## 7. 实验 1B 目的

- 对比 Double DQN 与 Dueling Double DQN 的评估行为。
- 在单个测试回合中，逐步记录 Dueling 网络内部的 V(s) 与四个动作 A(s,a)。
- 聚焦“高空自由落体”阶段，验证该阶段是否由 V(s) 主导而动作优势差异接近 0。

## 8. 实验 1B 设计（与你提出的 1B 对齐）

1. 训练 Double DQN 与 Dueling Double DQN（或加载已有权重）。
2. 使用相同随机种子运行一个 Evaluation Episode。
3. 在 Dueling 模型的每一步记录：
   - 动作
   - `V(s)`
   - `A(s,a0..a3)`
   - `max A(s,a)`
4. 自动截取“高空自由落体”窗口（优先条件：高海拔 + 下落 + 双腿未接触地面）。
5. 绘图：
   - 主图：`V(s)` 与 `max A(s,a)` 随时间变化
   - 辅图：Double DQN 与 Dueling DDQN 在同窗口的 `max Q` 对比

## 9. 实验 1B 文件说明

- `experiment_1B_dueling_value_advantage.py`：实验 1B 主脚本。
- 运行后输出目录（默认 `outputs_1b/`）包含：
  - `eval_trace_double_ddqn.csv`：Double DQN 整回合轨迹
  - `eval_trace_dueling_ddqn.csv`：Dueling DDQN 整回合轨迹（含 V 与 A）
  - `window_trace_double_ddqn.csv`：截取窗口内 Double DQN 轨迹
  - `window_trace_dueling_ddqn.csv`：截取窗口内 Dueling 轨迹
  - `dueling_value_vs_max_advantage.png`：你要求的核心图
  - `ddqn_vs_dueling_qmax_window.png`：窗口内 Qmax 对比图
  - `summary_1b.txt`：统计摘要（如 mean|V|、mean|A|、|V|/|A| 比值）

## 10. 实验 1B 运行方式

在项目根目录执行：

```bash
python examples/LunarLander_QBias_Study/experiment_1B_dueling_value_advantage.py
```

可选参数：

```bash
python examples/LunarLander_QBias_Study/experiment_1B_dueling_value_advantage.py \
  --episodes 600 \
  --seed 42 \
  --eval-seed 20260313 \
  --output-dir outputs_1b
```

如已训练好模型，可直接加载权重跳过训练：

```bash
python examples/LunarLander_QBias_Study/experiment_1B_dueling_value_advantage.py \
  --load-double-path path/to/double.pth \
  --load-dueling-path path/to/dueling.pth
```

## 11. 实验 1B 论文分析点（可直接使用）

- 在“无论采取什么动作对当前状态影响都较小”的高空阶段，`A(s,a)` 在四个动作间差异很小，说明动作优势项近似不敏感。
- 同时 `V(s)` 量级显著高于 `A(s,a)`，表示 Q 值主要由状态价值项贡献。
- 这说明 Dueling 架构能把学习重点放在更稳定的状态价值估计上，而不必在每一步精细拟合所有动作的 Q 值。
- 在 LunarLander 这类含冗余状态的任务中，该解耦通常带来更快收敛与更高样本效率。

本目录当前实现的是实验 1A。你后续可以在同目录新增实验 1B：

- 对比 Double DQN 与 Dueling Double DQN（可复用 `examples/LunarLander_DQN/train_dueling_double_dqn.py` 的网络定义）。
- 在固定状态集上额外记录 `V(s)` 与 `A(s,a)` 的统计量，分析 Dueling 是否更稳定地学习“状态价值”与“动作优势”。
