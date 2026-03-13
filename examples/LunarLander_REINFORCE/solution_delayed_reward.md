# Delayed Reward Structure Analysis – REINFORCE+Normalization on LunarLander-v3

## 1. Problem Formulation

In reinforcement learning the **credit assignment problem** asks: *which past
actions are responsible for the rewards the agent receives?*  When rewards are
**dense** (delivered at every time-step) the gradient signal is rich and
temporally localized.  When rewards are **delayed** or **sparse**, the agent
must assign credit across a long temporal gap, making learning dramatically
harder.

Formally, given a standard MDP \((S, A, P, R, \gamma)\) with episode length
\(T\), we consider a family of *reward delivery functions*
\(\tilde{R}_k\) parameterised by delay \(k\):

| Mode | Delivered reward at step \(t\) |
|------|-------------------------------|
| **Baseline** (dense, \(k{=}0\)) | \(\tilde{r}_t = r_t\) |
| **Dense (extra shaping)** | \(\tilde{r}_t = r_t + \Phi(s_t)\), additional potential-based bonus |
| **Medium delay** (\(k{=}K\)) | \(\tilde{r}_t = \sum_{i=t-K+1}^{t} r_i\) every \(K\) steps; 0 otherwise |
| **Sparse** (\(k{=}T\)) | \(\tilde{r}_t = 0\) for \(t < T\); \(\tilde{r}_T = \sum_{i=1}^{T} r_i\) |

The total undiscounted return is identical across all modes—only the *timing*
of information delivery changes.  This setup isolates the pure effect of reward
delay on learning dynamics.


## 2. Environment & Algorithm

### LunarLander-v3 (Discrete)

- **State** (8-dim): position, velocity, angle, angular velocity, leg contacts.
- **Actions** (4): noop, left engine, main engine, right engine.
- **Default reward**: potential-based shaping \(-100\sqrt{x^2+y^2}\),
  velocity/angle penalties, fuel cost, ±100 terminal bonus.
- Episode terminates on landing, crash, or the 1000-step time limit.

### REINFORCE with Return Normalization

REINFORCE is the simplest policy-gradient method:

- **Policy-only**: A single neural network \(\pi_\theta(a|s)\) outputs action
  probabilities.  There is **no critic / value network**.
- **Monte Carlo returns**: The full episode trajectory is collected before any
  update.  The discounted return-to-go
  \(G_t = \sum_{k=0}^{T-t-1} \gamma^k \tilde{r}_{t+k}\) is computed for
  each timestep.
- **Return normalization**: Returns are standardized per-episode:
  \(\tilde{G}_t = (G_t - \mu_G) / (\sigma_G + \epsilon)\).  This variance
  reduction technique stabilizes training by ensuring the gradient
  magnitudes are on a consistent scale.
- **Policy gradient**: \(\nabla J(\theta) \approx -\sum_t \log\pi_\theta(a_t|s_t) \cdot \tilde{G}_t\)

| Parameter | Value |
|-----------|-------|
| Hidden layers | 2 × 128 (ReLU) |
| Learning rate | 1 × 10⁻³ (Adam) |
| γ | 0.99 |
| Training episodes | 1000 (main) / 500 (robustness sweep) |
| Update frequency | Once per episode (full Monte Carlo) |

### Why REINFORCE is Special for This Study

Unlike DQN (off-policy, TD-based) and PPO (actor-critic, GAE-based), REINFORCE
has **no bootstrapping** and **no value function**.  It uses *complete episode
returns* for every update.  This creates a unique prediction:

1. **Theoretically more robust to delayed rewards** — since it always waits for
   the full trajectory, the gradient signal exists regardless of reward timing.
2. **Practically still degraded** — because return normalization equalizes
   gradient magnitudes.  Under sparse rewards, all timesteps receive nearly
   identical \(|\tilde{G}_t|\) (differing only by discounting), destroying
   temporal specificity in the gradient signal.


## 3. Experimental Design

### 3.1 Reward Configurations

Five reward delivery modes are compared:

1. **Baseline** – The unmodified LunarLander-v3 reward.
2. **Dense (extra shaping)** – Baseline + per-step bonus:
   \[
   \text{bonus}_t = 0.2(l_1 + l_2) - 0.3|x| - 0.2|\theta|
   \]
3. **Medium Delay (K=10)** – Rewards accumulated over 10 steps, then
   delivered as a single sum.
4. **Medium Delay (K=20)** – Same, with a 20-step accumulation window.
5. **Sparse (Terminal)** – The entire episode reward is delivered at
   termination; zero signal at all intermediate steps.

### 3.2 Robustness Sweep

REINFORCE+Norm is trained with medium-delay delivery for
\(K \in \{1, 5, 10, 15, 20, 30, 50, 100\}\) plus a sparse condition
(\(K = T\)).

### 3.3 Implementation: DelayedRewardWrapper

A lightweight wrapper sits between the agent and the Gymnasium environment.
It intercepts the reward signal and applies the delay logic while passing
all other quantities (states, termination flags) through unchanged.

```
Agent ↔ DelayedRewardWrapper ↔ robust_gymnasium.make("LunarLander-v3")
```

### 3.4 Metrics Collected

| Metric | Granularity | Purpose |
|--------|-------------|---------|
| Original episode reward | Per episode | Learning curve |
| Delivered reward | Per step | Reward timeline |
| G_t (return-to-go) | Per step | Value analog (no learned V(s)) |
| \|G̃_t\| (normalized return) | Per step | Gradient signal / credit assignment |
| Return-to-go variance | Per timestep | Estimation uncertainty |
| Final performance vs delay | Per delay value | Robustness profile |

**Note on metrics**: Since REINFORCE has no value network, Plot 3 shows the
Monte Carlo return-to-go \(G_t\) — the *ground truth* of what V(s) or Q(s,a)
would try to approximate.  Plot 4 uses \(|\tilde{G}_t|\) (the absolute
normalized return) as the credit assignment signal, since this is the actual
multiplier on each timestep's gradient in the REINFORCE update rule.


## 4. Visualizations

### Plot 1 – Reward Timeline (timestep vs reward)

Shows the **delivered** reward (bars) alongside the **original** reward (gray
line) for one evaluation episode under each mode.

**Insight**: Identical to DQN/PPO — this plot is algorithm-independent and
illustrates the raw sparsity pattern.

### Plot 2 – Return Variance (timestep vs Var(G_t))

Computes Var(return-to-go) at each timestep across 20 evaluation episodes.
Uses dual-panel layout (log scale + linear zoom excluding sparse).

**Insight**: REINFORCE is particularly affected by high return variance because
it has no baseline/critic to reduce variance.  The raw Monte Carlo return
is the only signal, so high variance directly translates to noisy gradients.

### Plot 3 – Return-to-Go Profile (timestep vs G_t)

Plots the mean return-to-go \(G_t\) across evaluation episodes.  This is
the ground-truth value that DQN's Q-network or PPO's critic *tries to learn*.

**Insight**: Under dense rewards, \(G_t\) is smooth and monotonically
decreasing (less future reward remaining).  Under sparse rewards, \(G_t\)
becomes a pure exponential decay \(\gamma^{T-t} R_\text{total}\) — all
temporal structure in the value landscape is erased.

### Plot 4 – Credit Assignment Heatmap (episodes × timesteps)

A heatmap of \(|\tilde{G}_t|\) for each (episode, timestep) pair.  For
REINFORCE, this is the direct gradient signal magnitude.

**Insight**: Under dense rewards, different timesteps receive different
gradient weights (credit is discriminative).  Under sparse rewards, after
normalization, \(|\tilde{G}_t|\) becomes nearly uniform across timesteps
(credit is spread indiscriminately), destroying temporal specificity.

### Plot 5 – Delayed Reward Robustness (delay steps vs performance)

The robustness sweep: x-axis is delay length \(K\), y-axis is mean episode
reward over the last 50 training episodes.

**Insight**: REINFORCE may show a different degradation profile than DQN/PPO.
Since it uses complete returns rather than bootstrapping, moderate delays may
cause less damage.  However, sparse rewards should still be devastating due to
the loss of temporal credit assignment specificity.

### Plot 6 – Learning Curve (training episode vs reward)

Smoothed training reward (original) over 1000 episodes for all five configs.

**Insight**: REINFORCE generally converges more slowly than DQN/PPO and
exhibits higher variance.  With 1000 episodes (vs 500 for DQN/PPO), we
give it a fair chance to converge under each reward structure.


## 5. Comparative Analysis: REINFORCE vs DQN vs PPO

| Aspect | DQN | PPO | REINFORCE+Norm |
|--------|-----|-----|----------------|
| Update type | TD (per-step) | Actor-critic (GAE) | Monte Carlo (per-episode) |
| Value function | Q-network | Critic V(s) | None |
| Bootstrapping | Yes (1-step TD) | Yes (GAE) | No |
| Variance reduction | Replay buffer | GAE + critic | Return normalization only |
| Theoretical delay sensitivity | High | Medium | Low (but practically still affected) |
| Credit assignment mechanism | TD-error | Advantage estimate | Raw normalized return |

**Key hypothesis**: REINFORCE should be the *most robust to moderate delays*
(since it already waits for complete episodes) but the *least efficient* overall
(since it has the highest gradient variance).  Under sparse rewards, all three
algorithms should fail, but for different reasons:
- **DQN**: TD-error = 0 at non-terminal steps → no gradient propagation
- **PPO**: Critic fails to learn → advantage estimates become noise
- **REINFORCE**: Normalized returns become uniform → gradient is temporally blind


## 6. Practical Implications

- **REINFORCE has a natural advantage with delayed rewards** because it never
  bootstraps — it always uses the true Monte Carlo return.  This makes it a
  reasonable baseline for delayed-reward domains where value-based methods fail.

- **Return normalization is a double-edged sword**: it stabilizes training
  under dense rewards but *homogenizes the gradient signal* under sparse
  rewards, removing the temporal discrimination needed for credit assignment.

- **For delayed-reward tasks**, consider combining REINFORCE's Monte Carlo
  returns with a learned baseline (i.e., REINFORCE with baseline / A2C) to
  get both the complete-return robustness and variance reduction.


## 7. Files

| File | Description |
|------|-------------|
| `train_delayed_reward.py` | Training and evaluation script |
| `visualize_delayed_reward.py` | Generates all 6 visualizations |
| `solution_delayed_reward.md` | This document |
| `results/delayed_reward_reinforce/experiment_results.pkl` | Serialized experiment data |
| `results/delayed_reward_reinforce/reinforce_*.pth` | Trained model checkpoints |
| `results/delayed_reward_reinforce/1_reward_timeline.png` | Plot 1 |
| `results/delayed_reward_reinforce/2_return_variance.png` | Plot 2 |
| `results/delayed_reward_reinforce/3_value_prediction.png` | Plot 3 |
| `results/delayed_reward_reinforce/4_credit_assignment.png` | Plot 4 |
| `results/delayed_reward_reinforce/5_robustness.png` | Plot 5 |
| `results/delayed_reward_reinforce/6_learning_curve.png` | Plot 6 |


## 8. Reproduction

```bash
conda activate robustgymnasium
cd Robust-Gymnasium

# Phase 1+2: train all configurations (~20-30 min on GPU)
python examples/LunarLander_REINFORCE/train_delayed_reward.py

# Generate all 6 plots
python examples/LunarLander_REINFORCE/visualize_delayed_reward.py
```

## 9. References

- Williams, "Simple statistical gradient-following algorithms for
  connectionist reinforcement learning," *Machine Learning*, 1992.
- Ng, Harada & Russell, "Policy invariance under reward transformations,"
  *ICML*, 1999.
- Arjona-Medina et al., "RUDDER: Return Decomposition for Delayed Rewards,"
  *NeurIPS*, 2019.
- Gu et al., "Robust Gymnasium: A Unified Modular Benchmark for Robust
  Reinforcement Learning," *ICLR*, 2025.


实验结果总结

配置	训练 Avg(last100)	评估 Avg
Baseline (Default)	77.0	159.3
Medium Delay (K=10)	80.8	107.0
Medium Delay (K=20)	42.9	-41.6
Dense (Extra Shaping)	-5.5	-9.0
Sparse (Terminal Only)	-272.2	-692.9

REINFORCE 独特发现（与 DQN/PPO 的关键差异）

对中等延迟更鲁棒：REINFORCE 在 K=10 时仍能达到 107 分（评估），而 DQN 在 K=10 时就已经降到 -17 分。这验证了"蒙特卡洛方法天然不依赖 bootstrapping"的优势。

鲁棒性曲线退化更平缓（图5）：delay=115 几乎水平，delay=2050 开始下滑，delay=100 才急剧恶化。相比 DQN 在 K=15 就断崖式下跌，REINFORCE 的容忍阈值明显更高。

信用分配热力图（图4）：由于使用归一化回报 |G̃_t| 而非 TD-error，REINFORCE 的梯度信号在时间步之间分布更均匀，但 Sparse 模式下信号完全同质化——所有时间步获得几乎相同的梯度权重，丧失了时间辨别能力。

回报-前瞻曲线（图3）：Sparse 模式下 G_t 呈现剧烈的负值指数下降（因为坠毁积累的大量负奖励集中在终止时），直接显示了稀疏奖励对值景观的毁灭性扭曲。

Dense shaping 对 REINFORCE 反而不利（-5.5 vs Baseline 77.0）：额外的 shaping 引入了与原始奖励信号不一致的梯度方向，导致纯策略梯度方法更容易被误导——这与 DQN 中 Dense 表现最好形成鲜明对比。