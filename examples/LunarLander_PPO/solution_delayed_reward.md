# Delayed Reward Structure Analysis – PPO on LunarLander-v3

## 1. Problem Formulation

The **credit assignment problem** is a central challenge in reinforcement
learning: which past actions are responsible for the rewards the agent
observes?  This experiment systematically varies the *temporal density* of
reward delivery while keeping total episode return invariant, isolating the
pure effect of delay on learning.

We study five reward delivery modes on the LunarLander-v3 (Discrete) task
using a Proximal Policy Optimization (PPO) agent — an on-policy,
actor-critic method that uses Generalized Advantage Estimation (GAE) for
variance-reduced policy gradients.

| Mode | Delivered reward at step t |
|------|---------------------------|
| **Baseline** (dense, K=0) | r̃_t = r_t |
| **Dense** (extra shaping) | r̃_t = r_t + Φ(s_t), additional per-step bonus |
| **Medium delay** (K=10/20) | r̃_t = Σ r_i every K steps; 0 otherwise |
| **Sparse** (K=T) | r̃_t = 0 for t < T; r̃_T = Σ r_i at episode end |


## 2. Algorithm: PPO with GAE

### Architecture

| Component | Specification |
|-----------|--------------|
| Actor | MLP 8→64→64→4, Tanh activations, softmax output |
| Critic | MLP 8→64→64→1, Tanh activations |
| Learning rate | 3 × 10⁻⁴ (Adam, separate for actor/critic) |
| Clip range ε | 0.2 |
| GAE λ | 0.95 |
| γ | 0.99 |
| Mini-batch size | 64 |
| PPO epochs | 10 per rollout |
| Rollout length | 2048 steps |
| Grad clip | 0.5 |
| Training episodes | 500 (main) / 300 (robustness sweep) |

### Why PPO + Delayed Rewards is Interesting

PPO's advantage estimation relies on the **temporal difference** signal
δ_t = r_t + γ V(s_{t+1}) − V(s_t).  When rewards are delayed:

- Most δ_t collapse to ≈ γV(s') − V(s) (near-zero gradient signal)
- GAE accumulates these near-zero deltas, producing noisy advantages
- The policy gradient becomes high-variance and low-signal
- Unlike DQN's replay buffer which averages over many transitions, PPO
  discards data after each update — making it *more sensitive* to reward
  sparsity


## 3. Experimental Design

### 3.1 Reward Configurations

Identical to the DQN experiment for direct comparison:

1. **Baseline** – Unmodified LunarLander-v3 reward
2. **Dense** – Baseline + bonus: 0.2·(leg1+leg2) − 0.3·|x| − 0.2·|θ|
3. **Medium K=10** – Rewards accumulated, delivered every 10 steps
4. **Medium K=20** – Rewards accumulated, delivered every 20 steps
5. **Sparse** – Entire episode reward delivered at termination only

### 3.2 Robustness Sweep

PPO trained with medium-delay delivery for
K ∈ {1, 5, 10, 15, 20, 30, 50, 100} plus sparse (K=T).

### 3.3 Evaluation Metrics

| Metric | Key | Meaning for PPO |
|--------|-----|-----------------|
| V(s) | `q_values` | Critic's state value prediction |
| \|TD-error\| | `td_errors` | \|r + γV(s') − V(s)\| |
| Original reward | `rewards` | True environment reward |
| Delivered reward | `delivered_rewards` | What PPO trains on |


## 4. Visualizations

### Plot 1 – Reward Timeline
Shows delivered vs original reward per timestep.  Same structure as
the DQN version with y-axis clipped to ±30 and overflow annotations.

### Plot 2 – Return Variance
Var(return-to-go) at each timestep across 20 evaluation episodes.
PPO's on-policy nature may show different variance profiles than DQN.

### Plot 3 – Value Prediction: V(s) Over Episode
Mean V(s) from the critic network.  Compared to DQN's Q(s,a), this
shows the critic's estimate of expected future return from each state.

### Plot 4 – Credit Assignment Heatmap (log scale)
TD-error magnitude heatmap with logarithmic color normalization.
PPO's TD-errors |r + γV(s') − V(s)| directly reveal where the
advantage signal concentrates.

### Plot 5 – Robustness Curve
Final performance vs delay length.  PPO may degrade differently from
DQN due to on-policy data requirements and GAE's sensitivity to
reward density.

### Plot 6 – Learning Curve
Training episode reward for all configurations.  PPO's sample
efficiency characteristics under delayed rewards.


## 5. PPO vs DQN Under Delayed Rewards

| Aspect | DQN | PPO |
|--------|-----|-----|
| Data reuse | Replay buffer (off-policy) | Single rollout (on-policy) |
| Value target | r + γ max Q(s',a') | r + γ V(s') via GAE |
| Credit assignment | TD(0) backup | GAE(γ,λ) multi-step |
| Expected robustness | Moderate | Lower (on-policy, no replay) |

**Hypothesis**: PPO should be *more* affected by reward delay than DQN
because:
1. On-policy data is used once and discarded — noisy advantages from
   delayed rewards cannot be averaged over many samples
2. GAE relies on a chain of TD-errors; zeros in the reward signal break
   the information chain
3. The policy gradient has inherently higher variance than Q-learning's
   regression objective


## 6. Files

| File | Description |
|------|-------------|
| `train_delayed_reward.py` | PPO training and evaluation |
| `visualize_delayed_reward.py` | Generates all 6 visualizations |
| `solution_delayed_reward.md` | This document |
| `results/delayed_reward_ppo/experiment_results.pkl` | Experiment data |
| `results/delayed_reward_ppo/ppo_*.pth` | Model checkpoints |
| `results/delayed_reward_ppo/1–6_*.png` | Visualization plots |


## 7. Reproduction

```bash
conda activate robustgymnasium
cd Robust-Gymnasium

# Train all configurations (~15-20 min on GPU)
python examples/LunarLander_PPO/train_delayed_reward.py

# Generate all 6 plots
python examples/LunarLander_PPO/visualize_delayed_reward.py
```


## 8. References

- Schulman et al., "Proximal Policy Optimization Algorithms," arXiv, 2017.
- Schulman et al., "High-Dimensional Continuous Control Using Generalized
  Advantage Estimation," ICLR, 2016.
- Arjona-Medina et al., "RUDDER: Return Decomposition for Delayed Rewards,"
  NeurIPS, 2019.
- Gu et al., "Robust Gymnasium: A Unified Modular Benchmark for Robust
  Reinforcement Learning," ICLR, 2025.


Results (results/delayed_reward_ppo/)
6 visualization PNGs + 5 model checkpoints + experiment data pickle

PPO Experimental Results
Configuration	Train Avg (last 100)	Eval Avg
Baseline (Default)	110.4	223.7
Dense (Extra Shaping)	28.6	-31.2
Medium Delay (K=10)	-106.9	-106.6
Medium Delay (K=20)	-41.0	-221.2
Sparse (Terminal Only)	-294.7	-805.5

Key Observations (PPO vs DQN)

PPO Baseline outperforms DQN Baseline in eval (223.7 vs 159.2) despite similar training curves, demonstrating PPO's policy optimization strength with dense rewards.

PPO is more sensitive to reward delay than DQN — the performance gap between Baseline and Sparse is larger for PPO (1029 points) vs DQN (690 points), confirming the hypothesis that on-policy methods suffer more from sparse credit assignment.

Value Prediction (Plot 3) reveals a stark contrast: PPO's Baseline critic learns a smooth, high V(s) trajectory (70 peak), while Sparse produces deeply negative values (-150), showing complete failure to estimate state value. Medium delay agents have near-flat V(s) ≈ 0.

Credit Assignment Heatmap (Plot 4) with log scale shows PPO Dense has the most uniform TD-error distribution, while Sparse concentrates all signal at the terminal step — a pattern consistent with DQN but with sharper contrast.

Robustness Curve (Plot 5) shows monotonic degradation from delay=1 (-20) to sparse (-240), with a steeper slope than DQN, confirming PPO's greater vulnerability to delayed rewards