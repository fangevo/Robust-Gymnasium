# DQN Solution for LunarLander-v3 (Discrete) — Technical Document

## 1. Problem Description

**Environment:** Robust Gymnasium `LunarLander-v3` (Discrete)

The agent controls a lunar lander attempting to land safely on a landing pad. The episode ends when the lander crashes, lands, or exceeds the step limit (1000 steps). A score ≥ 200 (averaged over 100 episodes) is considered solved.

| Property | Detail |
|---|---|
| Observation space | `Box(8,)` — position (x, y), velocity (vx, vy), angle, angular velocity, left/right leg contact |
| Action space | `Discrete(4)` — 0: nop, 1: left engine, 2: main engine, 3: right engine |
| Reward threshold | 200 |
| Max steps | 1000 |

**Perturbation setting:** None (`noise_factor="none"`, `noise_sigma=0.0`). This serves as the baseline before introducing robust perturbations.

---

## 2. Algorithm: Deep Q-Network (DQN)

### 2.1 Core Idea

DQN (Mnih et al., 2015) extends tabular Q-learning to high-dimensional state spaces by approximating the action-value function $Q(s, a)$ with a neural network. Two key techniques stabilize training:

1. **Experience Replay** — transitions are stored in a buffer and sampled uniformly at random, breaking temporal correlation.
2. **Target Network** — a separate slowly-updated copy of the Q-network provides stable TD targets.

### 2.2 Bellman Target

The loss for a sampled mini-batch is:

$$L(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left( r + \gamma \max_{a'} Q_{\bar{\theta}}(s', a') - Q_\theta(s, a) \right)^2 \right]$$

where $Q_\theta$ is the online network, $Q_{\bar{\theta}}$ is the target network, and $\gamma$ is the discount factor.

### 2.3 Epsilon-Greedy Exploration

$$a_t = \begin{cases} \text{random action} & \text{with probability } \epsilon \\ \arg\max_a Q_\theta(s_t, a) & \text{otherwise} \end{cases}$$

Epsilon decays multiplicatively each episode: $\epsilon_{k+1} = \max(\epsilon_{\min},\; \epsilon_k \times d)$ where $d = 0.995$.

### 2.4 Soft Target Update

Instead of periodic hard copies, we use Polyak averaging after every learning step:

$$\bar{\theta} \leftarrow \tau \theta + (1 - \tau) \bar{\theta}, \quad \tau = 10^{-3}$$

---

## 3. Network Architecture

```
Input (8) → Linear(128) → ReLU → Linear(128) → ReLU → Linear(4) → Q-values
```

A simple two-hidden-layer MLP. No batch normalization or dropout — the problem is low-dimensional and does not require heavy regularization.

---

## 4. Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| Learning rate | 5×10⁻⁴ | Standard for DQN on low-dim tasks |
| Discount γ | 0.99 | Long-horizon task benefits from high γ |
| Batch size | 64 | Good trade-off between gradient noise and speed |
| Replay buffer | 100,000 | Sufficient to decorrelate samples |
| Soft-update τ | 10⁻³ | Slow target updates for stability |
| ε start / end / decay | 1.0 / 0.01 / 0.995 | Ensures broad early exploration, converging to near-greedy |
| Hidden dim | 128 | Enough capacity for 8-dim input |
| Learn every | 4 steps | Reduces computation without hurting convergence |
| Total episodes | 600 | Sufficient to approach and nearly solve the task |
| Gradient clip | 1.0 | Prevents large gradient spikes |

---

## 5. Implementation Details

### 5.1 File Structure

```
examples/LunarLander_DQN/
├── train_dqn.py           # Complete DQN training script
└── results/
    ├── training_curves.png # Reward & epsilon plots
    ├── dqn_final.pth       # Final model weights
    └── dqn_solved.pth      # Model at solve point (if reached)
```

### 5.2 Environment Interface

Robust Gymnasium's `step()` requires a dict-based input:

```python
robust_input = {
    "action": action,
    "robust_type": "action",
    "robust_config": args,   # argparse namespace from robust_setting
}
observation, reward, terminated, truncated, info = env.step(robust_input)
```

With `noise_factor="none"` and `noise_sigma=0.0`, no perturbation is applied.

### 5.3 Key Design Choices

- **No double DQN / dueling / PER**: We use standard DQN to establish a clean baseline. These extensions can be added later for comparison.
- **Gradient clipping** (`max_norm=1.0`): Stabilizes early training when the Q-values are still poorly calibrated.
- **Matplotlib `Agg` backend**: Allows headless plotting on servers without a display.

---

## 6. Training Results

Training over **600 episodes** with **CUDA** acceleration:

| Milestone | Episode | Avg Reward (100 ep) |
|---|---|---|
| Early exploration | 100 | −135 |
| Policy improving | 300 | −31 |
| Approaching solve | 500 | +135 |
| Final | 600 | +188 |

The agent steadily improved from random behaviour (~−190) to near-solved performance (~188). With additional episodes or hyperparameter tuning, it would cross the 200 threshold.

### Training Curves

The generated `results/training_curves.png` contains:
- **Top panel**: Per-episode reward (blue) and 100-episode moving average (orange), with the solve threshold (green dashed line at 200).
- **Bottom panel**: Epsilon decay curve showing exploration rate decreasing from 1.0 to ~0.05.

---

## 7. How to Run

```bash
conda activate robustgymnasium
cd Robust-Gymnasium
python examples/LunarLander_DQN/train_dqn.py
```

Results are saved to `examples/LunarLander_DQN/results/`.

---

## 8. Future Extensions

| Extension | Purpose |
|---|---|
| Add perturbation (`noise_factor="state"`) | Evaluate DQN robustness under observation noise |
| Double DQN | Reduce Q-value overestimation |
| Dueling architecture | Better value/advantage decomposition |
| Prioritized experience replay | Focus learning on high-TD-error transitions |
| Compare robust vs. non-robust training | Core research question of Robust Gymnasium |

---

## References

1. Mnih, V., et al. (2015). *Human-level control through deep reinforcement learning*. Nature, 518(7540), 529–533.
2. Gu, S., et al. (2025). *Robust Gymnasium: A Unified Modular Benchmark for Robust Reinforcement Learning*. ICLR 2025.
