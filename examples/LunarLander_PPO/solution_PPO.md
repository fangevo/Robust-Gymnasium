# PPO Solution for LunarLander-v3 (Discrete) — Technical Document

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

## 2. Algorithm: Proximal Policy Optimization (PPO)

### 2.1 Core Idea

PPO (Schulman et al., 2017) is an on-policy actor-critic algorithm that optimizes policies via multiple epochs of mini-batch updates while constraining policy change. Unlike DQN's value-based approach, PPO directly learns a stochastic policy $\pi_\theta(a|s)$.

**Key advantages over DQN:**
1. **On-policy learning** — more stable and sample-efficient for continuous control
2. **Direct policy optimization** — naturally handles stochastic policies
3. **Clipped objective** — prevents destructively large policy updates

### 2.2 Clipped Surrogate Objective

The PPO loss combines policy loss, value loss, and entropy bonus:

$$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

where:
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ is the probability ratio
- $\hat{A}_t$ is the advantage estimate (GAE)
- $\epsilon = 0.2$ is the clip range

**Value loss:**
$$L^{VF} = \mathbb{E}_t \left[ \left( V_\phi(s_t) - \hat{R}_t \right)^2 \right]$$

**Total loss:**
$$L = L^{CLIP} - c_1 L^{VF} + c_2 H[\pi_\theta]$$

where $H[\pi_\theta]$ is the policy entropy for exploration.

### 2.3 Generalized Advantage Estimation (GAE)

PPO uses GAE($\lambda$) to compute advantages:

$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the TD error.

---

## 3. Network Architecture

**Actor Network (Policy):**