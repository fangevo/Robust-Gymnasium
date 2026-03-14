"""Utilities and agents for Experiment 2: on-policy PPO vs off-policy DQN variants.

This module provides a unified training loop and a consistent logging schema so
that algorithm comparisons are fair and reproducible.
"""

from __future__ import annotations

import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import robust_gymnasium as gym
from robust_gymnasium.configs.robust_setting import get_config


ALL_ALGOS = ["DQN", "DoubleDQN", "DuelingDDQN", "PPO"]


@dataclass
class TrainConfig:
    env_name: str = "LunarLander-v3"
    solve_score: float = 200.0
    moving_avg_window: int = 100
    max_episodes: int = 1000
    max_steps_per_episode: int = 1000
    gamma: float = 0.99

    # Robust-Gymnasium perturbation defaults
    # Default is a perturbed setting to satisfy Experiment 2 robustness runs.
    noise_factor: str = "state"
    noise_type: str = "gauss"
    noise_sigma: float = 0.05
    noise_mu: float = 0.0
    robust_type: str = "action"

    # DQN-family defaults
    lr_dqn: float = 5e-4
    batch_size: int = 64
    buffer_size: int = 100_000
    learning_starts: int = 5000
    tau: float = 1e-3
    update_every: int = 4
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay: float = 0.997
    q_hidden_dim: int = 128

    # Dueling + PER defaults
    per_alpha: float = 0.4
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    dueling_hidden_dim: int = 128
    dueling_max_grad_norm: float = 1.0

    # PPO defaults
    lr_ppo: float = 1e-4
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_coef: float = 0.5
    entropy_coef_start: float = 0.02
    entropy_coef_end: float = 0.005
    ppo_target_kl: float = 0.03
    ppo_batch_size: int = 64
    ppo_epochs: int = 4
    ppo_rollout_steps: int = 1024
    ppo_max_grad_norm: float = 0.5


@dataclass
class TrainingTrace:
    algorithm: str
    seed: int
    episode_rewards: List[float]
    moving_avg_rewards: List[float]
    total_timesteps: List[int]
    wall_times: List[float]
    convergence_episode: Optional[int]
    convergence_timestep: Optional[int]
    convergence_wall_time: Optional[float]


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            np.array(s, dtype=np.float32),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.array(ns, dtype=np.float32),
            np.array(d, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer: List = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        max_prio = float(self.priorities[: self.size].max()) if self.size > 0 else 1.0
        if self.size < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float):
        prios = self.priorities[: self.size] ** self.alpha
        probs = prios / prios.sum()
        indices = np.random.choice(self.size, batch_size, replace=False, p=probs)
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()

        batch = [self.buffer[i] for i in indices]
        s, a, r, ns, d = zip(*batch)
        return (
            np.array(s, dtype=np.float32),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.array(ns, dtype=np.float32),
            np.array(d, dtype=np.float32),
            indices,
            weights.astype(np.float32),
        )

    def update_priorities(self, indices, td_errors: np.ndarray, eps: float = 1e-6):
        for idx, err in zip(indices, td_errors):
            self.priorities[idx] = float(abs(err)) + eps

    def __len__(self):
        return self.size


class MLPQNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DuelingQNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.feature = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU())
        self.value = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature(x)
        v = self.value(feat)
        a = self.advantage(feat)
        return v + a - a.mean(dim=1, keepdim=True)


class BaseAgent:
    def select_action(self, state: np.ndarray) -> int:
        raise NotImplementedError

    def observe(self, state, action, reward, next_state, done):
        raise NotImplementedError

    def finish_episode(self):
        pass


class DQNAgent(BaseAgent):
    def __init__(self, state_dim: int, action_dim: int, device: torch.device, cfg: TrainConfig):
        self.device = device
        self.action_dim = action_dim
        self.cfg = cfg

        self.qnet = MLPQNet(state_dim, action_dim, hidden=cfg.q_hidden_dim).to(device)
        self.target_net = MLPQNet(state_dim, action_dim, hidden=cfg.q_hidden_dim).to(device)
        self.target_net.load_state_dict(self.qnet.state_dict())

        self.optimizer = optim.Adam(self.qnet.parameters(), lr=cfg.lr_dqn)
        self.buffer = ReplayBuffer(cfg.buffer_size)
        self.epsilon = cfg.eps_start
        self.step_count = 0

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return int(self.qnet(s).argmax(dim=1).item())

    def _compute_target(self, next_states_t: torch.Tensor, rewards_t: torch.Tensor, dones_t: torch.Tensor):
        with torch.no_grad():
            max_next_q = self.target_net(next_states_t).max(dim=1, keepdim=True)[0]
            return rewards_t + self.cfg.gamma * max_next_q * (1.0 - dones_t)

    def observe(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)
        self.step_count += 1
        if self.step_count < self.cfg.learning_starts:
            return
        if self.step_count % self.cfg.update_every != 0:
            return
        if len(self.buffer) < self.cfg.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.cfg.batch_size)
        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.qnet(states_t).gather(1, actions_t)
        target = self._compute_target(next_states_t, rewards_t, dones_t)
        loss = F.mse_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.qnet.parameters(), max_norm=1.0)
        self.optimizer.step()

        for tp, op in zip(self.target_net.parameters(), self.qnet.parameters()):
            tp.data.copy_(self.cfg.tau * op.data + (1.0 - self.cfg.tau) * tp.data)

    def finish_episode(self):
        self.epsilon = max(self.cfg.eps_end, self.epsilon * self.cfg.eps_decay)


class DoubleDQNAgent(DQNAgent):
    def _compute_target(self, next_states_t: torch.Tensor, rewards_t: torch.Tensor, dones_t: torch.Tensor):
        with torch.no_grad():
            next_actions = self.qnet(next_states_t).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states_t).gather(1, next_actions)
            return rewards_t + self.cfg.gamma * next_q * (1.0 - dones_t)


class DuelingDDQNAgent(BaseAgent):
    def __init__(self, state_dim: int, action_dim: int, device: torch.device, cfg: TrainConfig):
        self.device = device
        self.action_dim = action_dim
        self.cfg = cfg

        self.qnet = DuelingQNet(state_dim, action_dim, hidden=cfg.dueling_hidden_dim).to(device)
        self.target_net = DuelingQNet(state_dim, action_dim, hidden=cfg.dueling_hidden_dim).to(device)
        self.target_net.load_state_dict(self.qnet.state_dict())

        self.optimizer = optim.Adam(self.qnet.parameters(), lr=cfg.lr_dqn)
        self.buffer = PrioritizedReplayBuffer(cfg.buffer_size, alpha=cfg.per_alpha)
        self.epsilon = cfg.eps_start
        self.step_count = 0
        self.training_progress = 0.0

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return int(self.qnet(s).argmax(dim=1).item())

    def observe(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)
        self.step_count += 1
        if self.step_count < self.cfg.learning_starts:
            return
        if self.step_count % self.cfg.update_every != 0:
            return
        if len(self.buffer) < self.cfg.batch_size:
            return

        beta = min(
            self.cfg.per_beta_end,
            self.cfg.per_beta_start + (self.cfg.per_beta_end - self.cfg.per_beta_start) * self.training_progress,
        )

        states, actions, rewards, next_states, dones, indices, weights = self.buffer.sample(
            self.cfg.batch_size, beta=beta
        )

        s = torch.tensor(states, dtype=torch.float32, device=self.device)
        a = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        r = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        ns = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        d = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        w = torch.tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_pred = self.qnet(s).gather(1, a)
        with torch.no_grad():
            best_next_action = self.qnet(ns).argmax(dim=1, keepdim=True)
            q_next = self.target_net(ns).gather(1, best_next_action)
            q_target = r + self.cfg.gamma * q_next * (1.0 - d)

        td_error = (q_pred - q_target).detach().cpu().numpy().flatten()
        element_loss = F.smooth_l1_loss(q_pred, q_target, reduction="none")
        loss = (w * element_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.qnet.parameters(), max_norm=self.cfg.dueling_max_grad_norm)
        self.optimizer.step()

        self.buffer.update_priorities(indices, td_error)
        for tp, op in zip(self.target_net.parameters(), self.qnet.parameters()):
            tp.data.copy_(self.cfg.tau * op.data + (1.0 - self.cfg.tau) * tp.data)

    def finish_episode(self):
        self.epsilon = max(self.cfg.eps_end, self.epsilon * self.cfg.eps_decay)


class PPOActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(x), dim=-1)


class PPOCritic(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class PPOAgent(BaseAgent):
    def __init__(self, state_dim: int, action_dim: int, device: torch.device, cfg: TrainConfig):
        self.device = device
        self.cfg = cfg
        self.actor = PPOActor(state_dim, action_dim).to(device)
        self.critic = PPOCritic(state_dim).to(device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=cfg.lr_ppo)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=cfg.lr_ppo)

        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.dones: List[float] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []

        self._last_log_prob = 0.0
        self._last_value = 0.0
        self.training_progress = 0.0

    def select_action(self, state: np.ndarray) -> int:
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            probs = self.actor(s)
            dist = Categorical(probs)
            action = dist.sample()
            value = self.critic(s)
        self._last_log_prob = float(dist.log_prob(action).item())
        self._last_value = float(value.item())
        return int(action.item())

    def observe(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.log_probs.append(self._last_log_prob)
        self.values.append(self._last_value)

        if len(self.states) >= self.cfg.ppo_rollout_steps:
            self._update(next_state)

    def _compute_gae(self, rewards, dones, values, next_value: float):
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_val = next_value if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + self.cfg.gamma * next_val * (1.0 - dones[t]) - values[t]
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values
        return advantages, returns

    def _update(self, next_state):
        states = np.asarray(self.states, dtype=np.float32)
        actions = np.asarray(self.actions, dtype=np.int64)
        rewards = np.asarray(self.rewards, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        old_log_probs = np.asarray(self.log_probs, dtype=np.float32)
        values = np.asarray(self.values, dtype=np.float32)

        next_state_t = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            next_value = float(self.critic(next_state_t).item())

        advantages, returns = self._compute_gae(rewards, dones, values, next_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device)
        old_log_probs_t = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)

        n_samples = len(states)
        indices = np.arange(n_samples)
        entropy_coef = (
            self.cfg.entropy_coef_start
            + (self.cfg.entropy_coef_end - self.cfg.entropy_coef_start) * self.training_progress
        )

        stop_early = False
        for _ in range(self.cfg.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, n_samples, self.cfg.ppo_batch_size):
                end = start + self.cfg.ppo_batch_size
                batch_idx = indices[start:end]

                probs = self.actor(states_t[batch_idx])
                dist = Categorical(probs)
                new_log_probs = dist.log_prob(actions_t[batch_idx])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_probs_t[batch_idx])
                surr1 = ratio * advantages_t[batch_idx]
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.cfg.clip_range,
                    1.0 + self.cfg.clip_range,
                ) * advantages_t[batch_idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                approx_kl = (old_log_probs_t[batch_idx] - new_log_probs).mean().item()

                value_pred = self.critic(states_t[batch_idx])
                value_loss = F.mse_loss(value_pred, returns_t[batch_idx])

                total_loss = (
                    policy_loss
                    + self.cfg.value_coef * value_loss
                    - entropy_coef * entropy
                )

                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.ppo_max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.ppo_max_grad_norm)
                self.actor_optim.step()
                self.critic_optim.step()

                if approx_kl > self.cfg.ppo_target_kl:
                    stop_early = True
                    break
            if stop_early:
                break

        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()

    def finish_episode(self):
        # Keep partial trajectories between episodes; update happens after enough steps.
        pass


def _build_robust_args(cfg: TrainConfig):
    args = get_config().parse_args([])
    args.noise_factor = cfg.noise_factor
    args.noise_type = cfg.noise_type
    args.noise_sigma = cfg.noise_sigma
    args.noise_mu = cfg.noise_mu
    return args


def parse_algorithms(raw_algorithms: Sequence[str]) -> List[str]:
    normalized = []
    for item in raw_algorithms:
        key = item.strip()
        if not key:
            continue
        if key not in ALL_ALGOS:
            raise ValueError(f"Unsupported algorithm: {key}. Choose from: {ALL_ALGOS}")
        normalized.append(key)
    if not normalized:
        raise ValueError("At least one algorithm must be provided.")
    return normalized


def make_agent(algo_name: str, state_dim: int, action_dim: int, device: torch.device, cfg: TrainConfig) -> BaseAgent:
    if algo_name == "DQN":
        return DQNAgent(state_dim, action_dim, device, cfg)
    if algo_name == "DoubleDQN":
        return DoubleDQNAgent(state_dim, action_dim, device, cfg)
    if algo_name == "DuelingDDQN":
        return DuelingDDQNAgent(state_dim, action_dim, device, cfg)
    if algo_name == "PPO":
        return PPOAgent(state_dim, action_dim, device, cfg)
    raise ValueError(f"Unknown algorithm: {algo_name}")


def train_agent(
    algo_name: str,
    seed: int,
    cfg: TrainConfig,
    device: torch.device,
    stop_on_convergence: bool,
) -> TrainingTrace:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make(cfg.env_name)
    robust_args = _build_robust_args(cfg)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = make_agent(algo_name, state_dim, action_dim, device, cfg)

    rewards: List[float] = []
    moving_avg_rewards: List[float] = []
    total_timesteps: List[int] = []
    wall_times: List[float] = []

    recent = deque(maxlen=cfg.moving_avg_window)
    t0 = time.perf_counter()
    env_steps = 0

    convergence_episode: Optional[int] = None
    convergence_timestep: Optional[int] = None
    convergence_wall_time: Optional[float] = None

    for episode in range(1, cfg.max_episodes + 1):
        # Normalized progress for PER beta schedule.
        if isinstance(agent, DuelingDDQNAgent):
            agent.training_progress = episode / cfg.max_episodes
        elif isinstance(agent, PPOAgent):
            agent.training_progress = episode / cfg.max_episodes

        state, _ = env.reset(seed=seed + episode)
        ep_reward = 0.0

        for _ in range(cfg.max_steps_per_episode):
            action = agent.select_action(state)
            robust_input = {
                "action": action,
                "robust_type": cfg.robust_type,
                "robust_config": robust_args,
            }
            next_state, reward, terminated, truncated, _ = env.step(robust_input)
            done = terminated or truncated

            agent.observe(state, action, float(reward), next_state, done)
            state = next_state
            ep_reward += float(reward)
            env_steps += 1

            if done:
                break

        agent.finish_episode()

        rewards.append(ep_reward)
        recent.append(ep_reward)
        moving_avg = float(np.mean(recent))
        moving_avg_rewards.append(moving_avg)
        total_timesteps.append(env_steps)
        wall_times.append(time.perf_counter() - t0)

        if (
            convergence_episode is None
            and len(recent) == cfg.moving_avg_window
            and moving_avg >= cfg.solve_score
        ):
            convergence_episode = episode
            convergence_timestep = env_steps
            convergence_wall_time = wall_times[-1]
            if stop_on_convergence:
                break

    env.close()

    return TrainingTrace(
        algorithm=algo_name,
        seed=seed,
        episode_rewards=rewards,
        moving_avg_rewards=moving_avg_rewards,
        total_timesteps=total_timesteps,
        wall_times=wall_times,
        convergence_episode=convergence_episode,
        convergence_timestep=convergence_timestep,
        convergence_wall_time=convergence_wall_time,
    )
