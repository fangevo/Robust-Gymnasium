# Experiment 2: On-policy vs Off-policy on LunarLander

This folder implements a full comparison between:
- On-policy: PPO (Actor-Critic)
- Off-policy: DQN, Double DQN, Dueling DDQN+PER (value-based)

Goal:
- Compare sample efficiency and time efficiency (Experiment 2A)
- Compare exploration behavior and training stability variance (Experiment 2B)

## 1. Why this experiment is meaningful

PPO and DQN-family methods optimize different objectives and use data differently:
- DQN-family reuses old transitions from replay buffer, often improving sample efficiency.
- PPO updates policy with fresh trajectories, usually with smoother policy improvement and better late-stage stability.

This experiment quantifies those differences using the same task and a unified training/logging pipeline.

## 2. Folder structure

- common_agents.py: shared agents and unified training loop
- run_experiment_2a.py: convergence-oriented comparison (timesteps vs wall-clock)
- run_experiment_2b.py: multi-seed variance and stability analysis
- outputs/: generated automatically after running scripts

## 3. Experiment 2A (Sample efficiency vs Time efficiency)

### 3.1 Protocol

- Environment: LunarLander-v3 (discrete) under perturbation
- Default perturbation: state Gaussian noise (noise_factor=state, noise_sigma=0.05)
- Algorithms: DQN, DoubleDQN, DuelingDDQN, PPO
- Convergence criterion: moving average reward over 100 episodes >= 200
- Stop rule: each algorithm stops immediately at first convergence
- Metrics:
  - Total timesteps to convergence
  - Wall-clock time to convergence

### 3.2 Run command

From repository root:

```bash
python examples/experiment_2_on_off_policy/run_experiment_2a.py \
  --algorithms DQN,DoubleDQN,DuelingDDQN,PPO \
  --seed 42 \
  --max-episodes 1500 \
  --solve-score 200 \
  --window-size 100 \
  --noise-factor state \
  --noise-type gauss \
  --noise-sigma 0.05
```

Important:
- All options must use the `--key value` format (for example `--seed 42`).
- In this repository, some environment modules parse command-line args at import-time.
  These scripts already isolate that behavior internally, so the above command is safe.

### 3.3 Main outputs

- outputs/exp2a/figures/learning_curves_side_by_side.png
  - Left panel: Reward vs Timesteps
  - Right panel: Reward vs Wall-clock Time
- outputs/exp2a/exp2a_summary.csv
- outputs/exp2a/exp2a_summary.json
- outputs/exp2a/curves/*.csv (per-algorithm full learning trace)

## 4. Experiment 2B (Exploration and Stability)

### 4.1 Protocol

- Same perturbed environment and algorithm set
- Multi-seed training: 3 to 5 seeds recommended (default is 5 seeds)
- Fixed training budget: max_episodes is fixed for all algorithms
- Analysis windows:
  - Late phase variance: last 500 episodes
  - Boxplot window: last 100 episodes

### 4.2 Run command

```bash
python examples/experiment_2_on_off_policy/run_experiment_2b.py \
  --algorithms DQN,DoubleDQN,DuelingDDQN,PPO \
  --seeds 42,52,62,72,82 \
  --max-episodes 1000 \
  --late-window 500 \
  --boxplot-window 100 \
  --noise-factor state \
  --noise-type gauss \
  --noise-sigma 0.05
```

### 4.3 Main outputs

- outputs/exp2b/figures/shaded_learning_curves.png
  - Mean reward curve with +-1 std shading across seeds
- outputs/exp2b/figures/last100_boxplot.png
  - Distribution of last 100 episode rewards
- outputs/exp2b/exp2b_variance_summary.csv
- outputs/exp2b/exp2b_variance_summary.json
- outputs/exp2b/raw/*.csv (per-seed traces)

## 5. Improvements over the initial proposal

The scripts include additional controls to make the comparison publication-ready:

1. Unified implementation and logging
- Same logging fields for all algorithms: reward, moving average, timesteps, wall-clock.
- Same max steps per episode and same convergence definition.

2. Fair budget settings
- 2A uses convergence stop to compare "how fast to solve".
- 2B uses fixed episode budget to compare "how stable after enough training".

3. Explicit late-phase risk indicators
- Besides variance, 2B reports a crash-like spike ratio in late training:
  - fraction of rewards <= -100 in the last 500 episodes.

4. Reproducibility
- Seed is controlled and recorded.
- Raw CSV traces are always exported for re-plotting in notebook/paper.

5. Perturbation-first setup
- By default, both scripts run with non-zero perturbation.
- You can switch to clean environment by setting --noise-factor none --noise-sigma 0.0.

## 6. Suggested paper analysis points

For 2A:
- If DQN-family converges with fewer timesteps, this supports replay-driven sample efficiency.
- If PPO reaches similar reward with lower wall-clock, discuss reduced replay/target-network overhead and stable batch updates.

For 2B:
- If DQN-family has larger std bands and more low-score outliers, link to epsilon-greedy late random actions.
- If PPO has narrower variance and tighter boxplot IQR, link to entropy-regularized policy-distribution exploration.

## 7. Notes and recommended practice

- Hardware affects wall-clock time. Record CPU/GPU model in your report.
- For strong claims, repeat 2A with multiple seeds and report mean +- std of convergence timesteps/time.
- For significance testing, you can add Mann-Whitney U test on final-window returns.

## 8. If results look abnormal

- PPO catastrophic collapse:
  - This often indicates overly aggressive policy updates in noisy settings.
  - The current default config is already stabilized in code:
    - lower PPO learning rate
    - fewer PPO epochs per rollout
    - KL early stopping (`ppo_target_kl`)
    - entropy coefficient schedule with non-zero floor

- Dueling DDQN underperforming DQN:
  - In single-seed runs, this can be pure seed variance.
  - In noisy settings, PER can over-focus noisy TD spikes if learning starts too early.
  - The current default config is already stabilized in code:
    - replay warmup (`learning_starts`)
    - milder PER alpha
    - smaller dueling hidden size
    - tighter gradient clipping
    - slower epsilon decay

## 9. Quick checklist before writing the paper

- Run 2A once for visual trend and qualitative conclusion.
- Run 2B with at least 5 seeds.
- Use CSV/JSON summaries to build your tables.
- Keep one figure per claim:
  - efficiency claim -> 2A side-by-side curves + convergence table
  - stability claim -> shaded curves + boxplot + variance table
