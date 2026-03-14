"""Experiment 2B: exploration mechanism and training stability (variance analysis).

Each algorithm is trained with multiple random seeds. The script generates:
1) Shaded learning curves (mean +/- 1 std)
2) Boxplot for last N episode returns
3) Summary table for late-phase variance and crash-like spikes

Outputs:
- output_dir/raw/<algo>_seed<seed>.csv
- output_dir/figures/shaded_learning_curves.png
- output_dir/figures/last100_boxplot.png
- output_dir/exp2b_variance_summary.csv
- output_dir/exp2b_variance_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


LOCAL_ALL_ALGOS = ["DQN", "DoubleDQN", "DuelingDDQN", "PPO"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 2B: multi-seed stability analysis")
    parser.add_argument(
        "--algorithms",
        type=str,
        default=",".join(LOCAL_ALL_ALGOS),
        help="Comma-separated list: DQN,DoubleDQN,DuelingDDQN,PPO",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,52,62,72,82",
        help="Comma-separated random seeds, e.g. 42,52,62",
    )
    parser.add_argument("--max-episodes", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--solve-score", type=float, default=200.0)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--late-window", type=int, default=500)
    parser.add_argument("--boxplot-window", type=int, default=100)
    parser.add_argument(
        "--noise-factor",
        type=str,
        default="state",
        choices=["none", "state", "reward", "action"],
        help="Perturbation target in Robust-Gymnasium.",
    )
    parser.add_argument(
        "--noise-type",
        type=str,
        default="gauss",
        help="Noise distribution type (e.g. gauss).",
    )
    parser.add_argument(
        "--noise-sigma",
        type=float,
        default=0.05,
        help="Noise standard deviation or scale.",
    )
    parser.add_argument(
        "--noise-mu",
        type=float,
        default=0.0,
        help="Noise mean.",
    )
    parser.add_argument(
        "--robust-type",
        type=str,
        default="action",
        help="robust_type field passed to env.step input.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="examples/experiment_2_on_off_policy/outputs/exp2b",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    return parser.parse_args()


def ensure_dirs(output_dir: str) -> Dict[str, str]:
    raw_dir = os.path.join(output_dir, "raw")
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    return {"raw": raw_dir, "figures": figures_dir}


def save_trace_csv(path: str, trace) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "moving_avg_reward", "total_timesteps", "wall_time_sec"])
        for idx, reward in enumerate(trace.episode_rewards):
            writer.writerow([
                idx + 1,
                reward,
                trace.moving_avg_rewards[idx],
                trace.total_timesteps[idx],
                trace.wall_times[idx],
            ])


def to_2d_array(sequences: List[List[float]], target_len: int) -> np.ndarray:
    arr = np.full((len(sequences), target_len), np.nan, dtype=np.float32)
    for i, seq in enumerate(sequences):
        upto = min(len(seq), target_len)
        arr[i, :upto] = np.asarray(seq[:upto], dtype=np.float32)
    return arr


def main() -> None:
    args = parse_args()

    # robust_gymnasium env modules parse sys.argv at import-time in this repo.
    # Keep only script name to avoid argparse conflicts with this experiment CLI.
    sys.argv = [sys.argv[0]]

    from common_agents import TrainConfig, parse_algorithms, train_agent

    algorithms = parse_algorithms(args.algorithms.split(","))
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    dirs = ensure_dirs(args.output_dir)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    cfg = TrainConfig(
        solve_score=args.solve_score,
        moving_avg_window=args.window_size,
        max_episodes=args.max_episodes,
        max_steps_per_episode=args.max_steps,
        noise_factor=args.noise_factor,
        noise_type=args.noise_type,
        noise_sigma=args.noise_sigma,
        noise_mu=args.noise_mu,
        robust_type=args.robust_type,
    )

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Algorithms: {algorithms}")
    print(f"[INFO] Seeds: {seeds}")
    print(
        f"[INFO] Perturbation: factor={cfg.noise_factor}, type={cfg.noise_type}, "
        f"sigma={cfg.noise_sigma}, mu={cfg.noise_mu}, robust_type={cfg.robust_type}"
    )

    all_results: Dict[str, List] = {algo: [] for algo in algorithms}

    for algo in algorithms:
        for seed in seeds:
            print(f"\n[RUN] 2B | {algo} | seed={seed}")
            trace = train_agent(
                algo_name=algo,
                seed=seed,
                cfg=cfg,
                device=device,
                stop_on_convergence=False,
            )
            all_results[algo].append(trace)
            raw_path = os.path.join(dirs["raw"], f"{algo}_seed{seed}.csv")
            save_trace_csv(raw_path, trace)

    # Shaded curve: mean +/- std of per-episode return.
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(1, args.max_episodes + 1)

    summary_rows = []
    boxplot_data = []

    for algo in algorithms:
        traces = all_results[algo]
        rewards_by_seed = [t.episode_rewards for t in traces]
        reward_arr = to_2d_array(rewards_by_seed, target_len=args.max_episodes)

        mean_curve = np.nanmean(reward_arr, axis=0)
        std_curve = np.nanstd(reward_arr, axis=0)

        ax.plot(x, mean_curve, linewidth=2, label=algo)
        ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, alpha=0.20)

        late_values = reward_arr[:, -args.late_window:]
        per_seed_late_std = np.nanstd(late_values, axis=1)
        avg_late_std = float(np.nanmean(per_seed_late_std))

        per_seed_crash_rate = np.nanmean((late_values <= -100.0).astype(np.float32), axis=1)
        avg_crash_rate = float(np.nanmean(per_seed_crash_rate))

        last100 = reward_arr[:, -args.boxplot_window:]
        boxplot_data.append(last100[~np.isnan(last100)].tolist())

        summary_rows.append(
            {
                "algorithm": algo,
                "n_seeds": len(traces),
                "episodes_per_seed": args.max_episodes,
                "late_window": args.late_window,
                "boxplot_window": args.boxplot_window,
                "mean_return_last_100": float(np.nanmean(last100)),
                "std_return_last_100": float(np.nanstd(last100)),
                "avg_seed_std_last_500": avg_late_std,
                "avg_seed_crash_rate_last_500": avg_crash_rate,
            }
        )

    ax.axhline(args.solve_score, color="black", linestyle="--", linewidth=1, label="Solve threshold")
    ax.set_title("Experiment 2B: Multi-seed Learning Curves (mean +/- 1 std)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    shaded_path = os.path.join(dirs["figures"], "shaded_learning_curves.png")
    fig.savefig(shaded_path, dpi=180)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.boxplot(boxplot_data, labels=algorithms, showfliers=True)
    ax2.axhline(args.solve_score, color="black", linestyle="--", linewidth=1)
    ax2.set_title(f"Last {args.boxplot_window} Episode Rewards by Algorithm")
    ax2.set_ylabel("Episode Reward")
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    boxplot_path = os.path.join(dirs["figures"], "last100_boxplot.png")
    fig2.savefig(boxplot_path, dpi=180)
    plt.close(fig2)

    csv_path = os.path.join(args.output_dir, "exp2b_variance_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    json_path = os.path.join(args.output_dir, "exp2b_variance_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    print(f"\n[OK] 2B results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
