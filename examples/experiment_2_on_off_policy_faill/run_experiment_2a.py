"""Experiment 2A: sample efficiency vs wall-clock efficiency.

For each algorithm, training stops at convergence when moving average reward over
`window_size` episodes exceeds `solve_score`. The script then compares:
1) total environment timesteps to convergence
2) wall-clock time to convergence

Outputs:
- output_dir/curves/<algo>_seed<seed>.csv
- output_dir/figures/learning_curves_side_by_side.png
- output_dir/exp2a_summary.csv
- output_dir/exp2a_summary.json
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
import torch


LOCAL_ALL_ALGOS = ["DQN", "DoubleDQN", "DuelingDDQN", "PPO"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 2A: PPO vs DQN family")
    parser.add_argument(
        "--algorithms",
        type=str,
        default=",".join(LOCAL_ALL_ALGOS),
        help="Comma-separated list: DQN,DoubleDQN,DuelingDDQN,PPO",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--solve-score", type=float, default=200.0)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--max-episodes", type=int, default=1500)
    parser.add_argument("--max-steps", type=int, default=1000)
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
        default="examples/experiment_2_on_off_policy/outputs/exp2a",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    return parser.parse_args()


def ensure_dirs(output_dir: str) -> Dict[str, str]:
    curves_dir = os.path.join(output_dir, "curves")
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(curves_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    return {"curves": curves_dir, "figures": figures_dir}


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


def plot_side_by_side(traces, solve_score: float, output_path: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for trace in traces:
        ax1.plot(trace.total_timesteps, trace.moving_avg_rewards, linewidth=2, label=trace.algorithm)
        ax2.plot(trace.wall_times, trace.moving_avg_rewards, linewidth=2, label=trace.algorithm)

    ax1.axhline(solve_score, color="black", linestyle="--", linewidth=1, label="Solve threshold")
    ax2.axhline(solve_score, color="black", linestyle="--", linewidth=1)

    ax1.set_title("A. Reward vs Timesteps")
    ax1.set_xlabel("Timesteps")
    ax1.set_ylabel("Reward (moving average)")
    ax1.grid(alpha=0.3)

    ax2.set_title("B. Reward vs Wall-clock Time")
    ax2.set_xlabel("Wall-clock Time (s)")
    ax2.set_ylabel("Reward (moving average)")
    ax2.grid(alpha=0.3)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    # robust_gymnasium env modules parse sys.argv at import-time in this repo.
    # Keep only script name to avoid argparse conflicts with this experiment CLI.
    sys.argv = [sys.argv[0]]

    from common_agents import TrainConfig, parse_algorithms, train_agent

    algorithms = parse_algorithms(args.algorithms.split(","))
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

    traces = []
    summary_rows: List[Dict] = []

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Algorithms: {algorithms}")
    print(
        f"[INFO] Perturbation: factor={cfg.noise_factor}, type={cfg.noise_type}, "
        f"sigma={cfg.noise_sigma}, mu={cfg.noise_mu}, robust_type={cfg.robust_type}"
    )

    for algo in algorithms:
        print(f"\n[RUN] 2A | {algo} | seed={args.seed}")
        trace = train_agent(
            algo_name=algo,
            seed=args.seed,
            cfg=cfg,
            device=device,
            stop_on_convergence=True,
        )
        traces.append(trace)

        curve_path = os.path.join(dirs["curves"], f"{algo}_seed{args.seed}.csv")
        save_trace_csv(curve_path, trace)

        summary_rows.append(
            {
                "algorithm": algo,
                "seed": args.seed,
                "episodes_ran": len(trace.episode_rewards),
                "converged": trace.convergence_episode is not None,
                "convergence_episode": trace.convergence_episode,
                "convergence_timesteps": trace.convergence_timestep,
                "convergence_wall_time_sec": trace.convergence_wall_time,
                "final_moving_avg": trace.moving_avg_rewards[-1] if trace.moving_avg_rewards else None,
                "final_reward": trace.episode_rewards[-1] if trace.episode_rewards else None,
            }
        )

    fig_path = os.path.join(dirs["figures"], "learning_curves_side_by_side.png")
    plot_side_by_side(traces, solve_score=args.solve_score, output_path=fig_path)

    csv_path = os.path.join(args.output_dir, "exp2a_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    json_path = os.path.join(args.output_dir, "exp2a_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    print(f"\n[OK] 2A results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
