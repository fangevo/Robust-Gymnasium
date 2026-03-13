# Strict REINFORCE Solution (LunarLander-v3)

This folder contains a strict standard REINFORCE implementation for Robust Gymnasium LunarLander-v3.

## What "strict standard" means here

1. Policy-only training (no critic / no value function network).
2. No epsilon-greedy exploration.
3. No value loss curve.
4. Policy update is done once per episode using full-trajectory Monte Carlo returns.

The objective is:

L_policy = -sum_t log pi_theta(a_t | s_t) * G_t

where G_t is the discounted return from step t to the end of the episode.

## File

- train_REINFORC.py: strict REINFORCE training script.

## Outputs

After training, the script saves:

- results/training_curves.png
- results/reinforce_solved.pth (if solved threshold reached)
- results/reinforce_final.pth

## Visualization

training_curves.png contains:

1. Episode reward curve.
2. Moving average reward curve (window = 100 episodes).

This matches the strict REINFORCE requirement you specified.

## Run

From the repository root:

python examples/LunarLander_REINFORC/train_REINFORC.py

