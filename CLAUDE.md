# Project: expiration-aware scheduling for remote driving (CarRacing)

## Goal
Reproduce Section VI of our paper: a remote-driving loop where a vehicle (sender) decides when to transmit camera frames over an unreliable, costly channel to a remote controller (receiver). Compare an expiration-aware SAQ scheduler against periodic and JIT baselines.

## Environment
- Gymnasium CarRacing-v3 (Box2D), continuous actions, new random track each episode.
- Action: 2-D, [steering, throttle], throttle in [-1,1] merges gas (>0) and brake (<0). Map to the env's 3-D action internally.
- Each control action is repeated for 6 physics frames (action repeat = 6). One "step" below = one control step.
- Reward: env default (1000/N per tile visited, small per-frame penalty). Episode ends on track completion or leaving the track.
- Observation at the sender: current frame y_n (96x96, grayscale) and vector s_n:
  body-frame velocity (2), speed (1), yaw rate (1), heading (1), four wheel angular velocities (4), steering angle (1), plus lateral offset from lane center (1) and signed curvature of the next 20 track segments (20). Extract these from env.unwrapped (car.hull, car.wheels, track).

## Channel and receiver
- Channel: one frame per use, succeeds i.i.d. with prob p_s, costs c per attempt. Action return link is reliable.
- Receiver keeps the last K_BUF delivered frames plus Delta_n = age (in steps) of the newest delivered frame. With no delivery, the buffer is frozen and Delta_n increments.
- K_BUF = 4  # TODO: paper text says 4, figure says 6. Confirm.

## Controller (trained once, then frozen)
- IMPALA-style residual CNN encoder, 3 stages with 16, 32, 32 channels, over the K_BUF stacked delivered frames.
- Flatten -> linear to 256 -> LayerNorm -> concat [Delta_n, previous action] -> GRU.
- Gaussian policy head with tanh squashing; 2-layer value head. About 1.6M parameters total.
- PPO, 512 parallel envs, rollout segments of 128 steps, recurrent state carried across segments.
- Adaptive LR from sampled KL in [2e-5, 1.2e-3], KL target 0.02, early stop when sampled KL > 1.5 x target.
- Curriculum: delivery probability goes linearly from 1.0 to 0.3 over the first 40% of 15M steps, then stays at 0.3.  # TODO: confirm constant 0.3 after, or random p_s per env.
- During training, deliveries are random Bernoulli (no scheduler).

## Expiration time (oracle)
- For a frame y_n delivered at step n, freeze the receiver buffer and let the controller act on it for k steps. The expiration time is the largest k with r_0 - r_k <= eps, following eq. (tau-star) in the paper.
- TODO: define r_k exactly for CarRacing (e.g., per-step env reward, or divergence from the fresh-frame rollout). Write it here.
- Box2D envs cannot be deep-copied. To measure from state n, reset with the same seed and replay the recorded action sequence up to n, then branch. Verify determinism with a unit test before using it.
- Cap horizons at K_MAX = 20.  # TODO: confirm.

## Expiration predictor (sender side, causal)
- Inputs: y_n and s_n only. No receiver state, no channel feedback.
- CNN trunk on y_n, small MLP on s_n, concatenate, head with K_MAX outputs; output k = P(frame still useful after k steps). Train as a bounded ordinal / survival target (cumulative binary cross-entropy), NOT regression.
- Readout: largest k with predicted survival >= 0.5.
- Train on states visited under the scheduler being evaluated, not only on fully observed nominal trajectories (this matters; see paper C5).  # TODO: describe how you do this (e.g., iterate: collect under scheduler, retrain).

## Schedulers
1. Periodic: transmit every m steps, m in {1,2,3,4,6}.
2. Predicted JIT: transmit when the receiver's held frame expires according to T_hat.
3. SAQ (online, average reward): state (T_r, T_hat), T_r = remaining lifetime of the receiver's frame. Reward r_hat per step with an unexpired receiver frame, minus c per transmission. Update rule = Algorithm 1 in the paper (two-timescale: Q step alpha_n, gain step beta_n, beta_n/alpha_n -> 0). Never transmit when T_r > T_hat. Exploration prob 0.15, lr decays 0.5 -> 0.02, ties break toward transmit.
- r_hat = per-step return difference between always-transmit and never-transmit, measured on the plant.
- Training tracks and evaluation tracks are disjoint (separate seed ranges).

## Evaluation
- Periodic table: p_s in {0.2,0.4,0.6,0.8,1.0} x m in {1,2,3,4,6}; 200 paired tracks per cell (same seeds across all cells). Report mean return +/- s.e. and fraction of track completed.
- Main figure: p_s in {0.1,...,0.9}, c/r_hat in {0.2,0.5,0.8,1.0}. Baselines interpolated to SAQ's transmission rate.  # TODO: interpolation method (linear between neighbouring m / eps?).

## Engineering rules (always follow)
- Every long-running script shows live progress with tqdm and prints partial results as each group of work finishes (per p_s, per cell, per checkpoint), so runs can be tracked without waiting for the end.
- Save results incrementally (CSV/JSON per cell) so a killed job loses at most one cell. Scripts must resume from existing results.
- All randomness seeded; log seeds. Paired evaluation = identical track seeds across compared methods.
- Config via YAML or argparse; no hard-coded paths.
- Cluster: SLURM. TODO: fill in partition, account, GPU type, module loads / conda env name. Provide an sbatch script per stage.
- Checkpoint the controller every N updates; training must resume from the last checkpoint.
- Add a --smoke flag to every script that runs a tiny version (few envs, few steps) in under 2 minutes.
