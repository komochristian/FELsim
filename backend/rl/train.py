"""
train.py — Training script for BeamTuningEnv and BeamCalibrationEnv
====================================================================

Both modes use SAC (Soft Actor-Critic) from Stable-Baselines3.

Quick start
-----------
    # Tuning mode (default)
    python train.py --mode tuning --timesteps 300000 --run-name tune_v1

    # Calibration mode
    python train.py --mode calibration --timesteps 500000 --run-name calib_v1

    # Evaluate a saved model
    python train.py --mode tuning --eval-only --load-path runs/tune_v1/best_model

    # Use a real HDF5 database instead of synthetic data
    python train.py --mode tuning --db-path measurements.h5

Install
-------
    pip install stable-baselines3 gymnasium numpy torch tensorboard
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback, CallbackList, CheckpointCallback, EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from beam_utils import MeasurementDatabase
from beam_tuning_env import BeamTuningEnv
from beam_calibration_env import BeamCalibrationEnv


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class DomainMetricsCallback(BaseCallback):
    """Logs mode-specific metrics to TensorBoard every log_freq steps."""

    def __init__(self, mode: str, log_freq: int = 200, verbose: int = 0):
        super().__init__(verbose)
        self.mode = mode
        self.log_freq = log_freq
        self._buf: list[dict] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if self.mode == "tuning":
                if info.get("distance") is not None:
                    self._buf.append({
                        "dist":   info["distance"],
                        "solved": info.get("solved", False),
                    })
            else:
                if info.get("phase") == "correct":
                    self._buf.append({"reward": info.get("reward", 0.0)})

        if self.n_calls % self.log_freq == 0 and self._buf:
            w = self._buf[-100:]
            if self.mode == "tuning":
                self.logger.record("beam/distance_mean",
                                   float(np.mean([x["dist"]   for x in w])))
                self.logger.record("beam/success_rate",
                                   float(np.mean([x["solved"] for x in w])))
            else:
                self.logger.record("calib/mean_reward",
                                   float(np.mean([x["reward"] for x in w])))
        return True


class ProgressCallback(BaseCallback):
    def __init__(self, total: int, freq: int = 5000, verbose: int = 0):
        super().__init__(verbose)
        self._total = total
        self._freq = freq
        self._t0 = time.time()

    def _on_step(self) -> bool:
        if self.n_calls % self._freq == 0:
            pct     = 100 * self.num_timesteps / self._total
            elapsed = time.time() - self._t0
            eta     = elapsed / max(self.num_timesteps, 1) * \
                      (self._total - self.num_timesteps)
            print(f"  [{pct:5.1f}%]  steps={self.num_timesteps:>9,}  "
                  f"elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m")
        return True


# ---------------------------------------------------------------------------
# Environment factories
# ---------------------------------------------------------------------------

def _make_tuning_env(db, env_kwargs, seed):
    def _init():
        env = BeamTuningEnv(database=db, **env_kwargs)
        env = RecordEpisodeStatistics(env)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def _make_calib_env(db, env_kwargs, seed):
    def _init():
        env = BeamCalibrationEnv(database=db, **env_kwargs)
        env = RecordEpisodeStatistics(env)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def build_vec_env(mode, db, env_kwargs, n_envs=4, seed=0, use_subproc=False):
    factory_fn = _make_tuning_env if mode == "tuning" else _make_calib_env
    factories  = [factory_fn(db, env_kwargs, seed + i) for i in range(n_envs)]
    VecCls     = SubprocVecEnv if use_subproc else DummyVecEnv
    vec        = VecCls(factories)
    vec = VecNormalize(vec, norm_obs=True, norm_reward=True, clip_obs=10.0)
    return vec


# ---------------------------------------------------------------------------
# SAC configuration
# ---------------------------------------------------------------------------

def make_sac_kwargs(mode: str) -> dict:
    common = dict(
        policy        = "MlpPolicy",
        policy_kwargs = dict(net_arch=[256, 256, 256]),
        learning_rate = 3e-4,
        batch_size    = 512,
        tau           = 0.005,
        gamma         = 0.99,
        ent_coef      = "auto",
        target_entropy = "auto",
        verbose        = 1,
    )
    if mode == "tuning":
        common.update(buffer_size=300_000, learning_starts=5_000,
                      train_freq=1, gradient_steps=1)
    else:
        # Calibration has sparse/delayed reward — more replay helps
        common.update(buffer_size=500_000, learning_starts=10_000,
                      train_freq=4, gradient_steps=4)
    return common


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    run_dir = Path("runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*62}")
    print(f"  Mode      : {args.mode.upper()}")
    print(f"  Run       : {args.run_name}")
    print(f"  Timesteps : {args.timesteps:,}")
    print(f"  Envs      : {args.n_envs}")
    print(f"{'='*62}\n")

    # Database
    if args.db_path:
        ext = Path(args.db_path).suffix.lower()
        if ext in (".h5", ".hdf5"):
            db = MeasurementDatabase.from_hdf5(args.db_path)
        else:
            db = MeasurementDatabase.from_numpy(args.db_path)
        print(f"Loaded database: {len(db)} entries")
    else:
        db = MeasurementDatabase.from_synthetic(500, n_particles=args.n_particles,
                                                seed=args.seed)
        print(f"Synthetic database: {len(db)} entries")

    # Env kwargs per mode
    if args.mode == "tuning":
        env_kwargs = dict(
            sensor_element_index = args.sensor_index,
            n_sample_obs         = args.n_sample_obs,
            w_moment             = 1.0,
            w_penalty            = 1e-3,
            success_threshold    = args.success_threshold,
            success_bonus        = 10.0,
            max_steps            = args.max_steps,
            randomise_init_k     = True,
        )
    else:
        env_kwargs = dict(
            sensor_element_index = args.sensor_index,
            probe_budget         = args.probe_budget,
            n_history            = args.n_history,
            max_quad_fault       = args.max_quad_fault,
            max_monitor_fault    = args.max_monitor_fault,
        )

    train_env = build_vec_env(args.mode, db, env_kwargs,
                              n_envs=args.n_envs, seed=args.seed,
                              use_subproc=args.subproc)
    eval_env  = build_vec_env(args.mode, db, env_kwargs, n_envs=1,
                              seed=args.seed + 9999)
    eval_env.obs_rms  = train_env.obs_rms
    eval_env.ret_rms  = train_env.ret_rms
    eval_env.training = False

    sac_kwargs = make_sac_kwargs(args.mode)
    sac_kwargs["tensorboard_log"] = str(run_dir / "logs" / "tb")

    if args.load_path:
        model = SAC.load(args.load_path, env=train_env)
    else:
        model = SAC(env=train_env, **sac_kwargs)

    print(f"Policy:\n{model.policy}\n")

    ckpt_cb = CheckpointCallback(
        save_freq         = max(args.timesteps // 20, 2000),
        save_path         = str(run_dir / "checkpoints"),
        name_prefix       = f"{args.mode}_sac",
        save_vecnormalize = True,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = str(run_dir / "best_model"),
        log_path             = str(run_dir / "logs" / "eval"),
        eval_freq            = max(args.timesteps // 40, 1000),
        n_eval_episodes      = 10,
        deterministic        = True,
        verbose              = 1,
    )
    cb = CallbackList([ckpt_cb, eval_cb,
                       DomainMetricsCallback(args.mode),
                       ProgressCallback(args.timesteps)])

    print("Training ...  (Ctrl-C to stop early)\n")
    t0 = time.time()
    try:
        model.learn(
            total_timesteps     = args.timesteps,
            callback            = cb,
            reset_num_timesteps = not bool(args.load_path),
            tb_log_name         = args.run_name,
            progress_bar        = False,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.")

    print(f"\nDone in {(time.time()-t0)/60:.1f} min.")
    model.save(str(run_dir / "final_model"))
    train_env.save(str(run_dir / "vec_normalize.pkl"))
    train_env.close()
    eval_env.close()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> None:
    run_dir  = Path("runs") / args.run_name
    vec_path = run_dir / "vec_normalize.pkl"
    db = MeasurementDatabase.from_synthetic(500, seed=args.seed + 1)

    if args.mode == "tuning":
        raw = BeamTuningEnv(
            database=db,
            sensor_element_index=args.sensor_index,
            n_sample_obs=args.n_sample_obs,
            success_threshold=args.success_threshold,
            max_steps=args.max_steps,
            render_mode="human",
        )
    else:
        raw = BeamCalibrationEnv(
            database=db,
            sensor_element_index=args.sensor_index,
            probe_budget=args.probe_budget,
            n_history=args.n_history,
            render_mode="human",
        )

    raw = Monitor(raw)
    vec = DummyVecEnv([lambda: raw])
    if vec_path.exists():
        vec = VecNormalize.load(str(vec_path), vec)
        vec.training = False
        vec.norm_reward = False
    else:
        warnings.warn("vec_normalize.pkl not found.")

    model = SAC.load(args.load_path, env=vec)
    print(f"Loaded: {args.load_path}\n")

    rewards, solved = [], []
    for ep in range(args.eval_episodes):
        obs, _ = vec.reset()
        ep_r, done, ep_solved = 0.0, False, False
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = vec.step(act)
            ep_r += float(r[0])
            done = bool(term[0] or trunc[0])
            if term[0]:
                ep_solved = True
        rewards.append(ep_r)
        solved.append(ep_solved)
        print(f"Ep {ep+1:3d} | reward={ep_r:+8.2f} | "
              f"{'solved' if ep_solved else 'timeout'}")

    print(f"\nMean reward: {np.mean(rewards):+.3f}  "
          f"Success rate: {100*np.mean(solved):.1f}%")
    vec.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode",       choices=["tuning", "calibration"], default="tuning")
    p.add_argument("--run-name",   default="beam_run")
    p.add_argument("--eval-only",  action="store_true")
    p.add_argument("--load-path",  default="")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--db-path",    default="")
    p.add_argument("--n-particles",type=int, default=1000)
    p.add_argument("--timesteps",  type=int, default=300_000)
    p.add_argument("--n-envs",     type=int, default=4)
    p.add_argument("--subproc",    action="store_true")
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--sensor-index",  type=int, default=-1)
    p.add_argument("--max-steps",     type=int, default=100)
    p.add_argument("--success-threshold", type=float, default=0.05)
    p.add_argument("--n-sample-obs",  type=int, default=50)
    p.add_argument("--probe-budget",  type=int, default=8)
    p.add_argument("--n-history",     type=int, default=5)
    p.add_argument("--max-quad-fault",type=float, default=2.0)
    p.add_argument("--max-monitor-fault", type=float, default=2e-3)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.eval_only:
        if not args.load_path:
            raise ValueError("--load-path required with --eval-only")
        evaluate(args)
    else:
        train(args)