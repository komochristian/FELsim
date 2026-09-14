import os
from py_compile import main
from tuning_env import Tuning_env
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from beamline import *
from stable_baselines3 import PPO



print("CPU cores available:", os.cpu_count())

import torch


print("\n--- Running the Trained Agent ---")

# 1. Recreate the environment configuration used during training
# (Make sure setup matches the training environment exactly)
dummy_target_sigma = {
    "sigma_x": 3.1,
    "sigma_y": 5.7
}

# ------- GOALS -----------------------------
# beamline[1]: -0.17
# beamline[3]: -0.35

dummy_beamline = [
    driftLattice(length=0.5),
    qpdLattice(current=1),
    driftLattice(length=0.5),
    qpfLattice(current=1),
    driftLattice(length=0.1)
]

dummy_monitor_indices = [0, 2, 4]
dummy_quad_indices = [1, 3]

loaded_model = PPO.load("beamline_sigma_tuning_model")

env = Tuning_env(
    target_sigma=dummy_target_sigma,
    beamline=dummy_beamline,
    monitor_indices=dummy_monitor_indices,
    quad_indices=dummy_quad_indices,
)

episodes_to_test = 5

for ep in range(episodes_to_test):
    obs, info = env.reset()

    print(f"\nEpisode {ep + 1}:")

    step = 0

    while True:
        step += 1

        # Predict the continuous array of quadrupole currents
        action, _states = loaded_model.predict(
            obs,
            deterministic=True
        )

        # Apply actions simultaneously to all quadrupoles
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"  Step {step}:")
        print(f"    Actions chosen (Normalized current): {np.round(action, 4)}")
        print(f"    Amperages: {((np.round(action, 4) + 1.0) / 2.0) * 10}")
        print(f"    Resulting Last Monitor Noisy Sigma X: {obs['sigma_x'][-1][0]:.4f}")
        print(f"    Resulting Last Monitor Noisy Sigma Y: {obs['sigma_y'][-1][0]:.4f}")
        print(f"    Reward received: {reward:.4f}")
        print(f"    Relative Error X: {info.get('relative_error_x', 0.0):.4f}")
        print(f"    Relative Error Y: {info.get('relative_error_y', 0.0):.4f}")

        # Stop only when the environment says the episode is finished
        if terminated or truncated:
            print(f"  Episode ended after {step} step(s).")
            print(f"  Truncated: {truncated} Terminated: {terminated}")
            break

env.close()
