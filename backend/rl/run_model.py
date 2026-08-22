import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from stable_baselines3 import PPO
from beamline import driftLattice, qpdLattice, qpfLattice
from tuning_env import Tuning_env

# 1. Recreate the environment configuration used during training
# (Make sure setup matches the training environment exactly)
target_sigma = {"sigma_x": 1.5, "sigma_y": 1.2}
beamline = [
    driftLattice(length=1),
    qpdLattice(current=1),
    driftLattice(length=1),
    qpfLattice(current=1),
    driftLattice(length=1),
]
monitor_indices = [0, 1, 4]
quad_indices = [1, 3]

env = Tuning_env(
    target_sigma=target_sigma,
    beamline=beamline,
    monitor_indices=monitor_indices,
    quad_indices=quad_indices
)

model_path = "beamline_sigma_tuning_model"
model = PPO.load(model_path)

num_episodes = 5

for episode in range(num_episodes):
    obs, info = env.reset()
    terminated, truncated = False, False
    total_reward = 0.0

    print(f"\n--- Episode {episode + 1} ---")

    while not (terminated or truncated):
        # Pass deterministic=True to use the optimal policy without exploration noise
        action, _states = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Extract measured spot sizes at the final monitor
        final_sigma_x = obs["sigma_x"][-1][0]
        final_sigma_y = obs["sigma_y"][-1][0]

        print(f"  Actions (Amps): {np.round(action, 4)}")
        print(
            f"  Final Monitor Sigma X: {final_sigma_x:.4f} (Target: {target_sigma['sigma_x']})"
        )
        print(
            f"  Final Monitor Sigma Y: {final_sigma_y:.4f} (Target: {target_sigma['sigma_y']})"
        )
        print(f"  Step Reward: {reward:.4f}")
        print(f"  Target Met: {info.get('is_success', False)}")

    print(f"Total Episode Reward: {total_reward:.4f}")

env.close()