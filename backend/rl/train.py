import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from beamline import *
import os
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

# Import your custom environment and physics modules
from tuning_env import Tuning_env
# from beamline import *

def main():
    # ==========================================
    # 1. SETUP THE ENVIRONMENT WITH SIGMA TARGETS
    # ==========================================
    dummy_target_sigma = {
        "sigma_x": 2.15,  # Targeted horizontal standard deviation size
        "sigma_y": 2.15   # Targeted vertical standard deviation size
    }

    dummy_beamline = [
            driftLattice(length = 1),
            qpdLattice(current = 1),
            driftLattice(length = 1),
            qpfLattice(current = 1),
            driftLattice(length = 0.5)
        ] 
    dummy_monitor_indices = [4]
    dummy_quad_indices = [3]

    num_cpu = os.cpu_count()
    env_kwargs = dict(
            target_sigma=dummy_target_sigma,
            beamline=dummy_beamline,
            monitor_indices=dummy_monitor_indices,
            quad_indices=dummy_quad_indices,
        )
    vec_env = make_vec_env(
            Tuning_env,
            n_envs=15,
            env_kwargs=env_kwargs,
            vec_env_cls=SubprocVecEnv,
    )

    # ==========================================
    # 2. TRAIN THE MODEL
    # ==========================================
    print("Starting Training...")

    # "MultiInputPolicy" is still required because observation_space is a gym.spaces.Dict
    model = PPO("MultiInputPolicy", vec_env, verbose=1)

    model.learn(total_timesteps=2000000)

    # Save the trained brain to a file
    model.save("beamline_sigma_tuning_model")
    print("Training complete and model saved!")

    # ==========================================
    # 3. RUN THE TRAINED MODEL (EVALUATION)
    # ==========================================
    print("\n--- Running the Trained Agent ---")

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
        
        # Predict the continuous array of quadrupole currents
        action, _states = loaded_model.predict(obs, deterministic=True)
        
        # Apply actions simultaneously to all quadrupoles
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Episode {ep + 1}:")
        print(f"  Actions chosen (Amps for each Quad): {np.round(action, 4)}")
        print(f"  Resulting Last Monitor Noisy Sigma X: {obs['sigma_x'][-1][0]:.4f}")
        print(f"  Resulting Last Monitor Noisy Sigma Y: {obs['sigma_y'][-1][0]:.4f}")
        print(f"  Reward received: {reward:.4f}")
        print(f"  Success Bonus Reached: {info.get('is_success', False)}\n")

    env.close()

if __name__ == "__main__":
    main()