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
        "sigma_x": 3.1,  # Targeted horizontal standard deviation size
        "sigma_y": 5.7,   # Targeted vertical standard deviation size
    }
    # ------- GOALS -----------------------------
    # beamline[1]: -0.17
    # beamline[3]: -0.35

    dummy_beamline = [
        driftLattice(length = 0.5),
        qpdLattice(current = 1),
        driftLattice(length = 0.5),
        qpfLattice(current = 1),
        driftLattice(length = 0.1)
    ]
    dummy_monitor_indices = [0, 2, 4]
    dummy_quad_indices = [1, 3]

    num_cpu = os.cpu_count()
    env_kwargs = dict(
            target_sigma=dummy_target_sigma,
            beamline=dummy_beamline,
            monitor_indices=dummy_monitor_indices,
            quad_indices=dummy_quad_indices,
        )
    vec_env = make_vec_env(
            Tuning_env,
            n_envs=14,
            env_kwargs=env_kwargs,
            vec_env_cls=SubprocVecEnv,
    )

    # ==========================================
    # 2. TRAIN THE MODEL
    # ==========================================
    print("Starting Training...")

    # "MultiInputPolicy" is still required because observation_space is a gym.spaces.Dict
    model = PPO("MultiInputPolicy", vec_env, ent_coef=0.05, verbose=1)

    model.learn(total_timesteps=3000000)

    # Save the trained brain to a file
    model.save("beamline_sigma_tuning_model")
    print("Training complete and model saved!")

if __name__ == "__main__":
    main()