import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

# Import your custom environment and physics modules
from gym import Tuning_env
# from beamline import *

# ==========================================
# 1. SETUP THE ENVIRONMENT WITH SIGMA TARGETS
# ==========================================
# CHANGED: We now specify target beam spot sizes (sigmas) instead of Twiss parameters
dummy_target_sigma = {
    "sigma_x": 1.5,  # Targeted horizontal standard deviation size
    "sigma_y": 1.2   # Targeted vertical standard deviation size
}

dummy_beamline = [] # Insert your populated list of lattice segments
dummy_monitor_indices = [2, 5, 8]
dummy_quad_indices = [1, 3, 6]

# Instantiate the environment using our new target_sigma dictionary
env = Tuning_env(
    target_sigma=dummy_target_sigma,
    beamline=dummy_beamline,
    monitor_indices=dummy_monitor_indices,
    quad_indices=dummy_quad_indices
)

# ==========================================
# 2. TRAIN THE MODEL
# ==========================================
print("Starting Training...")

# "MultiInputPolicy" is still required because observation_space is a gym.spaces.Dict
model = PPO("MultiInputPolicy", env, verbose=1)

# Train the agent (10,000 episodes for our 1-shot setup)
model.learn(total_timesteps=10000)

# Save the trained brain to a file
model.save("beamline_sigma_tuning_model")
print("Training complete and model saved!")

# ==========================================
# 3. RUN THE TRAINED MODEL (EVALUATION)
# ==========================================
print("\n--- Running the Trained Agent ---")

loaded_model = PPO.load("beamline_sigma_tuning_model")
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