import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

# Import your custom environment and physics modules
# (Assuming your env is saved in a file named `tuning_env.py`)
# from tuning_env import Tuning_env
# from beamline import * # ==========================================
# 1. SETUP THE ENVIRONMENT
# ==========================================
# (These are placeholders - plug in your actual beamline and targets here)
dummy_target_twiss = {
    "alpha": [0.0], 
    "beta": [15.0], 
    "gamma": [0.0]
}
dummy_beamline = [] # Insert your populated list of lattice segments
dummy_monitor_indices = [2, 5, 8]
dummy_quad_indices = [1, 3, 6]

# Instantiate the environment
env = Tuning_env(
    target_twiss=dummy_target_twiss,
    beamline=dummy_beamline,
    monitor_indices=dummy_monitor_indices,
    quad_indices=dummy_quad_indices
)

# ==========================================
# 2. TRAIN THE MODEL
# ==========================================
print("Starting Training...")

# Because your observation_space is a gym.spaces.Dict, you MUST use "MultiInputPolicy"
model = PPO("MultiInputPolicy", env, verbose=1)

# Train the agent (10,000 timesteps means 10,000 episodes for a 1-shot env)
model.learn(total_timesteps=10000)

# Save the trained brain to a file
model.save("beamline_tuning_model")
print("Training complete and model saved!")

# ==========================================
# 3. RUN THE TRAINED MODEL
# ==========================================
print("\n--- Running the Trained Agent ---")

# Load the saved model (useful if you run this in a separate file later)
loaded_model = PPO.load("beamline_tuning_model")

# Standard RL evaluation loop
episodes_to_test = 5

for ep in range(episodes_to_test):
    # 1. Reset the environment for a fresh beam
    obs, info = env.reset()
    
    # 2. Ask the model to predict the best actions (quad currents) based on the observation
    # deterministic=True tells the model to pick its absolute best guess, no random exploration
    action, _states = loaded_model.predict(obs, deterministic=True)
    
    # 3. Apply the actions to the environment
    obs, reward, terminated, truncated, info = env.step(action)
    
    # 4. Print the results!
    print(f"Episode {ep + 1}:")
    print(f"  Actions chosen (Amps): {action}")
    print(f"  Reward received: {reward:.4f}")
    print(f"  Success Bonus Reached: {info.get('is_success', False)}\n")

# Close the environment when finished
env.close()