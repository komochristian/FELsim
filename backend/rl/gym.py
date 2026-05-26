import gymnasium as gym
import numpy as np
from ebeam import beam
from beamline import *

class Tuning_env(gym.Env):
    def __init__(self, target_twiss, beamline, monitor_indices, quad_indices):
        super().__init__()

        self._target_twiss = target_twiss
        self.target_alpha = np.array(self._target_twiss["alpha"], dtype=np.float32)
        self.target_beta = np.array(self._target_twiss["beta"], dtype=np.float32)
        self.target_gamma = np.array(self._target_twiss["gamma"], dtype=np.float32)
        
        self._monitor_locations = monitor_indices
        self.quad_indices = quad_indices
        self._beamline = beamline
        self.CURRENT_MAX = 10.0
        self._NUM_PARTICLES = 1000
        self.ebeam = beam()

        # Combine Type Validation and Noise Allocation into a single pass
        for idx, seg in enumerate(self._beamline):            
            if idx in self.quad_indices and not isinstance(seg, (qpdLattice, qpfLattice)):
                raise TypeError(f"Quad segment at index {idx} is not a valid lattice.") 
            
            if idx in self._monitor_locations:
                # --- SIMULATE MONITOR DEGRADATION HERE ---
                # 1. White Noise (Random hardware jitter per measurement)
                # We use a percentage of the expected baseline scale (e.g., 2% noise)
                seg.alpha_noise = np.random.normal(0, 1.0 * 0.02, size=3)
                seg.beta_noise  = np.random.normal(0, 1.0 * 0.02, size=3)
                seg.gamma_noise = np.random.normal(0, 0.1 * 0.02, size=3)

        # Spaces (Fixed current_vals shape bug)
        self.observation_space = gym.spaces.Dict({
            "current_vals": gym.spaces.Box(low=0, high=10, shape=(len(self.quad_indices),), dtype=np.float32),
            "alpha": gym.spaces.Box(low=-1e6, high=1e6, shape=(len(monitor_indices), 3), dtype=np.float32),
            "beta":  gym.spaces.Box(low=-1e6, high=1e6, shape=(len(monitor_indices), 3), dtype=np.float32),
            "gamma": gym.spaces.Box(low=-1e6, high=1e6, shape=(len(monitor_indices), 3), dtype=np.float32),
            "target_alpha": gym.spaces.Box(low=-1e6, high=1e6, shape=(1,), dtype=np.float32),
            "target_beta":  gym.spaces.Box(low=-1e6, high=1e6, shape=(1,), dtype=np.float32),
            "target_gamma": gym.spaces.Box(low=-1e6, high=1e6, shape=(1,), dtype=np.float32),
        })
        self.action_space = gym.spaces.Discrete(len(quad_indices))

    def _get_obs(self):
        alpha_list, beta_list, gamma_list = [], [], []
        current_list = [self._beamline[q_idx].current for q_idx in self.quad_indices]
        
        # Fresh local copy of particles so tracking tracking transformations don't mutate globally
        local_particles = self.particles.copy()
        
        for idx, seg in enumerate(self._beamline):
            local_particles = np.array(seg.useMatrice(local_particles))
            
            if idx in self._monitor_locations:
                twiss = self.ebeam.getXYZ(local_particles)[3]
                
                # Apply degradation attributes directly
                alpha_list.append(twiss[self.ebeam.LABEL_MAPPING['alpha']] + seg.alpha_noise)
                beta_list.append(twiss[self.ebeam.LABEL_MAPPING['beta']] + seg.beta_noise)
                gamma_list.append(twiss[self.ebeam.LABEL_MAPPING['gamma']] + seg.gamma_noise)
        
        return {
            "current_vals": np.array(current_list, dtype=np.float32),
            "alpha": np.array(alpha_list, dtype=np.float32),
            "beta":  np.array(beta_list, dtype=np.float32),
            "gamma": np.array(gamma_list, dtype=np.float32),
            "target_alpha": self.target_alpha,
            "target_beta":  self.target_beta,
            "target_gamma": self.target_gamma,
        }

    def _get_info(self):
        return {"particles": self.particles}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Particle generation initialization
        self.particles = self.ebeam.gen_6d_gaussian(0, [1,1,1,1,0.1,100], self._NUM_PARTICLES)

        # Set default values and re-roll degradation parameters together
        for idx, seg in enumerate(self._beamline):            
            if idx in self.quad_indices:
                seg.current = 0.0889
            if idx in self._monitor_locations:
                seg.alpha_noise = np.random.normal(0, 0.02, size=3)
                seg.beta_noise  = np.random.normal(0, 0.02, size=3)
                seg.gamma_noise = np.random.normal(0, 0.002, size=3)

        return self._get_obs(), self._get_info()
    
    def _calculate_reward(self, obs):
        # Compare last monitor index reading arrays cleanly
        # 1. Extract the measured values at the final monitor index [-1] 
        # and average across the 3 planes (X, Y, Z) to compare against your scalar target
        curr = [np.mean(obs["alpha"][-1]), np.mean(obs["beta"][-1]), np.mean(obs["gamma"][-1])]
        targ = [obs["target_alpha"][0], obs["target_beta"][0], obs["target_gamma"][0]]
        
        # 2. Calculate absolute percentage error for each parameter
        # Adding a tiny 1e-8 to prevent division-by-zero errors if a target is 0
        errors = [abs(c - t) / (abs(t) + 1e-8) for c, t in zip(curr, targ)]
        reward = -sum(errors)
        
        # 3. The 10% Margin Bonus condition
        # 0.10 represents 10% deviation
        if all(e <= 0.10 for e in errors):
            reward += 10.0  
            
        return float(reward)

    def step(self, action):
        # Map Discrete action int to the corresponding beamline quad index
        target_quad_idx = self.quad_indices[action]
        
        # Set a dummy test adjustment step value (e.g., incrementing current by +0.1A)
        # Alternatively change this to map directly to absolute current scales
        new_current = np.clip(self._beamline[target_quad_idx].current + 0.1, 0, self.CURRENT_MAX)
        self._beamline[target_quad_idx].current = new_current

        obs = self._get_obs()
        reward = self._calculate_reward(obs)
        
        info = self._get_info()
        info['is_success'] = reward > 0
        
        return obs, reward, True, False, info