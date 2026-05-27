import gymnasium as gym
import numpy as np
from ebeam import beam
from beamline import *

class Tuning_env(gym.Env):
    # CHANGED: target_twiss parameter replaced with target_sigma dict containing {"sigma_x": float, "sigma_y": float}
    def __init__(self, target_sigma, beamline, monitor_indices, quad_indices):
        super().__init__()

        self.target_sigma_x = np.array([target_sigma["sigma_x"]], dtype=np.float32)
        self.target_sigma_y = np.array([target_sigma["sigma_y"]], dtype=np.float32)
        
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
                # CHANGED: Set up monitor noise parameters for the physical sigma channels
                seg.sigma_x_noise = np.random.normal(0, 0.02)
                seg.sigma_y_noise = np.random.normal(0, 0.02)

        # CHANGED: Observation space altered to output only physical, noisy sigma data
        self.observation_space = gym.spaces.Dict({
            "current_vals": gym.spaces.Box(low=0, high=10, shape=(len(self.quad_indices),), dtype=np.float32),
            "sigma_x": gym.spaces.Box(low=0, high=1e4, shape=(len(monitor_indices), 1), dtype=np.float32),
            "sigma_y": gym.spaces.Box(low=0, high=1e4, shape=(len(monitor_indices), 1), dtype=np.float32),
            "target_sigma_x": gym.spaces.Box(low=0, high=1e4, shape=(1,), dtype=np.float32),
            "target_sigma_y": gym.spaces.Box(low=0, high=1e4, shape=(1,), dtype=np.float32),
        })
        
        self.action_space = gym.spaces.Box(low=0.0, high=self.CURRENT_MAX, shape=(len(quad_indices),), dtype=np.float32)

    def _get_obs(self):
        sigma_x_list, sigma_y_list = [], []
        current_list = [self._beamline[q_idx].current for q_idx in self.quad_indices]
        
        local_particles = self.particles.copy()
        
        for idx, seg in enumerate(self._beamline):
            local_particles = np.array(seg.useMatrice(local_particles))
            
            if idx in self._monitor_locations:
                # CHANGED: Extract actual standard deviations directly from the particle coordinate matrices
                # local_particles layout columns: [x, x_prime, y, y_prime, z, z_prime]
                true_sigma_x = np.std(local_particles[:, 0], ddof=1)
                true_sigma_y = np.std(local_particles[:, 2], ddof=1)
                
                # Apply simulated degradation attributes directly to diagnostic reads
                sigma_x_list.append([true_sigma_x + seg.sigma_x_noise])
                sigma_y_list.append([true_sigma_y + seg.sigma_y_noise])
        
        return {
            "current_vals": np.array(current_list, dtype=np.float32),
            "sigma_x": np.array(sigma_x_list, dtype=np.float32),
            "sigma_y": np.array(sigma_y_list, dtype=np.float32),
            "target_sigma_x": self.target_sigma_x,
            "target_sigma_y": self.target_sigma_y,
        }

    def _get_info(self):
        return {"particles": self.particles}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.particles = self.ebeam.gen_6d_gaussian(0, [1,1,1,1,0.1,100], self._NUM_PARTICLES)

        for idx, seg in enumerate(self._beamline):            
            if idx in self.quad_indices:
                seg.current = 0.0889
            if idx in self._monitor_locations:
                # Re-roll degradation noise settings at start of run
                seg.sigma_x_noise = np.random.normal(0, 0.02)
                seg.sigma_y_noise = np.random.normal(0, 0.02)

        return self._get_obs(), self._get_info()
    
    def _calculate_reward(self, obs):
        # CHANGED: Evaluate rewards based entirely on the last monitor's transverse sizes [-1]
        curr_sx = obs["sigma_x"][-1][0]
        curr_sy = obs["sigma_y"][-1][0]
        
        targ_sx = obs["target_sigma_x"][0]
        targ_sy = obs["target_sigma_y"][0]
        
        #  Add 1e-8 to avoid division by zero
        err_x = abs(curr_sx - targ_sx) / (abs(targ_sx) + 1e-8)
        err_y = abs(curr_sy - targ_sy) / (abs(targ_sy) + 1e-8)
        
        reward = -(err_x + err_y)
        
        # 10% Margin Bonus condition for beam spot size constraints
        if err_x <= 0.10 and err_y <= 0.10:
            reward += 10.0  
            
        return float(reward)

    def step(self, action):
        clamped_currents = np.clip(action, 0.0, self.CURRENT_MAX)

        for q_idx, new_current in zip(self.quad_indices, clamped_currents):
            self._beamline[q_idx].current = float(new_current)

        obs = self._get_obs()
        reward = self._calculate_reward(obs)
        
        info = self._get_info()
        info['is_success'] = reward > 0
        
        return obs, reward, True, False, info