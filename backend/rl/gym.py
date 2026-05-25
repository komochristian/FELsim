import gymnasium as gym
import numpy as np
from ebeam import beam
from beamline import *

#  Notes:
#  1. Assume monitor_indices and quad_indices contain the indices for each respective
#  2. Assume for monitor indices that the monitor is placed directly at the end of the
#     segment specified by the indice

class Tuning_env(gym.Env):
    def __init__(self, target_twiss, beamline, monitor_indices, quad_indices):
        super().__init__()

        self._target_twiss = target_twiss
        self.target_alpha = self._target_twiss["alpha"]
        self.target_beta = self._target_twiss["beta"]
        self.target_gamma = self._target_twiss["gamma"]
        self._monitor_locations = monitor_indices
        self._beamline = beamline
        self.CURRENT_MAX = 10 # Amps
        self._NUM_PARTICLES = 1000

        self.ebeam = beam()
        self.particles = self.ebeam.gen_6d_gaussian(0,[1,1,1,1,0.1,100], self._NUM_PARTICLES)

        self.quad_indices = quad_indices
        for idx, seg in enumerate(self._beamline):            
            if idx in self.quad_indices and not (isinstance(seg, qpdLattice) or isinstance(seg, qpfLattice)):
                raise TypeError(f"Quad segment at index {idx} is not a qpdLattice or qpfLattice.") 

        for idx, seg in enumerate(self._beamline):
            if idx in self._monitor_locations:
                # --- SIMULATE MONITOR DEGRADATION HERE ---
                # 1. White Noise (Random hardware jitter per measurement)
                # We use a percentage of the expected baseline scale (e.g., 2% noise)
                seg.alpha_noise = np.random.normal(0, 1.0 * 0.02, size=3)
                seg.beta_noise  = np.random.normal(0, 1.0 * 0.02, size=3)
                seg.gamma_noise = np.random.normal(0, 0.1 * 0.02, size=3)

        self.observation_space = gym.spaces.Dict(
            {
                "current_vals": gym.spaces.Box(low=0, high=10, shape=len(self.quad_indices,), dtype=np.float32),
                "alpha": gym.spaces.Box(low=-1e6, high=1e6, shape=(len(monitor_indices), 3), dtype=np.float32),
                "beta": gym.spaces.Box(low=-1e6, high=1e6, shape=(len(monitor_indices), 3), dtype=np.float32),
                "gamma": gym.spaces.Box(low=-1e6, high=1e6, shape=(len(monitor_indices), 3), dtype=np.float32),
                "target_alpha": gym.spaces.Box(low=-1e6, high=1e6, shape=(1,), dtype=np.float32),
                "target_beta": gym.spaces.Box(low=-1e6, high=1e6, shape=(1,), dtype=np.float32),
                "target_gamma": gym.spaces.Box(low=-1e6, high=1e6, shape=(1,), dtype=np.float32),
            }
        )
        self.action_space = gym.spaces.Discrete(len(quad_indices))

    def _get_obs(self):
        # 1. Initialize lists to collect data only at monitor locations
        alpha_list = []
        beta_list = []
        gamma_list = []
        current_list = []
        
        for idx, seg in enumerate(self._beamline):
            self.particles = np.array(seg.useMatrice(self.particles))

            if idx in self.quad_indices:
                current_list.append(seg.current)
            
            if idx in self._monitor_locations:
                twiss = self.ebeam.getXYZ(self.particles)[3]
                
                # Extract the X, Y, and Z components for each parameter
                # Assumes your label mapping returns an index or array slice for [X, Y, Z]
                a_xyz = twiss[self.ebeam.LABEL_MAPPING['alpha']]
                b_xyz = twiss[self.ebeam.LABEL_MAPPING['beta']]
                g_xyz = twiss[self.ebeam.LABEL_MAPPING['gamma']]
                
                # --- MONITOR DEGRADATION ---
                # 2. Calibration Drift / Bias (Optional: consistent offset for this episode)
                # If you want a monitor to be systematically "wrong" by a fixed amount:
                # alpha_bias = 0.05 
                
                # Apply the degradation noise to the true physical values
                a_xyz_degraded = a_xyz + seg.alpha_noise
                b_xyz_degraded = b_xyz + seg.beta_noise
                g_xyz_degraded = g_xyz + seg.gamma_noise
                # ----------------------------------------
                
                alpha_list.append(a_xyz_degraded)
                beta_list.append(b_xyz_degraded)
                gamma_list.append(g_xyz_degraded)
        
        # 2. Convert collected lists to NumPy arrays and cast to float32
        # This will result in the required shape: (len(monitor_indices), 3)
        obs_dict = {
            "current_vals": np.array(current_list, dtype=np.float32),
            "alpha": np.array(alpha_list, dtype=np.float32),
            "beta": np.array(beta_list, dtype=np.float32),
            "gamma": np.array(gamma_list, dtype=np.float32),
            "target_alpha": np.array(self.target_alpha, dtype=np.float32),
            "target_beta": np.array(self.target_beta, dtype=np.float32),
            "target_gamma": np.array(self.target_gamma, dtype=np.float32),
        }
        
        return obs_dict

    def _get_info(self):
        return {
            "particles": self.particles
        }

    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        baseline_mean = 0.0
        baseline_std  = np.array([1.0, 1.0, 1.0, 1.0, 0.1, 100.0], dtype=np.float32)

        #  Uncomment to add random noise to distribution
        # mean_noise = np.random.uniform(-0.05, 0.05, size=6) 
        # std_noise_factor = np.random.uniform(0.95, 1.05, size=6)
        # baseline_mean += mean_noise
        # baseline_std *= std_noise_factor
    
        self.particles = self.ebeam.gen_6d_gaussian(
            mean=baseline_mean, 
            std_dev=baseline_std, 
            num_particles=self._NUM_PARTICLES
        )

        for idx, seg in enumerate(self._beamline):            
            if idx in self.quad_indices:
                seg.current = 0.0889

        for idx, seg in enumerate(self._beamline):
            if idx in self._monitor_locations:
                # --- SIMULATE MONITOR DEGRADATION HERE ---
                # 1. White Noise (Random hardware jitter per measurement)
                # We use a percentage of the expected baseline scale (e.g., 2% noise)
                seg.alpha_noise = np.random.normal(0, 1.0 * 0.02, size=3)
                seg.beta_noise  = np.random.normal(0, 1.0 * 0.02, size=3)
                seg.gamma_noise = np.random.normal(0, 0.1 * 0.02, size=3)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def _calculate_reward(self, obs):
        """
        Calculates a dense reward based on the percentage error from targets,
        giving a large bonus if all parameters are within a 10% margin.
        """
        # 1. Extract the measured values at the final monitor index [-1] 
        # and average across the 3 planes (X, Y, Z) to compare against your scalar target
        current_a = np.mean(obs["alpha"][-1])
        current_b = np.mean(obs["beta"][-1])
        current_g = np.mean(obs["gamma"][-1])
        
        # Extract targets as scalars
        target_a = obs["target_alpha"][0]
        target_b = obs["target_beta"][0]
        target_g = obs["target_gamma"][0]
        
        # 2. Calculate absolute percentage error for each parameter
        # Adding a tiny 1e-8 to prevent division-by-zero errors if a target is 0
        err_a = abs(current_a - target_a) / (abs(target_a) + 1e-8)
        err_b = abs(current_b - target_b) / (abs(target_b) + 1e-8)
        err_g = abs(current_g - target_g) / (abs(target_g) + 1e-8)
        
        # 3. Base Reward: Penalize the model based on total percentage error
        # (Perfect score = 0, worse performance = increasingly negative)
        total_error = err_a + err_b + err_g
        reward = -total_error 
        
        # 4. The 10% Margin Bonus condition
        # 0.10 represents 10% deviation
        if err_a <= 0.10 and err_b <= 0.10 and err_g <= 0.10:
            reward += 10.0  # Give a significant positive bonus for achieving the goal!
            
        return float(reward)

    def step(self, action):
        currents = np.clip(action, 0, 10)

        for idx, current in currents:
            self._beamline[idx].current = current

        obs, _ = self._get_obs()
        info = self._get_info()

        reward = self._calculate_reward(obs)

        terminated = True
        truncated = False
        info['is_success'] = reward > 0
        

        return  obs, reward, terminated, truncated, info
    

        





    



