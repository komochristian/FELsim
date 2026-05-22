import gymnasium as gym
import numpy as np
from ebeam import beam
from beamline import *


class Tuning_env(gym.Env):
    def __init__(self, target_twiss, beamline, monitor_indices, quad_indices):
        super().__init__()

        self._target_twiss = target_twiss
        self._monitor_locations = monitor_indices
        self._beamline = beamline
        self.CURRENT_MAX = 10 # Amps
        self._NUM_PARTICLES = 1000

        self.ebeam = beam()
        self.particles = self.ebeam.gen_6d_gaussian(0,[1,1,1,1,0.1,100], self._NUM_PARTICLES)

        self.quad_indices = quad_indices
        for idx, seg in enumerate(self._beamline):            
            if idx in self._monitor_locations and not (isinstance(seg, qpdLattice) or isinstance(seg, qpfLattice)):
                raise TypeError(f"Segment at index {idx} is not a qpdLattice or qpfLattice and is at a monitor location.") 
        

        self.observation_space = gym.spaces.Dict(
            {
                "alpha": gym.spaces.Box(low=-1e6, high=1e6, shape=(len(monitor_indices), 3), dtype=np.float32),
                "beta": gym.spaces.Box(low=-1e6, high=1e6, shape=(len(monitor_indices), 3), dtype=np.float32),
                "gamma": gym.spaces.Box(low=-1e6, high=1e6, shape=(len(monitor_indices), 3), dtype=np.float32),
            }
        )

        self.action_space = gym.spaces.Discrete(len(quad_indices))

    def _get_obs(self):
        # 1. Initialize lists to collect data only at monitor locations
        alpha_list = []
        beta_list = []
        gamma_list = []
        
        for idx, seg in enumerate(self._beamline):
            self.particles = np.array(seg.useMatrice(self.particles))
            
            if idx in self._monitor_locations:
                twiss = self.ebeam.getXYZ(self.particles)[3]
                
                # Extract the X, Y, and Z components for each parameter
                # Assumes your label mapping returns an index or array slice for [X, Y, Z]
                a_xyz = twiss[self.ebeam.LABEL_MAPPING['alpha']]
                b_xyz = twiss[self.ebeam.LABEL_MAPPING['beta']]
                g_xyz = twiss[self.ebeam.LABEL_MAPPING['gamma']]
                
                # --- SIMULATE MONITOR DEGRADATION HERE ---
                # 1. White Noise (Random hardware jitter per measurement)
                # We use a percentage of the expected baseline scale (e.g., 2% noise)
                alpha_noise = np.random.normal(0, 1.0 * 0.02, size=3)
                beta_noise  = np.random.normal(0, 1.0 * 0.02, size=3)
                gamma_noise = np.random.normal(0, 0.1 * 0.02, size=3)
                
                # 2. Calibration Drift / Bias (Optional: consistent offset for this episode)
                # If you want a monitor to be systematically "wrong" by a fixed amount:
                # alpha_bias = 0.05 
                
                # Apply the degradation noise to the true physical values
                a_xyz_degraded = a_xyz + alpha_noise
                b_xyz_degraded = b_xyz + beta_noise
                g_xyz_degraded = g_xyz + gamma_noise
                # ----------------------------------------
                
                alpha_list.append(a_xyz_degraded)
                beta_list.append(b_xyz_degraded)
                gamma_list.append(g_xyz_degraded)
        
        # 2. Convert collected lists to NumPy arrays and cast to float32
        # This will result in the required shape: (len(monitor_indices), 3)
        obs_dict = {
            "alpha": np.array(alpha_list, dtype=np.float32),
            "beta": np.array(beta_list, dtype=np.float32),
            "gamma": np.array(gamma_list, dtype=np.float32),
        }
        
        return obs_dict
    
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











         

    

        
