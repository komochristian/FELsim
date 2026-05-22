import gymnasium as gym
import numpy as np
from ebeam import beam


class Tuning_env(gym.Env):
    def __init__(self, target_twiss, beamline, monitor_indices, quad_indices):
        super().__init__()

        self._target_twiss = target_twiss
        self._monitor_locations = monitor_indices
        self._beamline = beamline

        self.ebeam = beam()
        self.particles = self.ebeam.gen_6d_gaussian(0,[1,1,1,1,0.1,100], 1000)

        self.quad_indices = quad_indices

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
                
                alpha_list.append(a_xyz)
                beta_list.append(b_xyz)
                gamma_list.append(g_xyz)
        
        # 2. Convert collected lists to NumPy arrays and cast to float32
        # This will result in the required shape: (len(monitor_indices), 3)
        obs_dict = {
            "alpha": np.array(alpha_list, dtype=np.float32),
            "beta": np.array(beta_list, dtype=np.float32),
            "gamma": np.array(gamma_list, dtype=np.float32),
        }
        
        return obs_dict



         

    

        
