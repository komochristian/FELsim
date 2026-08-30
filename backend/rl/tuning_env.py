import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import numpy as np
from ebeam import beam
from beamline import *

class Tuning_env(gym.Env):
    # target_twiss parameter replaced with target_sigma dict containing {"sigma_x": float, "sigma_y": float}
    def __init__(self, target_sigma, beamline, monitor_indices, quad_indices):
        super().__init__()

        self.target_sigma_x = np.array([target_sigma["sigma_x"]], dtype=np.float32)
        self.target_sigma_y = np.array([target_sigma["sigma_y"]], dtype=np.float32)
        
        self._monitor_locations = monitor_indices
        self.quad_indices = quad_indices
        self._beamline = beamline
        self.CURRENT_MAX = 10.0
        self._NUM_PARTICLES = 1000
        self.PARTICLE_STD_ABS_STDEV_NOISE_MM = 0.02
        self.PARTICLE_STD_SCALE_STDEV_NOISE_PERCENTAGE = 0.04
        self.DEFAULT_QUAD_CURRENT_AMPS = 0.0889
        self.ebeam = beam()

        if not beamline:
            raise ValueError("The beamline array cannot be empty.")
        if not monitor_indices:
            raise ValueError("monitor_indices cannot be empty. The agent needs at least one monitor to observe.")
        if not quad_indices:
            raise ValueError("quad_indices cannot be empty. The agent needs at least one quadrupole to control.")
        if "sigma_x" not in target_sigma or "sigma_y" not in target_sigma:
            raise KeyError("target_sigma dictionary must explicitly contain 'sigma_x' and 'sigma_y' keys.")
        if target_sigma["sigma_x"] <= 0 or target_sigma["sigma_y"] <= 0:
            raise ValueError(f"Target sigmas must be strictly greater than zero. Received: {target_sigma}")  

        if not all(isinstance(i, int) for i in monitor_indices + quad_indices):
            raise TypeError("All indices in monitor_indices and quad_indices must be integers.")

        if len(quad_indices) != len(set(quad_indices)):
            raise ValueError("Duplicate indices found in quad_indices. Each quadrupole must only be listed once.")
            
        if len(monitor_indices) != len(set(monitor_indices)):
            raise ValueError("Duplicate indices found in monitor_indices. Each monitor must only be listed once.") 
        
        beamline_length = len(beamline)
        for idx in monitor_indices:
            if idx < 0 or idx >= beamline_length:
                raise IndexError(f"Monitor index {idx} is out of bounds for a beamline of length {beamline_length}.")
        for idx in quad_indices:
            if idx < 0 or idx >= beamline_length:
                raise IndexError(f"Quadrupole index {idx} is out of bounds for a beamline of length {beamline_length}.")

        for idx in quad_indices:
            if not isinstance(beamline[idx], (qpdLattice, qpfLattice)):
                raise TypeError(f"Lattice component at index {idx} was flagged as a quad, but is type {type(beamline[idx]).__name__}.")
            
        if monitor_indices != sorted(monitor_indices):
            raise ValueError("monitor_indices must be sorted in ascending order to ensure reward calculations target the final screen.")
            
        if quad_indices != sorted(quad_indices):
            raise ValueError("quad_indices must be sorted in ascending order down the beamline.")

        # Ensure the agent isn't tuning a magnet it cannot observe
        if max(quad_indices) > max(monitor_indices):
            raise ValueError(
                f"Blind magnet detected! The final quadrupole is at index {max(quad_indices)}, "
                f"but the final monitor is at index {max(monitor_indices)}. "
                "The agent must have a monitor placed after the final quadrupole to observe its effects."
            )

        # Spaces
        self.observation_space = gym.spaces.Dict({
            "current_vals": gym.spaces.Box(low=0, high=10, shape=(len(self.quad_indices),), dtype=np.float32),
            "sigma_x": gym.spaces.Box(low=0, high=1e4, shape=(len(monitor_indices), 1), dtype=np.float32),
            "sigma_y": gym.spaces.Box(low=0, high=1e4, shape=(len(monitor_indices), 1), dtype=np.float32),
            "target_sigma_x": gym.spaces.Box(low=0, high=1e4, shape=(1,), dtype=np.float32),
            "target_sigma_y": gym.spaces.Box(low=0, high=1e4, shape=(1,), dtype=np.float32),
        })
        
        # FIXED BUG 2: Normalize action space to [-1, 1] as recommended by Gymnasium
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(len(quad_indices),), dtype=np.float32)

    def _get_obs(self):
        sigma_x_list, sigma_y_list = [], []
        current_list = [self._beamline[q_idx].current for q_idx in self.quad_indices]
        
        local_particles = self.particles.copy()
        
        for idx, seg in enumerate(self._beamline):
            local_particles = np.array(seg.useMatrice(local_particles))
            
            if idx in self._monitor_locations:
                true_sigma_x = self.ebeam.std(local_particles, 'x')
                true_sigma_y = self.ebeam.std(local_particles, 'y')
                
                # Apply simulated degradation attributes directly to diagnostic reads
                sigma_x_list.append([true_sigma_x*seg.sigma_x_scale_noise_percentage + seg.sigma_x_abs_noise_mm])
                sigma_y_list.append([true_sigma_y*seg.sigma_y_scale_noise_percentage + seg.sigma_y_abs_noise_mm])
        
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
        # FIXED BUG 1: Properly pass the seed up to Gym's built-in generator
        super().reset(seed=seed)
        
        # Seed NumPy's global random state if an explicit environment seed is requested.
        # This guarantees gen_6d_gaussian generates identical particle spreads during verification passes.
        if seed is not None:
            np.random.seed(seed)
        
        self.particles = self.ebeam.gen_6d_gaussian(0, [1,1,1,1,0.1,100], self._NUM_PARTICLES)
        sigma_x_abs_noise_mm = np.random.normal(scale=self.PARTICLE_STD_ABS_STDEV_NOISE_MM, loc=0.0)
        sigma_y_abs_noise_mm = np.random.normal(scale=self.PARTICLE_STD_ABS_STDEV_NOISE_MM, loc=0.0)
        sigma_x_scale_noise_percentage = np.random.normal(scale=self.PARTICLE_STD_SCALE_STDEV_NOISE_PERCENTAGE, loc=1.0)
        sigma_y_scale_noise_percentage = np.random.normal(scale=self.PARTICLE_STD_SCALE_STDEV_NOISE_PERCENTAGE, loc=1.0)

        for idx, seg in enumerate(self._beamline):
            # Reset all quadrupoles model can control to default current value.            
            if idx in self.quad_indices:
                seg.current = self.DEFAULT_QUAD_CURRENT_AMPS
            if idx in self._monitor_locations:
                # Noise included in object so we can make noise unique to each monitor in future with calibration mode
                seg.sigma_x_abs_noise_mm = sigma_x_abs_noise_mm
                seg.sigma_y_abs_noise_mm = sigma_y_abs_noise_mm
                seg.sigma_x_scale_noise_percentage = sigma_x_scale_noise_percentage
                seg.sigma_y_scale_noise_percentage = sigma_y_scale_noise_percentage 
        return self._get_obs(), self._get_info()
    
    def _calculate_reward(self, obs):
        # CHANGED: Evaluate rewards based entirely on the last monitor's transverse sizes [-1]
        curr_sx = obs["sigma_x"][-1][0]
        curr_sy = obs["sigma_y"][-1][0]
        
        targ_sx = obs["target_sigma_x"][0]
        targ_sy = obs["target_sigma_y"][0]
        
        #  Add 1e-8 to avoid division by zero
        relative_err_x = abs(curr_sx - targ_sx) / (abs(targ_sx) + 1e-8)
        relative_err_y = abs(curr_sy - targ_sy) / (abs(targ_sy) + 1e-8)

        # print("err_x:", err_x, "err_y:", err_y)
        
        reward = -(relative_err_x + relative_err_y)
        if relative_err_x > 2 or relative_err_y > 2:
            reward -= 10.0

        if relative_err_x > 1.5 or relative_err_y > 1.5:
            reward -= 4.0
        
        reward += 1/((relative_err_x + 1e-8)**0.75)
        reward += 1/((relative_err_y + 1e-8)**0.75)
            
        return float(reward)

    def step(self, action):
        # FIXED BUG 2: Unscale incoming action array from [-1, 1] to actual physical currents [0, 10.0 Amps]
        # Formula: physical_value = ((action + 1) / 2) * (max - min) + min
        scaled_currents = ((action + 1.0) / 2.0) * self.CURRENT_MAX
        clamped_currents = np.clip(scaled_currents, 0.0, self.CURRENT_MAX)

        for q_idx, new_current in zip(self.quad_indices, clamped_currents):
            self._beamline[q_idx].current = float(new_current)

        obs = self._get_obs()
        reward = self._calculate_reward(obs)
        
        info = self._get_info()
        info['is_success'] = reward > 0
        
        return obs, reward, True, False, info
    

if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env

    dummy_target = {"sigma_x": 1.5, 
                    "sigma_y": 1.5
                    }
    beamline_list = [
        driftLattice(length = 1),
        qpdLattice(current = 1),
        driftLattice(length = 1),
        qpfLattice(current = 1)
    ]
    monitor_indice = [0, 1, 2]
    quad_indices = [1]

    tuning_env = Tuning_env(
        dummy_target,
        beamline_list,
        monitor_indice,
        quad_indices
    )


    # This will catch many common issues
    try:
        check_env(tuning_env)
        print("Environment passes all checks!")
    except Exception as e:
        print(f"Environment has issues: {e}")