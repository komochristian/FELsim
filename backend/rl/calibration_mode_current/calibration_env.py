import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import numpy as np
from ebeam import beam
from beamline import *
import copy
from backend.rl.calibration_mode_current.calibration_reference import *


class CalibrationSMDPEnv(gym.Env):
    def __init__(self):
        # Action space: Discrete menu of measurement recipes (Options)
        self.action_space = gym.spaces.Discrete(NUM_RECIPES)
        
        # Observation space: 14 estimates, 105 covariance, 6 currents, time budget
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(127,))
        
        # Initialize physical and diagnostic models[cite: 1]
        self.diagnostic = OTRScreen() #[cite: 1]
        self.analysis = OTRGaussianAnalysis() #[cite: 1]
        self.max_budget_seconds = 100.0

    def reset(self, seed=None):
        # Reset master clock, beam, and hidden machine state[cite: 1]
        self.clock = Clock() #[cite: 1]
        self.beam = initial_beam(seed) #[cite: 1]
        self.errors = benchmark_quad_errors() #[cite: 1]
        self.currents = INITIAL_CURRENTS.copy() #[cite: 1]
        self.screen_inserted = False 
        
        # Initialize belief state (Estimator tracks the 14 unknowns)
        self.estimator = Estimator() 
        
        return self._get_observation(), {}

    def step(self, action):
        # 1. Decode agent action into physical commands
        target_currents, n_pulses = self._decode_recipe(action)
        
        # 2. Execute measurement (advances clock for settling, insertion, pulses, and latency)[cite: 1]
        frames, measurements = execute_measurement(
            machine=self, 
            diagnostic=self.diagnostic, 
            analysis=self.analysis, 
            estimator=self.estimator, 
            currents=target_currents, 
            n_pulses=n_pulses, 
            quad_rng=self.np_random, 
            camera_rng=self.np_random, 
            config=GaussianConfig(), #[cite: 1]
            charge_latency=self._charge_latency #[cite: 1]
        )

        # 3. Check temporal boundaries
        terminated = self.clock.t >= self.max_budget_seconds
        
        # 4. SMDP Reward Calculation: (Uncertainty reduction) - (Elapsed time cost)
        reward = self._calculate_reward(measurements)
        
        return self._get_observation(), reward, terminated, False, {}

    def _get_observation(self):
        # Fetch the 14 estimates and 105 covariance elements, appending current physical state
        estimates, covariance = self.estimator.get_current_belief()
        remaining_time = self.max_budget_seconds - self.clock.t
        return np.concatenate([estimates, covariance, self.currents, [remaining_time]])

    def _charge_latency(self, kind, func):
        # Wrapper to time live computational steps and deduct them from the master clock[cite: 1]
        start_time = measure_real_time()
        result = func()
        self.clock.advance(measure_real_time() - start_time) #[cite: 1]
        return result

    def action_masks(self):
        # MaskablePPO interface: Zero out actions that require more time than the remaining budget
        remaining = self.max_budget_seconds - self.clock.t
        return [self._estimate_recipe_cost(a) <= remaining for a in range(self.action_space.n)]