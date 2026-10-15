import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import time
import gymnasium as gym
import numpy as np
from ebeam import beam
from beamline import *
import copy
from backend.rl.calibration_mode_current.calibration_reference import *
from backend.rl.calibration_mode_current.data_classes import Clock, GaussianConfig
from otr_estimator import Estimator

NUM_RECIPES = 20


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
        # Receive currents to change, and number of macropulses to simulate
        # scan_steps: list(list(list(current_vals), n_pulses))
        scan_steps = self._decode_recipe(action)
        
        # 2. Execute measurement (advances clock for settling, insertion, pulses, and latency)[cite: 1]
        frames, measurements = execute_measurement(
            machine=self, 
            diagnostic=self.diagnostic, 
            analysis=self.analysis, 
            estimator=self.estimator, 
            scan_steps=scan_steps,
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
        start_time = time.time()
        result = func()
        self.clock.advance(time.time() - start_time) #[cite: 1]
        return result

    def action_masks(self):
        # MaskablePPO interface: Zero out actions that require more time than the remaining budget
        remaining = self.max_budget_seconds - self.clock.t
        return [self._estimate_recipe_cost(a) <= remaining for a in range(self.action_space.n)]
    
    def _decode_recipe(self, action_val):
        """
        Decodes a discrete action ID into a sequence of steps (recipe_steps) 
        and specifies the diagnostic type.
        
        Returns:
            recipe_steps: List of tuples -> [(target_currents, n_pulses), ...]
            diagnostic_kind: str -> "otr"
        """
        currents = self.currents.copy()
        
        # --- BLOCK 1: Baseline Acquisitions (Indices 0 - 1) ---
        if action_val == 0:
            # Acquire 1 pulse at current settings
            return [(currents, 1)]
        elif action_val == 1:
            # Acquire 5 pulses at current settings
            return [(currents, 5)]

        # --- BLOCK 2: OAT Scans (Indices 2 - 7) ---
        # 6 quadrupoles total. Sweeps chosen quad across 5 steps from 0.0 A to 10.0 A.
        elif 2 <= action_val <= 7:
            quad_idx = action_val - 2
            scan_values = np.linspace(0.0, 10.0, 5)
            
            recipe_steps = []
            for val in scan_values:
                step_currents = currents.copy()
                step_currents[quad_idx] = val
                # Acquire 1 pulse per scan step
                recipe_steps.append((step_currents, 1))
                
            return recipe_steps

        # --- BLOCK 3: Dynamic Micro-Actions (Indices 8 - 19) ---
        # 6 quads * 2 directions (+0.1A and -0.1A) = 12 actions.
        elif 8 <= action_val <= 19:
            adjusted_action = action_val - 8
            quad_idx = adjusted_action // 2
            direction = 1 if (adjusted_action % 2 == 0) else -1
            
            # Apply +/- 0.1 A bump
            step_currents = currents.copy()
            step_currents[quad_idx] += direction * 0.1
            
            # Safety clamp to keep within the valid 0-10A scan range
            step_currents[quad_idx] = np.clip(step_currents[quad_idx], 0.0, 10.0)
            
            # Execute 1 pulse acquisition for the micro-action
            return [(step_currents, 1)]

        else:
            raise ValueError(f"Invalid action value: {action_val}")