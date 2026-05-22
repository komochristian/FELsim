"""
beam_tuning_env.py — Tuning Mode Environment
=============================================

The agent adjusts quadrupole k-values to steer the beam toward a target
distribution.  The initial beam comes from a retrospective measurement
database (or a fixed stored measurement), NOT from random particle
generation each step.  This matches the project description: the virtual
accelerator replays real (or realistic) beam conditions.

Episode flow
------------
  reset()
    1. Sample one measurement from the database → use its particles as
       the initial bunch entering the beamline.
    2. Sample or use a fixed target distribution (also from database or
       from a target Twiss specification).
    3. Reset quad k-values (random or nominal).

  step(action)
    1. Apply new absolute k-values from action.
    2. Run simulation with the stored bunch through the current beamline.
    3. Read sensor at downstream monitor/screen.
    4. Compute reward based on how close observed moments are to target.
    5. Terminate if within success_threshold or max_steps reached.
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from beam_utils import (
    MeasurementDatabase,
    BeamMeasurement,
    compute_moments,
    moment_distance,
    obs_vector,
    find_quad_indices,
    get_beamline,
    set_quad_k,
    simulate,
    read_at,
    generate_particles,
)


class BeamTuningEnv(gym.Env):
    """
    Tuning Mode: correct quadrupole k-values so that the beam observed at
    a downstream sensor matches a desired target distribution.

    Parameters
    ----------
    database : MeasurementDatabase
        Retrospective database of beam measurements.  One entry is drawn
        at each reset() to serve as the initial bunch.
    target : BeamMeasurement or None
        Fixed target distribution.  If None, a second database entry is
        drawn at each reset() as the target (variable-target training).
    quad_indices : list[int] or None
        Which beamline elements are controllable quads.  Auto-detected if None.
    sensor_element_index : int
        Beamline element at which to read the downstream sensor (-1 = last).
    k_bounds : (float, float)
        Global (k_min, k_max) for all quads [1/m²].
    per_quad_bounds : list[(float,float)] or None
        Per-quad overrides to k_bounds.
    n_sample_obs : int
        Number of raw particle coordinates (x, x', y, y') in the observation.
    w_moment : float
        Weight on Twiss/moment matching in the reward.
    w_penalty : float
        Weight on quadrupole excursion penalty  Σ(k − k_nominal)².
    k_nominal : np.ndarray or None
        Reference k-values for the excursion penalty (defaults to zeros).
    success_threshold : float
        Moment distance below which the episode is considered solved.
    success_bonus : float
        One-time bonus added to reward on solve.
    max_steps : int
        Episode length limit.
    randomise_init_k : bool
        If True, randomise k-values at reset; else start from k_nominal.
    render_mode : str or None
        "human" prints a one-line summary each step.
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        database:             MeasurementDatabase,
        target:               Optional[BeamMeasurement] = None,
        quad_indices:         Optional[list[int]] = None,
        sensor_element_index: int = -1,
        k_bounds:             tuple[float, float] = (-10.0, 10.0),
        per_quad_bounds:      Optional[list[tuple[float, float]]] = None,
        n_sample_obs:         int = 50,
        w_moment:             float = 1.0,
        w_penalty:            float = 1e-3,
        k_nominal:            Optional[np.ndarray] = None,
        success_threshold:    float = 0.05,
        success_bonus:        float = 10.0,
        max_steps:            int = 100,
        randomise_init_k:     bool = True,
        render_mode:          Optional[str] = None,
    ):
        super().__init__()

        self._db              = database
        self._fixed_target    = target
        self.sensor_idx       = sensor_element_index
        self.n_sample_obs     = n_sample_obs
        self.w_moment         = w_moment
        self.w_penalty        = w_penalty
        self.success_threshold = success_threshold
        self.success_bonus    = success_bonus
        self.max_steps        = max_steps
        self.randomise_init_k = randomise_init_k
        self.render_mode      = render_mode

        # Beamline
        self._beamline = get_beamline()
        if quad_indices is None:
            quad_indices = find_quad_indices(self._beamline)
        if not quad_indices:
            raise ValueError("No quadrupoles found. Pass quad_indices explicitly.")
        self.quad_indices = quad_indices
        self.n_quads      = len(quad_indices)

        # k bounds
        k_lo, k_hi = k_bounds
        self.k_min = np.full(self.n_quads, k_lo, dtype=np.float64)
        self.k_max = np.full(self.n_quads, k_hi, dtype=np.float64)
        if per_quad_bounds:
            for i, (lo, hi) in enumerate(per_quad_bounds):
                self.k_min[i] = lo; self.k_max[i] = hi

        self.k_nominal = (np.zeros(self.n_quads, dtype=np.float64)
                          if k_nominal is None else np.asarray(k_nominal, np.float64))

        # Spaces
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.n_quads,), dtype=np.float32)

        obs_dim = 6 + 6 + 3 + 3 + 3 + self.n_sample_obs * 4 + self.n_quads
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Internal state
        self._k_current:        np.ndarray = self.k_nominal.copy()
        self._init_particles:   np.ndarray = np.zeros((1, 6))
        self._target_moments:   dict       = {}
        self._step_count:       int        = 0
        self._prev_dist:        float      = np.inf

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._step_count = 0
        self._prev_dist  = np.inf

        # --- Draw initial bunch from database ---
        entry = self._db.sample(self.np_random)
        self._init_particles = entry.particles.copy()

        # --- Draw / set target ---
        if self._fixed_target is not None:
            target_particles = self._fixed_target.particles
        else:
            # Sample a different entry as target (keep sampling until distinct)
            for _ in range(10):
                tgt_entry = self._db.sample(self.np_random)
                if not np.array_equal(tgt_entry.particles, entry.particles):
                    break
            target_particles = tgt_entry.particles

        self._target_moments = compute_moments(target_particles)

        # --- Initialise quad k-values ---
        if self.randomise_init_k:
            self._k_current = self.np_random.uniform(self.k_min, self.k_max)
        else:
            self._k_current = self.k_nominal.copy()
        self._apply_k()

        obs  = self._get_obs()
        info = {"step": 0, "distance": None, "k_values": self._k_current.copy()}
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        # Map normalised action → physical k
        action = np.clip(np.asarray(action, np.float32), -1.0, 1.0)
        self._k_current = (
            self.k_min + (action.astype(np.float64) + 1.0) / 2.0
            * (self.k_max - self.k_min)
        )
        self._apply_k()
        self._step_count += 1

        obs  = self._get_obs()
        dist, reward = self._reward()

        terminated = dist < self.success_threshold
        if terminated:
            reward += self.success_bonus
        truncated = self._step_count >= self.max_steps
        self._prev_dist = dist

        info = {
            "step":     self._step_count,
            "distance": dist,
            "k_values": self._k_current.copy(),
            "solved":   terminated,
        }
        if self.render_mode == "human":
            tag = "  ✓ SOLVED" if terminated else ""
            k_s = " ".join(f"{k:+.3f}" for k in self._k_current)
            print(f"[{self._step_count:3d}] dist={dist:.5f}  "
                  f"reward={reward:+.4f}  k=[{k_s}]{tag}")

        return obs, float(reward), terminated, truncated, info

    def render(self) -> None: pass
    def close(self)  -> None: pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_k(self) -> None:
        for i, q_idx in enumerate(self.quad_indices):
            set_quad_k(q_idx, float(self._k_current[i]))

    def _simulate_and_read(self) -> np.ndarray:
        result = simulate(self._init_particles, self._beamline)
        return read_at(result, self.sensor_idx)

    def _get_obs(self) -> np.ndarray:
        coords = self._simulate_and_read()
        return obs_vector(coords, self.n_sample_obs,
                          self._k_current, self.k_min, self.k_max, self.np_random)

    def _reward(self) -> tuple[float, float]:
        coords = self._simulate_and_read()
        m_obs  = compute_moments(coords)
        dist   = moment_distance(m_obs, self._target_moments)

        improvement = self._prev_dist - dist if np.isfinite(self._prev_dist) else 0.0
        excursion   = float(np.sum((self._k_current - self.k_nominal) ** 2))

        reward = (self.w_moment  * improvement
                  - self.w_penalty * excursion)
        return dist, float(reward)

    @property
    def current_k(self) -> np.ndarray:
        return self._k_current.copy()

    @property
    def target_moments(self) -> dict:
        return self._target_moments