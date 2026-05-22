"""
beam_utils.py
=============
Shared utilities for both BeamTuningEnv and BeamCalibrationEnv.

Covers:
  - The interface contract your simulation must satisfy (mocked here)
  - MeasurementDatabase: loads / replays retrospective beam measurements
  - Beam physics helpers: moments, Twiss, emittance, distribution distance
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


# ============================================================================
#  ACCELERATOR SIMULATION INTERFACE  (replace mocks with your real imports)
# ============================================================================
#
#  Expected signatures:
#
#   get_beamline() -> list[Component]
#       Returns the beamline as an ordered list of element objects.
#       Each element must have a `.kind` attribute in
#       {"drift", "quadrupole", "dipole", "monitor", ...}
#       Quadrupoles must additionally have `.index: int`.
#
#   generate_particles(n, mean, sigma, seed) -> np.ndarray  (n, 6)
#       Samples a fresh bunch.  Used ONLY when no database entry is
#       available (e.g. during random-target episodes).
#
#   set_quad_k(quad_index, k_value) -> None
#   get_quad_k(quad_index) -> float
#
#   simulate(particles, beamline) -> SimResult
#   read_at(sim_result, element_index) -> np.ndarray  (n, 6)
#       element_index == -1  →  last element (default screen).
#
# ============================================================================

class _MockDrift:
    kind = "drift"
    def __init__(self, length: float): self.length = length

class _MockQuad:
    kind = "quadrupole"
    def __init__(self, k: float, length: float, index: int):
        self.k = k; self.length = length; self.index = index

class _MockDipole:
    kind = "dipole"
    def __init__(self, angle: float, length: float):
        self.angle = angle; self.length = length

class _MockMonitor:
    kind = "monitor"
    def __init__(self, index: int, offset: float = 0.0):
        self.index = index; self.offset = offset   # offset = calibration error

_BEAMLINE: list = []
_QUAD_K: dict[int, float] = {}
_MONITOR_OFFSETS: dict[int, float] = {}   # used by calibration env

def get_beamline() -> list:
    global _BEAMLINE, _QUAD_K, _MONITOR_OFFSETS
    if not _BEAMLINE:
        rng = np.random.default_rng(0)
        elements: list = []
        quad_idx = 0
        mon_idx  = 0
        for segment in range(5):
            elements.append(_MockDrift(0.5))
            k0 = rng.uniform(-2.0, 2.0)
            elements.append(_MockQuad(k0, 0.1, quad_idx))
            _QUAD_K[quad_idx] = k0
            quad_idx += 1
            elements.append(_MockDrift(0.3))
            elements.append(_MockMonitor(mon_idx, offset=0.0))
            _MONITOR_OFFSETS[mon_idx] = 0.0
            mon_idx += 1
        elements.append(_MockDrift(1.0))
        elements.append(_MockDipole(np.pi / 8, 0.3))
        _BEAMLINE = elements
    return _BEAMLINE

def generate_particles(
    n: int = 1000,
    mean: Optional[np.ndarray] = None,
    sigma: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if mean  is None: mean  = np.zeros(6)
    if sigma is None: sigma = np.array([1e-3, 1e-4, 1e-3, 1e-4, 1e-3, 1e-4])
    return (rng.standard_normal((n, 6)) * sigma + mean).astype(np.float64)

def set_quad_k(quad_index: int, k_value: float) -> None:
    global _BEAMLINE, _QUAD_K
    _QUAD_K[quad_index] = k_value
    for el in _BEAMLINE:
        if isinstance(el, _MockQuad) and el.index == quad_index:
            el.k = k_value

def get_quad_k(quad_index: int) -> float:
    return _QUAD_K.get(quad_index, 0.0)

def simulate(particles: np.ndarray, beamline: list) -> Any:
    coords = particles.copy()
    snapshots = [coords.copy()]
    for el in beamline:
        if el.kind == "drift":
            coords[:, 0] += el.length * coords[:, 1]
            coords[:, 2] += el.length * coords[:, 3]
        elif el.kind == "quadrupole":
            f = 1.0 / (el.k * el.length) if abs(el.k * el.length) > 1e-9 else 1e9
            coords[:, 1] -= coords[:, 0] / f
            coords[:, 3] += coords[:, 2] / f
        elif el.kind == "dipole":
            coords[:, 1] += el.angle * 0.5
        elif el.kind == "monitor":
            # Apply monitor offset (calibration error) to stored snapshot
            c = coords.copy()
            c[:, 0] += getattr(el, "offset", 0.0)
            c[:, 2] += getattr(el, "offset", 0.0)
            snapshots.append(c)
            continue
        snapshots.append(coords.copy())

    class _Result:
        snapshots_ = snapshots
        beamline_  = beamline
    return _Result()

def read_at(sim_result: Any, element_index: int) -> np.ndarray:
    idx = element_index if element_index >= 0 else len(sim_result.snapshots_) - 1
    idx = min(idx, len(sim_result.snapshots_) - 1)
    return sim_result.snapshots_[idx].copy()

# ============================================================================
#  END MOCK BLOCK
# ============================================================================


# ---------------------------------------------------------------------------
# Measurement database
# ---------------------------------------------------------------------------

@dataclass
class BeamMeasurement:
    """
    One entry in the retrospective database.

    particles : (n, 6) float64   — stored particle coordinates
    quad_k    : dict[int, float] — quad settings at time of measurement
    metadata  : dict             — timestamp, run number, etc. (optional)
    """
    particles : np.ndarray
    quad_k    : dict[int, float] = field(default_factory=dict)
    metadata  : dict             = field(default_factory=dict)


class MeasurementDatabase:
    """
    Retrospective database of beam measurements.

    In production: load real measurement files (HDF5, CSV, …).
    Here: generates a synthetic database of N entries so the code runs
    without real data.

    Usage
    -----
        db = MeasurementDatabase.from_synthetic(n_entries=200)
        # or
        db = MeasurementDatabase.from_hdf5("measurements.h5")

        measurement = db.sample(rng)   # random entry
        measurement = db[42]           # specific entry
    """

    def __init__(self, entries: list[BeamMeasurement]):
        if not entries:
            raise ValueError("Database must contain at least one measurement.")
        self._entries = entries

    # --- Loaders ---

    @classmethod
    def from_synthetic(
        cls,
        n_entries:   int   = 200,
        n_particles: int   = 1000,
        seed:        int   = 0,
    ) -> "MeasurementDatabase":
        """
        Generate a synthetic database with varied beam conditions.
        Replace with your real loader in production.
        """
        rng = np.random.default_rng(seed)
        entries = []
        for i in range(n_entries):
            sigma_x = rng.uniform(0.5e-3, 2.5e-3)
            sigma_y = rng.uniform(0.5e-3, 2.5e-3)
            mean    = rng.uniform(-1e-3, 1e-3, size=6)
            sigma   = np.array([sigma_x, sigma_x*0.1,
                                 sigma_y, sigma_y*0.1,
                                 1e-3,    1e-4])
            particles = generate_particles(n_particles, mean=mean,
                                           sigma=sigma, seed=int(rng.integers(0, 2**31)))
            # Simulate realistic quad settings for this entry
            quad_k = {qi: float(rng.uniform(-3.0, 3.0))
                      for qi in range(5)}   # 5 quads in mock beamline
            entries.append(BeamMeasurement(particles=particles,
                                           quad_k=quad_k,
                                           metadata={"entry_id": i}))
        return cls(entries)

    @classmethod
    def from_hdf5(cls, path: str, n_particles: int = 1000) -> "MeasurementDatabase":
        """
        Load from an HDF5 file.
        Expected structure:
            /entry_<i>/particles   (n, 6) dataset
            /entry_<i>/quad_k      (n_quads,) dataset
            /entry_<i>/quad_ids    (n_quads,) dataset  [integer quad indices]
        """
        import h5py   # type: ignore
        entries = []
        with h5py.File(path, "r") as f:
            for key in sorted(f.keys()):
                g = f[key]
                particles = g["particles"][:]
                quad_ids  = g["quad_ids"][:]
                quad_vals = g["quad_k"][:]
                quad_k    = dict(zip(quad_ids.tolist(), quad_vals.tolist()))
                entries.append(BeamMeasurement(particles=particles, quad_k=quad_k))
        return cls(entries)

    @classmethod
    def from_numpy(cls, path: str) -> "MeasurementDatabase":
        """
        Load from a .npz file saved as:
            np.savez("db.npz", particles=..., quad_k=..., quad_ids=...)
        where particles is (N_entries, n_particles, 6).
        """
        data = np.load(path, allow_pickle=True)
        entries = []
        for i in range(len(data["particles"])):
            qk = dict(zip(data["quad_ids"].tolist(), data["quad_k"][i].tolist()))
            entries.append(BeamMeasurement(particles=data["particles"][i], quad_k=qk))
        return cls(entries)

    # --- Access ---

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> BeamMeasurement:
        return self._entries[idx]

    def sample(self, rng: np.random.Generator) -> BeamMeasurement:
        return self._entries[int(rng.integers(0, len(self._entries)))]


# ---------------------------------------------------------------------------
# Beam physics helpers
# ---------------------------------------------------------------------------

def compute_moments(coords: np.ndarray) -> dict[str, np.ndarray]:
    """
    Compute statistical moments, RMS sizes, geometric emittances,
    and Courant-Snyder (Twiss) parameters from a (n, 6) particle array.

    Returns
    -------
    dict with keys:
        mean       (6,)  — first moments
        sigma      (6,)  — RMS sizes
        emittance  (3,)  — geometric emittance (x, y, z planes)
        twiss_x    (3,)  — (beta_x, alpha_x, gamma_x)
        twiss_y    (3,)  — (beta_y, alpha_y, gamma_y)
    """
    mean    = coords.mean(axis=0)
    centered = coords - mean
    sigma   = centered.std(axis=0)

    def _plane(u: np.ndarray, up: np.ndarray):
        s_uu  = float(np.mean(u  * u))
        s_uup = float(np.mean(u  * up))
        s_upup = float(np.mean(up * up))
        eps   = np.sqrt(max(s_uu * s_upup - s_uup**2, 1e-30))
        beta  =  s_uu  / eps
        alpha = -s_uup / eps
        gamma =  s_upup / eps
        return eps, beta, alpha, gamma

    eps_x, bx, ax, gx = _plane(centered[:, 0], centered[:, 1])
    eps_y, by, ay, gy = _plane(centered[:, 2], centered[:, 3])
    eps_z, *_         = _plane(centered[:, 4], centered[:, 5])

    return {
        "mean":      mean.astype(np.float32),
        "sigma":     sigma.astype(np.float32),
        "emittance": np.array([eps_x, eps_y, eps_z], dtype=np.float32),
        "twiss_x":   np.array([bx, ax, gx],          dtype=np.float32),
        "twiss_y":   np.array([by, ay, gy],           dtype=np.float32),
    }


def moment_distance(m_obs: dict, m_tgt: dict) -> float:
    """Weighted L2 distance between two moment dicts."""
    d  = float(np.sum((m_obs["mean"]      - m_tgt["mean"])      ** 2))
    d += float(np.sum((m_obs["sigma"]     - m_tgt["sigma"])     ** 2)) * 1e2
    d += float(np.sum((m_obs["emittance"] - m_tgt["emittance"]) ** 2)) * 1e6
    d += float(np.sum((m_obs["twiss_x"]   - m_tgt["twiss_x"])   ** 2)) * 10.0
    d += float(np.sum((m_obs["twiss_y"]   - m_tgt["twiss_y"])   ** 2)) * 10.0
    return float(np.sqrt(max(d, 0.0)))


def obs_vector(
    coords:       np.ndarray,
    n_sample_obs: int,
    k_values:     np.ndarray,
    k_min:        np.ndarray,
    k_max:        np.ndarray,
    rng:          np.random.Generator,
) -> np.ndarray:
    """
    Build the flat observation vector shared by both environments.

    Layout
    ------
    mean(6) | sigma(6) | emittance(3) | twiss_x(3) | twiss_y(3)
    | raw_particles(n_sample*4) | k_norm(n_quads)
    """
    m = compute_moments(coords)

    if n_sample_obs > 0 and len(coords) >= n_sample_obs:
        idx = rng.integers(0, len(coords), size=n_sample_obs)
        raw = coords[idx, :4].flatten().astype(np.float32)
    else:
        raw = np.zeros(n_sample_obs * 4, dtype=np.float32)

    k_norm = (2.0 * (k_values - k_min) / np.maximum(k_max - k_min, 1e-9) - 1.0
              ).astype(np.float32)

    vec = np.concatenate([
        m["mean"], m["sigma"], m["emittance"],
        m["twiss_x"], m["twiss_y"],
        raw, k_norm,
    ]).astype(np.float32)

    return np.clip(np.nan_to_num(vec, nan=0.0, posinf=1e6, neginf=-1e6), -1e6, 1e6)


def find_quad_indices(beamline: list) -> list[int]:
    """Return the beamline positions of all quadrupoles."""
    return [i for i, el in enumerate(beamline)
            if getattr(el, "kind", None) == "quadrupole"]


def find_monitor_indices(beamline: list) -> list[int]:
    """Return the beamline positions of all monitors."""
    return [i for i, el in enumerate(beamline)
            if getattr(el, "kind", None) == "monitor"]