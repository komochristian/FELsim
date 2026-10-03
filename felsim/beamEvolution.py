from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import bisect


@dataclass
class ElementInfo:
    """Beamline element metadata for schematic drawing."""
    element_type: str
    s_start: float
    s_end: float
    length: float
    color: str
    index: int
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BeamEvolution:
    """
    Container for beam evolution data from any simulator.
    Maintains s_positions in sorted order automatically.

    Attributes
    ----------
    s_positions : List[float]
        Longitudinal positions where data was sampled [m]
        **Automatically maintained in sorted order**
    particles : Dict[float, np.ndarray]
        Particle distributions keyed by s-position
        Each array is (N, 6) in FELsim coordinates
    twiss : Dict[float, dict]
        Twiss parameters keyed by s-position
        Format: {'x': {'beta': ..., 'alpha': ..., ...}, 'y': {...}}
    elements : List[ElementInfo]
        Beamline element information for schematic
    total_length : float
        Total beamline length [m]
    num_particles : int
        Number of particles in distribution
    simulator_name : str
        Name of simulator that produced data
    beam_energy : float
        Beam kinetic energy [MeV]
    """
    s_positions: List[float] = field(default_factory=list)
    particles: Dict[float, np.ndarray] = field(default_factory=dict)
    twiss: Dict[float, dict] = field(default_factory=dict)
    elements: List[ElementInfo] = field(default_factory=list)
    total_length: float = 0.0
    num_particles: int = 0
    simulator_name: str = ""
    beam_energy: float = 0.0

    # Tolerance for considering two s-positions identical [m]
    _s_tolerance: float = field(default=1e-9, repr=False)

    def __post_init__(self):
        """Ensure s_positions is sorted on initialization."""
        if self.s_positions and self.s_positions != sorted(self.s_positions):
            # Sort and check for near-duplicates
            self.s_positions.sort()
            self._check_duplicates()

    def _check_duplicates(self):
        """Check for s-positions within tolerance and warn."""
        for i in range(len(self.s_positions) - 1):
            if abs(self.s_positions[i + 1] - self.s_positions[i]) < self._s_tolerance:
                raise ValueError(
                    f"Duplicate s-positions within tolerance: "
                    f"s[{i}]={self.s_positions[i]:.12f}, "
                    f"s[{i + 1}]={self.s_positions[i + 1]:.12f} "
                    f"(diff={abs(self.s_positions[i + 1] - self.s_positions[i]):.2e} m)"
                )

    def add_sample(self, s: float, particles: Optional[np.ndarray] = None,
                   twiss: Optional[dict] = None):
        """
        Add beam data at position s, maintaining sorted order.

        Parameters
        ----------
        s : float
            Longitudinal position [m]
        particles : np.ndarray, optional
            Particle distribution (N, 6)
        twiss : dict, optional
            Twiss parameters {'x': {...}, 'y': {...}}
        """
        # Check if s already exists within tolerance
        nearest_s, nearest_idx = self._find_nearest_s(s)
        if nearest_s is not None and abs(s - nearest_s) < self._s_tolerance:
            raise ValueError(
                f"s={s:.12f} m already exists as s={nearest_s:.12f} m "
                f"(within tolerance {self._s_tolerance:.2e} m). "
                f"Use update_sample() to modify existing data."
            )

        # Insert s in sorted position using bisect
        insert_idx = bisect.bisect_left(self.s_positions, s)
        self.s_positions.insert(insert_idx, s)

        # Add associated data
        if particles is not None:
            self.particles[s] = particles
        if twiss is not None:
            self.twiss[s] = twiss

    def update_sample(self, s: float, particles: Optional[np.ndarray] = None,
                      twiss: Optional[dict] = None):
        """
        Update beam data at existing s-position (or nearest within tolerance).

        Parameters
        ----------
        s : float
            Longitudinal position [m]
        particles : np.ndarray, optional
            Particle distribution (N, 6)
        twiss : dict, optional
            Twiss parameters {'x': {...}, 'y': {...}}
        """
        nearest_s, _ = self._find_nearest_s(s)
        if nearest_s is None or abs(s - nearest_s) > self._s_tolerance:
            raise KeyError(
                f"No s-position exists near s={s:.12f} m "
                f"(tolerance {self._s_tolerance:.2e} m). "
                f"Use add_sample() to add new data."
            )

        # Update using the actual stored s-value
        if particles is not None:
            self.particles[nearest_s] = particles
        if twiss is not None:
            self.twiss[nearest_s] = twiss

    def _find_nearest_s(self, s: float) -> tuple[Optional[float], Optional[int]]:
        """Find s-position nearest to target s using binary search."""
        if not self.s_positions:
            return None, None

        # Binary search for insertion point
        idx = bisect.bisect_left(self.s_positions, s)

        # Check boundaries
        if idx == 0:
            return self.s_positions[0], 0
        if idx == len(self.s_positions):
            return self.s_positions[-1], len(self.s_positions) - 1

        # Compare neighbors
        before = self.s_positions[idx - 1]
        after = self.s_positions[idx]

        if abs(s - before) <= abs(s - after):
            return before, idx - 1
        else:
            return after, idx

    def get_particles_at(self, s: float, tolerance: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Get particle distribution at or nearest to given s-position.

        Parameters
        ----------
        s : float
            Target longitudinal position [m]
        tolerance : float, optional
            Maximum allowed distance to nearest s-position [m]
            If None, uses internal tolerance

        Returns
        -------
        particles : np.ndarray or None
            Particle distribution (N, 6), or None if no data within tolerance
        """
        if tolerance is None:
            tolerance = self._s_tolerance

        nearest_s, _ = self._find_nearest_s(s)

        if nearest_s is None:
            return None

        if abs(nearest_s - s) <= tolerance:
            return self.particles.get(nearest_s)

        return None

    def get_twiss_at(self, s: float, tolerance: Optional[float] = None) -> Optional[dict]:
        """
        Get Twiss parameters at or nearest to given s-position.

        Parameters
        ----------
        s : float
            Target longitudinal position [m]
        tolerance : float, optional
            Maximum allowed distance to nearest s-position [m]

        Returns
        -------
        twiss : dict or None
            Twiss parameters, or None if no data within tolerance
        """
        if tolerance is None:
            tolerance = self._s_tolerance

        nearest_s, _ = self._find_nearest_s(s)

        if nearest_s is None:
            return None

        if abs(nearest_s - s) <= tolerance:
            return self.twiss.get(nearest_s)

        return None

    def get_twiss_evolution(self) -> pd.DataFrame:
        """
        Extract Twiss evolution as DataFrame for analysis/export.
        s_positions already sorted, so no need to sort again.

        Returns
        -------
        pd.DataFrame
            Columns: s, beta_x, beta_y, alpha_x, alpha_y, gamma_x, gamma_y,
                     epsilon_x, epsilon_y, envelope_x, envelope_y,
                     dispersion_x, dispersion_y
        """
        data = []

        # s_positions is guaranteed sorted
        for s in self.s_positions:
            if s not in self.twiss:
                continue

            t = self.twiss[s]

            # Calculate envelopes (simplified - no dispersion contribution)
            eps_x = t['x'].get('emittance', 0) * 1e-6  # π·mm·mrad → π·m·rad
            eps_y = t['y'].get('emittance', 0) * 1e-6
            env_x = 1e3 * np.sqrt(eps_x * t['x']['beta']) if eps_x > 0 else 0
            env_y = 1e3 * np.sqrt(eps_y * t['y']['beta']) if eps_y > 0 else 0

            data.append({
                's': s,
                'beta_x': t['x']['beta'],
                'beta_y': t['y']['beta'],
                'alpha_x': t['x']['alpha'],
                'alpha_y': t['y']['alpha'],
                'gamma_x': t['x']['gamma'],
                'gamma_y': t['y']['gamma'],
                'epsilon_x': t['x'].get('emittance', 0),
                'epsilon_y': t['y'].get('emittance', 0),
                'dispersion_x': t['x'].get('dispersion', 0),
                'dispersion_y': t['y'].get('dispersion', 0),
                'envelope_x': env_x,
                'envelope_y': env_y
            })

        return pd.DataFrame(data)

    def __repr__(self):
        return (f"BeamEvolution({self.simulator_name}, "
                f"{len(self.s_positions)} samples, "
                f"{self.num_particles} particles, "
                f"L={self.total_length:.3f} m)")
