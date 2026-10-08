from calibration_reference import REP_RATE_HZ
from dataclasses import dataclass
import numpy as np
import math

@dataclass
class Clock:
    t: float = 0.
    rep_rate_hz: float = REP_RATE_HZ

    def advance(self, seconds):
        if not np.isfinite(seconds) or seconds < 0:
            raise ValueError("Clock advances must be finite and nonnegative")
        self.t += seconds

    def pulse_times(self, n):
        """Timestamps for n machine pulses, not RF bunches."""
        if not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError("Choose an integer number of pulses >= 1")
        # Whole pulse slots; guarantees integration >= n/f, even at an exact tick.
        start_slot = math.ceil(self.t * self.rep_rate_hz)
        return (start_slot + np.arange(1, n+1)) / self.rep_rate_hz

@dataclass
class QuadErrors:
    b0: np.ndarray             # m^-2; constant within an episode
    sigma_k: np.ndarray        # m^-2; independent machine-pulse-to-pulse RMS
    A: np.ndarray              # m^-2; zero means no drift
    P: np.ndarray              # seconds; positive even when A = 0
    phi: np.ndarray            # radians

    def sample(self, k_nominal, t, rng):
        b = self.b0 + self.A * np.sin(2 * np.pi * t / self.P + self.phi)
        k_actual = k_nominal + b + self.sigma_k * rng.normal(size=6)
        if not np.all(np.isfinite(k_actual)) or np.any(k_actual < 0):
            raise ValueError("Outside the initial positive-strength model")
        return k_actual


@dataclass
class RawFrame:
    image: np.ndarray          # electrons/pixel integrated over one bunch train
    t: float                   # master-clock machine-pulse timestamp
    currents: np.ndarray       # commands; never actual hidden k
    diagnostic: str = "DC2.OTR.032"
    bunches_integrated: int = BUNCHES_PER_MACROPULSE
    bunch_charge_C: float = BUNCH_CHARGE_C
    macropulse_duration_s: float = MACROPULSE_DURATION_S
    machine_period_s: float = 1 / REP_RATE_HZ
    camera_readouts: int = 1


@dataclass(frozen=True)
class GaussianConfig:
    # Fixed validated ROI; use the same configuration in every procedure.
    roi: tuple = (0, 1200, 0, 1920)  # row start/end, column start/end
    background_border: int = 8
    max_reduced_chi2: float = 3.0
    edge_margin_sigma: float = 3.0
    min_signal_snr: float = 5.0
    max_background_signal_sigma: float = .1


@dataclass
class Measurement:
    sigma_xy: np.ndarray       # [sigma_x, sigma_y], m in nominal beam coordinates
    covariance_xy: np.ndarray  # approximate fit uncertainty, m^2
    background_variance: float # electrons^2, for read-noise estimation
    background_dof: int
    valid: bool
    reduced_chi2: float
    t: float
    currents: np.ndarray
    method: str = "otr_gaussian_v1"
