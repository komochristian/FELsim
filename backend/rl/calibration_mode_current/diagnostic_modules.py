from dataclasses import dataclass
from enum import Enum
import numpy as np
import time

class QualityFlag(Enum):
    GOOD = "GOOD"
    DEGRADED = "DEGRADED"
    SATURATED = "SATURATED"
    FAULT = "FAULT"

@dataclass
class MeasurementPacket:
    diagnostic_id: str
    location_z: float
    raw_signal: np.ndarray
    observables: dict          # e.g., {'sigma_x': 0.45, 'mean_x': 0.02}
    units: dict                # e.g., {'sigma_x': 'mm', 'mean_x': 'mm'}
    covariance: np.ndarray     # Uncertainty matrix for observables
    quality_flag: QualityFlag
    validity: bool
    timestamp: float
    latency: float
    state: str                 # e.g., "READY", "ARMED", "ERROR"
    version: str
    _ground_truth: dict        # Isolated simulation truth for debugging/evals

class NoiseModel:
    def __init__(self, seed=None, noise_std=0.02, offset=0.0, scale_drift=1.0, resolution=0.001):
        self.rng = np.random.default_rng(seed)
        self.noise_std = noise_std
        self.offset = offset
        self.scale_drift = scale_drift
        self.resolution = resolution

    def apply(self, signal: np.ndarray) -> np.ndarray:
        # Scale drift and baseline offset
        corrupted = (signal * self.scale_drift) + self.offset
        
        # Gaussian statistical noise
        corrupted += self.rng.normal(0, self.noise_std, size=signal.shape)
        
        # Digitization / sensor spatial resolution quantization
        if self.resolution > 0:
            corrupted = np.round(corrupted / self.resolution) * self.resolution
            
        return corrupted


class FaultModel:
    def __init__(self, dropout_prob=0.0, saturation_limit=10.0, roi_shift_pixels=0, is_stuck=False):
        self.dropout_prob = dropout_prob
        self.saturation_limit = saturation_limit
        self.roi_shift_pixels = roi_shift_pixels
        self.is_stuck = is_stuck
        self.stuck_signal = None

    def apply(self, signal: np.ndarray, rng: np.random.Generator):
        # 1. Hardware Dropout
        if rng.random() < self.dropout_prob:
            return np.zeros_like(signal), QualityFlag.FAULT, False

        corrupted = signal.copy()
        quality = QualityFlag.GOOD

        # 2. ROI Shift Error
        if self.roi_shift_pixels != 0:
            corrupted = np.roll(corrupted, self.roi_shift_pixels)

        # 3. Saturation Limit
        if np.any(corrupted >= self.saturation_limit):
            corrupted = np.clip(corrupted, 0, self.saturation_limit)
            quality = QualityFlag.SATURATED

        # 4. Stuck Readout Value
        if self.is_stuck:
            if self.stuck_signal is None:
                self.stuck_signal = corrupted
            return self.stuck_signal.copy(), QualityFlag.DEGRADED, True

        return corrupted, quality, True

class VirtualDiagnostic:
    def __init__(self, diagnostic_id: str, location_z: float, noise_model=None, fault_model=None, seed=None, version="1.0.0"):
        self.diagnostic_id = diagnostic_id
        self.location_z = location_z
        self.noise_model = noise_model or NoiseModel()
        self.fault_model = fault_model or FaultModel()
        self.rng = np.random.default_rng(seed)
        self.version = version
        self.state = "READY"

    def observation_model(self, particles: np.ndarray):
        """Generates ideal raw signal (e.g., 1D projection) and extracts ground truth."""
        x_coords = particles[:, 0]
        counts, _ = np.histogram(x_coords, bins=30, range=(-5, 5))
        
        ground_truth = {
            "mean_x": float(np.mean(x_coords)),
            "sigma_x": float(np.std(x_coords))
        }
        return counts.astype(np.float64), ground_truth

    def analysis_model(self, raw_signal: np.ndarray):
        """Fits corrupted signal to extract measured scalars and covariance."""
        bins = np.linspace(-5, 5, len(raw_signal))
        total = np.sum(raw_signal) + 1e-9
        
        mean_x = np.sum(bins * raw_signal) / total
        var_x = np.sum(raw_signal * (bins - mean_x)**2) / total
        sigma_x = np.sqrt(max(var_x, 0.0))

        observables = {"mean_x": float(mean_x), "sigma_x": float(sigma_x)}
        covariance = np.diag([0.005, 0.005])  # Estimated fit variance
        return observables, covariance

    def measure(self, particles: np.ndarray) -> MeasurementPacket:
        start_time = time.time()

        # 1. Ideal beam response
        clean_signal, ground_truth = self.observation_model(particles)

        # 2. Hardware noise & discrete faults
        noisy_signal = self.noise_model.apply(clean_signal)
        corrupted_signal, quality_flag, validity = self.fault_model.apply(noisy_signal, self.rng)

        # 3. Data reduction fit
        observables, covariance = self.analysis_model(corrupted_signal)

        return MeasurementPacket(
            diagnostic_id=self.diagnostic_id,
            location_z=self.location_z,
            raw_signal=corrupted_signal,
            observables=observables,
            units={"mean_x": "mm", "sigma_x": "mm"},
            covariance=covariance,
            quality_flag=quality_flag,
            validity=validity,
            timestamp=time.time(),
            latency=time.time() - start_time,
            state=self.state,
            version=self.version,
            _ground_truth=ground_truth
        )