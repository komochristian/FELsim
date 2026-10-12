"""Ref functions for
"""
from dataclasses import dataclass
from typing import Protocol
import math
import numpy as np
from backend.rl.calibration_mode_current.data_classes import QuadErrors, RawFrame, Measurement, GaussianConfig

QUAD_INDICES = (1, 3, 10, 16, 18, 20)
INITIAL_CURRENTS = np.array([
    .84, 1.05, 3.98,
    2.05, 4.88, 3.48,
])
RF_HZ = 2.856e9
K_PER_AMP_40_MEV = 19.9379220911
INITIAL_K = K_PER_AMP_40_MEV * INITIAL_CURRENTS
QUAD_SETTLE_S = 2.0
OTR_INSERT_S = OTR_RETRACT_S = 1.0
REP_RATE_HZ = 1.0                 # Machine pulses per second
BUNCH_REPETITION_HZ = RF_HZ       # Every RF bucket filled by one bunch.
BUNCH_CHARGE_C = 60e-12          #  charge per bunch
MACROPULSE_DURATION_S = 5e-6     # 5 us pulse train
_bunch_count = BUNCH_REPETITION_HZ * MACROPULSE_DURATION_S
if not np.isclose(_bunch_count, round(_bunch_count), rtol=0, atol=1e-9):
    raise ValueError("Specify RF bucket timing for a noninteger bunch train")
BUNCHES_PER_MACROPULSE = int(round(_bunch_count))  # 14,280
MACROPULSE_CHARGE_C = BUNCH_CHARGE_C * BUNCHES_PER_MACROPULSE  # 856.8 nC
ENERGY_MEV = 40.0
REST_MEV = .51099895
QE = .6                       # Provisional effective quantum efficiency.
NOMINAL_MAGNIFICATION = .825 / .18333333333333334
PIXEL_PITCH_M = 5.86e-6


def initial_beam(seed=12345):
    """Notebook prescription, one fixed realization; units remain FELSim's."""
    gamma = 1 + ENERGY_MEV / REST_MEV
    beta_gamma = np.sqrt(gamma**2 - 1)
    divergence_mrad = (8 / beta_gamma) / .8
    beam = np.random.default_rng(seed).normal(
        0, [.8, divergence_mrad, .8, divergence_mrad, 5.712, 5.0],
        size=(100_000, 6),
    )
    beam[:, 5] += 5e9 * beam[:, 4] / RF_HZ
    return beam

def benchmark_quad_errors():
    return QuadErrors(
        b0=INITIAL_K * np.array([.01, -.015, .02, -.025, .03, -.035]),
        sigma_k=INITIAL_K * np.array([.005, .0075, .01, .0125, .015, .02]),
        A=np.zeros(6), P=np.full(6, 300.), phi=np.zeros(6),
    )


def quad_k(quad, current, k_actual=None):
    """Add to beamline.py; call from BOTH numerical quad tracking paths."""
    k = (abs(quad.Q * quad.G * current /
             (quad.M * quad.C * quad.beta * quad.gamma))
         if k_actual is None else float(k_actual))
    if not np.isfinite(k) or k < 0:
        raise ValueError("Invalid effective quadrupole strength")
    return k


def track_one_pulse(line, input_beam, currents, errors, t, rng):
    """Track one representative bunch for a complete 5 us machine pulse.
    All bunches initially share this distribution and the same six quad errors.
    line is already truncated/configured for 40 MeV; no new aperture model.
    """
    nominal = np.array([quad_k(line[i], I)
                        for i, I in zip(QUAD_INDICES, currents)])
    effective = errors.sample(nominal, t, rng)
    overrides = dict(zip(QUAD_INDICES, zip(currents, effective)))
    beam = input_beam.copy()
    for i, element in enumerate(line):
        args = {}
        if i in overrides:
            I, k_actual = overrides[i]
            args = dict(current=I, k_actual=k_actual)
        beam = element.useMatrice(beam, **args)
    return beam


def to_otr_coordinates(beam):
    result = np.array(beam, dtype=float, copy=True)
    result[:, :4] *= 1e-3       # mm/mrad -> m/rad
    dt = beam[:, 4] / (1000 * RF_HZ)
    gamma = 1 + ENERGY_MEV / REST_MEV
    beta = np.sqrt(1 - 1 / gamma**2)
    result[:, 4] = beta * 299792458.0 * dt  # longitudinal length, unused by forward()
    kinetic = ENERGY_MEV * (1 + beam[:, 5] * 1e-3)
    if np.any(kinetic <= 0):
        raise ValueError("Nonpositive particle kinetic energy")
    p = np.sqrt(kinetic * (kinetic + 2 * REST_MEV))
    p0 = np.sqrt(ENERGY_MEV * (ENERGY_MEV + 2 * REST_MEV))
    result[:, 5] = p / p0 - 1   # relative momentum, not relative kinetic energy
    return result



class BeamDiagnostic(Protocol):
    interceptive: bool

    def acquire(self, beam, *, t, currents, rng) -> RawFrame: ...


class OTRScreen:
    interceptive = True

    def __init__(self, sigma_read=3., z2=.827):
        from otr_diagnostic_v3 import OTRDiagnostic as OpticalModel
        self._sigma_read = float(sigma_read)
        self._optics = OpticalModel(
            reference_kinetic_energy_MeV=40.,
            optical_transport={"lens_to_sensor_m": z2},
        )

    def expected_photons_per_machine_period(self, beam):
        """Expected photons/pixel from the whole 5 us bunch train in one period.
        The remaining beam-off part of the period contributes no beam photons.
        Identical bunches and fixed focusing within the train are assumed.
        """
        import torch
        with torch.no_grad():
            photons_per_bunch = self._optics(
                to_otr_coordinates(beam), bunch_charge_C=BUNCH_CHARGE_C,
            ).cpu().numpy()
        return sum_identical_bunch_images(photons_per_bunch)

    def acquire(self, beam, *, t, currents, rng):
        #  Generate expected beam shape as an np.array
        #  return an array of expected photos per pixel over all bunches in time period
        #  Ex. at 2.856 GHz, expected photon data includes all 14280 bunches in each 5 microsecond period
        #  the bunch is fired.
        photons = self.expected_photons_per_machine_period(beam)

        #  Generate OTR snapshot frame of beam distribution with poisson and normal bellcurve noise
        #  (so it is assumed that the end of the beamline contains the OTR screen)
        image = sample_cmos_frame(photons, self._sigma_read, rng)
        return RawFrame(image, t, np.array(currents, copy=True))


def sum_identical_bunch_images(photons_per_bunch,
                              bunch_count=BUNCHES_PER_MACROPULSE):
    """Sum expected signals before detection; never synthesize bunch readouts."""
    if not isinstance(bunch_count, (int, np.integer)) or bunch_count < 1:
        raise ValueError("bunch_count must be a positive integer")
    photons = np.asarray(photons_per_bunch, dtype=float)
    if not np.isfinite(photons).all() or np.any(photons < 0):
        raise ValueError("Expected photon counts must be finite and nonnegative")
    return bunch_count * photons


def sample_cmos_frame(photons_per_period, sigma_read, rng):
    """One frame per machine pulse: one Poisson draw and one camera readout.
    Image is electrons/pixel. No saturation/gain/dark-current model yet.
    For N software-averaged frames at fixed signal:
    Var(mean) = (QE * photons_per_period + sigma_read**2) / N.
    Within-pulse quad errors are common to all bunches and do not average away.
    """
    photons = np.asarray(photons_per_period, dtype=float)
    if not np.isfinite(photons).all() or np.any(photons < 0):
        raise ValueError("Expected photon counts must be finite and nonnegative")
    if not np.isfinite(sigma_read) or sigma_read < 0:
        raise ValueError("Read-noise RMS must be finite and nonnegative")
    return rng.poisson(QE * photons) + rng.normal(0, sigma_read, photons.shape)



def gaussian_image(parameters, shape):
    """Axis-aligned Gaussian integrated over pixels, plus constant background."""
    from scipy.special import ndtr
    log_signal, col0, row0, log_sc, log_sr, background = parameters
    rows, cols = shape
    pc = np.diff(ndtr((np.arange(cols + 1) - .5 - col0) / np.exp(log_sc)))
    pr = np.diff(ndtr((np.arange(rows + 1) - .5 - row0) / np.exp(log_sr)))
    return np.exp(log_signal) * pr[:, None] * pc[None, :] + background


class OTRGaussianAnalysis:
    diagnostic_kind = "otr"

    def analyze(self, raw, config=GaussianConfig()):
        from scipy.optimize import least_squares

        #  Estimate background level and check camera boundaries
        r0, r1, c0, c1 = config.roi
        image = np.asarray(raw.image[r0:r1, c0:c1], dtype=float)
        rows, cols = image.shape
        if min(rows, cols) < 8 or not np.isfinite(image).all():
            raise ValueError("Invalid OTR image/ROI")
        border = config.background_border
        if not 1 <= border < min(rows, cols) / 2:
            raise ValueError("Invalid background border")
        background_mask = np.ones(image.shape, dtype=bool)
        background_mask[border:-border, border:-border] = False
        dark = image[background_mask]
        background = np.median(dark)
        dark_var = max(float(np.var(dark, ddof=1)), 1e-12)

        #  Initial guess of the centroid and spread of beamline with zero, first, and second order summation
        rr, cc = np.indices(image.shape)
        weights = np.maximum(image - background, 0)  # initialization only
        total = weights.sum()
        if total <= 0:
            raise ValueError("No positive signal for initialization")
        col0, row0 = (weights * cc).sum()/total, (weights * rr).sum()/total
        sc = np.sqrt((weights * (cc-col0)**2).sum()/total)
        sr = np.sqrt((weights * (rr-row0)**2).sum()/total)
        initial = [np.log(total), col0, row0,
                   np.log(np.clip(sc, .5, cols/3)),
                   np.log(np.clip(sr, .5, rows/3)), background]
        lower = [-40, 0, 0, np.log(.3), np.log(.3), -np.inf]
        upper = [60, cols-1, rows-1, np.log(cols), np.log(rows), np.inf]

        #  Run reweighting with two passes to improve predictions.
        #  Two feasible weighted-least-squares passes. Never read hidden sigma_read.
        noise = np.sqrt(np.maximum(image-background, 0) + dark_var)
        for _ in range(2):
            fit = least_squares(
                lambda p: ((gaussian_image(p, image.shape)-image)/noise).ravel(),
                initial, bounds=(lower, upper), x_scale="jac", max_nfev=150,
            )
            model = gaussian_image(fit.x, image.shape)
            initial = fit.x
            next_noise = np.sqrt(np.maximum(model-fit.x[5], 0) + dark_var)
            if _ == 0:
                noise = next_noise

        #  Measure quaility of prediction with chi2, construct fisher information,
        #  Calculate scaled parameter covariance matrix
        residual = (model-image) / noise
        chi2 = float(np.sum(residual**2) / max(image.size-6, 1))
        information = fit.jac.T @ fit.jac
        full_rank = np.linalg.matrix_rank(information) == 6
        covariance = np.linalg.pinv(information) * max(1., chi2)

        #  Transfer pixel space into physical beam size properties
        scale = PIXEL_PITCH_M / NOMINAL_MAGNIFICATION
        # At nominal geometry rows=P -> beam x; columns=S -> beam y.
        sx, sy = np.exp(fit.x[[4, 3]]) * scale
        derivative = np.zeros((2, 6))
        derivative[0, 4], derivative[1, 3] = sx, sy
        covariance_xy = derivative @ covariance @ derivative.T
        col0, row0 = fit.x[1:3]
        sc, sr = np.exp(fit.x[3:5])
        margin = config.edge_margin_sigma

        #  run final quaility checking on data
        contained = (min(col0-border, cols-1-border-col0) > margin*sc and
                     min(row0-border, rows-1-border-row0) > margin*sr)
        dark_clean = (np.max(model[background_mask]-fit.x[5]) <
                      config.max_background_signal_sigma*np.sqrt(dark_var))
        signal_snr = 1 / np.sqrt(max(covariance[0, 0], 1e-30))
        valid = bool(fit.success and full_rank and contained and dark_clean and
                     chi2 < config.max_reduced_chi2 and
                     signal_snr > config.min_signal_snr)
        measurement = Measurement(
            np.array([sx, sy]), covariance_xy, dark_var, dark.size-1,
            valid, chi2, raw.t, raw.currents.copy(),
        )
        return measurement, {"image": image, "fit": model, "residual": image-model}

    @staticmethod
    def results(debug):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3)
        for ax, key in zip(axes, ("image", "fit", "residual")):
            ax.imshow(debug[key], origin="lower")
            ax.set_title(key)
        return fig


DIAGNOSTIC_TYPES = {"otr": OTRScreen}
ANALYSIS_METHODS = {("otr", "gaussian_v1"): OTRGaussianAnalysis}



def execute_measurement(machine, diagnostic, analysis, estimator, currents,
                        n_pulses, quad_rng, camera_rng, config, charge_latency):
    """Scheduler integration example. machine owns line, beam, errors, clock,
    currents, screen_inserted. n_pulses counts complete machine pulses, each
    containing BUNCHES_PER_MACROPULSE bunches. One camera readout per pulse.
    charge_latency(kind, callable) executes and
    charges measured or replayed online latency; it is not simulation timing.
    Check remaining budget INCLUDING processing/retraction before calling.
    """
    #  Insert OTR screen, and set machine currents if not equal. Subtract time bank 
    if not np.array_equal(currents, machine.currents):
        machine.currents = np.array(currents, copy=True)
        machine.clock.advance(QUAD_SETTLE_S)
    if not machine.screen_inserted:
        machine.clock.advance(OTR_INSERT_S)
        machine.screen_inserted = True

    frames = []

    # Iterate over each timestamp for each pulse that occurs every second
    for t in machine.clock.pulse_times(n_pulses):
        machine.clock.advance(t - machine.clock.t)

        # Propogate pulse through beamline with new currents and error model
        # k strength of quadropole is calculated and used
        beam = track_one_pulse(machine.line, machine.beam, machine.currents,
                               machine.errors, t, quad_rng)

        # Add data frame capturing pulse data with diagnostic tool
        frames.append(diagnostic.acquire(beam, t=t, currents=machine.currents,
                                         rng=camera_rng))
    # Analysis happens after acquisition; no artificial gaps between its pulses.
    # Prepare analysis data to be used by the estimator
    results = charge_latency("analysis", lambda: [analysis.analyze(f, config)
                                                  for f in frames])
    measurements = [m for m, debug in results]
    charge_latency("estimator", lambda: estimator.update(measurements))
    return frames, measurements


def width_negative_log_likelihood(measurements, predict_mean_cov):
    """predict_mean_cov(m) supplies C_total (quad spread + camera error), NOT
    its square root. Candidate parameters belong to that forward-model closure.
    Use a matching likelihood for background variance to estimate read noise.
    """
    loss = 0.
    for m in measurements:
        if not m.valid:
            continue
        mean, covariance = predict_mean_cov(m)
        factor = np.linalg.cholesky(covariance)
        whitened = np.linalg.solve(factor, m.sigma_xy-mean)
        loss += .5 * (whitened @ whitened +
                       2 * np.log(np.diag(factor)).sum())
    return float(loss)


def read_noise_negative_log_likelihood(measurements, sigma_read):
    """Gaussian empty-pixel variance likelihood, up to constants independent
    of sigma_read. Combination with the width likelihood is an approximation:
    image fitting also uses the estimated background variance.
    """
    if not np.isfinite(sigma_read) or sigma_read <= 0:
        return np.inf
    variance = sigma_read**2
    return float(sum(.5*m.background_dof *
                     (np.log(variance) + m.background_variance/variance)
                     for m in measurements if m.valid))
