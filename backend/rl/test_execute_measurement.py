#!/usr/bin/env python3
"""
test_execute_measurement.py -- standalone sandbox for execute_measurement()

Needs only numpy. No FELSim, torch, scipy, gymnasium or otr_diagnostic_v3.

WHAT IS IN HERE
  1. Constants / dataclasses ... copied from calibration_reference.py and
                                 data_classes.py (logic unchanged).
  2. Placeholders .............. simple stand-ins for everything unfinished
                                 (beam tracking, OTR camera, image analysis,
                                 estimator, latency charging, the "machine").
  3. execute_measurement() ..... your function, identical logic. Every line
                                 starting with `dprint(` is debug-only; delete
                                 them all and you have the original function.
  4. Test harness + SCENARIOS .. edit / add scenarios to try different inputs.

RUN IT
  python test_execute_measurement.py --list
  python test_execute_measurement.py                      # runs "baseline"
  python test_execute_measurement.py --scenario scan
  python test_execute_measurement.py --scenario all --quiet
  python test_execute_measurement.py --scenario custom    # edit CUSTOM below
  python test_execute_measurement.py --latency live       # real wall-clock latency

TIME COST CHEAT-SHEET (1 Hz machine, default constants)
  changed currents  -> +2 s settle          (QUAD_SETTLE_S)
  screen retracted  -> +1 s insertion       (OTR_INSERT_S)
  pulses            -> aligned to WHOLE-second slots; the first pulse is the
                       slot AFTER the clock's current position (see Clock.pulse_times)
  analysis/estimator latency -> charged to the clock AFTER the last pulse

SWAPPING IN REAL PIECES LATER
  execute_measurement() takes its collaborators as arguments, so in run_scenario()
  you can replace PlaceholderOTRScreen / PlaceholderGaussianAnalysis /
  PlaceholderEstimator with your real OTRScreen / OTRGaussianAnalysis / Estimator.
  track_one_pulse is looked up as a module-level name; replace the placeholder
  below (or import the real one) when FELSim is wired in.
"""
import argparse
import math
import time
from dataclasses import dataclass

import numpy as np

# ============================================================================
# 0. DEBUG PRINT HELPERS
# ============================================================================
VERBOSE = True


def dprint(msg="", level=0):
    """Debug print. Indented by `level`. Turned off by --quiet."""
    if VERBOSE:
        print("    " * level + msg)


def fmt(a, nd=3):
    return np.array2string(np.asarray(a), precision=nd, suppress_small=True,
                           separator=", ")


# ============================================================================
# 1. CONSTANTS  (values copied from calibration_reference.py)
# ============================================================================
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
REP_RATE_HZ = 1.0                      # machine pulses per second
BUNCH_CHARGE_C = 60e-12
MACROPULSE_DURATION_S = 5e-6
BUNCHES_PER_MACROPULSE = int(round(RF_HZ * MACROPULSE_DURATION_S))   # 14,280
ENERGY_MEV = 40.0
REST_MEV = .51099895

# Placeholder-only constants (the fake camera)
PLACEHOLDER_BINS = 40                  # fake sensor is 40 x 40 pixels
PLACEHOLDER_EXTENT_MM = 4.0            # sensor covers +/- 4 mm  (0.2 mm / pixel)
PLACEHOLDER_MIN_SIGNAL = 10_000        # "valid" if more than this many counts


# ============================================================================
# 1b. DATACLASSES  (copied from data_classes.py, logic unchanged)
# ============================================================================
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


class DebugClock(Clock):
    """Identical to Clock, but prints every time advance (debug only)."""

    def advance(self, seconds):
        before = self.t
        super().advance(seconds)
        dprint(f"[clock] {before:7.3f}s -> {self.t:7.3f}s   (+{seconds:.3f}s)", 2)


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
    roi: tuple = (0, 1200, 0, 1920)
    background_border: int = 8
    max_reduced_chi2: float = 3.0
    edge_margin_sigma: float = 3.0
    min_signal_snr: float = 5.0
    max_background_signal_sigma: float = .1


@dataclass
class Measurement:
    sigma_xy: np.ndarray       # [sigma_x, sigma_y], m
    covariance_xy: np.ndarray  # approximate fit uncertainty, m^2
    background_variance: float
    background_dof: int
    valid: bool
    reduced_chi2: float
    t: float
    currents: np.ndarray
    method: str = "otr_gaussian_v1"


# ============================================================================
# 1c. REFERENCE HELPERS  (copied from calibration_reference.py)
# ============================================================================
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


# ============================================================================
# 2. PLACEHOLDERS  (stand-ins for everything not implemented yet)
# ============================================================================
def measure_real_time():
    """Placeholder for the undefined measure_real_time() in the env."""
    return time.perf_counter()


def track_one_pulse(line, input_beam, currents, errors, t, rng):
    """PLACEHOLDER for FELSim tracking of one machine pulse.

    REAL parts kept: the hidden quad-error draw (QuadErrors.sample).
    FAKE parts: nominal k (real code: quad_k(line[i], I)) and the beam
    'optics', which is just a toy scaling of x and y so you can SEE currents
    and quad errors change the measured width.
    """
    nominal = K_PER_AMP_40_MEV * np.asarray(currents, dtype=float)   # fake quad_k
    effective = errors.sample(nominal, t, rng)                       # real draw
    dprint(f"[placeholder track] nominal k[0:3]  = {fmt(nominal[:3])}", 2)
    dprint(f"[placeholder track] effective k[0:3]= {fmt(effective[:3])}"
           f"   (offset + pulse jitter)", 2)

    beam = input_beam.copy()
    # TOY optics: width reacts to quad-0 / quad-1 current and to quad-0 error.
    x_scale = (1 + 0.10 * (currents[0] - INITIAL_CURRENTS[0])) \
        * (1 + 0.02 * (effective[0] - nominal[0]))
    y_scale = (1 - 0.10 * (currents[1] - INITIAL_CURRENTS[1])) \
        * (1 + 0.02 * (effective[1] - nominal[1]))
    beam[:, 0] *= x_scale
    beam[:, 2] *= y_scale
    dprint(f"[placeholder track] toy scale x={x_scale:.4f}  y={y_scale:.4f}", 2)
    return beam


class PlaceholderOTRScreen:
    """PLACEHOLDER camera: 2-D histogram of the beam (x = rows, y = columns)
    plus Gaussian read noise. No optics, no photons, no PSF."""
    interceptive = True

    def __init__(self, sigma_read=3.):
        self._sigma_read = float(sigma_read)

    def acquire(self, beam, *, t, currents, rng):
        e = PLACEHOLDER_EXTENT_MM
        counts, _, _ = np.histogram2d(
            beam[:, 0], beam[:, 2],
            bins=PLACEHOLDER_BINS, range=[[-e, e], [-e, e]])
        image = counts + rng.normal(0, self._sigma_read, counts.shape)
        dprint(f"[placeholder OTR.acquire] t={t:.3f}s  particles on sensor="
               f"{int(counts.sum())}  image shape={image.shape}", 2)
        return RawFrame(image, t, np.array(currents, copy=True))


class PlaceholderGaussianAnalysis:
    """PLACEHOLDER analysis: RMS widths from image moments (no least-squares
    fit, no quality gates). Returns (Measurement, debug_dict) like the real one."""
    diagnostic_kind = "otr"

    def analyze(self, raw, config=GaussianConfig()):
        image = np.asarray(raw.image, dtype=float)
        border = 2
        mask = np.ones(image.shape, dtype=bool)
        mask[border:-border, border:-border] = False
        dark = image[mask]
        background = float(np.median(dark))
        # Ignore pixels within 3 sigma of the dark noise, otherwise clipping the
        # noise at zero adds a small positive bias to the widths.
        signal = image - background
        weights = np.where(signal > 3 * np.std(dark, ddof=1), signal, 0.0)
        total = float(weights.sum())
        pixel_m = 2 * PLACEHOLDER_EXTENT_MM / PLACEHOLDER_BINS * 1e-3

        if total > PLACEHOLDER_MIN_SIGNAL:
            rr, cc = np.indices(image.shape)
            row0 = (weights * rr).sum() / total
            col0 = (weights * cc).sum() / total
            sx = np.sqrt((weights * (rr - row0)**2).sum() / total) * pixel_m
            sy = np.sqrt((weights * (cc - col0)**2).sum() / total) * pixel_m
            valid = True
        else:
            sx = sy = 0.0
            valid = False

        measurement = Measurement(
            np.array([sx, sy]),
            np.diag([(0.01 * sx)**2, (0.01 * sy)**2]),   # fake 1% fit error
            float(np.var(dark, ddof=1)), dark.size - 1,
            valid, 1.0, raw.t, raw.currents.copy(),
        )
        dprint(f"[placeholder analyze] frame t={raw.t:.3f}s  total signal={total:,.0f}"
               f"  valid={valid}  sigma_xy={fmt(np.array([sx, sy]) * 1e3)} mm", 2)
        return measurement, {"image": image}


class PlaceholderEstimator:
    """PLACEHOLDER estimator: just records measurements and prints a running
    mean of the valid widths. Returns None, like the real update()."""

    def __init__(self):
        self.history = []
        self.update_calls = 0

    def update(self, measurements):
        self.update_calls += 1
        self.history.extend(measurements)
        valid_now = [m for m in measurements if m.valid]
        valid_all = [m for m in self.history if m.valid]
        dprint(f"[placeholder estimator.update] call #{self.update_calls}: received "
               f"{len(measurements)} measurement(s), {len(valid_now)} valid; "
               f"history now {len(self.history)} ({len(valid_all)} valid)", 2)
        if valid_all:
            mean_sigma = np.mean([m.sigma_xy for m in valid_all], axis=0) * 1e3
            dprint(f"[placeholder estimator.update] running mean sigma_xy = "
                   f"{fmt(mean_sigma)} mm", 2)


def make_charge_latency(clock, mode="fixed", fixed_s=None):
    """PLACEHOLDER for env._charge_latency(kind, func).

    mode="fixed": charge a constant time per kind (reproducible; mimics the
                  'replay a measured latency profile' idea in the plan).
    mode="live" : charge real wall-clock time, like the env's current version.
    """
    fixed_s = fixed_s or {"analysis": 0.30, "estimator": 0.20}

    def charge_latency(kind, func):
        dprint(f"[latency] running '{kind}' (mode={mode})", 2)
        start = measure_real_time()
        result = func()
        real = measure_real_time() - start
        charged = real if mode == "live" else fixed_s.get(kind, 0.0)
        dprint(f"[latency] '{kind}' took {real*1e3:.1f} ms real; charging "
               f"{charged:.3f}s to the master clock", 2)
        clock.advance(charged)
        return result

    return charge_latency


class MockMachine:
    """PLACEHOLDER for the env object that execute_measurement() treats as
    `machine`. Holds the six attributes the function touches."""

    def __init__(self, initial_currents=None, screen_inserted=False,
                 start_time=0.0, seed=12345, beam_offset_mm=0.0):
        self.clock = DebugClock(t=start_time)
        self.line = None                        # real: FELSim line, elements 0..21
        self.beam = initial_beam(seed)          # fixed beam, as the plan requires
        if beam_offset_mm:
            self.beam[:, 0] += beam_offset_mm   # push beam off the sensor (test aid)
        self.errors = benchmark_quad_errors()   # hidden truth
        self.currents = (INITIAL_CURRENTS.copy() if initial_currents is None
                         else np.array(initial_currents, dtype=float))
        self.screen_inserted = screen_inserted

    def describe(self):
        return (f"t={self.clock.t:.3f}s  screen_inserted={self.screen_inserted}  "
                f"currents={fmt(self.currents)}")


# ============================================================================
# 3. execute_measurement  -- ORIGINAL LOGIC, DEBUG PRINTS ADDED
#    (only `dprint(...)` lines and `else:` branches containing only dprint
#     were added; no functional line was changed)
# ============================================================================
def execute_measurement(machine, diagnostic, analysis, estimator, currents,
                        n_pulses, quad_rng, camera_rng, config, charge_latency):
    """Scheduler integration example. machine owns line, beam, errors, clock,
    currents, screen_inserted. n_pulses counts complete machine pulses, each
    containing BUNCHES_PER_MACROPULSE bunches. One camera readout per pulse.
    charge_latency(kind, callable) executes and
    charges measured or replayed online latency; it is not simulation timing.
    Check remaining budget INCLUDING processing/retraction before calling.
    """
    dprint("=" * 74)
    dprint(f"execute_measurement(n_pulses={n_pulses})")
    dprint(f"requested currents : {fmt(currents)}", 1)
    dprint(f"machine on entry   : {machine.describe()}", 1)

    # ---- STEP 1: change quad setpoints (costs settling time) ---------------
    if not np.array_equal(currents, machine.currents):
        dprint(f"STEP 1: requested != current setpoints -> apply them and wait "
               f"QUAD_SETTLE_S={QUAD_SETTLE_S}s", 1)
        machine.currents = np.array(currents, copy=True)
        machine.clock.advance(QUAD_SETTLE_S)
    else:
        dprint("STEP 1: currents unchanged -> NO settling time charged", 1)

    # ---- STEP 2: make sure the OTR screen is in the beam -------------------
    if not machine.screen_inserted:
        dprint(f"STEP 2: screen is OUT -> insert it, wait OTR_INSERT_S="
               f"{OTR_INSERT_S}s", 1)
        machine.clock.advance(OTR_INSERT_S)
        machine.screen_inserted = True
    else:
        dprint("STEP 2: screen already IN -> NO insertion time charged", 1)

    # ---- STEP 3: acquire n_pulses machine pulses ---------------------------
    frames = []
    dprint(f"STEP 3: acquire {n_pulses} pulse(s). clock t={machine.clock.t:.3f}s; "
           f"pulses fall on whole machine-period slots, first one is slot "
           f"{math.ceil(machine.clock.t * machine.clock.rep_rate_hz) + 1}", 1)
    for t in machine.clock.pulse_times(n_pulses):
        dprint(f"pulse at t={t:.3f}s: clock is {machine.clock.t:.3f}s, so wait "
               f"{t - machine.clock.t:.3f}s for the slot", 2)
        machine.clock.advance(t - machine.clock.t)
        beam = track_one_pulse(machine.line, machine.beam, machine.currents,
                               machine.errors, t, quad_rng)
        frames.append(diagnostic.acquire(beam, t=t, currents=machine.currents,
                                         rng=camera_rng))
        dprint(f"-> frame {len(frames)}/{n_pulses} stored (t={frames[-1].t:.3f}s)", 2)

    # ---- STEP 4: analyse all frames (latency charged AFTER acquisition) ----
    # Analysis happens after acquisition; no artificial gaps between its pulses.
    dprint(f"STEP 4: analyse {len(frames)} frame(s); clock t={machine.clock.t:.3f}s", 1)
    results = charge_latency("analysis", lambda: [analysis.analyze(f, config)
                                                  for f in frames])
    measurements = [m for m, debug in results]
    dprint(f"-> {len(measurements)} measurement(s), "
           f"{sum(m.valid for m in measurements)} valid", 2)

    # ---- STEP 5: feed the estimator (latency charged) ----------------------
    dprint(f"STEP 5: estimator update; clock t={machine.clock.t:.3f}s", 1)
    charge_latency("estimator", lambda: estimator.update(measurements))

    dprint(f"DONE. machine on exit: {machine.describe()}", 1)
    dprint("=" * 74)
    return frames, measurements


# ============================================================================
# 4. TEST HARNESS
# ============================================================================
def with_current(index, value, base=None):
    """Copy of the initial currents (or `base`) with one quad changed."""
    out = (INITIAL_CURRENTS if base is None else np.asarray(base)).astype(float).copy()
    out[index] = value
    return out


def run_scenario(name, steps, description="", initial_currents=None,
                 screen_inserted=False, start_time=0.0, latency_mode="fixed",
                 beam_offset_mm=0.0, seed=12345, n_particles_note=None):
    """Build a mock machine, then call execute_measurement once per step.

    steps : list of (currents, n_pulses). currents=None means "leave unchanged".
            All steps share ONE machine, clock and estimator, like a real episode.
    """
    print("\n" + "#" * 74)
    print(f"# SCENARIO: {name}")
    if description:
        print(f"# {description}")
    print("#" * 74)

    machine = MockMachine(initial_currents, screen_inserted, start_time,
                          seed, beam_offset_mm)
    diagnostic = PlaceholderOTRScreen()
    analysis = PlaceholderGaussianAnalysis()
    estimator = PlaceholderEstimator()
    config = GaussianConfig()
    quad_rng = np.random.default_rng(seed + 1)       # independent random streams
    camera_rng = np.random.default_rng(seed + 2)
    charge_latency = make_charge_latency(machine.clock, latency_mode)

    rows = []
    for i, (currents, n_pulses) in enumerate(steps, start=1):
        if currents is None:
            currents = machine.currents.copy()
        t_start = machine.clock.t
        try:
            frames, measurements = execute_measurement(
                machine, diagnostic, analysis, estimator, currents, n_pulses,
                quad_rng, camera_rng, config, charge_latency)
        except Exception as exc:                      # show failures, don't crash
            print(f"\n!! call {i} raised {type(exc).__name__}: {exc}")
            print(f"!! machine state AFTER the failure: {machine.describe()}")
            rows.append((i, t_start, machine.clock.t, None, None, f"ERROR: {exc}"))
            break
        valid = [m for m in measurements if m.valid]
        sig = (np.mean([m.sigma_xy for m in valid], axis=0) * 1e3
               if valid else None)
        rows.append((i, t_start, machine.clock.t, len(frames), len(valid), sig))

    print("\nSUMMARY  --  " + name)
    for i, t0, t1, nf, nv, sig in rows:
        if nf is None:
            print(f"  call {i}: t {t0:7.3f} -> {t1:7.3f} s   {sig}")
        else:
            sig_txt = (f"mean sigma_xy = ({sig[0]:.3f}, {sig[1]:.3f}) mm"
                       if sig is not None else "no valid measurement")
            print(f"  call {i}: t {t0:7.3f} -> {t1:7.3f} s  (elapsed {t1-t0:6.3f}s)"
                  f"  frames={nf}  valid={nv}/{nf}  {sig_txt}")
    print(f"  total machine time used: {machine.clock.t - start_time:.3f}s | "
          f"estimator saw {len(estimator.history)} measurements in "
          f"{estimator.update_calls} update(s)")
    return machine, estimator


# ============================================================================
# 5. SCENARIOS  --  edit these / add your own
# ============================================================================
I0 = INITIAL_CURRENTS

# ---- EDIT ME: your own experiment -------------------------------------------
CUSTOM = dict(
    description="Edit CUSTOM in the script to try your own inputs.",
    initial_currents=None,        # None -> INITIAL_CURRENTS; or a 6-element list
    screen_inserted=False,        # is the OTR already in the beam at the start?
    start_time=0.0,               # master-clock time at the start (seconds)
    beam_offset_mm=0.0,           # shift beam in x; ~10 pushes it off the sensor
    steps=[                       # (currents or None, n_pulses) per call
        (with_current(0, 1.0), 3),
        (None, 2),
    ],
)

SCENARIOS = {
    "baseline": dict(
        description="Screen OUT, currents unchanged, 1 pulse. Expect: insertion + "
                    "wait for slot, NO settling.",
        steps=[(None, 1)]),
    "change_currents": dict(
        description="Quad 0 +0.2 A. Expect: settle 2s + insertion 1s + pulse.",
        steps=[(with_current(0, I0[0] + 0.2), 1)]),
    "screen_already_in": dict(
        description="Screen already IN, currents unchanged, 5 pulses. Expect: "
                    "no settle, no insertion, 5 consecutive 1 Hz slots.",
        screen_inserted=True, steps=[(None, 5)]),
    "scan": dict(
        description="Four-point quad-0 scan, 2 pulses each, one shared clock. "
                    "Expect: screen inserted once; settle charged each change.",
        steps=[(with_current(0, v), 2) for v in (0.64, 0.84, 1.04, 1.24)]),
    "exact_tick": dict(
        description="Clock starts exactly on a pulse tick (t=1.0). The first "
                    "pulse still lands at t=2.0, never at the current tick.",
        start_time=1.0, screen_inserted=True, steps=[(None, 1)]),
    "mid_period": dict(
        description="Clock starts mid-period (t=0.4). Watch the wait to the slot.",
        start_time=0.4, screen_inserted=True, steps=[(None, 1)]),
    "off_screen": dict(
        description="Beam shifted 10 mm off the sensor -> placeholder analysis "
                    "flags it invalid; estimator gets 0 valid measurements.",
        beam_offset_mm=10.0, steps=[(None, 2)]),
    "bad_pulses": dict(
        description="n_pulses=0 -> ValueError from Clock.pulse_times. Note the "
                    "machine state left behind (settle/insert already charged).",
        steps=[(with_current(0, I0[0] + 0.1), 0)]),
    "negative_k": dict(
        description="Quads 1,3,5 commanded to 0 A -> hidden offsets push k "
                    "negative -> ValueError from QuadErrors.sample.",
        steps=[(with_current(5, 0.0, with_current(3, 0.0, with_current(1, 0.0))), 1)]),
    "custom": CUSTOM,
}


def main():
    global VERBOSE
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--scenario", default="baseline",
                        help="name from SCENARIOS, or 'all'")
    parser.add_argument("--list", action="store_true", help="list scenarios")
    parser.add_argument("--quiet", action="store_true",
                        help="hide debug prints, show summaries only")
    parser.add_argument("--latency", choices=["fixed", "live"], default="fixed",
                        help="fixed = replayed constants; live = wall-clock")
    args = parser.parse_args()

    if args.list:
        for name, cfg in SCENARIOS.items():
            print(f"{name:18s} {cfg.get('description', '')}")
        return

    VERBOSE = not args.quiet
    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    for name in names:
        if name not in SCENARIOS:
            raise SystemExit(f"Unknown scenario '{name}'. Use --list.")
        if args.scenario == "all" and name == "custom":
            continue
        run_scenario(name, latency_mode=args.latency, **SCENARIOS[name])


if __name__ == "__main__":
    main()