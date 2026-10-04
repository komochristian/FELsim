"""Reproducible optimization fixtures for the UH beamline.

`historical_40mev` rebuilds the 23-current, 12-stage Nelder-Mead matching of
backend/test/UHM_beamline_opt.py with the start beam it used before 3835895.
`cosy_fit_40mev` runs the 11-stage COSY FIT path of backend/test/UHM_beamline_opt_cosy.py
on the same start beam; `cross_objectives` judges any current set on both objectives.
"""
import copy
import re
import time

import numpy as np

from felsim import paths
from felsim.beamline import lattice, qpdLattice, qpfLattice
from felsim.beamOptimizer import beamOptimizer
from felsim.ebeam import beam
from felsim.excelElements import ExcelElements
from felsim.parameters import load_reference_parameters

# Legacy start beam of UHM_beamline_opt.py (not the reference master, listed in its repo_deviations_to_fix)
HISTORICAL_40MEV = {
    "kinetic_energy_MeV": 40.0,
    "emittance_n_mm_mrad": 8.0,
    "sigma_x_mm": 0.8,
    "sigma_y_mm": 0.8,
    "sigma_t_ps": 2.0,
    "energy_spread_percent": 0.5,
    "chirp_per_s": 5e9,
    "n_particles": 1000,
    "n_elements": 118,
}

# COSY FIT path of UHM_beamline_opt_cosy.py (DA order, FIT limits, algorithm 1 = gradient on DA)
COSY_FIT_40MEV = {"order": 3, "fringe_field_order": 0, "nmax": 1000, "eps": 1e-8, "nalgorithm": 1}
COSY_STAGE5_START = {37: 0.28, 35: 2.65, 33: 2.69}  # A, the S1 multistart seed of the COSY script
COSY_UNDULATOR_X = {"alpha": 0.47, "beta_m": 1.4}  # horizontal undulator-entrance target of the COSY script

# Downstream quads set equal to the double-triplet currents after stage 5 (mirror symmetry about index 38)
MIRROR = {43: 33, 41: 35, 39: 37}


def undulator_beta_m(gamma):
    """Natural vertical focusing beta gamma / k_beta of the MkV planar undulator, k_beta = K k_u."""
    und = load_reference_parameters()["machine"]["undulator"]["primary_mkv"]
    lambda_u = und["period"]["value"] * 1e-2  # cm -> m
    return gamma * lambda_u / (2 * np.pi * und["K"])


def historical_40mev_beam(seed=0, p=HISTORICAL_40MEV):
    """6D Gaussian start beam in FELsim units (mm, mrad, 1e-3 RF period, 1e-3 dW/W), seeded."""
    relat = lattice(1, fringeType=None)
    relat.setE(E=p["kinetic_energy_MeV"])
    epsilon = p["emittance_n_mm_mrad"] / (relat.gamma * relat.beta)  # geometric, mm mrad
    f_rf = load_reference_parameters()["machine"]["rf_frequency"]["value"] * 1e9  # GHz -> Hz
    std = [p["sigma_x_mm"], epsilon / p["sigma_x_mm"],
           p["sigma_y_mm"], epsilon / p["sigma_y_mm"],
           p["sigma_t_ps"] * 1e-9 * f_rf,
           p["energy_spread_percent"] * 10]
    np.random.seed(seed)
    particles = beam().gen_6d_gaussian(0, std, p["n_particles"])
    particles[:, 5] += p["chirp_per_s"] * particles[:, 4] / f_rf  # chirp exactly as in the historical script
    return particles, relat


def historical_40mev_stages(beta_ym):
    """The 12 stages of UHM_beamline_opt.py: {name, variables, startPoint, objectives}."""
    def nm(starts):
        return {v: {"bounds": (0, 10), "start": s} for v, s in starts.items()}

    def disp(i):
        return {i: [{"measure": ["x", "dispersion"], "goal": 0, "weight": 1}]}

    def yw(i):
        return {i: [{"measure": ["y", "alpha"], "goal": 0, "weight": 1},
                    {"measure": ["y", "beta"], "goal": 0.1, "weight": 0.5}]}

    def xw(i, w=0.5):
        return {i: [{"measure": ["x", "alpha"], "goal": 0, "weight": 1},
                    {"measure": ["x", "beta"], "goal": 0.1, "weight": w}]}

    def var(idx_names):
        return {i: [n, "current", lambda num: num] for i, n in idx_names}

    return [
        {"name": "First quadrupole doublet", "variables": var([(1, "I"), (3, "I2")]),
         "startPoint": nm({"I": 1, "I2": 1}), "objectives": {**xw(8, 0.0), **yw(9)}},
        {"name": "First chromaticity quad", "variables": var([(10, "I")]),
         "startPoint": nm({"I": 1}), "objectives": disp(15)},
        {"name": "Quadrupole triplet 1", "variables": var([(16, "I"), (18, "I2"), (20, "I3")]),
         "startPoint": nm({"I": 2, "I2": 5, "I3": 3}), "objectives": {**xw(25), **yw(26)}},
        {"name": "Second chromaticity quad", "variables": var([(27, "I")]),
         "startPoint": nm({"I": 1}), "objectives": disp(32)},
        {"name": "Double quadrupole triplet", "variables": var([(37, "I"), (35, "I2"), (33, "I3")]),
         "startPoint": nm({"I": 2, "I2": 2, "I3": 2}),
         "objectives": {37: [{"measure": ["x", "alpha"], "goal": 0, "weight": 1},
                             {"measure": ["y", "alpha"], "goal": 0, "weight": 1},
                             {"measure": ["x", "envelope"], "goal": 2.0, "weight": 1},
                             {"measure": ["y", "envelope"], "goal": 2.0, "weight": 1}]}},
        {"name": "Third chromaticity quad", "variables": var([(50, "I")]),
         "startPoint": nm({"I": 1}), "objectives": disp(55)},
        {"name": "IP doublet", "variables": var([(56, "I"), (58, "I2")]),
         "startPoint": nm({"I": 2, "I2": 2}),
         "objectives": {59: [{"measure": ["x", "envelope"], "goal": 0.0, "weight": 1},
                             {"measure": ["y", "envelope"], "goal": 0.0, "weight": 1}]}},
        {"name": "Post-IP doublet", "variables": var([(61, "I"), (63, "I2")]),
         "startPoint": nm({"I": 2, "I2": 2}), "objectives": {**xw(68), **yw(69)}},
        {"name": "Fourth chromaticity quad", "variables": var([(70, "I")]),
         "startPoint": nm({"I": 1}), "objectives": disp(75)},
        {"name": "Quadrupole triplet 2", "variables": var([(76, "I"), (78, "I2"), (80, "I3")]),
         "startPoint": nm({"I": 2, "I2": 2, "I3": 2}), "objectives": {**xw(85), **yw(86)}},
        {"name": "Fifth chromaticity quad", "variables": var([(87, "I")]),
         "startPoint": nm({"I": 1}), "objectives": disp(92)},
        {"name": "Final triplet to undulator", "variables": var([(93, "I"), (95, "I2"), (97, "I3")]),
         "startPoint": nm({"I": 2, "I2": 2, "I3": 2}),
         "objectives": {117: [{"measure": ["x", "alpha"], "goal": 0, "weight": 1},
                              {"measure": ["y", "alpha"], "goal": 0, "weight": 1},
                              {"measure": ["x", "beta"], "goal": beta_ym, "weight": 1},
                              {"measure": ["y", "beta"], "goal": beta_ym, "weight": 1}]}},
    ]


def historical_40mev(seed=0, p=HISTORICAL_40MEV):
    """Beamline (118 elements from beam_excel/), seeded start beam and the 12 stages of the historical run."""
    particles, relat = historical_40mev_beam(seed, p)
    elements = ExcelElements(paths.EXCEL_DIR / "Beamline_elements.xlsx").create_beamline()
    line = relat.changeBeamType("electron", p["kinetic_energy_MeV"], elements)[:p["n_elements"]]
    beta_ym = undulator_beta_m(relat.gamma)
    return {"line": line, "particles": particles, "stages": historical_40mev_stages(beta_ym),
            "beta_ym_m": beta_ym, "seed": seed, "params": dict(p)}


def stage_loss(stage, moments):
    """Weighted mean squared residual of beamOptimizer._optiSpeed for given moments {(i, plane, measure): value}."""
    terms = [g["weight"] * (moments[(i, g["measure"][0], measure_name(g))] - g["goal"]) ** 2
             for i, goals in stage["objectives"].items() for g in goals]
    return float(np.sum(terms) / len(terms))


def measure_name(goal):
    """Measure name of an objective goal, before or after beamOptimizer.calc replaced it by a function."""
    m = goal["measure"][1]
    return m if isinstance(m, str) else {"disper": "dispersion"}.get(m.__name__, m.__name__)


def target_moments(line, particles, stages):
    """Track through `line` and return every stage target {(i, plane, measure): value}."""
    eb = beam()
    fn = {"alpha": eb.alpha, "beta": eb.beta, "envelope": eb.envelope, "dispersion": eb.disper}
    wanted = {}
    for st in stages:
        for i, goals in st["objectives"].items():
            wanted.setdefault(i, set()).update((g["measure"][0], measure_name(g)) for g in goals)
    out = {}
    for i, el in enumerate(line):
        particles = el.useMatrice(particles)
        for plane, m in wanted.get(i, ()):
            out[(i, plane, m)] = float(fn[m](particles, plane))
    return out


def run_stages(fix, method="Nelder-Mead", stages=None, clock=time.perf_counter):
    """Run the stages in order with beamOptimizer.calc, as in UHM_beamline_opt.py; return per-stage records."""
    line, particles = fix["line"], fix["particles"]
    opti = beamOptimizer(line, particles)
    records = []
    for k, st in enumerate(fix["stages"] if stages is None else stages):
        t0 = clock()
        res = opti.calc(method, st["variables"], st["startPoint"], st["objectives"])
        if st["name"] == "Double quadrupole triplet":
            for dst, src in MIRROR.items():
                line[dst].current = line[src].current
        records.append({"stage": k + 1, "name": st["name"], "wall_s": clock() - t0,
                        "fun": float(res.fun), "nfev": int(res.nfev), "nit": int(res.nit),
                        "success": bool(res.success), "message": str(res.message),
                        "currents_A": {int(i): float(line[i].current) for i in st["variables"]}})
    return records


def cosy_40mev_stages(beta_ym):
    """The 11 COSY FIT stages: FELsim stages 1-10 (stage 5 started at the S1 seed), then 87 joined to the final triplet."""
    stages = copy.deepcopy(historical_40mev_stages(beta_ym)[:10])
    names = stages[4]["variables"]
    for i, v in COSY_STAGE5_START.items():
        stages[4]["startPoint"][names[i][0]]["start"] = v
    final = {"name": "Chromaticity 5 + final triplet",
             "variables": {87: ["Ic", "current", lambda num: num], 93: ["I", "current", lambda num: num],
                           95: ["I2", "current", lambda num: num], 97: ["I3", "current", lambda num: num]},
             "startPoint": {"Ic": {"bounds": (0, 10), "start": 4}, "I": {"bounds": (0, 10), "start": 2},
                            "I2": {"bounds": (0, 10), "start": 2}, "I3": {"bounds": (0, 10), "start": 2}},
             "objectives": {92: [{"measure": ["x", "dispersion"], "goal": 0, "weight": 0.5}],
                            117: [{"measure": ["x", "alpha"], "goal": COSY_UNDULATOR_X["alpha"], "weight": 1},
                                  {"measure": ["y", "alpha"], "goal": 0, "weight": 1},
                                  {"measure": ["x", "beta"], "goal": COSY_UNDULATOR_X["beta_m"], "weight": 1},
                                  {"measure": ["y", "beta"], "goal": beta_ym, "weight": 1}]}}
    return stages + [final]


def to_cosy_stage(stage, start=None):
    """Convert a fixture stage to the cosyOptHelper format; `start` {index: A} overrides the start currents."""
    names = {i: v[0] for i, v in stage["variables"].items()}
    sp = {n: {"start": stage["startPoint"][n]["start"], "bounds": tuple(stage["startPoint"][n]["bounds"])}
          for n in names.values()}
    for i, n in names.items():
        if start is not None and i in start:
            sp[n]["start"] = float(start[i])
    out = {"name": stage["name"], "variables": names, "start_point": sp,
           "objectives": {i: [{"measure": [g["measure"][0], measure_name(g)], "goal": g["goal"], "weight": g["weight"]}
                              for g in goals] for i, goals in stage["objectives"].items()}}
    if stage["name"] == "Double quadrupole triplet":
        out["mirror"] = dict(MIRROR)
    return out


def cosy_start_twiss(p=HISTORICAL_40MEV):
    """Geometric emittance (mm mrad) and waist beta0 = sigma^2 / eps (m) of the historical beam, as in the COSY script."""
    relat = lattice(1, fringeType=None)
    relat.setE(E=p["kinetic_energy_MeV"])
    eps = p["emittance_n_mm_mrad"] / (relat.gamma * relat.beta)
    return eps, p["sigma_x_mm"] ** 2 / eps, relat


POST_FIT_ELEM = re.compile(r"POST-FIT elem (\d+): ax=\s*(\S+) ay=\s*(\S+) bx=\s*(\S+) by=\s*(\S+) Dx=\s*(\S+)")
POST_FIT_VAR = re.compile(r"POST-FIT S(\d+)_(\w+)=\s*(\S+)")


def parse_post_fit(log):
    """Twiss {MAP_ARR index: (ax, ay, bx, by, Dx)} and fitted values {(stage, name): A} printed after each FIT block."""
    twiss = {int(m[0]): tuple(float(v) for v in m[1:]) for m in POST_FIT_ELEM.findall(log)}
    values = {(int(k), n): float(v) for k, n, v in POST_FIT_VAR.findall(log)}
    return twiss, values


def cosy_moments(twiss, index_map, eps_mm_mrad):
    """Stage-target moments {(i, plane, measure): value} from COSY map Twiss at FELsim indices (beta m, D m, envelope mm)."""
    out = {}
    for i, c in index_map.items():
        if c + 1 not in twiss:
            continue
        ax, ay, bx, by, dx = twiss[c + 1]
        for plane, a, b in (("x", ax, bx), ("y", ay, by)):
            out[(i, plane, "alpha")] = a
            out[(i, plane, "beta")] = b
            out[(i, plane, "envelope")] = 1e3 * np.sqrt(eps_mm_mrad * 1e-6 * b)
        out[(i, "x", "dispersion")] = dx
    return out


def run_cosy(stages, p=HISTORICAL_40MEV, settings=COSY_FIT_40MEV, start=None, output_dir=None, clock=time.perf_counter):
    """One COSY run of the given stages as sequential FIT blocks; returns currents, map moments, log and wall time."""
    from felsim.cosyAdapter import COSYAdapter
    from felsim.cosyOptHelper import add_stages, parse_beamline_felsim_indexed

    excel = str(paths.EXCEL_DIR / "Beamline_elements.xlsx")
    eps, beta0, _ = cosy_start_twiss(p)
    adapter = COSYAdapter(lattice_path=excel, mode="transfer_matrix",
                          config={"simulation": {"KE": p["kinetic_energy_MeV"], "order": settings["order"],
                                                 "dimensions": 3}},
                          fringe_field_order=settings["fringe_field_order"],
                          output_dir=str(output_dir or paths.default_output_dir("cosy")))
    sim = adapter.get_native_simulator()
    sim.beamline = parse_beamline_felsim_indexed(excel)[:p["n_elements"]]
    sim.set_geometric_emittance(eps)
    sim.set_initial_twiss(beta_x=beta0, alpha_x=0.0, beta_y=beta0, alpha_y=0.0)
    sim.fit_nmax, sim.fit_eps, sim.fit_nalgorithm = settings["nmax"], settings["eps"], settings["nalgorithm"]
    sim.fit_combined_mse = True
    cstages = [to_cosy_stage(st, start) for st in stages]
    index_map = add_stages(sim, cstages)["index_map"]
    t0 = clock()
    res = sim.run_simulation()
    wall = clock() - t0
    twiss, values = parse_post_fit(res["log"])
    currents = {}
    for k, st in enumerate(cstages, 1):
        for i, n in st["variables"].items():
            currents[i] = values[(k, n)]
        for dst, src in st.get("mirror", {}).items():
            currents[dst] = currents[src]
    return {"currents_A": currents, "moments": cosy_moments(twiss, index_map, eps), "wall_s": wall,
            "log": res["log"], "index_map": index_map}


def cosy_fit_40mev(p=HISTORICAL_40MEV, settings=COSY_FIT_40MEV, output_dir=None, stage_timing=True):
    """11-stage COSY FIT on the historical beam: currents, per-stage COSY objective and timings.

    COSY prints no per-block clock, so with `stage_timing` the prefixes 1..k are rerun and
    stage k is timed as wall(1..k) - wall(1..k-1); stage 1 includes the COSY start-up.
    """
    eps, beta0, relat = cosy_start_twiss(p)
    stages = cosy_40mev_stages(undulator_beta_m(relat.gamma))
    full = run_cosy(stages, p, settings, output_dir=output_dir)
    walls = [run_cosy(stages[:k], p, settings, output_dir=output_dir)["wall_s"] for k in range(1, len(stages))] \
        if stage_timing else []
    walls.append(full["wall_s"])
    per_stage = np.diff([0.0] + walls).tolist() if stage_timing else None
    return {"currents_A": full["currents_A"], "stages": stages, "eps_mm_mrad": eps, "beta0_m": beta0,
            "stage_loss": [stage_loss(st, full["moments"]) for st in stages], "moments": full["moments"],
            "timing_s": {"total": full["wall_s"], "cumulative_prefix": walls if stage_timing else None,
                         "per_stage": per_stage},
            "log": full["log"]}


def set_quad_current(line, i, amps):
    """Set a quadrupole current; a negative current reverses the polarity (QPF <-> QPD at |I|), as in COSY's MQ."""
    if amps < 0:
        el = copy.copy(line[i])
        el.__class__ = qpdLattice if isinstance(el, qpfLattice) else qpfLattice
        line[i] = el
    line[i].current = abs(float(amps))


def felsim_moments(currents, stages, seed=0, p=HISTORICAL_40MEV):
    """FELsim particle-tracking moments of the seeded historical beam at the given currents {index: A}."""
    fix = historical_40mev(seed, p)
    for i, v in currents.items():
        set_quad_current(fix["line"], int(i), v)
    return target_moments(fix["line"], fix["particles"], stages)


def cross_objectives(currents, seed=0, p=HISTORICAL_40MEV, settings=COSY_FIT_40MEV, output_dir=None, cosy=True):
    """Per-stage FELsim 12-stage and COSY 11-stage objectives at `currents`, by FELsim tracking and by COSY maps.

    The COSY evaluation reruns the FIT blocks with Nmax = 0, started at `currents`.
    """
    _, _, relat = cosy_start_twiss(p)
    beta_ym = undulator_beta_m(relat.gamma)
    fel, cos = historical_40mev_stages(beta_ym), cosy_40mev_stages(beta_ym)
    m = felsim_moments(currents, fel + cos, seed, p)
    out = {"felsim_tracking": {"felsim_12": [stage_loss(st, m) for st in fel],
                               "cosy_11": [stage_loss(st, m) for st in cos]}}
    if cosy:
        r = run_cosy(cos, p, {**settings, "nmax": 0}, start=currents, output_dir=output_dir)
        moved = max(abs(r["currents_A"][i] - float(currents[i])) for i in r["currents_A"])
        out["cosy_map"] = {"felsim_12": [stage_loss(st, r["moments"]) for st in fel],
                           "cosy_11": [stage_loss(st, r["moments"]) for st in cos], "max_current_move_A": moved}
    return out
