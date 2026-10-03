"""Reproducible optimization fixtures for the UH beamline.

`historical_40mev` rebuilds the 23-current, 12-stage Nelder-Mead matching of
backend/test/UHM_beamline_opt.py with the start beam it used before 3835895.
"""
import time

import numpy as np

from felsim import paths
from felsim.beamline import lattice
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
