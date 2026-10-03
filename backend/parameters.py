"""UH reference beam parameters loaded from parameters/UH_reference_parameters.yaml."""
from pathlib import Path

import yaml

REFERENCE_PATH = Path(__file__).resolve().parent.parent / "parameters" / "UH_reference_parameters.yaml"

_TO_SI = {"MeV": 1e6, "pC": 1e-12, "um": 1e-6, "ps": 1e-12, "percent": 1e-2, "m": 1.0}


def load_reference_parameters(path=None) -> dict:
    """Return the full UH reference parameter set (meta, madey_beam, inputs, machine)."""
    with open(path or REFERENCE_PATH) as f:
        return yaml.safe_load(f)


def _si(value, unit):
    if unit not in _TO_SI:
        raise ValueError(f"Unknown unit in reference parameters: {unit!r}")
    return value * _TO_SI[unit]


def chicane_inputs(path=None) -> dict:
    """Start values at the diagnostic-chicane entrance in SI units (energy in eV, spread as a fraction)."""
    p = load_reference_parameters(path)["inputs"]["chicane_entrance"]
    emit, twiss = p["normalized_emittance"], p["twiss"]
    return {
        "kinetic_energy_eV": _si(p["kinetic_energy"]["value"], p["kinetic_energy"]["unit"]),
        "bunch_charge_C": _si(p["bunch_charge"]["value"], p["bunch_charge"]["unit"]),
        "emittance_n_x_m": _si(emit["x"], emit["unit"]),
        "emittance_n_y_m": _si(emit["y"], emit["unit"]),
        "sigma_t_s": _si(p["bunch_length_rms"]["value"], p["bunch_length_rms"]["unit"]),
        "energy_spread_rms": _si(p["energy_spread_rms"]["value"], p["energy_spread_rms"]["unit"]),
        "beta_x_m": _si(twiss["beta_x"], twiss["unit_beta"]),
        "beta_y_m": _si(twiss["beta_y"], twiss["unit_beta"]),
        "alpha_x": twiss["alpha_x"],
        "alpha_y": twiss["alpha_y"],
    }
