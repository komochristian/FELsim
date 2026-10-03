"""
Electron-beam energy deposition: stopping power, range, dose and heating.

Two study cases are supported and kept deliberately separate because the
physics assumptions differ:

A. Beam-pipe / beam-dump study  ->  :func:`beam_pipe_heating`
   A (usually defocused) beam strikes a metal surface, stainless steel by
   default.  The wall may be thinner than the electron range, so the code
   transports the electrons through the wall along the CSDA path and splits
   the energy into: deposited by collisions (heats the wall), radiated as
   bremsstrahlung (mostly escapes a thin wall), and transmitted.  Outputs are
   the absorbed power, the peak surface energy density per macropulse and the
   adiabatic temperature rise.

B. FLASH radiotherapy study  ->  :func:`flash_dose`
   A focused or unfocused beam enters a water or skin target that is much
   thinner than the range.  The dose is the mass collision stopping power
   times the fluence:  D[Gy] = 1.602e-10 * (S_col/rho)[MeV cm2/g] * Phi[cm^-2].
   Outputs are dose per macropulse and per bunch, instantaneous and mean dose
   rates, and the number of pulses / time to reach a prescription.

Stopping powers, CSDA ranges and radiation yields are interpolated (log-log)
from NIST ESTAR tables stored in ``felsim/data/estar/``.  Stainless steel is
built from the Fe, Cr and Ni tables by Bragg additivity.  An analytic
Rohrlich-Carlson (Bethe) collision stopping power with the Sternheimer
density-effect correction is provided as an independent cross-check only.

Conventions
-----------
* Energies in MeV, stopping powers in MeV cm2/g (mass) or MeV/cm (linear),
  ranges in g/cm2 or cm, densities in g/cm3.
* Beam parameters are SI on input (A, s, Hz, m).  Fluence is per cm2.
* ``sigma_x``/``sigma_y`` are RMS sizes for a Gaussian profile.  The peak
  fluence of a Gaussian beam is N / (2 pi sigma_x sigma_y).  For a
  ``"uniform"`` profile they are the ellipse semi-axes.

Limits of the model (read before quoting numbers)
-------------------------------------------------
* CSDA (continuous slowing down) transport: no multiple scattering, no range
  straggling, no lateral spreading of the beam inside the target.  Valid when
  the target is thin compared with the range (the FLASH case) and adequate for
  energy bookkeeping in a thick dump.  Do not use the depth-dose curve near
  the end of the range: the real fall-off is gradual (scattering shortens the
  penetration, range straggling from rare large bremsstrahlung losses lengthens
  it), and in water above ~10 MeV the practical range actually exceeds the
  CSDA range (about 1.05x at 20 MeV and 1.2x at 40 MeV from the ICRU-35
  range-energy relation); in heavier materials at a few MeV it is shorter.
* Backscatter is neglected (a few percent of the energy for steel near normal
  incidence).  At grazing incidence (more than ~75 deg from the wall normal)
  multiple scattering turns a large fraction of the electrons back out of the
  wall within the first millimetres, so the straight-line path length and the
  "stopped in wall" result of :func:`beam_pipe_heating` are upper bounds on
  the deposited energy; use a Monte Carlo code for those angles.  The surface
  energy density per pulse (Phi_peak * S_col) does not depend on the angle.
* Bremsstrahlung re-absorption in the target is a user parameter
  (``brems_absorbed_fraction``): 0 for a thin wall, up to ~0.5 for a thick
  high-Z dump.
* Heating is adiabatic per macropulse.  Thermal conduction between pulses is
  not modelled; the "mean heating rate" is an upper bound.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from felsim import paths

# ---------------------------------------------------------------------------
# Constants (CODATA 2018)
# ---------------------------------------------------------------------------
E_CHARGE = 1.602176634e-19       # C
MEV_TO_J = 1.602176634e-13       # J
GY_PER_MEV_PER_G = 1.602176634e-10  # 1 MeV/g = 1.602e-13 J / 1e-3 kg
ME_C2_MEV = 0.51099895000        # electron rest energy, MeV
R_E_CM = 2.8179403262e-13        # classical electron radius, cm
N_A = 6.02214076e23
ESTAR_E_MIN = 0.01               # MeV, lower edge of the ESTAR tables
ESTAR_E_MAX = 1000.0             # MeV
GRAZING_WARN_DEG = 75.0          # beyond this incidence angle straight-line transport is unreliable

DATA_DIR = paths.ESTAR_DIR


# ---------------------------------------------------------------------------
# NIST ESTAR tables
# ---------------------------------------------------------------------------
class EstarTable:
    """Log-log interpolation of a NIST ESTAR electron table.

    Columns: kinetic energy (MeV), collision / radiative / total mass stopping
    power (MeV cm2/g), CSDA range (g/cm2), radiation yield (fraction of the
    initial kinetic energy converted to bremsstrahlung while slowing to rest).
    """

    def __init__(self, energy, collision, radiative, csda, rad_yield, source=""):
        self.energy = np.asarray(energy, float)
        self.collision_sp = np.asarray(collision, float)
        self.radiative_sp = np.asarray(radiative, float)
        self.csda = np.asarray(csda, float)
        self.rad_yield = np.asarray(rad_yield, float)
        self.source = source
        self._logE = np.log(self.energy)
        self._logR = np.log(self.csda)

    # -- construction -------------------------------------------------------
    @classmethod
    def from_csv(cls, name: str) -> "EstarTable":
        path = DATA_DIR / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"ESTAR table {path} not found")
        with open(path) as f:
            header = f.readline().strip("# \n")
        df = pd.read_csv(path, comment="#")
        return cls(df["energy_MeV"], df["collision_MeV_cm2_g"], df["radiative_MeV_cm2_g"],
                   df["csda_range_g_cm2"], df["radiation_yield"], source=header)

    @classmethod
    def mixture(cls, components: Sequence[Tuple["EstarTable", float]], source="") -> "EstarTable":
        """Bragg additivity: mass stopping powers are weighted by mass fraction.

        CSDA range and radiation yield are re-integrated on the common grid.
        Weights are normalised to one.
        """
        w = np.array([c[1] for c in components], float)
        w /= w.sum()
        E = components[0][0].energy
        col = sum(wi * t.collision(E) for t, wi in zip((c[0] for c in components), w))
        rad = sum(wi * t.radiative(E) for t, wi in zip((c[0] for c in components), w))
        tot = col + rad
        # Fine log grid for the integrals; start values are the weighted averages at E_min.
        Ef = np.logspace(np.log10(E[0]), np.log10(E[-1]), 4001)
        tot_f = np.exp(np.interp(np.log(Ef), np.log(E), np.log(tot)))
        rad_f = np.exp(np.interp(np.log(Ef), np.log(E), np.log(rad)))
        R0 = sum(wi * t.csda_range(E[0]) for (t, _), wi in zip(components, w))
        Y0 = sum(wi * t.radiation_yield(E[0]) for (t, _), wi in zip(components, w))
        dR = np.concatenate([[0.0], np.cumsum(0.5 * (1 / tot_f[1:] + 1 / tot_f[:-1]) * np.diff(Ef))])
        frac = rad_f / tot_f
        dYE = np.concatenate([[0.0], np.cumsum(0.5 * (frac[1:] + frac[:-1]) * np.diff(Ef))])
        R = R0 + np.interp(np.log(E), np.log(Ef), dR)
        Y = (Y0 * E[0] + np.interp(np.log(E), np.log(Ef), dYE)) / E
        return cls(E, col, rad, R, Y, source=source)

    # -- evaluation ---------------------------------------------------------
    def _interp(self, y, E):
        E = np.clip(np.asarray(E, float), ESTAR_E_MIN, ESTAR_E_MAX)
        return np.exp(np.interp(np.log(E), self._logE, np.log(y)))

    def collision(self, E):
        """Mass collision stopping power, MeV cm2/g."""
        return self._interp(self.collision_sp, E)

    def radiative(self, E):
        """Mass radiative (bremsstrahlung) stopping power, MeV cm2/g."""
        return self._interp(self.radiative_sp, E)

    def total(self, E):
        """Mass total stopping power, MeV cm2/g."""
        return self.collision(E) + self.radiative(E)

    def csda_range(self, E):
        """CSDA range, g/cm2."""
        return self._interp(self.csda, E)

    def radiation_yield(self, E):
        """Fraction of E converted to bremsstrahlung while slowing to rest."""
        return self._interp(self.rad_yield, E)

    def energy_from_range(self, R):
        """Inverse of :meth:`csda_range`: kinetic energy (MeV) with CSDA range R (g/cm2)."""
        R = np.asarray(R, float)
        out = np.exp(np.interp(np.log(np.clip(R, self.csda[0], self.csda[-1])), self._logR, self._logE))
        return np.where(R <= self.csda[0], 0.0, out)


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Material:
    """Target material.

    density        g/cm3
    heat_capacity  J/(g K)  (specific heat at room temperature)
    limit_C        temperature limit for a heating study (melting point for
                   metals, boiling for water), degrees C
    estar          ((table name, mass fraction), ...) used to build the ESTAR data
    Z_over_A, I_eV, sternheimer  are used only by the analytic Bethe cross-check
    """
    name: str
    density: float
    heat_capacity: float
    limit_C: Optional[float]
    estar: Tuple[Tuple[str, float], ...]
    Z_over_A: float
    I_eV: float
    sternheimer: Optional[Dict[str, float]] = None

    def table(self) -> EstarTable:
        return _get_table(self)


# Sternheimer density-effect parameters (Sternheimer, Berger, Seltzer 1984; as
# tabulated by the Particle Data Group).  Keys: Cbar, X0, X1, a, k, d0.
_STERNHEIMER = {
    "water":    dict(Cbar=3.5017, X0=0.2400, X1=2.8004, a=0.09116, k=3.4773, d0=0.00),
    "aluminum": dict(Cbar=4.2395, X0=0.1708, X1=3.0127, a=0.08024, k=3.6345, d0=0.12),
    "copper":   dict(Cbar=4.4190, X0=-0.0254, X1=3.2792, a=0.14339, k=2.9044, d0=0.08),
    "iron":     dict(Cbar=4.2911, X0=-0.0012, X1=3.1531, a=0.14680, k=2.9632, d0=0.12),
}

MATERIALS: Dict[str, Material] = {
    "Water": Material("Water", 1.000, 4.184, 100.0, (("water", 1.0),),
                      Z_over_A=0.55509, I_eV=75.0, sternheimer=_STERNHEIMER["water"]),
    # ICRP skin as used by ESTAR (rho = 1.1 g/cm3); specific heat of skin ~3.4-3.7 kJ/kg/K
    "Skin": Material("Skin", 1.100, 3.5, None, (("skin_icrp", 1.0),),
                     Z_over_A=0.54932, I_eV=72.7),
    "Soft Tissue": Material("Soft Tissue", 1.000, 3.6, None, (("soft_tissue_icrp", 1.0),),
                            Z_over_A=0.55121, I_eV=72.3),
    "Aluminum": Material("Aluminum", 2.699, 0.897, 660.0, (("aluminum", 1.0),),
                         Z_over_A=0.48181, I_eV=166.0, sternheimer=_STERNHEIMER["aluminum"]),
    "Copper": Material("Copper", 8.960, 0.385, 1085.0, (("copper", 1.0),),
                       Z_over_A=0.45636, I_eV=322.0, sternheimer=_STERNHEIMER["copper"]),
    "Iron": Material("Iron", 7.874, 0.449, 1538.0, (("iron", 1.0),),
                     Z_over_A=0.46557, I_eV=286.0, sternheimer=_STERNHEIMER["iron"]),
    # AISI 304: Fe ~70 %, Cr 19 %, Ni 9.5 %, Mn ~2 % (folded into Fe). Melting range 1400-1450 C.
    "Stainless Steel": Material("Stainless Steel", 7.90, 0.500, 1400.0,
                                (("iron", 0.715), ("chromium", 0.190), ("nickel", 0.095)),
                                Z_over_A=0.4659, I_eV=283.0),
}

_TABLE_CACHE: Dict[str, EstarTable] = {}


def _get_table(material: Material) -> EstarTable:
    if material.name not in _TABLE_CACHE:
        comps = [(EstarTable.from_csv(n), w) for n, w in material.estar]
        if len(comps) == 1:
            _TABLE_CACHE[material.name] = comps[0][0]
        else:
            _TABLE_CACHE[material.name] = EstarTable.mixture(
                comps, source=f"Bragg additivity of NIST ESTAR tables: {material.estar}")
    return _TABLE_CACHE[material.name]


def get_material(m) -> Material:
    if isinstance(m, Material):
        return m
    try:
        return MATERIALS[m]
    except KeyError:
        raise KeyError(f"Unknown material {m!r}. Known: {list(MATERIALS)}") from None


# ---------------------------------------------------------------------------
# Stopping power and range: ESTAR-based public helpers
# ---------------------------------------------------------------------------
def collision_stopping_power(E_MeV, material="Water"):
    """Mass collision stopping power S_col/rho, MeV cm2/g (NIST ESTAR)."""
    return get_material(material).table().collision(E_MeV)


def radiative_stopping_power(E_MeV, material="Water"):
    """Mass radiative stopping power S_rad/rho, MeV cm2/g (NIST ESTAR)."""
    return get_material(material).table().radiative(E_MeV)


def total_stopping_power(E_MeV, material="Water"):
    """Mass total stopping power, MeV cm2/g (NIST ESTAR)."""
    return get_material(material).table().total(E_MeV)


def linear_stopping_power(E_MeV, material="Water", kind="total"):
    """Linear stopping power rho * S/rho, MeV/cm."""
    mat = get_material(material)
    t = mat.table()
    sp = {"total": t.total, "collision": t.collision, "radiative": t.radiative}[kind](E_MeV)
    return sp * mat.density


def csda_range(E_MeV, material="Water", unit="cm"):
    """CSDA range from NIST ESTAR, in 'cm' or 'g/cm2'."""
    mat = get_material(material)
    R = mat.table().csda_range(E_MeV)
    return R / mat.density if unit == "cm" else R


def radiation_yield(E_MeV, material="Water"):
    """Fraction of the kinetic energy emitted as bremsstrahlung when fully stopped."""
    return get_material(material).table().radiation_yield(E_MeV)


def stopping_power_table(material="Water", energies_MeV=None) -> pd.DataFrame:
    """Convenience DataFrame of stopping powers, range and yield vs energy."""
    mat = get_material(material)
    t = mat.table()
    E = t.energy if energies_MeV is None else np.asarray(energies_MeV, float)
    return pd.DataFrame({
        "Energy (MeV)": E,
        "S_col/rho (MeV cm2/g)": t.collision(E),
        "S_rad/rho (MeV cm2/g)": t.radiative(E),
        "S_tot/rho (MeV cm2/g)": t.total(E),
        "S_col (MeV/cm)": t.collision(E) * mat.density,
        "S_tot (MeV/cm)": t.total(E) * mat.density,
        "CSDA range (g/cm2)": t.csda_range(E),
        "CSDA range (cm)": t.csda_range(E) / mat.density,
        "Radiation yield": t.radiation_yield(E),
    })


# ---------------------------------------------------------------------------
# Analytic cross-checks (not used by the studies)
# ---------------------------------------------------------------------------
def _sternheimer_generic(material: Material) -> Dict[str, float]:
    """Sternheimer-Peierls (1971) general rules for a condensed material."""
    hbar_wp = 28.816 * math.sqrt(material.density * material.Z_over_A)  # eV
    Cbar = 2 * math.log(material.I_eV / hbar_wp) + 1
    if material.I_eV < 100:
        X1 = 2.0
        X0 = 0.2 if Cbar < 3.681 else 0.326 * Cbar - 1.0
    else:
        X1 = 3.0
        X0 = 0.2 if Cbar < 5.215 else 0.326 * Cbar - 1.5
    k = 3.0
    a = (Cbar - 4.6052 * X0) / (X1 - X0) ** k
    return dict(Cbar=Cbar, X0=X0, X1=X1, a=a, k=k, d0=0.0)


def density_effect_delta(E_MeV, material="Water"):
    """Sternheimer density-effect correction delta(E) for electrons."""
    mat = get_material(material)
    p = mat.sternheimer or _sternheimer_generic(mat)
    tau = np.asarray(E_MeV, float) / ME_C2_MEV
    X = np.log10(np.sqrt(tau * (tau + 2)))  # log10(beta*gamma)
    # np.where evaluates every branch, so keep the power's base non-negative outside its own branch.
    mid = 4.6052 * X - p["Cbar"] + p["a"] * np.maximum(p["X1"] - X, 0.0) ** p["k"]
    delta = np.where(X >= p["X1"], 4.6052 * X - p["Cbar"],
             np.where(X >= p["X0"], mid, p["d0"] * 10 ** (2 * (X - p["X0"]))))
    return delta


def bethe_collision_stopping_power(E_MeV, material="Water", density_effect=True):
    """Analytic mass collision stopping power for electrons (Rohrlich-Carlson,
    ICRU Report 37), MeV cm2/g.  Independent cross-check of the ESTAR tables;
    agrees to ~1-3 % above 1 MeV for materials with tabulated Sternheimer
    parameters, a few percent otherwise.
    """
    mat = get_material(material)
    E = np.asarray(E_MeV, float)
    tau = E / ME_C2_MEV
    gamma = tau + 1
    beta2 = 1 - 1 / gamma ** 2
    I = mat.I_eV * 1e-6 / ME_C2_MEV  # in units of m c2
    F = 1 - beta2 + (tau ** 2 / 8 - (2 * tau + 1) * math.log(2)) / (tau + 1) ** 2
    bracket = np.log(tau ** 2 * (tau + 2) / (2 * I ** 2)) + F
    if density_effect:
        bracket = bracket - density_effect_delta(E, mat)
    prefactor = 2 * math.pi * R_E_CM ** 2 * ME_C2_MEV * N_A * mat.Z_over_A / beta2
    return prefactor * bracket


def katz_penfold_range(E_MeV):
    """Empirical CSDA-like range for electrons in low-Z materials, g/cm2
    (Katz & Penfold 1952).  Valid roughly 0.01-20 MeV; cross-check only."""
    E = np.asarray(E_MeV, float)
    low = 0.412 * E ** (1.265 - 0.0954 * np.log(E))
    high = 0.530 * E - 0.106
    return np.where(E < 2.5, low, high)


def compare_with_estar(material="Water", energies_MeV=(1, 2, 5, 10, 20, 50, 100)) -> pd.DataFrame:
    """Side-by-side of the analytic Bethe collision stopping power and ESTAR."""
    E = np.asarray(energies_MeV, float)
    estar = collision_stopping_power(E, material)
    bethe = bethe_collision_stopping_power(E, material)
    return pd.DataFrame({"Energy (MeV)": E, "ESTAR S_col/rho": estar, "Bethe S_col/rho": bethe,
                         "Ratio Bethe/ESTAR": bethe / estar})


# ---------------------------------------------------------------------------
# Beam description
# ---------------------------------------------------------------------------
@dataclass
class Beam:
    """Pulsed electron beam (macropulse structure of an S-band linac).

    energy_MeV   kinetic energy
    I_pulse_A    macropulse (average-over-macropulse) current
    T_pulse_s    macropulse duration
    f_rep_Hz     macropulse repetition rate
    sigma_x_m, sigma_y_m   RMS sizes at the target (Gaussian) or semi-axes (uniform)
    f_bunch_Hz   bunch repetition frequency inside the macropulse
    profile      "gaussian" or "uniform"
    """
    energy_MeV: float
    I_pulse_A: float
    T_pulse_s: float
    f_rep_Hz: float
    sigma_x_m: float = 1e-3
    sigma_y_m: float = 10e-3
    f_bunch_Hz: float = 2.856e9
    profile: str = "gaussian"

    def __post_init__(self):
        if self.profile not in ("gaussian", "uniform"):
            raise ValueError("profile must be 'gaussian' or 'uniform'")

    # charge / particle numbers
    @property
    def Q_macropulse_C(self) -> float:
        return self.I_pulse_A * self.T_pulse_s

    @property
    def N_electrons(self) -> float:
        return self.Q_macropulse_C / E_CHARGE

    @property
    def n_bunches(self) -> float:
        return self.f_bunch_Hz * self.T_pulse_s

    @property
    def Q_bunch_C(self) -> float:
        return self.I_pulse_A / self.f_bunch_Hz

    @property
    def I_avg_A(self) -> float:
        return self.I_pulse_A * self.T_pulse_s * self.f_rep_Hz

    @property
    def duty_cycle(self) -> float:
        return self.T_pulse_s * self.f_rep_Hz

    # energy / power carried by the beam (before the target)
    @property
    def E_macropulse_J(self) -> float:
        return self.N_electrons * self.energy_MeV * MEV_TO_J

    @property
    def P_avg_W(self) -> float:
        return self.E_macropulse_J * self.f_rep_Hz

    @property
    def P_pulse_W(self) -> float:
        """Power during the macropulse."""
        return self.energy_MeV * 1e6 * self.I_pulse_A

    # transverse
    @property
    def effective_area_cm2(self) -> float:
        """Area such that N / area is the peak fluence.

        Gaussian: 2 pi sigma_x sigma_y (peak of the normalised 2-D Gaussian).
        Uniform ellipse: pi a b.
        """
        sx, sy = self.sigma_x_m * 100, self.sigma_y_m * 100
        return 2 * math.pi * sx * sy if self.profile == "gaussian" else math.pi * sx * sy

    @property
    def peak_fluence_cm2(self) -> float:
        """Electrons per cm2 at the beam centre, per macropulse."""
        return self.N_electrons / self.effective_area_cm2

    def fluence_at(self, x_m=0.0, y_m=0.0) -> float:
        """Fluence per macropulse at transverse position (x, y)."""
        if self.profile == "uniform":
            inside = (x_m / self.sigma_x_m) ** 2 + (y_m / self.sigma_y_m) ** 2 <= 1
            return self.peak_fluence_cm2 if inside else 0.0
        return self.peak_fluence_cm2 * math.exp(-0.5 * ((x_m / self.sigma_x_m) ** 2 + (y_m / self.sigma_y_m) ** 2))

    def summary(self) -> Dict[str, float]:
        return {
            "Energy (MeV)": self.energy_MeV,
            "Macropulse current (mA)": self.I_pulse_A * 1e3,
            "Macropulse length (us)": self.T_pulse_s * 1e6,
            "Rep rate (Hz)": self.f_rep_Hz,
            "Charge per macropulse (nC)": self.Q_macropulse_C * 1e9,
            "Charge per bunch (pC)": self.Q_bunch_C * 1e12,
            "Bunches per macropulse": self.n_bunches,
            "Electrons per macropulse": self.N_electrons,
            "Average current (uA)": self.I_avg_A * 1e6,
            "Beam power in pulse (MW)": self.P_pulse_W * 1e-6,
            "Average beam power (W)": self.P_avg_W,
            "sigma_x (mm)": self.sigma_x_m * 1e3,
            "sigma_y (mm)": self.sigma_y_m * 1e3,
            "Effective area (cm2)": self.effective_area_cm2,
            "Peak fluence per macropulse (1/cm2)": self.peak_fluence_cm2,
        }


# ---------------------------------------------------------------------------
# CSDA slab transport (shared by both studies)
# ---------------------------------------------------------------------------
def slab_energy_balance(E0_MeV: float, material, path_length_cm: Optional[float]) -> Dict[str, float]:
    """Energy bookkeeping for one electron entering a slab along a straight path.

    Integrates dE/ds = -rho S_tot(E) and dE_rad/ds = rho S_rad(E) along the
    path.  Returns MeV per electron: collision (locally deposited), radiative
    (bremsstrahlung produced), transmitted (residual kinetic energy leaving the
    slab), plus the stopping depth if the electron stops inside.

    path_length_cm=None means a thick target (electron fully stopped), for
    which the ESTAR radiation yield is used directly.
    """
    mat = get_material(material)
    t = mat.table()
    rho = mat.density
    R0_cm = t.csda_range(E0_MeV) / rho
    if E0_MeV <= ESTAR_E_MIN:
        return dict(collision=E0_MeV, radiative=0.0, transmitted=0.0, stopped=True, stop_depth_cm=0.0)
    if path_length_cm is None or path_length_cm >= R0_cm:
        Y = float(t.radiation_yield(E0_MeV))
        return dict(collision=E0_MeV * (1 - Y), radiative=E0_MeV * Y, transmitted=0.0,
                    stopped=True, stop_depth_cm=float(R0_cm))

    def rhs(s, y):
        E = max(y[0], ESTAR_E_MIN)
        return [-rho * float(t.total(E)), rho * float(t.radiative(E))]

    def stopped(s, y):
        return y[0] - ESTAR_E_MIN
    stopped.terminal = True

    sol = solve_ivp(rhs, (0.0, path_length_cm), [E0_MeV, 0.0], events=stopped,
                    rtol=1e-8, atol=1e-10, max_step=path_length_cm / 50)
    E_exit = max(float(sol.y[0, -1]), 0.0)
    E_rad = float(sol.y[1, -1])
    if sol.t_events[0].size:  # stopped inside the slab (within the tolerance of the table edge)
        E_exit = 0.0
    E_col = E0_MeV - E_exit - E_rad
    return dict(collision=E_col, radiative=E_rad, transmitted=E_exit,
                stopped=E_exit == 0.0, stop_depth_cm=float(sol.t[-1]) if E_exit == 0.0 else float("nan"))


def energy_at_depth(E0_MeV: float, material, depth_cm) -> np.ndarray:
    """Kinetic energy along the CSDA path after a path length ``depth_cm`` (0 when stopped)."""
    mat = get_material(material)
    t = mat.table()
    R0 = t.csda_range(E0_MeV)
    return t.energy_from_range(R0 - mat.density * np.asarray(depth_cm, float))


# ---------------------------------------------------------------------------
# Case A: beam-pipe / beam-dump heating
# ---------------------------------------------------------------------------
def beam_pipe_heating(beam: Beam, material="Stainless Steel", wall_thickness_m: Optional[float] = None,
                      incidence_deg: float = 0.0, brems_absorbed_fraction: float = 0.0) -> Dict[str, float]:
    """Energy deposition of a (defocused) beam hitting a metal wall.

    Parameters
    ----------
    beam                     Beam at the impact point (sigma_x/sigma_y measured in the wall plane).
    material                 wall material, default stainless steel.
    wall_thickness_m         None for a thick dump; otherwise the wall thickness.
    incidence_deg            angle between the beam and the wall NORMAL (0 = normal
                             incidence, 89 = grazing).  Straight-line path = t / cos(theta);
                             the footprint stretches by 1/cos(theta), which lowers the
                             number of electrons per cm2 of wall and hence the absorbed
                             power per cm2, but NOT the energy per gram at the surface
                             (each electron travels 1/cos(theta) further per unit depth).
                             Above GRAZING_WARN_DEG the straight-line transport is
                             unreliable (multiple scattering turns a large fraction of
                             the electrons back out of the wall or through it)
                             and a warning is issued; see the module docstring.
    brems_absorbed_fraction  fraction of the bremsstrahlung energy re-absorbed in
                             the wall (0 thin wall, ~0.3-0.5 thick high-Z dump).

    Returns a dict with per-electron energy split, absorbed power, peak surface
    energy density per macropulse and the adiabatic temperature rise.
    """
    mat = get_material(material)
    theta = math.radians(incidence_deg)
    cos_t = math.cos(theta)
    if cos_t <= 1e-6:
        raise ValueError("incidence_deg must be < 90 (angle from the wall normal)")
    if incidence_deg > GRAZING_WARN_DEG:
        warnings.warn(f"incidence_deg={incidence_deg:g}: straight-line (CSDA) transport is unreliable at grazing "
                      "incidence; multiple scattering turns a large fraction of the electrons back out of the wall. "
                      "Treat the deposited energy as an upper bound and use a Monte Carlo code.", stacklevel=2)
    path_cm = None if wall_thickness_m is None else wall_thickness_m * 100 / cos_t

    split = slab_energy_balance(beam.energy_MeV, mat, path_cm)
    E_abs = split["collision"] + brems_absorbed_fraction * split["radiative"]
    E_rad_escape = (1 - brems_absorbed_fraction) * split["radiative"]

    # Electrons per cm2 of wall: the beam footprint is stretched by 1/cos(theta).
    fluence_wall = beam.peak_fluence_cm2 * cos_t
    S_col_surface = float(collision_stopping_power(beam.energy_MeV, mat))  # MeV cm2/g
    # Peak energy density at the entrance surface per macropulse (J/g == kGy).  This uses the
    # track-length fluence Phi_peak, not the planar wall fluence: there are cos(theta) fewer
    # electrons per cm2 of wall, but each travels 1/cos(theta) further per unit depth, so the
    # energy per gram in the surface layer is independent of the incidence angle.
    dose_surface_J_per_g = beam.peak_fluence_cm2 * S_col_surface * MEV_TO_J
    dT_pulse = dose_surface_J_per_g / mat.heat_capacity  # K, adiabatic

    P_abs = E_abs * MEV_TO_J * beam.N_electrons * beam.f_rep_Hz  # W
    P_abs_density = fluence_wall * beam.f_rep_Hz * E_abs * MEV_TO_J  # W/cm2 at beam centre
    P_rad = E_rad_escape * MEV_TO_J * beam.N_electrons * beam.f_rep_Hz
    P_trans = split["transmitted"] * MEV_TO_J * beam.N_electrons * beam.f_rep_Hz

    R_cm = float(csda_range(beam.energy_MeV, mat, "cm"))
    out = {
        "Material": mat.name,
        "Energy (MeV)": beam.energy_MeV,
        "Wall thickness (mm)": float("inf") if wall_thickness_m is None else wall_thickness_m * 1e3,
        "Incidence from normal (deg)": incidence_deg,
        "Path length in wall (mm)": float("inf") if path_cm is None else path_cm * 10,
        "CSDA range (mm)": R_cm * 10,
        "Electron stopped in wall": bool(split["stopped"]),
        "Energy deposited per electron (MeV)": E_abs,
        "Energy radiated & escaping per electron (MeV)": E_rad_escape,
        "Energy transmitted per electron (MeV)": split["transmitted"],
        "Fraction of beam energy absorbed": E_abs / beam.energy_MeV,
        "Absorbed energy per macropulse (J)": E_abs * MEV_TO_J * beam.N_electrons,
        "Absorbed power (W)": P_abs,
        "Escaping bremsstrahlung power (W)": P_rad,
        "Transmitted beam power (W)": P_trans,
        "Incident beam power (W)": beam.P_avg_W,
        "Peak absorbed power density (W/cm2)": P_abs_density,
        "Wall fluence per macropulse (1/cm2)": fluence_wall,
        "Peak surface dose per macropulse (kGy)": dose_surface_J_per_g,  # 1 J/g = 1 kGy
        "Adiabatic temperature rise per macropulse (K)": dT_pulse,
        "Mean heating rate without conduction (K/s)": dT_pulse * beam.f_rep_Hz,
        "Material temperature limit (C)": mat.limit_C,
    }
    return out


# ---------------------------------------------------------------------------
# Case B: FLASH dose in a thin water / skin target
# ---------------------------------------------------------------------------
def dose_per_fluence(E_MeV, material="Water"):
    """Dose per unit fluence, Gy cm2 (= 1.602e-10 * S_col/rho)."""
    return GY_PER_MEV_PER_G * collision_stopping_power(E_MeV, material)


def flash_dose(beam: Beam, material="Water", prescription_Gy: float = 10.0,
               target_thickness_m: Optional[float] = None) -> Dict[str, float]:
    """Dose and dose-rate figures for a thin water/skin target.

    The entrance dose per macropulse at the beam centre is
        D = 1.602e-10 * (S_col/rho)(E) * Phi_peak      [Gy]
    Depth variation inside a thin target is small and slightly negative:
    S_col/rho decreases as the electron slows towards the stopping-power
    minimum near 1.7 MeV (about 1 % over 2 cm of water at 40 MeV, 3 % at
    10 MeV); use :func:`depth_dose` for the profile.

    prescription_Gy    dose to deliver (FLASH studies typically 10 Gy or more,
                       at a mean rate above ~40 Gy/s and total time below ~200 ms).
    target_thickness_m  optional; if given, the energy fraction deposited in the
                       target and the exit energy are reported.
    """
    mat = get_material(material)
    E = beam.energy_MeV
    S_col = float(collision_stopping_power(E, mat))
    D_pulse = GY_PER_MEV_PER_G * S_col * beam.peak_fluence_cm2  # Gy per macropulse
    D_bunch = D_pulse / beam.n_bunches
    rate_inst = D_pulse / beam.T_pulse_s          # Gy/s during the macropulse
    rate_mean = D_pulse * beam.f_rep_Hz           # Gy/s averaged over many pulses

    n_pulses = prescription_Gy / D_pulse
    n_pulses_int = math.ceil(n_pulses - 1e-12)
    delivery_time = beam.T_pulse_s if n_pulses_int <= 1 else (n_pulses_int - 1) / beam.f_rep_Hz + beam.T_pulse_s
    Q_needed = beam.Q_macropulse_C * n_pulses
    # Spot size (area scale) that would deliver the prescription in exactly one macropulse.
    area_one_pulse = beam.effective_area_cm2 * D_pulse / prescription_Gy

    out = {
        "Material": mat.name,
        "Energy (MeV)": E,
        "S_col/rho (MeV cm2/g)": S_col,
        "CSDA range (cm)": float(csda_range(E, mat, "cm")),
        "Peak fluence per macropulse (1/cm2)": beam.peak_fluence_cm2,
        "Dose per macropulse (Gy)": D_pulse,
        "Dose per bunch (Gy)": D_bunch,
        "Dose rate in macropulse (Gy/s)": rate_inst,
        "Mean dose rate (Gy/s)": rate_mean,
        "Prescription (Gy)": prescription_Gy,
        "Macropulses for prescription": n_pulses,
        "Delivery time for prescription (s)": delivery_time,
        "Charge for prescription (nC)": Q_needed * 1e9,
        "Effective area for prescription in one pulse (cm2)": area_one_pulse,
        "Adiabatic temperature rise per macropulse (mK)": D_pulse / (mat.heat_capacity * 1e3) * 1e3,
    }
    if target_thickness_m is not None:
        split = slab_energy_balance(E, mat, target_thickness_m * 100)
        out.update({
            "Target thickness (cm)": target_thickness_m * 100,
            "Exit energy (MeV)": split["transmitted"],
            "Fraction of beam energy deposited in target": split["collision"] / E,
            "Fraction radiated in target": split["radiative"] / E,
            "Electron stopped in target": bool(split["stopped"]),
        })
    return out


def depth_dose(beam: Beam, material="Water", depth_cm=None, normalise=False) -> pd.DataFrame:
    """Entrance-axis dose vs depth from CSDA slowing down (per macropulse).

    Valid for depths well inside the range (no scattering, no straggling, the
    fluence is taken constant with depth).  Near the end of the range the
    CSDA curve is not realistic: the real fall-off is gradual because of
    scattering and range straggling, and the steep rise of S_col as E -> 0 in
    the last millimetres is a CSDA artefact.  See the module docstring for the
    relation between the practical range and the CSDA range.
    """
    mat = get_material(material)
    R = float(csda_range(beam.energy_MeV, mat, "cm"))
    z = np.linspace(0, R * 1.05, 300) if depth_cm is None else np.asarray(depth_cm, float)
    Ez = energy_at_depth(beam.energy_MeV, mat, z)
    D = np.where(Ez > 0, GY_PER_MEV_PER_G * collision_stopping_power(np.maximum(Ez, ESTAR_E_MIN), mat)
                 * beam.peak_fluence_cm2, 0.0)
    if normalise:
        D = D / D[0]
    return pd.DataFrame({"Depth (cm)": z, "Energy (MeV)": Ez,
                         "Dose per macropulse (Gy)" if not normalise else "Relative dose": D})


# ---------------------------------------------------------------------------
# Parameter scans
# ---------------------------------------------------------------------------
def _scan(fn, energies, currents_A, pulse_lengths_s, rep_rates_Hz, base: Beam, **kw) -> pd.DataFrame:
    rows: List[Dict] = []
    for E in np.atleast_1d(energies):
        for r in np.atleast_1d(rep_rates_Hz):
            for T in np.atleast_1d(pulse_lengths_s):
                for I in np.atleast_1d(currents_A):
                    b = Beam(float(E), float(I), float(T), float(r), base.sigma_x_m, base.sigma_y_m,
                             base.f_bunch_Hz, base.profile)
                    row = {"Energy (MeV)": float(E), "Beam Current (mA)": float(I) * 1e3,
                           "Pulse Duration (us)": float(T) * 1e6, "Repetition Rate (Hz)": float(r)}
                    res = fn(b, **kw)
                    row.update({k: v for k, v in res.items() if k not in row})
                    rows.append(row)
    return pd.DataFrame(rows)


def scan_beam_pipe(energies_MeV, currents_A, pulse_lengths_s, rep_rates_Hz, base: Beam,
                   material="Stainless Steel", **kw) -> pd.DataFrame:
    """Grid scan of :func:`beam_pipe_heating`. Extra keywords are forwarded."""
    return _scan(beam_pipe_heating, energies_MeV, currents_A, pulse_lengths_s, rep_rates_Hz, base,
                 material=material, **kw)


def scan_flash(energies_MeV, currents_A, pulse_lengths_s, rep_rates_Hz, base: Beam,
               material="Water", **kw) -> pd.DataFrame:
    """Grid scan of :func:`flash_dose`. Extra keywords are forwarded."""
    return _scan(flash_dose, energies_MeV, currents_A, pulse_lengths_s, rep_rates_Hz, base,
                 material=material, **kw)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_stopping_power(material="Water", ax=None, show=True):
    import matplotlib.pyplot as plt
    mat = get_material(material)
    df = stopping_power_table(mat)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(df["Energy (MeV)"], df["S_col/rho (MeV cm2/g)"], label="collision")
    ax.loglog(df["Energy (MeV)"], df["S_rad/rho (MeV cm2/g)"], label="radiative")
    ax.loglog(df["Energy (MeV)"], df["S_tot/rho (MeV cm2/g)"], "k", label="total")
    Eb = np.logspace(0, 3, 60)
    ax.loglog(Eb, bethe_collision_stopping_power(Eb, mat), ":", label="collision, analytic Bethe")
    ax.set_xlabel("Kinetic energy (MeV)")
    ax.set_ylabel("Mass stopping power (MeV cm$^2$/g)")
    ax.set_title(f"Electron stopping power in {mat.name} (NIST ESTAR)")
    ax.grid(True, which="both", ls=":", lw=0.5)
    ax.legend()
    if show:
        plt.tight_layout(); plt.show()
    return ax


def plot_range(materials=("Water", "Skin", "Aluminum", "Stainless Steel", "Copper"), ax=None, show=True):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    E = np.logspace(-1, 2.5, 100)
    for m in materials:
        ax.loglog(E, csda_range(E, m, "cm"), label=get_material(m).name)
    ax.loglog(E, katz_penfold_range(E), "k:", label="Katz-Penfold (g/cm$^2$, low Z)")
    ax.set_xlabel("Kinetic energy (MeV)")
    ax.set_ylabel("CSDA range (cm)")
    ax.set_title("Electron CSDA range (NIST ESTAR)")
    ax.grid(True, which="both", ls=":", lw=0.5)
    ax.legend()
    if show:
        plt.tight_layout(); plt.show()
    return ax


def plot_scan_grid(df: pd.DataFrame, y_column: str, logy=False, show=True):
    """One panel per (rep rate, pulse length); lines per current, x = energy.

    Works for the output of :func:`scan_beam_pipe` and :func:`scan_flash`.
    """
    import matplotlib.pyplot as plt
    reps = np.sort(df["Repetition Rate (Hz)"].unique())[::-1]
    Ts = np.sort(df["Pulse Duration (us)"].unique())[::-1]
    Is = np.sort(df["Beam Current (mA)"].unique())
    fig, axes = plt.subplots(len(reps), len(Ts), figsize=(4 * len(Ts), 3 * len(reps)),
                             sharex=True, squeeze=False)
    for i, r in enumerate(reps):
        for j, T in enumerate(Ts):
            ax = axes[i, j]
            sub = df[(df["Repetition Rate (Hz)"] == r) & (df["Pulse Duration (us)"] == T)]
            for I in Is:
                d = sub[sub["Beam Current (mA)"] == I].sort_values("Energy (MeV)")
                ax.plot(d["Energy (MeV)"], d[y_column], label=f"{I:g} mA")
            ax.set_title(f"{r:g} Hz, {T:g} us", fontsize=9)
            if logy:
                ax.set_yscale("log")
            ax.grid(True, ls=":", lw=0.5)
    axes[0, 0].legend(fontsize=8)
    fig.supxlabel("Beam energy (MeV)")
    fig.supylabel(y_column)
    if show:
        plt.tight_layout(); plt.show()
    return fig


def plot_charge_per_macropulse(I_pulse_range_A, T_pulse_values_s, f_bunch_Hz=2.856e9, show=True):
    """Charge per macropulse (and per bunch on a second axis) vs macropulse current."""
    import matplotlib.pyplot as plt
    I = np.asarray(I_pulse_range_A, float)
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    for ls, T in zip(["-", "--", "-.", ":"], T_pulse_values_s):
        ax1.plot(I * 1e3, I * T * 1e9, "k", ls=ls, label=f"{T*1e6:g} us macropulse")
    ax1.set_xlabel("Macropulse current (mA)")
    ax1.set_ylabel("Charge per macropulse (nC)")
    ax2 = ax1.twinx()
    ax2.plot(I * 1e3, I / f_bunch_Hz * 1e12, "b", alpha=0.6, label=f"per bunch at {f_bunch_Hz*1e-9:g} GHz")
    ax2.set_ylabel("Charge per bunch (pC)", color="b")
    ax1.grid(True, ls=":", lw=0.5)
    ax1.legend(loc="upper left"); ax2.legend(loc="lower right")
    fig.suptitle("Charge per macropulse and per bunch")
    if show:
        plt.tight_layout(); plt.show()
    return fig


# ---------------------------------------------------------------------------
# Facade kept for scripts that used the old class name
# ---------------------------------------------------------------------------
class beamUtility:
    """Thin convenience wrapper around :class:`Beam`, :func:`beam_pipe_heating`
    and :func:`flash_dose` for interactive use.

    >>> u = beamUtility(energy_MeV=40, I_pulse_A=0.1, T_pulse_s=4e-6, f_rep_Hz=10,
    ...                 sigma_x_m=1e-3, sigma_y_m=10e-3)
    >>> u.flash_study("Water")["Dose per macropulse (Gy)"]
    """

    def __init__(self, energy_MeV=40.0, I_pulse_A=0.1, T_pulse_s=4e-6, f_rep_Hz=1.0,
                 sigma_x_m=1e-3, sigma_y_m=10e-3, f_bunch_Hz=2.856e9, profile="gaussian"):
        self.beam = Beam(energy_MeV, I_pulse_A, T_pulse_s, f_rep_Hz, sigma_x_m, sigma_y_m, f_bunch_Hz, profile)

    def beam_pipe_study(self, material="Stainless Steel", wall_thickness_m=None, incidence_deg=0.0,
                        brems_absorbed_fraction=0.0) -> pd.Series:
        return pd.Series(beam_pipe_heating(self.beam, material, wall_thickness_m, incidence_deg,
                                           brems_absorbed_fraction))

    def flash_study(self, material="Water", prescription_Gy=10.0, target_thickness_m=None) -> pd.Series:
        return pd.Series(flash_dose(self.beam, material, prescription_Gy, target_thickness_m))

    def chargePerMacropulse(self, I_pulse_range_A, T_pulse_values_s):
        return plot_charge_per_macropulse(I_pulse_range_A, T_pulse_values_s, self.beam.f_bunch_Hz)


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    base = Beam(energy_MeV=40.0, I_pulse_A=0.1, T_pulse_s=4e-6, f_rep_Hz=10.0, sigma_x_m=1e-3, sigma_y_m=10e-3)
    print("Beam:\n", pd.Series(base.summary()).to_string(), "\n")

    print("Case A: 40 MeV, 100 mA, 4 us, 10 Hz, defocused to 5 mm x 20 mm rms on a 2 mm stainless wall at 80 deg")
    wall = Beam(40.0, 0.1, 4e-6, 10.0, sigma_x_m=5e-3, sigma_y_m=20e-3)
    print(pd.Series(beam_pipe_heating(wall, "Stainless Steel", wall_thickness_m=2e-3, incidence_deg=80)).to_string(), "\n")

    print("Case B: same beam focused to 1 mm x 10 mm rms into water, 10 Gy prescription")
    print(pd.Series(flash_dose(base, "Water", prescription_Gy=10.0, target_thickness_m=0.02)).to_string(), "\n")

    print("Analytic Bethe vs ESTAR, water:\n", compare_with_estar("Water").to_string(index=False))
