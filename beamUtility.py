"""
beamUtility.py — shared constants/equations + modular classes for:
- BeamPower (power & deposition)
- BeamRadiation (Inverse Compton Scattering)
- AlphaMagnet (refined per scaling relations with LAS overlays)


"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import norm
    from scipy.integrate import quad
except Exception:
    norm = None
    quad = None


# ------------------------------
# Constants & shared equations
# ------------------------------

@dataclass(frozen=True)
class PhysicsEqCst:
    e: float = 1.602176634e-19
    me: float = 9.10938356e-31
    mp: float = 1.67262192595e-27
    c: float = 299_792_458.0
    epsilon_0: float = 8.854187817e-12
    NA: float = 6.02214076e23
    h: float = 6.62607015e-34
    me_c2_MeV: float = 0.510_998_95

    # Alpha-magnet geometry / constants (used for trajectory sketch & y_max estimate)
    THETA_ALPHA_DEG: float = 40.70991
    THETA_ALPHA_RAD: float = np.deg2rad(40.70991)

    # Empirical/closed-form coefficients (units consistent with g in T/m)
    # s_alpha = S_COEFF * sqrt(beta*gamma / g), x_max = X_COEFF * sqrt(beta*gamma / g)
    S_COEFF: float = 0.19165   # [m] * sqrt(T/m)
    X_COEFF: float = 0.07505   # [m] * sqrt(T/m)

    @property
    def MeV_to_J(self) -> float:
        return self.e * 1e6
    @property
    def me_c2_J(self) -> float:
        return self.me * self.c ** 2
    @property
    def r_e(self) -> float:
        return self.e ** 2 / (4 * np.pi * self.epsilon_0 * self.me * self.c ** 2)
    @property
    def sigma_T(self) -> float:
        return (8.0 * np.pi / 3.0) * self.r_e ** 2

    # --- Conversions ---
    @staticmethod
    def us_to_s(x_us): return np.asarray(x_us, dtype=float) * 1e-6
    @staticmethod
    def ms_to_s(x_ms): return np.asarray(x_ms, dtype=float) * 1e-3
    @staticmethod
    def mA_to_A(x_mA): return np.asarray(x_mA, dtype=float) * 1e-3
    @staticmethod
    def A_to_mA(x_A): return np.asarray(x_A, dtype=float) * 1e3
    @staticmethod
    def um_to_m(x_um): return np.asarray(x_um, dtype=float) * 1e-6
    @staticmethod
    def m_to_um(x_m): return np.asarray(x_m, dtype=float) * 1e6
    @staticmethod
    def eV_to_J(x_eV): return np.asarray(x_eV, dtype=float) * 1.602176634e-19
    @staticmethod
    def J_to_eV(x_J): return np.asarray(x_J, dtype=float) / 1.602176634e-19

    # --- Reusable physics building-blocks ---
    def gamma_from_Ee_MeV(self, E_e_MeV: float) -> float:
        gamma = 1.0 + float(E_e_MeV) / self.me_c2_MeV
        return gamma

    @staticmethod
    def beta_gamma_from_E_MeV(E_MeV) -> np.ndarray:
        """Robust converter for Matplotlib secondary axis: clips negatives, avoids NaNs."""
        E = np.asarray(E_MeV, dtype=float)
        E = np.where(np.isfinite(E), E, 0.0)
        E = np.maximum(E, 0.0)
        gamma = 1.0 + E / PhysicsEqCst.me_c2_MeV
        val = 1.0 - 1.0 / (gamma * gamma)
        val = np.clip(val, 0.0, None)
        beta = np.sqrt(val)
        return beta * gamma

    @staticmethod
    def E_MeV_from_beta_gamma(bg) -> np.ndarray:
        """Inverse mapping used by secondary axis, handles scalars/arrays safely."""
        bg = np.asarray(bg, dtype=float)
        bg = np.where(np.isfinite(bg), bg, 0.0)
        bg = np.maximum(bg, 0.0)
        gamma = np.sqrt(1.0 + bg * bg)
        E = (gamma - 1.0) * PhysicsEqCst.me_c2_MeV
        return E

    def photon_energy_J_from_lambda_um(self, lambda_um: float) -> float:
        lam = lambda_um * 1e-6
        return self.h * self.c / lam

    def gaussian_overlap_area_m2(self, sigma_x_um: float, sigma_y_um: float) -> float:
        return 2.0 * np.pi * (sigma_x_um * 1e-6) * (sigma_y_um * 1e-6)

    def Egamma_max_J(self, gamma: float, E_L_J: float) -> float:
        xi = 4 * gamma * E_L_J / self.me_c2_J
        return (4 * gamma ** 2 * E_L_J) / (1 + xi)

    def Egamma_vs_theta_J(self, gamma: float, E_L_J: float, theta: np.ndarray) -> np.ndarray:
        xi = 4 * gamma * E_L_J / self.me_c2_J
        return (4 * gamma ** 2 * E_L_J) / (1 + xi + (gamma ** 2) * (theta ** 2))

    def thomson_dsigma_dOmega(self, theta: np.ndarray) -> np.ndarray:
        return 0.5 * self.r_e ** 2 * (1.0 + np.cos(theta) ** 2)

    # --- Bunch/macropulse helpers shared by classes ---
    def electrons_per_bunch(self, Q_bunch_pC: float) -> float:
        return Q_bunch_pC * 1e-12 / self.e

    @staticmethod
    def nbunches_in_macropulse(tau_macro_us: float, f_RF_Hz: float) -> float:
        return tau_macro_us * 1e-6 * f_RF_Hz

    def macro_current_A(self, N_e_total: float, tau_macro_us: float) -> float:
        return N_e_total * self.e / (tau_macro_us * 1e-6)


# ------------------------------
# Shared user inputs
# ------------------------------

@dataclass
class SharedInputs:
    t_pulse_us: np.ndarray
    currents_mA: np.ndarray
    rep_rate_Hz: np.ndarray
    energies_MeV: np.ndarray

    @property
    def t_pulse_s(self) -> np.ndarray:
        return PhysicsEqCst.us_to_s(self.t_pulse_us)
    @property
    def currents_A(self) -> np.ndarray:
        return PhysicsEqCst.mA_to_A(self.currents_mA)

    def current_linspace_A(self, n: int = 50) -> np.ndarray:
        imax = np.max(self.currents_A) if self.currents_A.size else 0.0
        return np.linspace(0.0, imax, n)


# ------------------------------
# BeamPower
# ------------------------------

class BeamPower:
    materials = {
        "Aluminum": {"density": 2700.0, "specific_heat": 900.0, "stopping_power": 2.7, "heat_capacity": 0.897,
                     "atomic_number": 13, "ionization_potential": 166, "atomic_mass": 26.98},
        "Copper": {"density": 8960.0, "specific_heat": 385.0, "stopping_power": 5.0, "heat_capacity": 0.385,
                   "atomic_number": 29, "ionization_potential": 322, "atomic_mass": 63.55},
        "Stainless Steel": {"density": 7850.0, "specific_heat": 500.0, "stopping_power": 3.5, "heat_capacity": 0.500,
                            "atomic_number": 26, "ionization_potential": 233, "atomic_mass": 55.85},
    }
    _sp_converted = False

    @classmethod
    def _ensure_stopping_power_SI(cls, K: PhysicsEqCst):
        if not cls._sp_converted:
            for props in cls.materials.values():
                props["stopping_power"] *= (K.MeV_to_J * 1e6) / props["density"]
            cls._sp_converted = True

    def __init__(self, beam_type: str = "electron", sigma_x: float = 1e-3, sigma_y: float = 10e-3,
                 K: PhysicsEqCst | None = None, inputs: SharedInputs | None = None):
        self.K = K or PhysicsEqCst()
        self.inputs = inputs
        self._ensure_stopping_power_SI(self.K)

        self.PARTICLES = {
            "electron": {"m": self.K.me, "q": self.K.e, "E0": self.K.me_c2_J},
            "proton":   {"m": self.K.mp, "q": self.K.e, "E0": self.K.mp * self.K.c ** 2},
        }
        if beam_type not in self.PARTICLES:
            raise ValueError(f"beam_type must be one of {list(self.PARTICLES)}")
        self.beam_info = self.PARTICLES[beam_type]
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

    def chargePerMacropulse(self, I_pulse_range: np.ndarray | None = None, T_pulse_values: list | np.ndarray | None = None, f_bunch: float = 2.856e9):
        if I_pulse_range is None:
            if self.inputs is None:
                raise ValueError("Either provide I_pulse_range or initialize BeamPower with SharedInputs.")
            I_pulse_range = self.inputs.current_linspace_A(50)
        if T_pulse_values is None:
            if self.inputs is None:
                raise ValueError("Either provide T_pulse_values or initialize BeamPower with SharedInputs.")
            T_pulse_values = self.inputs.t_pulse_s

        line_styles = ['-', '--', '-.', ':']
        fig, ax1 = plt.subplots(figsize=(10, 5))
        for idx, T_pulse in enumerate(T_pulse_values):
            Q_macropulse = (np.asarray(I_pulse_range) * T_pulse) * 1e12  # pC
            ax1.plot(self.K.A_to_mA(I_pulse_range), Q_macropulse,
                     linestyle=line_styles[idx % len(line_styles)],
                     label=f'Q per Macropulse ({T_pulse * 1e6:.1f} μs)')
        ax1.set_xlabel("Macropulse Current (mA)")
        ax1.set_ylabel("Charge per Macropulse (pC)")
        ax1.grid(True)
        ax1.legend(loc='upper left')
        fig.suptitle("Charge per Macropulse vs Beam Current")
        plt.tight_layout()
        plt.show()

    def getPowerDF(
        self,
        I_pulse_range: np.ndarray,
        T_pulse_values: np.ndarray,
        rep_rate_values: np.ndarray,
        E_energy_range: np.ndarray,
        plot_type: str = "Power",
        penetration_depth: float = 20e-3,
        plot: bool = True,
        material_for_temp: str = "Copper",
    ) -> pd.DataFrame:
        if material_for_temp not in self.materials:
            raise ValueError(f"material_for_temp must be one of {list(self.materials)}")

        beam_area = np.pi * (6 * self.sigma_x) * (6 * self.sigma_y)
        beam_volume = beam_area * penetration_depth
        beam_volume_cm3 = beam_volume * 1e6

        power_results = []
        for E in E_energy_range:
            for r in rep_rate_values:
                for T_pulse in T_pulse_values:
                    for I_pulse in I_pulse_range:
                        Q_macropulse = I_pulse * T_pulse
                        N_electrons = Q_macropulse / self.K.e
                        E_pulse = N_electrons * (E * self.K.MeV_to_J)
                        P_beam = E_pulse * r  # W
                        temp_rise = {}
                        for material, props in self.materials.items():
                            mass_g = beam_volume_cm3 * props["density"] / 1000  # g
                            temp_rise[material] = P_beam / (mass_g * props["heat_capacity"])  # °C/s
                        power_results.append([E, I_pulse * 1e3, r, T_pulse * 1e6, P_beam,
                                              temp_rise["Copper"], temp_rise["Aluminum"], temp_rise["Stainless Steel"]])

        columns = ["Energy (MeV)", "Beam Current (mA)", "Repetition Rate (Hz)", "Pulse Duration (μs)", "Power (W)",
                   "Temp Rise Copper (°C/s)", "Temp Rise Aluminum (°C/s)", "Temp Rise Stainless Steel (°C/s)"]
        df_power = pd.DataFrame(power_results, columns=columns)

        if plot:
            h_size = rep_rate_values.size
            v_size = T_pulse_values.size
            fig, axes = plt.subplots(h_size, v_size, figsize=(12, 12), sharex=True)
            axes = np.atleast_2d(axes)
            for i, r in enumerate(rep_rate_values[::-1]):
                max_y = 0.0
                for j, T_pulse in enumerate(T_pulse_values[::-1]):
                    ax = axes[i, j]
                    subset = df_power[(df_power["Repetition Rate (Hz)"] == r) &
                                      (df_power["Pulse Duration (μs)"] == T_pulse * 1e6)]
                    if not subset.empty:
                        for I_pulse in I_pulse_range:
                            data = subset[subset["Beam Current (mA)"] == I_pulse * 1e3]
                            if plot_type == "Power":
                                y_data = data["Power (W)"].values
                                ylabel = "Power (W)"
                            else:
                                y_col = f"Temp Rise {material_for_temp} (°C/s)"
                                y_data = data[y_col].values
                                ylabel = y_col
                            ax.plot(data["Energy (MeV)"].values, y_data, label=f"{I_pulse*1e3:.0f} mA")
                            if len(y_data):
                                max_y = max(max_y, float(np.max(y_data)))
                    ax.set_title(f"{r:.1f} Hz, {T_pulse*1e6:.1f} μs", fontsize=9)
                    ax.grid(True)
                    if i == 0 and j == 0:
                        ax.legend(fontsize=8, title="Current")
                for j in range(v_size):
                    if max_y > 0:
                        axes[i, j].set_ylim(0, max_y * 1.1)

            fig.text(0.5, 0.02, "Beam Energy (MeV)", ha='center', fontsize=12)
            fig.text(0.02, 0.5, ylabel, va='center', rotation='vertical', fontsize=12)
            fig.suptitle(f"Beam {plot_type}", fontsize=14)
            plt.tight_layout()
            plt.show()

        return df_power

    def model_Grunn(self, material: str, E_energy_range: np.ndarray) -> pd.DataFrame:
        rho = self.materials[material]["density"] / 1000.0  # g/cm^3
        results = [[material, E, (0.1 * E ** 1.5) / rho] for E in E_energy_range]
        return pd.DataFrame(results, columns=["Material", "Energy (MeV)", "Penetration Depth (cm)"])

    def model_Bethe(self, material: str, E_energy_range: np.ndarray) -> pd.DataFrame:
        props = self.materials[material]
        rho = props["density"] / 1000.0  # g/cm^3
        Z = props["atomic_number"]
        A = props["atomic_mass"]  # g/mol
        I = props["ionization_potential"] * self.K.e  # J
        n_e = (self.K.NA * rho / A) * Z * 1e6  # electrons/m^3

        rows = []
        for E in E_energy_range:
            E_J = E * self.K.MeV_to_J
            gamma = 1 + (E_J / self.K.me_c2_J)
            beta2 = 1 - (1 / gamma ** 2)
            beta = np.sqrt(max(beta2, 1e-16))

            log_term = (2 * self.K.me_c2_J * beta2) / I
            log_term = max(log_term, 1e-12)

            stopping_power_J_per_m = (4 * np.pi * self.K.e ** 3 * Z * n_e) / (self.K.me * self.K.c ** 2 * beta2) * np.log(log_term)
            stopping_power_MeV_per_mm = stopping_power_J_per_m / self.K.MeV_to_J / 1000.0

            R_mm = (E) / max(stopping_power_MeV_per_mm, 1e-12)
            rows.append([material, E, max(R_mm / 10.0, 0.0), stopping_power_MeV_per_mm])

        return pd.DataFrame(rows, columns=["Material", "Energy (MeV)", "Penetration Depth (cm)", "Stopping Power (MeV/mm)"])

    def compute_deposition_profile(self, energy: float, material: str):
        if norm is None:
            raise ImportError("Requires scipy.stats.norm")
        df_bethe = self.model_Bethe(material, np.array([energy]))
        R_cm = df_bethe["Penetration Depth (cm)"].values[0]
        x_range = np.linspace(0, max(R_cm + 2.0, 2.0), 200)
        sigma = max(R_cm * 0.2, 1e-6)
        dep = norm.pdf(x_range, R_cm, sigma)
        dep /= np.max(dep) if np.max(dep) > 0 else 1.0
        return x_range, dep

    def plot_deposition_profile(self, energy: float, material: str):
        x_range, dep = self.compute_deposition_profile(energy, material)
        plt.figure(figsize=(8, 5))
        plt.plot(x_range, dep, label=f"{energy} MeV in {material}")
        plt.xlabel("Depth (cm)")
        plt.ylabel("Relative Deposition")
        plt.title(f"Electron Deposition Profile in {material}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_penetration_depth(self, material: str, E_energy_range: np.ndarray = None):
        if E_energy_range is None:
            E_energy_range = np.logspace(-1, 2, 100)
        df_grunn = self.model_Grunn(material, E_energy_range)
        df_bethe = self.model_Bethe(material, E_energy_range)
        plt.figure(figsize=(8, 5))
        plt.xscale("log")
        plt.plot(df_grunn["Energy (MeV)"], df_grunn["Penetration Depth (cm)"], linestyle='--', label=f"{material} - Grunn")
        plt.plot(df_bethe["Energy (MeV)"], df_bethe["Penetration Depth (cm)"], linestyle='-', label=f"{material} - Bethe")
        plt.xlabel("Beam Energy (MeV)")
        plt.ylabel("Penetration Depth (cm)")
        plt.title(f"Penetration Depth in {material}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


# ------------------------------
# BeamRadiation (ICS)
# ------------------------------

class BeamRadiation:
    def __init__(self, K: PhysicsEqCst | None = None):
        self.K = K or PhysicsEqCst()

    def ics_yield_per_pulse(self, N_e: float, N_ph: float, A_overlap_m2: float) -> float:
        return self.K.sigma_T * N_e * N_ph / max(A_overlap_m2, 1e-30)

    def fraction_above_threshold(self, gamma: float, E_L_J: float, E_thresh_J: float) -> float:
        if quad is None:
            raise ImportError("Requires scipy.integrate.quad")
        num = 4 * gamma ** 2 * E_L_J / E_thresh_J - 1.0 - 4 * gamma * E_L_J / self.K.me_c2_J
        if num <= 0:
            return 0.0
        theta_thresh = np.sqrt(num) / gamma
        integrand = lambda th: float(self.K.thomson_dsigma_dOmega(np.array([th]))[0]) * np.sin(th)
        sigma_sel, _ = quad(integrand, 0.0, theta_thresh)
        return sigma_sel / self.K.sigma_T

    def quick_report_and_plots(
        self,
        E_e_MeV: float = 45.0,
        Q_bunch_pC: float = 60.0,
        f_RF_Hz: float = 2.856e9,
        tau_macro_us: float = 4.0,
        rep_rate_Hz: float = 10.0,
        lambda_L_um: float = 3.0,
        E_L_mJ: float = 10.0,
        sigma_um: float = 30.0,
        E_thresh_eV: float = 1e4,
        make_plots: bool = True,
    ) -> dict:
        K = self.K
        gamma = K.gamma_from_Ee_MeV(E_e_MeV)
        N_e_per_bunch = K.electrons_per_bunch(Q_bunch_pC)
        N_bunches = K.nbunches_in_macropulse(tau_macro_us, f_RF_Hz)
        N_e_total = N_e_per_bunch * N_bunches
        I_pulse = K.macro_current_A(N_e_total, tau_macro_us)

        E_L_photon_J = K.photon_energy_J_from_lambda_um(lambda_L_um)
        N_photon = (E_L_mJ * 1e-3) / E_L_photon_J
        A_overlap = K.gaussian_overlap_area_m2(sigma_um, sigma_um)

        N_ICS_per_pulse = self.ics_yield_per_pulse(N_e_total, N_photon, A_overlap)
        N_ICS_per_sec = N_ICS_per_pulse * rep_rate_Hz

        E_gamma_max_J = K.Egamma_max_J(gamma, E_L_photon_J)
        E_gamma_max_eV = K.J_to_eV(E_gamma_max_J)

        frac_above = self.fraction_above_threshold(gamma, E_L_photon_J, K.eV_to_J(E_thresh_eV)) if quad else np.nan
        N_above = frac_above * N_ICS_per_sec if np.isfinite(frac_above) else np.nan

        report = {
            "gamma": gamma,
            "electron_current_mA": I_pulse * 1e3,
            "N_e_total": N_e_total,
            "N_photon": N_photon,
            "A_overlap_m2": A_overlap,
            "N_ICS_per_pulse": N_ICS_per_pulse,
            "N_ICS_per_sec": N_ICS_per_sec,
            "E_gamma_max_eV": E_gamma_max_eV,
            "fraction_above_thresh": frac_above,
            "N_ICS_above_thresh_per_s": N_above,
        }

        if make_plots:
            theta_vals = np.linspace(0, 5.0 / gamma, 2000)
            d_sigma_vals = self.K.thomson_dsigma_dOmega(theta_vals) * np.sin(theta_vals)
            d_sigma_norm = d_sigma_vals / self.K.sigma_T

            E_theta_keV = self.K.Egamma_vs_theta_J(gamma, E_L_photon_J, theta_vals) / self.K.e / 1e3
            num = 4 * gamma ** 2 * E_L_photon_J / (self.K.eV_to_J(E_thresh_eV)) - 1.0 - 4 * gamma * E_L_photon_J / self.K.me_c2_J
            theta_thresh = np.sqrt(num) / gamma if num > 0 else None

            plt.figure(figsize=(8, 5))
            plt.plot(theta_vals * 1e3, d_sigma_norm, label=r"(1/σ_T)(dσ/dΩ)·sinθ", color='blue')
            if theta_thresh is not None:
                plt.axvline(theta_thresh * 1e3, color='red', linestyle='--', label=f"{E_thresh_eV/1e3:.1f} keV cutoff")
                mask = (theta_vals <= theta_thresh)
                plt.fill_between(theta_vals[mask] * 1e3, d_sigma_norm[mask], color='blue', alpha=0.3, label="E ≥ threshold")
            plt.xlabel("Scattering angle θ [mrad]")
            plt.ylabel("Normalized angular distribution")
            plt.title("ICS Angular Distribution (Thomson)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

            theta_1_over_gamma_mrad = (1.0 / gamma) * 1e3
            plt.figure(figsize=(8, 5))
            plt.plot(theta_vals * 1e3, E_theta_keV, label=r"$E_\gamma(\theta)$")
            plt.axvline(theta_1_over_gamma_mrad, color='green', linestyle='--', label=r"$1/\gamma$")
            if theta_thresh is not None:
                plt.axvline(theta_thresh * 1e3, color='red', linestyle='--', label="Threshold angle")
                plt.axhline(E_thresh_eV / 1e3, color='gray', linestyle=':', linewidth=0.8)
            plt.xlabel("Scattering angle θ [mrad]")
            plt.ylabel("Scattered photon energy [keV]")
            plt.title("ICS Photon Energy vs Scattering Angle")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

            theta_vals2 = np.linspace(0, 5.0 / gamma, 4000)
            dtheta = theta_vals2[1] - theta_vals2[0]
            E_gamma_vals_keV = self.K.Egamma_vs_theta_J(gamma, E_L_photon_J, theta_vals2) / self.K.e / 1e3
            d_sigma_vals2 = self.K.thomson_dsigma_dOmega(theta_vals2) * np.sin(theta_vals2)
            weights = d_sigma_vals2 * dtheta
            E_bins = np.linspace(0, np.max(E_gamma_vals_keV), 300)
            hist_vals, bin_edges = np.histogram(E_gamma_vals_keV, bins=E_bins, weights=weights)
            E_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            norm_area = np.sum(hist_vals * np.diff(E_bins))
            if norm_area > 0:
                hist_vals = hist_vals / norm_area
            plt.figure(figsize=(8, 5))
            plt.plot(E_centers, hist_vals, label=r"(1/σ_T) dσ/dEγ", color='blue')
            plt.axvline(E_thresh_eV / 1e3, color='red', linestyle='--', label=f"{E_thresh_eV/1e3:.1f} keV threshold")
            maskE = (E_centers >= E_thresh_eV / 1e3)
            plt.fill_between(E_centers[maskE], hist_vals[maskE], alpha=0.3, color='blue', label="E ≥ threshold")
            plt.xlabel("Photon energy [keV]")
            plt.ylabel("Normalized distribution")
            plt.title("ICS Photon Energy Spectrum (Thomson)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

        return report


# ------------------------------
# AlphaMagnet (refined)
# ------------------------------

class AlphaMagnet:
    def __init__(self, grad_per_amp_T_per_m: float = 0.103754, nominal_current_A: float = 17.5, K: PhysicsEqCst | None = None):
        self.K = K or PhysicsEqCst()
        self.grad_per_amp = float(grad_per_amp_T_per_m)  # g(I) = kappa * I, kappa in (T/m)/A
        self.nominal_current_A = float(nominal_current_A)

    def g_from_current(self, I_A: float) -> float:
        return self.grad_per_amp * float(I_A)

    def s_alpha_from_bg_g(self, beta_gamma: np.ndarray, g_T_per_m: float) -> np.ndarray:
        return self.K.S_COEFF * np.sqrt(np.asarray(beta_gamma) / g_T_per_m)

    def x_max_from_bg_g(self, beta_gamma: np.ndarray, g_T_per_m: float) -> np.ndarray:
        return self.K.X_COEFF * np.sqrt(np.asarray(beta_gamma) / g_T_per_m)

    def y_max_est_from_xmax(self, x_m: np.ndarray) -> np.ndarray:
        return np.asarray(x_m) * np.tan(np.deg2rad(self.K.THETA_ALPHA_DEG) / 2.0)

    @staticmethod
    def dsalpha_dQp0(beta_gamma: np.ndarray, g_T_per_m: float) -> np.ndarray:
        S_COEFF = PhysicsEqCst.S_COEFF
        return (S_COEFF / 2.0) * (1.0 / np.sqrt(g_T_per_m)) * (np.asarray(beta_gamma) ** -0.5)

    @staticmethod
    def e_to_betagamma_axis(E_MeV: np.ndarray) -> np.ndarray:
        return PhysicsEqCst.beta_gamma_from_E_MeV(E_MeV)

    @staticmethod
    def betagamma_to_e_axis(bg: np.ndarray) -> np.ndarray:
        return PhysicsEqCst.E_MeV_from_beta_gamma(bg)

    def plot_s_x_y_vs_energy_with_LAS(
            self,
            currents_A=(10.0, 15.0, 17.5, 20.0),
            E_min_MeV: float = 0.05,
            E_max_MeV: float = 1.20,
            nE: int = 400,
            s_op_cm: float = 22.6,
            energies_for_traj_MeV=(0.6, 1.0, 1.2),
            currents_for_traj_A=(10.0, 17.5, 20.0),
    ) -> None:
        E_grid = np.linspace(E_min_MeV, E_max_MeV, nE)
        BG_grid = np.maximum(PhysicsEqCst.beta_gamma_from_E_MeV(E_grid), 1e-12)

        s_op_m = s_op_cm * 1e-2
        s_op_mm = s_op_cm * 10.0
        ratio = s_op_m / self.K.S_COEFF
        ratio2 = ratio * ratio

        # Optional vertical guides: energy where s_alpha == s_op for each current
        E_ref_by_I = []
        for I in currents_A:
            gI = self.g_from_current(I)
            bg_ref = gI * ratio2
            E_ref_by_I.append(float(PhysicsEqCst.E_MeV_from_beta_gamma(bg_ref)))

        # Reference x_max, y_max at s_op (independent of current)
        x_max_ref_m = self.K.X_COEFF * ratio
        x_max_ref_mm = x_max_ref_m * 1e3
        y_max_ref_m = self.y_max_est_from_xmax(x_max_ref_m)
        y_max_ref_mm = y_max_ref_m * 1e3

        fig, axs = plt.subplots(2, 2, figsize=(12, 9))
        ax1, ax2 = axs[0, 0], axs[0, 1]
        ax3, ax4 = axs[1, 0], axs[1, 1]

        # (1a) s_alpha vs E
        for I in currents_A:
            g = self.g_from_current(I)
            s_mm = 1e3 * self.s_alpha_from_bg_g(BG_grid, g)
            ax1.plot(E_grid, s_mm, label=f"I = {I:.1f} A")
        for I, E_ref in zip(currents_A, E_ref_by_I):
            ax1.axvline(E_ref, linestyle="--", linewidth=1)
            ax1.plot(E_ref, s_op_mm, marker="o")
        ax1.axhline(s_op_mm, linestyle="--", linewidth=1, label=f"op: ℓ = {s_op_cm:.1f} cm")
        ax1.set_xlabel("Kinetic energy E [MeV]")
        ax1.set_ylabel(r"$s_{\alpha}$ [mm]")
        ax1.grid(True)
        ax1.legend(fontsize=8)
        sec1 = ax1.secondary_xaxis('top',
                                   functions=(self.e_to_betagamma_axis, self.betagamma_to_e_axis))
        sec1.set_xlabel(r"$\beta\gamma$")

        # (1b) x_max vs E
        for I in currents_A:
            g = self.g_from_current(I)
            x_mm = 1e3 * self.x_max_from_bg_g(BG_grid, g)
            ax2.plot(E_grid, x_mm, label=f"I = {I:.1f} A")
        for I, E_ref in zip(currents_A, E_ref_by_I):
            ax2.axvline(E_ref, linestyle="--", linewidth=1)
            ax2.plot(E_ref, x_max_ref_mm, marker="o")
        ax2.axhline(x_max_ref_mm, linestyle="--", linewidth=1, label=f"op: $x_\\max$ = {x_max_ref_mm:.1f} mm")
        ax2.set_xlabel("Kinetic energy E [MeV]")
        ax2.set_ylabel(r"$x_{\max}$ [mm]")
        ax2.grid(True)
        ax2.legend(fontsize=8)
        sec2 = ax2.secondary_xaxis('top',
                                   functions=(self.e_to_betagamma_axis, self.betagamma_to_e_axis))
        sec2.set_xlabel(r"$\beta\gamma$")

        # (1c) y_max (estimate) vs E
        for I in currents_A:
            g = self.g_from_current(I)
            x_m = self.x_max_from_bg_g(BG_grid, g)
            y_mm = 1e3 * self.y_max_est_from_xmax(x_m)
            ax3.plot(E_grid, y_mm, label=f"I = {I:.1f} A")
        for I, E_ref in zip(currents_A, E_ref_by_I):
            ax3.axvline(E_ref, linestyle="--", linewidth=1)
            ax3.plot(E_ref, y_max_ref_mm, marker="o")
        ax3.axhline(y_max_ref_mm, linestyle="--", linewidth=1, label=f"op: $y_\\max$ ≈ {y_max_ref_mm:.1f} mm")
        ax3.set_xlabel("Kinetic energy E [MeV]")
        ax3.set_ylabel(r"$y_{\max}$ (est.) [mm]")
        ax3.grid(True)
        ax3.legend(fontsize=8)
        sec3 = ax3.secondary_xaxis('top',
                                   functions=(self.e_to_betagamma_axis, self.betagamma_to_e_axis))
        sec3.set_xlabel(r"$\beta\gamma$")

        # (1d) Alpha-trajectory (two "rosette" branches) for multiple E and I
        theta = np.deg2rad(self.K.THETA_ALPHA_DEG)

        def cubic_bezier(P0, P1, P2, P3, n=200):
            t = np.linspace(0.0, 1.0, n)
            return ((1 - t) ** 3)[:, None] * P0 + (3 * (1 - t) ** 2 * t)[:, None] * P1 + (3 * (1 - t) * t ** 2)[:,
                                                                                         None] * P2 + (t ** 3)[:,
                                                                                                      None] * P3

        def alpha_rosette_branches(xmax_m, ymax_m, Lctrl=None):
            """Two smooth branches that both start/end at (0,0) and touch there.
            Upper branch goes to (xmax,+ymax), lower to (xmax,-ymax)."""
            if Lctrl is None:
                Lctrl = 0.5 * xmax_m
            # Upper branch: O(0,0) -> A(xmax,+ymax) -> O(0,0)
            O = np.array([0.0, 0.0])
            A = np.array([xmax_m, ymax_m])
            C1 = O + Lctrl * np.array([np.cos(theta), np.sin(theta)])  # start tangent +θ
            C2 = A - Lctrl * np.array([np.cos(theta), np.sin(theta)])
            seg1 = cubic_bezier(O, C1, C2, A, n=200)
            C1b = A - Lctrl * np.array([np.cos(theta), -np.sin(theta)])  # end tangent −θ on return
            C2b = O + Lctrl * np.array([np.cos(theta), -np.sin(theta)])
            seg2 = cubic_bezier(A, C1b, C2b, O, n=200)
            upper = np.vstack([seg1, seg2])

            # Lower branch: O(0,0) -> B(xmax,-ymax) -> O(0,0)
            B = np.array([xmax_m, -ymax_m])
            C3 = O + Lctrl * np.array([np.cos(theta), -np.sin(theta)])  # start tangent −θ
            C4 = B - Lctrl * np.array([np.cos(theta), -np.sin(theta)])
            seg3 = cubic_bezier(O, C3, C4, B, n=200)
            C3b = B - Lctrl * np.array([np.cos(theta), np.sin(theta)])  # end tangent +θ on return
            C4b = O + Lctrl * np.array([np.cos(theta), np.sin(theta)])
            seg4 = cubic_bezier(B, C3b, C4b, O, n=200)
            lower = np.vstack([seg3, seg4])

            return upper, lower

        # Entrance/exit straight segments (on the LEFT, x<0), drawn once for reference
        L_edge = 0.25 * x_max_ref_m
        t_edge = np.linspace(0.0, 1.0, 50)
        # Direction vectors pointing to the left from the origin
        v_in = np.array([-np.cos(theta), np.sin(theta)])  # +θ
        v_out = np.array([-np.cos(theta), -np.sin(theta)])  # −θ
        line_in = (t_edge[:, None] * L_edge) * v_in  # from (0,0) toward x<0
        line_out = (t_edge[:, None] * L_edge) * v_out

        # Color by current; linestyle by energy
        colors = plt.cm.viridis(np.linspace(0.15, 0.95, len(currents_for_traj_A)))
        linestyles = {energies_for_traj_MeV[i]: ls for i, ls in
                      enumerate(["-", "--", ":"][:len(energies_for_traj_MeV)])}

        # Draw once: entrance/exit lines (x<0)
        ax4.plot(line_in[:, 0] * 1e3, line_in[:, 1] * 1e3, color="k", lw=1.2, alpha=0.7, label="entrance (+θ)")
        ax4.plot(line_out[:, 0] * 1e3, line_out[:, 1] * 1e3, color="k", lw=1.2, alpha=0.7, label="exit (−θ)")

        # Overlaid rosette branches for each (E, I)
        for ci, I in enumerate(currents_for_traj_A):
            g = self.g_from_current(I)
            for E in energies_for_traj_MeV:
                bg = PhysicsEqCst.beta_gamma_from_E_MeV(E)
                s_m = float(self.s_alpha_from_bg_g(bg, g))
                x_m = float(self.x_max_from_bg_g(bg, g))
                y_m = float(self.y_max_est_from_xmax(x_m))
                upper, lower = alpha_rosette_branches(x_m, y_m, Lctrl=0.5 * x_m)
                ax4.plot(upper[:, 0] * 1e3, upper[:, 1] * 1e3,
                         color=colors[ci], linestyle=linestyles[E], lw=1.6,
                         label=f"{E:.1f} MeV, I={I:.1f} A (upper)")
                ax4.plot(lower[:, 0] * 1e3, lower[:, 1] * 1e3,
                         color=colors[ci], linestyle=linestyles[E], lw=1.6, alpha=0.9,
                         label=f"{E:.1f} MeV, I={I:.1f} A (lower)")

        ax4.set_xlabel("x [mm]")
        ax4.set_ylabel("y [mm]")
        ax4.axis('equal')
        ax4.grid(True)
        ax4.legend(fontsize=8, ncol=2)

        fig.tight_layout()
        plt.show()

    def plot_alpha_compression_1343(
            self,
            currents_A=(10.0, 15.0, 17.5, 20.0),
            E_min_MeV: float = 0.05,
            E_max_MeV: float = 1.20,
            nE: int = 400,
            rf_freq_Hz: float = 2.856e9
    ) -> None:
        c = self.K.c
        me_c2 = self.K.me_c2_MeV
        T_rf_ps = (1.0 / rf_freq_Hz) * 1e12

        E_grid = np.linspace(E_min_MeV, E_max_MeV, nE)
        gamma = 1.0 + E_grid / me_c2
        beta = np.sqrt(1.0 - 1.0 / np.maximum(gamma ** 2, 1.0 + 1e-15))
        bg = np.maximum(PhysicsEqCst.beta_gamma_from_E_MeV(E_grid), 1e-12)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=False)

        # (A) ds/d(βγ) using Eq. (13.43)
        axA = axes[0]
        for I in currents_A:
            g = self.g_from_current(I)
            ds_dQ = self.K.X_COEFF / np.sqrt(g) / np.sqrt(bg)  # m per unit (βγ)  (Eq. 13.43)
            axA.plot(E_grid, 1e3 * ds_dQ, label=f"I={I:.1f} A")  # mm per unit (βγ)
        axA.set_xlabel("Kinetic energy E [MeV]")
        axA.set_ylabel(r"$\mathrm{d}s_\alpha/\mathrm{d}(\beta\gamma)$ [mm]")
        axA.grid(True)
        axA.legend(fontsize=8)
        secA = axA.secondary_xaxis('top',
                                   functions=(self.e_to_betagamma_axis, self.betagamma_to_e_axis))
        secA.set_xlabel(r"$\beta\gamma$")

        # (B) dt/dE (ps/keV)
        axB = axes[1]
        for I in currents_A:
            g = self.g_from_current(I)
            s_m = self.s_alpha_from_bg_g(bg, g)
            ds_dQ = self.K.X_COEFF / np.sqrt(g) / np.sqrt(bg)  # m/(βγ)
            dQ_dE = 1.0 / (beta * me_c2)  # (βγ)/MeV
            dβ_dE = 1.0 / (beta * gamma ** 3 * me_c2)  # 1/MeV
            dt_dE_s_per_MeV = (ds_dQ * dQ_dE) / (beta * c) - (s_m * dβ_dE) / (beta ** 2 * c)
            dt_dE_ps_per_keV = dt_dE_s_per_MeV * 1e9  # 1e12/1e3
            axB.plot(E_grid, dt_dE_ps_per_keV, label=f"I={I:.1f} A")
        axB.set_xlabel("Kinetic energy E [MeV]")
        axB.set_ylabel(r"$\mathrm{d}t/\mathrm{d}E$ [ps/keV]")
        axB.grid(True)
        axB.legend(fontsize=8)
        secB = axB.secondary_xaxis('top',
                                   functions=(self.e_to_betagamma_axis, self.betagamma_to_e_axis))
        secB.set_xlabel(r"$\beta\gamma$")

        # (C) per-mille normalized metrics
        axC = axes[2]
        for I in currents_A:
            g = self.g_from_current(I)
            s_m = self.s_alpha_from_bg_g(bg, g)
            ds_dQ = self.K.X_COEFF / np.sqrt(g) / np.sqrt(bg)
            dQ_dE = 1.0 / (beta * me_c2)
            dβ_dE = 1.0 / (beta * gamma ** 3 * me_c2)
            dt_dE_s_per_MeV = (ds_dQ * dQ_dE) / (beta * c) - (s_m * dβ_dE) / (beta ** 2 * c)
            dt_dE_ps_per_keV = dt_dE_s_per_MeV * 1e9
            # For a 1 keV energy step:
            dt_per_keV_over_Trf_permille = 1e-3 * (dt_dE_ps_per_keV / T_rf_ps)
            dE_over_E_permille = 1e-3 * (1.0 / (1000.0 * E_grid))  # (1 keV)/E times 1e-3
            axC.plot(E_grid, dt_per_keV_over_Trf_permille, label=f"I={I:.1f} A")
        # Also overlay 1e-3*(1 keV / E) for reference
        axC.plot(E_grid, 1e-3 * (1.0 / (1000.0 * E_grid)), linestyle="--", color="k",
                 label=r"$10^{-3}\cdot (\Delta E/E)$ with $\Delta E=1$ keV")
        axC.set_xlabel("Kinetic energy E [MeV]")
        axC.set_ylabel("per-mille (dimensionless)")
        axC.grid(True)
        axC.legend(fontsize=8)
        secC = axC.secondary_xaxis('top',
                                   functions=(self.e_to_betagamma_axis, self.betagamma_to_e_axis))
        secC.set_xlabel(r"$\beta\gamma$")

        fig.suptitle("Alpha Magnet: Eq. (13.43) sensitivity and normalized metrics", fontsize=13)
        fig.tight_layout()
        plt.show()


# ------------------------------
# Main demo: shared inputs -> classes
# ------------------------------

if __name__ == "__main__":
    # Shared constants/equations and user input arrays
    K = PhysicsEqCst()
    inputs = SharedInputs(
        t_pulse_us=np.array([2, 4, 6, 8], dtype=float),
        currents_mA=np.array([1, 10, 50, 100, 150, 170, 200], dtype=float),
        rep_rate_Hz=np.array([0.5, 1, 2, 4, 10], dtype=float),
        energies_MeV=np.array([0.7, 1.2, 20, 35, 40, 45], dtype=float),
    )

    print("User inputs (tables):")
    print("t_pulse (μs):", inputs.t_pulse_us)
    print("currents (mA):", inputs.currents_mA)
    print("rep_rate (Hz):", inputs.rep_rate_Hz)
    print("energies (MeV):", inputs.energies_MeV)
    print()

    # ----- BeamPower demos (figures remain as before) -----
    #bp = BeamPower(K=K, inputs=inputs)
    #bp.chargePerMacropulse()  # uses linspace 0..Imax and all t_pulse values
    #I_subset_A = K.mA_to_A(np.array([50, 100, 150, 170, 200], dtype=float))
    #df_power = bp.getPowerDF(I_pulse_range=I_subset_A, T_pulse_values=inputs.t_pulse_s, rep_rate_values=inputs.rep_rate_Hz,
    #    E_energy_range=inputs.energies_MeV, plot_type='Power', penetration_depth=20e-3, plot=True, material_for_temp='Copper')

    # ----- BeamRadiation (ICS) quick report + plots -----
    #br = BeamRadiation(K=K)
    #rep = br.quick_report_and_plots(make_plots=True)
    #print("ICS summary:", rep)

    # ----- AlphaMagnet full suite -----
    am = AlphaMagnet(grad_per_amp_T_per_m=0.103754, nominal_current_A=17.5, K=K)
    am.plot_s_x_y_vs_energy_with_LAS(
        currents_A=(7.5, 10.0, 15.0, 17.5, 20.0),
        E_min_MeV=0.02, E_max_MeV=1.20, nE=400,
    )
    am.plot_alpha_compression_1343(currents_A=(10.0, 15.0, 17.5, 20.0),
                                   E_min_MeV = 0.05, E_max_MeV = 1.20, nE = 400, rf_freq_Hz = 2.856e9)
