import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm

class beamUtility:
    e = 1.602176634e-19  # Elementary charge (C) [CODATA 2018]
    MeV_to_J = 1.602176634e-13  # Conversion factor from MeV to Joules [CODATA 2018]
    NA = 6.02214076e23  # Avogadro's number
    epsilon_0 = 8.854187817e-12  # Vacuum permittivity (F/m)
    me = 9.10938356e-31  # Electron mass (kg)
    m_p = 1.67262192595e-27  # Proton Mass (kg)
    c = 299792458.0  # Speed of light (m/s)

    # Material properties (Density in kg/m^3, Specific Heat in J/kg*K, Stopping Power in MeV cm^2/g, heat capacity in J/g°C)
    # Source: National Institute of Standards and Technology (NIST) Stopping Power Data
    materials = {
    "Aluminum": {"density": 2700.0, "specific_heat": 900.0, "stopping_power": 2.7, "heat_capacity": 0.897,
                 "atomic_number": 13, "ionization_potential": 166, "atomic_mass": 26.98},
    "Copper": {"density": 8960.0, "specific_heat": 385.0, "stopping_power": 5.0, "heat_capacity": 0.385,
               "atomic_number": 29, "ionization_potential": 322, "atomic_mass": 63.55},
    "Stainless Steel": {"density": 7850.0, "specific_heat": 500.0, "stopping_power": 3.5, "heat_capacity": 0.500,
                        "atomic_number": 26, "ionization_potential": 233, "atomic_mass": 55.85},
    }

    # Convert stopping power to SI units (J/m)
    for material, props in materials.items():
        props["stopping_power"] *= (MeV_to_J * 1e6) / props["density"]  # Convert MeV cm^2/g to J/m

    PARTICLES = {"electron": [me, e, (me * c ** 2)],
                    "proton": [m_p, e, (m_p * c ** 2)]}
    

    def __init__(self, beam_type = "electron", sigma_x = 1e-3, sigma_y = 10e-3) -> None:
        self.beam_information = self.PARTICLES[beam_type]

        # Beam transverse size (6 sigma contains ~95% of beam)
        self.sigma_x = sigma_x # sigma_x : 1 mm
        self.sigma_y = sigma_y # sigma_y : 10 mm

    #f_bunch: Bunch frequency (Hz)
    def chargePerMacropulse(self, I_pulse_range: list, T_pulse_values: list, f_bunch = 2.856e9):
        line_styles = ['-', '--', '-.']

        # Plotting beam properties
        fig, ax1 = plt.subplots(figsize=(10, 5))

        for idx, T_pulse in enumerate(T_pulse_values):
            Q_macropulse = np.zeros(len(I_pulse_range))
            Q_bunch = np.zeros(len(I_pulse_range))

            # Compute charge per macropulse (pC/macropulse) and charge per bunch (pC/bunch)
            bunches_per_macropulse = f_bunch * T_pulse
            for i, I_pulse in enumerate(I_pulse_range):
                Q_macropulse[i] = (I_pulse * T_pulse) * 1e12  # Charge per macropulse (pC)
                nb_bunch_per_pulse = f_bunch * T_pulse
                Q_bunch[i] = (I_pulse / f_bunch) * 1e12  # Charge per bunch (pC)

            # Plot charge per bunch
            # ax1.plot(I_pulse_range * 1e3, Q_bunch, linestyle=line_styles[idx], color='k', label=f'Charge per Bunch ({T_pulse * 1e6} us)')

            # Plot charge per macropulse
            ax1.plot(I_pulse_range * 1e3, Q_macropulse, alpha=1.0, linestyle=line_styles[idx], color='k', label=f'Charge per Macropulse ({T_pulse * 1e6} us)')

        ax1.set_xlabel("Macropulse Current (mA)")
        ax1.set_ylabel("Charge per Macropulse (pC)", color='k')
        ax1.tick_params(axis='y', labelcolor='k')
        ax1.grid()
        ax1.legend(loc='upper left')

        fig.suptitle("Charge per Macropulse vs Beam Current for Different Pulse Durations")
        plt.show()

    # penetration_depch : 1 mm
    def getPowerDF(self, I_pulse_range: np.array, T_pulse_values: np.array, rep_rate_values: np.array,
                   E_energy_range: np.array, plot_type = "Power", penetration_depth = 20e-3, plot = True):
        sigma_x = self.sigma_x
        sigma_y = self.sigma_y
        colors = ['lightskyblue', 'goldenrod', 'black', 'lightcoral']
        linestyles = ['-', '-', '--', '-']

        beam_area = np.pi * (6 * sigma_x) * (6 * sigma_y)  # Elliptical cross-section
        beam_volume = beam_area * penetration_depth  # Volume in m³
        beam_volume_cm3 = beam_volume * 1e6  # Convert m³ to cm³

        # Compute power deposition and temperature rise
        power_results = []

        for E in E_energy_range:
            for r in rep_rate_values:
                for T_pulse in T_pulse_values:
                    for I_pulse in I_pulse_range:
                        Q_macropulse = I_pulse * (T_pulse * 1e-6)  # Convert us to s
                        N_electrons = Q_macropulse / self.e  # Number of electrons per macropulse
                        E_pulse = N_electrons * (E * self.MeV_to_J)  # Energy per macropulse (J)
                        P_beam = E_pulse * r  # Power deposited (W)

                        temp_rise = {}
                        for material, props in self.materials.items():
                            mass = beam_volume_cm3 * props["density"] / 1000  # Mass in g
                            temp_rise[material] = P_beam / (mass * props["heat_capacity"])  # °C/s

                        power_results.append([E, I_pulse * 1e3, r, T_pulse, P_beam,
                                            temp_rise["Copper"], temp_rise["Aluminum"], temp_rise["Stainless Steel"]])

        columns = ["Energy (MeV)", "Beam Current (mA)", "Repetition Rate (Hz)", "Pulse Duration (us)", "Power (W)",
                "Temp Rise Copper (C/s)", "Temp Rise Aluminum (C/s)", "Temp Rise Stainless Steel (C/s)"]
        df_power = pd.DataFrame(power_results, columns=columns)

        if plot:
            h_size = rep_rate_values.size
            v_size = T_pulse_values.size
            fig, axes = plt.subplots(h_size, v_size, figsize=(12, 12), sharex=True)

            for i, r in enumerate(rep_rate_values[::-1]):
                max_y = 0
                for j, T_pulse in enumerate(T_pulse_values[::-1]):
                    ax = axes[i, j]
                    subset = df_power[(df_power["Repetition Rate (Hz)"] == r) & (df_power["Pulse Duration (us)"] == T_pulse)]
                    if not subset.empty:
                        for k, I_pulse in enumerate(I_pulse_range):
                            data = subset[subset["Beam Current (mA)"] == I_pulse * 1e3]
                            y_data = data["Power (W)"] if plot_type == "Power" else data[f"Temp Rise {material} (C/s)"]
                            ax.plot(data["Energy (MeV)"], y_data, linestyle=linestyles[k], color=colors[k])
                            max_y = max(max_y, y_data.max())
                    ax.set_title(f"{r} Hz, {T_pulse} us", fontsize=8)
                    ax.grid()
                for j in range(v_size):  # Apply independent y-scale for the row
                    axes[i, j].set_ylim(0, max_y * 1.1)

            ylabel = "Power (W)" if plot_type == "Power" else f"Temperature Rise ({material}) (C/s)"
            fig.text(0.5, 0.0, "Beam Energy (MeV)", ha='center', fontsize=12)
            fig.text(0., 0.5, ylabel, va='center', rotation='vertical', fontsize=12)
            fig.suptitle(f"Beam {plot_type}", fontsize=14)
            fig.legend(labels=[f"{I*1e3} mA" for I in I_pulse_range], loc='upper right', ncol=4)
            plt.tight_layout()
            plt.show()

        return df_power
    
        # Function to compute penetration depth using Grunn model
    def model_Grunn(self, material, E_energy_range):
        rho = self.materials[material]["density"]/1000  # Density in g/cm³
        results = [[material, E, (0.1 * E**1.5) / rho] for E in E_energy_range]
        return pd.DataFrame(results, columns=["Material", "Energy (MeV)", "Penetration Depth (cm)"])

    # Function to compute penetration depth and stopping power using Bethe model
    def model_Bethe(self, material, E_energy_range):
        props = self.materials[material]
        rho = props["density"]/1000  # Density in g/cm³
        Z = props["atomic_number"]
        A = props["atomic_mass"]  # Atomic mass in g/mol
        I = props["ionization_potential"] * self.e  # Convert eV to Joules
        n_e = (self.NA * rho / A) * Z * 1e6  # Convert electrons/cm³ to electrons/m³

        results = []

        for E in E_energy_range:
            E_J = E * self.MeV_to_J  # Convert energy to Joules
            gamma = 1 + (E_J / (self.me * self.c**2))
            beta = np.sqrt(1 - (1 / gamma**2))

            log_term = (2 * self.me * self.c**2 * beta**2) / I
            log_term = max(log_term, 1e-6)  # Avoid log errors

            stopping_power = (4 * np.pi * self.e**3 * Z * n_e) / (self.me * self.c**2 * beta**2) * np.log(log_term) / 100  # MeV/cm

            R = E_J / (stopping_power) if stopping_power > 0 else 0  # Penetration depth in cm

            stopping_power = stopping_power / self.MeV_to_J / 10

            results.append([material, E, max(R, 0), stopping_power])

        return pd.DataFrame(results, columns=["Material", "Energy (MeV)", "Penetration Depth (cm)", "Stopping Power (MeV/mm)"])

    # Function to compute electron deposition profile
    def compute_deposition_profile(self, energy, material):
        df_bethe = self.model_Bethe(material)
        closest_energy_idx = (df_bethe["Energy (MeV)"].sub(energy)).abs().idxmin()
        R = df_bethe.loc[closest_energy_idx, "Penetration Depth (cm)"]

        x_range = np.linspace(0, R + 2, 100)
        sigma = R * 0.2  # Assuming a spread of 20% around the penetration depth
        deposition_profile = norm.pdf(x_range, R, sigma)
        deposition_profile /= np.max(deposition_profile)  # Normalize

        return x_range, deposition_profile
    
    # Function to plot electron deposition profile
    def plot_deposition_profile(self, energy, material):
        x_range, deposition_profile = self.compute_deposition_profile(energy, material)
        plt.figure(figsize=(8, 5))
        plt.plot(x_range, deposition_profile, label=f"{energy} MeV in {material}", color='green')
        plt.xlabel("Depth (cm)")
        plt.ylabel("Relative Deposition Intensity")
        plt.title(f"Electron Deposition Profile in {material}")
        plt.legend()
        plt.grid()
        plt.show()

    # Function to plot penetration depth
    def plot_penetration_depth(self, material, df_grunn = None, df_bethe = None, E_energy_range = np.logspace(-1, 2, 100)):
        df_grunn = self.model_Grunn(material,E_energy_range) if df_grunn is None else df_grunn
        df_bethe = self.model_Bethe(material, E_energy_range) if df_bethe is None else df_bethe
        plt.figure(figsize=(8, 5))
        plt.xscale("log")
        plt.plot(df_grunn["Energy (MeV)"], df_grunn["Penetration Depth (cm)"], label=f"{material} - Grunn", linestyle='--', color='blue')
        plt.plot(df_bethe["Energy (MeV)"], df_bethe["Penetration Depth (cm)"], label=f"{material} - Bethe", linestyle='-', color='red')
        plt.xlabel("Beam Energy (MeV)")
        plt.ylabel("Penetration Depth (cm)")
        plt.title(f"Penetration Depth in {material}")
        plt.legend()
        plt.grid()
        plt.show()

    # Function to plot stopping power
    def plot_stopping_power(df, material):
        plt.figure(figsize=(8, 5))
        plt.xscale("log")
        plt.yscale("log")
        plt.plot(df["Energy (MeV)"], df["Stopping Power (MeV/mm)"], label=f"{material} - Bethe", linestyle='-', color='green')
        plt.xlabel("Beam Energy (MeV)")
        plt.ylabel("Stopping Power (MeV/mm)")
        plt.title(f"Stopping Power in {material}")
        plt.legend()
        plt.grid(True, which="both", linestyle='--', linewidth=0.5)
        plt.show()

if __name__ == "__main__":    
    obj = beamUtility()
    obj.chargePerMacropulse(np.linspace(0, 200e-3, 50), [2.0e-6, 5.0e-6, 8.0e-6])
    obj.getPowerDF(np.array([50e-3, 100e-3, 170e-3, 200e-3]), np.array([4.0, 6.0]),
                       rep_rate_values = np.array([1, 2]), E_energy_range = np.linspace(0, 50, 100),
                       plot_type = "Temperature")
    for material in obj.materials.keys():
        obj.plot_penetration_depth(material)
