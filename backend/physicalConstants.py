"""
Physical constants and particle properties for beam physics simulations.
Author: Eremey Valetov
Uses CODATA 2018 values for consistency and precision.
References:
NIST CODATA 2018: https://physics.nist.gov/cuu/Constants/
"""

from typing import Dict, Tuple, TypedDict, Optional


class ParticleProperties(TypedDict):
    """Type definition for particle properties dictionary."""
    mass: float
    charge: float
    rest_energy: float


class PhysicalConstants:
    """
    Physical constants (CODATA 2018 values) and particle properties.

    All constants use SI units unless otherwise specified.

    Attributes:
        Q (float): Elementary charge (C) [exact since 2019 SI]
        C (float): Speed of light (m/s) [exact by definition]
        h (float): Planck constant (J·s) [exact since 2019 SI]
        epsilon_0 (float): Vacuum permittivity (F/m)
        NA (float): Avogadro's number [exact since 2019 SI]
        M_e (float): Electron mass (kg)
        M_p (float): Proton mass (kg)
        M_AMU (float): Atomic mass unit (kg)
        E0_electron (float): Electron rest energy (MeV)
        E0_proton (float): Proton rest energy (MeV)
        MeV_to_J (float): MeV to Joules conversion [exact]
        f_RF_default (float): Default RF frequency (Hz)
        G_quad_default (float): Default quadrupole gradient (T/A/m)

    Examples:
        >>> props = PhysicalConstants.get_particle_properties()
        >>> electron_mass = props["electron"]["mass"]
        >>> print(f"Electron mass: {electron_mass:.6e} kg")
        Electron mass: 9.109384e-31 kg

        >>> gamma, beta = PhysicalConstants.relativistic_parameters(45.0, 0.51099895000)
        >>> print(f"γ = {gamma:.4f}, β = {beta:.6f}")
        γ = 89.0628, β = 0.999937
    """

    # =========================================================================
    # Fundamental Constants (CODATA 2018)
    # =========================================================================

    Q = 1.602176634e-19  # Elementary charge (C) [exact since 2019 SI redefinition]
    C = 299792458  # Speed of light (m/s) [exact by definition]
    h = 6.62607015e-34  # Planck constant (J·s) [exact since 2019 SI redefinition]
    epsilon_0 = 8.854187817e-12  # Vacuum permittivity (F/m)
    NA = 6.02214076e23  # Avogadro's number [exact since 2019 SI redefinition]

    # =========================================================================
    # Particle Masses (CODATA 2018)
    # =========================================================================

    M_e = 9.1093837015e-31  # Electron mass (kg)
    M_p = 1.67262192369e-27  # Proton mass (kg)
    M_AMU = 1.66053906660e-27  # Atomic mass unit (kg)

    # =========================================================================
    # Rest Energies (MeV) - CODATA 2018
    # =========================================================================

    E0_electron = 0.51099895000  # Electron rest energy (MeV)
    E0_proton = 938.27208816  # Proton rest energy (MeV)

    # =========================================================================
    # Conversion Factors
    # =========================================================================

    MeV_to_J = 1.602176634e-13  # MeV to Joules [exact]
    J_to_MeV = 1.0 / MeV_to_J  # Joules to MeV

    # =========================================================================
    # Accelerator-Specific Constants (UH FEL)
    # =========================================================================

    f_RF_default = 2856e6  # Default RF frequency (Hz)
    G_quad_default = 2.694  # Default quadrupole gradient (T/A/m)

    @classmethod
    def get_particle_properties(cls) -> Dict[str, ParticleProperties]:
        """
        Get particle properties as nested dictionaries (recommended interface).

        Returns:
            dict: Nested dictionary structure:
                {
                    "particle_name": {
                        "mass": float (kg),
                        "charge": float (C),
                        "rest_energy": float (MeV)
                    }
                }

        Examples:
            >>> props = PhysicalConstants.get_particle_properties()
            >>> electron = props["electron"]
            >>> print(f"Mass: {electron['mass']:.6e} kg")
            Mass: 9.109384e-31 kg
            >>> print(f"Charge: {electron['charge']:.6e} C")
            Charge: 1.602177e-19 C
            >>> print(f"Rest energy: {electron['rest_energy']:.9f} MeV")
            Rest energy: 0.510998950 MeV
        """
        return {
            "electron": {
                "mass": cls.M_e,
                "charge": cls.Q,
                "rest_energy": cls.E0_electron
            },
            "proton": {
                "mass": cls.M_p,
                "charge": cls.Q,
                "rest_energy": cls.E0_proton
            }
        }

    @classmethod
    def get_particle_properties_legacy(cls) -> Dict[str, list]:
        """
        Get particle properties in legacy list format [mass, charge, rest_energy].

        Provided for backwards compatibility with existing FELsim code.
        New code should use get_particle_properties() instead.

        Returns:
            dict: {particle_name: [mass_kg, charge_C, rest_energy_MeV]}

        Examples:
            >>> props = PhysicalConstants.get_particle_properties_legacy()
            >>> mass, charge, rest_energy = props["electron"]
            >>> print(f"Mass: {mass:.6e} kg")
            Mass: 9.109384e-31 kg
        """
        return {
            "electron": [cls.M_e, cls.Q, cls.E0_electron],
            "proton": [cls.M_p, cls.Q, cls.E0_proton]
        }

    @classmethod
    def get_particle(cls, particle_name: str) -> ParticleProperties:
        """
        Get properties for a specific particle.

        Args:
            particle_name: Name of particle ("electron" or "proton")

        Returns:
            dict: Particle properties {"mass": ..., "charge": ..., "rest_energy": ...}

        Raises:
            KeyError: If particle_name is not recognized

        Examples:
            >>> electron = PhysicalConstants.get_particle("electron")
            >>> print(electron["mass"])
            9.1093837015e-31
        """
        particles = cls.get_particle_properties()
        if particle_name not in particles:
            available = ", ".join(particles.keys())
            raise KeyError(f"Unknown particle: {particle_name}. "
                           f"Available particles: {available}")
        return particles[particle_name]

    @classmethod
    def compute_rest_energy(cls, mass_kg: float) -> float:
        """
        Compute rest energy from mass using E₀ = m₀c².

        Args:
            mass_kg: Mass in kilograms

        Returns:
            float: Rest energy in MeV

        Examples:
            >>> # Compute electron rest energy from mass
            >>> E0 = PhysicalConstants.compute_rest_energy(9.10938356e-31)
            >>> print(f"E₀ = {E0:.9f} MeV")
            E₀ = 0.510998942 MeV
        """
        energy_J = mass_kg * cls.C ** 2
        return energy_J * cls.J_to_MeV

    @classmethod
    def relativistic_parameters(cls, kinetic_energy_MeV: float,
                                rest_energy_MeV: float) -> Tuple[float, float]:
        """
        Calculate relativistic γ (gamma) and β (beta) factors.

        Definitions:
            γ = 1 + KE/E₀
            β = √(1 - 1/γ²) = v/c

        Args:
            kinetic_energy_MeV: Kinetic energy (MeV)
            rest_energy_MeV: Rest energy (MeV)

        Returns:
            tuple: (gamma, beta)
                - gamma: Lorentz factor (dimensionless)
                - beta: Velocity factor v/c (dimensionless, 0 < β < 1)

        Examples:
            >>> # 45 MeV electrons (using CODATA 2018 E₀ = 0.51099895000 MeV)
            >>> gamma, beta = PhysicalConstants.relativistic_parameters(45.0, 0.51099895000)
            >>> print(f"γ = {gamma:.4f}, β = {beta:.6f}")
            γ = 89.0628, β = 0.999937

            >>> # Using the class constant directly
            >>> gamma, beta = PhysicalConstants.relativistic_parameters(45.0, PhysicalConstants.E0_electron)
            >>> print(f"γ = {gamma:.4f}, β = {beta:.6f}")
            γ = 89.0628, β = 0.999937

            >>> # 100 MeV protons
            >>> gamma, beta = PhysicalConstants.relativistic_parameters(100.0, 938.27208816)
            >>> print(f"γ = {gamma:.6f}, β = {beta:.6f}")
            γ = 1.106579, β = 0.428195
        """
        gamma = 1.0 + (kinetic_energy_MeV / rest_energy_MeV)
        beta = (1.0 - (1.0 / gamma ** 2)) ** 0.5
        return gamma, beta

    @classmethod
    def momentum(cls, kinetic_energy_MeV: float,
                 rest_energy_MeV: float) -> float:
        """
        Calculate relativistic momentum pc.

        From relativistic energy-momentum relation:
            E² = (pc)² + (m₀c²)²

        For particle with kinetic energy KE:
            E = KE + E₀
            (pc)² = E² - E₀² = (KE + E₀)² - E₀²
                  = KE² + 2·KE·E₀
            pc = √(KE² + 2·KE·E₀)

        Args:
            kinetic_energy_MeV: Kinetic energy (MeV)
            rest_energy_MeV: Rest energy (MeV)

        Returns:
            float: Momentum × c in MeV

        Examples:
            >>> # 45 MeV electron momentum
            >>> pc = PhysicalConstants.momentum(45.0, 0.511)
            >>> print(f"pc = {pc:.3f} MeV")
            pc = 45.508 MeV

            >>> # Verify: for ultra-relativistic particles, pc ≈ KE
            >>> print(f"pc/KE = {pc/45.0:.6f}")
            pc/KE = 1.011292
        """
        return (kinetic_energy_MeV ** 2 + 2 * kinetic_energy_MeV * rest_energy_MeV) ** 0.5

    @classmethod
    def compute_isotope_properties(cls, mass_number: int,
                                   ion_charge: int) -> ParticleProperties:
        """
        Compute properties for an arbitrary ion (isotope with charge state).

        Args:
            mass_number: Atomic mass number A (number of nucleons)
            ion_charge: Ion charge state Z (e.g., 5 for C¹²⁺⁵)

        Returns:
            dict: Particle properties {"mass": kg, "charge": C, "rest_energy": MeV}

        Examples:
            >>> # Carbon-12 with 5+ charge state (C¹²⁺⁵)
            >>> carbon = PhysicalConstants.compute_isotope_properties(12, 5)
            >>> print(f"Mass: {carbon['mass']:.6e} kg")
            Mass: 1.992647e-26 kg
            >>> print(f"Charge: {carbon['charge']:.6e} C")
            Charge: 8.010883e-19 C
            >>> print(f"Rest energy: {carbon['rest_energy']:.3f} MeV")
            Rest energy: 11177.929 MeV
        """
        mass_kg = mass_number * cls.M_AMU
        charge_C = ion_charge * cls.Q
        rest_energy_MeV = cls.compute_rest_energy(mass_kg)

        return {
            "mass": mass_kg,
            "charge": charge_C,
            "rest_energy": rest_energy_MeV
        }

    @classmethod
    def parse_particle_specification(cls, particle_spec: str) -> ParticleProperties:
        """
        Parse particle specification string and return properties.

        Supported formats:
            - "electron" or "proton" - predefined particles
            - "A,Z" - isotope format (e.g., "12,5" for C¹²⁺⁵)

        Args:
            particle_spec: Particle specification string

        Returns:
            dict: Particle properties {"mass": kg, "charge": C, "rest_energy": MeV}

        Raises:
            ValueError: If particle_spec format is invalid
            KeyError: If predefined particle name is unknown

        Examples:
            >>> # Predefined particle
            >>> electron = PhysicalConstants.parse_particle_specification("electron")
            >>> print(electron["rest_energy"])
            0.51099895

            >>> # Custom isotope
            >>> carbon = PhysicalConstants.parse_particle_specification("12,5")
            >>> print(f"C¹²⁺⁵ rest energy: {carbon['rest_energy']:.1f} MeV")
            C¹²⁺⁵ rest energy: 11177.9 MeV
        """
        # Try predefined particles first
        particles = cls.get_particle_properties()
        if particle_spec in particles:
            return particles[particle_spec]

        # Try isotope format: "A,Z"
        try:
            parts = particle_spec.split(",")
            if len(parts) != 2:
                raise ValueError("Isotope format must be 'A,Z'")

            mass_number = int(parts[0].strip())
            ion_charge = int(parts[1].strip())

            if mass_number <= 0:
                raise ValueError(f"Mass number must be positive, got {mass_number}")
            if ion_charge <= 0:
                raise ValueError(f"Ion charge must be positive, got {ion_charge}")

            return cls.compute_isotope_properties(mass_number, ion_charge)

        except (ValueError, IndexError) as e:
            available = ", ".join(particles.keys())
            raise ValueError(
                f"Invalid particle specification: '{particle_spec}'. "
                f"Use predefined particle name ({available}) or isotope format 'A,Z' "
                f"(e.g., '12,5' for C¹²⁺⁵). Error: {e}"
            )


# ============================================================================
# Convenience functions for common operations
# ============================================================================

def get_electron() -> ParticleProperties:
    """Get electron properties (convenience function)."""
    return PhysicalConstants.get_particle("electron")


def get_proton() -> ParticleProperties:
    """Get proton properties (convenience function)."""
    return PhysicalConstants.get_particle("proton")


# ============================================================================
# Usage examples and validation
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Physical Constants Module - CODATA 2018")
    print("=" * 70)

    # Test nested dict interface
    print("\n1. Nested Dict Interface (Recommended):")
    props = PhysicalConstants.get_particle_properties()
    for particle_name, particle_props in props.items():
        print(f"\n{particle_name.capitalize()}:")
        print(f"  Mass:        {particle_props['mass']:.10e} kg")
        print(f"  Charge:      {particle_props['charge']:.10e} C")
        print(f"  Rest energy: {particle_props['rest_energy']:.10f} MeV")

    # Test legacy interface
    print("\n2. Legacy List Interface (Backwards Compatibility):")
    legacy_props = PhysicalConstants.get_particle_properties_legacy()
    mass, charge, rest_energy = legacy_props["electron"]
    print(f"Electron: mass={mass:.6e}, charge={charge:.6e}, E₀={rest_energy:.6f}")

    # Test relativistic parameters
    print("\n3. Relativistic Parameters:")
    KE = 45.0  # MeV
    electron = get_electron()
    gamma, beta = PhysicalConstants.relativistic_parameters(KE, electron["rest_energy"])
    pc = PhysicalConstants.momentum(KE, electron["rest_energy"])
    print(f"45 MeV electrons:")
    print(f"  γ = {gamma:.6f}")
    print(f"  β = {beta:.6f}")
    print(f"  pc = {pc:.3f} MeV")

    # Test isotope computation
    print("\n4. Custom Isotope (C¹²⁺⁵):")
    carbon = PhysicalConstants.compute_isotope_properties(12, 5)
    print(f"  Mass:        {carbon['mass']:.10e} kg")
    print(f"  Charge:      {carbon['charge']:.10e} C")
    print(f"  Rest energy: {carbon['rest_energy']:.3f} MeV")

    # Test parser
    print("\n5. Particle Specification Parser:")
    for spec in ["electron", "proton", "12,5"]:
        particle = PhysicalConstants.parse_particle_specification(spec)
        print(f"  '{spec}': E₀ = {particle['rest_energy']:.3f} MeV")

    # Validate rest energy computation
    print("\n6. Validation - Rest Energy Computation:")
    computed_E0 = PhysicalConstants.compute_rest_energy(PhysicalConstants.M_e)
    expected_E0 = PhysicalConstants.E0_electron
    error = abs(computed_E0 - expected_E0)
    print(f"  Computed: {computed_E0:.10f} MeV")
    print(f"  Expected: {expected_E0:.10f} MeV")
    print(f"  Error:    {error:.2e} MeV")
    print(f"  Status:   {'✓ PASS' if error < 1e-9 else '✗ FAIL'}")

    print("\n" + "=" * 70)