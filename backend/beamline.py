#   Authors: Christian Komo, Niels Bidault
from sympy import symbols, Matrix
import sympy as sp
import numpy as np
from scipy import interpolate
from scipy import optimize
import math
from physicalConstants import PhysicalConstants


# IMPORTANT NOTES:
#  by default every beam type is an electron beam type
#  NOTE: default fringe fields for now is noted as [[x list], [y list]],
#  ASSUMES that the measurement begins at 0
#  ex. [[0.01,0.02,0.03,0.95,1],[1.6,0.7,0.2,0.01,1]]
#  getSymbolicMatrice() must use all sympy methods and functions, NOT numpy

class lattice:
    __slots__ = (
        'name', 'E', 'E0', 'Q', 'M', 'C', 'f', 'M_AMU', 'k_MeV', 'm_p',
        'PARTICLES', 'gamma', 'beta', 'unitsF', 'color', 'fringeType',
        'startPos', 'endPos', 'chromatic', 'aperture_x', 'aperture_y', 'length',
    )

    def __init__(self, length, fringeType=None, name=None):
        '''
        parent class for accelerator beamline segment object

        Parameters
        ----------
        length : float
            Sets the physical length of the beamline element in meters.
        fringeType :
        name : str, optional
            Element label (e.g. 'LQ2', 'F1QA').
        '''
        self.name = name
        self.E = 45  # Kinetic energy (MeV/c^2)
        self.E0 = PhysicalConstants.E0_electron  # Electron rest energy (MeV/c^2)
        self.Q = PhysicalConstants.Q  # (C)
        self.M = PhysicalConstants.M_e  # (kg)
        self.C = PhysicalConstants.C  # Speed of light (m/s)
        self.f = PhysicalConstants.f_RF_default  # RF frequency (Hz)
        self.M_AMU = PhysicalConstants.M_AMU  # Atomic mass unit (kg)
        self.k_MeV = 1e-6 / self.Q  # Conversion factor (MeV / J)
        self.m_p = PhysicalConstants.M_p  # Proton Mass (kg)
        self.PARTICLES = {"electron": [self.M, self.Q, (self.M * self.C ** 2) * self.k_MeV],
                          "proton": [self.m_p, self.Q, (self.m_p * self.C ** 2) * self.k_MeV]}
        self.gamma = (1 + (self.E / self.E0))
        self.beta = np.sqrt(1 - (1 / (self.gamma ** 2)))
        self.unitsF = 10 ** 6  # Units factor used for conversions from (keV) to (ns)
        self.color = 'none'  # Color of beamline element when graphed
        self.fringeType = fringeType  # Each segment has no magnetic fringe by default
        self.startPos = None
        self.endPos = None
        self.chromatic = False  # Per-particle momentum-dependent focusing
        self.aperture_x = None  # Half-aperture in x [mm], None = no cut
        self.aperture_y = None  # Half-aperture in y [mm], None = no cut
        if isinstance(length, (int, float)) and not math.isnan(length) and length > 0:
            self.length = length
        else:
            raise ValueError("Invalid Parameter: Please enter a positive length parameter")

    def setE(self, E):
        '''
        Sets the kinetic energy (E) of the particle and updates dependent relativistic factors.

        Parameters
        ----------
        E : float
            New kinetic energy value (MeV/c^2).
        '''
        if not isinstance(E, (int, float)) or math.isnan(E) or math.isinf(E):
            raise ValueError(f"Invalid kinetic energy: {E}")
        if E <= 0:
            raise ValueError(f"Kinetic energy must be positive, got {E} MeV")
        self.E = E
        self.gamma = (1 + (self.E / self.E0))
        self.beta = np.sqrt(1 - (1 / (self.gamma ** 2)))

    def setMQE(self, mass, charge, restE):
        '''
        Sets the mass, charge, and rest energy of the particle, and updates
        dependent relativistic factors.

        Parameters
        ----------
        mass : float
            The new mass of the particle in kg.
        charge : float
            The new charge of the particle in Coulombs.
        restE : float
            The new rest energy of the particle in MeV.
        '''
        if mass <= 0:
            raise ValueError(f"Particle mass must be positive, got {mass} kg")
        if restE <= 0:
            raise ValueError(f"Rest energy must be positive, got {restE} MeV")
        self.M = mass
        self.Q = charge
        self.E0 = restE
        self.gamma = (1 + (self.E / self.E0))
        self.beta = np.sqrt(1 - (1 / (self.gamma ** 2)))

    def changeBeamType(self, particleType, kineticE, beamSegments=None):
        '''
        Changes the type of particle being simulated (e.g., "electron", "proton", or isotope).
        Updates the mass, charge, rest energy, and kinetic energy for the current segment
        and optionally for a list of other beamline segments.

        Parameters
        ----------
        particleType : str
            The type of particle. Either a predefined string ("electron", "proton")
            or an isotope string in the format "(isotope number),(ion charge)" (e.g., "12,5" for C12 5+).
        kineticE : float
            The kinetic energy for the new particle type in MeV/c^2.
        beamSegments : list[lattice], optional
            A list of other beamline segment objects whose particle properties
            should also be updated.

        Returns
        -------
        list[lattice] or None
            If `beamSegments` is provided, returns the updated list of beam segments.
            Otherwise, returns None.

        Raises
        ------
        TypeError
            If the `particleType` is not recognized or in an invalid isotope format.
        '''
        try:
            particleData = self.PARTICLES[particleType]
            self.setMQE(particleData[0], particleData[1], particleData[2])
            self.setE(kineticE)
            if beamSegments is not None:
                for seg in beamSegments:
                    seg.setMQE(particleData[0], particleData[1], particleData[2])
                    seg.setE(kineticE)
                return beamSegments
        except KeyError:
            try:
                isotopeData = particleType.split(",")
                A = int(isotopeData[0])
                Z = int(isotopeData[1])
                m_i = A * self.M_AMU
                q_i = Z * self.Q
                meV = (m_i * self.C ** 2) * self.k_MeV
                self.setMQE(m_i, q_i, meV)
                self.setE(kineticE)
                if beamSegments is not None:
                    for seg in beamSegments:
                        seg.setMQE(m_i, q_i, meV)
                        seg.setE(kineticE)
                    return beamSegments
            except (KeyError, ValueError, TypeError):
                raise TypeError("Invalid particle type/isotope")

    def getSymbolicMatrice(self, **kwargs):
        '''
        Returns the transfer matrix for the beamline element.
        Uses pure NumPy for numeric=True, SymPy for symbolic analysis.

        Parameters
        ----------
        **kwargs : dict
            Additional parameters specific to the child class's matrix calculation.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in the child class.
        '''
        numeric = kwargs.get('numeric', False)
        if numeric:
            return self._compute_numeric_matrix(**kwargs)
        else:
            return self._compute_symbolic_matrix(**kwargs)

    def _compute_numeric_matrix(self, **kwargs):
        '''
        Pure NumPy implementation for numeric matrix computation.
        Must be implemented by child classes.
        '''
        raise NotImplementedError("_compute_numeric_matrix not defined in child class")

    def _compute_symbolic_matrix(self, **kwargs):
        '''
        SymPy implementation for symbolic matrix computation.
        Must be implemented by child classes.
        '''
        raise NotImplementedError("_compute_symbolic_matrix not defined in child class")

    def useMatrice(self, val, **kwargs):
        '''
        Simulates the movement of particles through the segment by
        applying the segment's transfer matrix with numeric parameters.
        Vectorized for performance.

        Parameters
        ----------
        val : np.ndarray or list
            A 2D array representing the particle states. Each row is a particle,
            and columns correspond to phase space coordinates (e.g., [x, x', y, y', z, z']).
        **kwargs : dict
            Other segment-specific numeric parameters (e.g., `length`, `current`)
            that might override the segment's default properties for this specific simulation.

        Returns
        -------
        list
            A 2D list where each inner list represents the transformed state of a particle
            after passing through the segment.
        '''
        mat = self._compute_numeric_matrix(**kwargs)
        particles = np.asarray(val, dtype=np.float64)
        return (mat @ particles.T).T

    def apply_aperture(self, particles):
        '''
        Remove particles that exceed the element's aperture.

        Parameters
        ----------
        particles : np.ndarray
            (N, 6) particle array in FELsim coordinates [x(mm), x', y(mm), y', c5, c6].

        Returns
        -------
        np.ndarray
            Surviving particles (may be smaller than input).
        '''
        if self.aperture_x is None and self.aperture_y is None:
            return particles
        particles = np.asarray(particles, dtype=np.float64)
        mask = np.ones(particles.shape[0], dtype=bool)
        if self.aperture_x is not None:
            mask &= np.abs(particles[:, 0]) <= self.aperture_x
        if self.aperture_y is not None:
            mask &= np.abs(particles[:, 2]) <= self.aperture_y
        return particles[mask]


class driftLattice(lattice):
    __slots__ = ()

    def useMatrice(self, val, **kwargs):
        if not self.chromatic:
            return super().useMatrice(val, **kwargs)
        particles = np.asarray(val, dtype=np.float64)
        l = kwargs.get('length', self.length)
        delta = particles[:, 5]
        gamma_p = (self.E * (1 + delta * 1e-3) + self.E0) / self.E0
        beta_p = np.sqrt(np.maximum(1 - 1 / gamma_p**2, 1e-30))
        M56 = -(l * self.f / (self.C * beta_p * gamma_p * (gamma_p + 1)))
        out = particles.copy()
        out[:, 0] = particles[:, 0] + l * particles[:, 1]
        out[:, 2] = particles[:, 2] + l * particles[:, 3]
        out[:, 4] = particles[:, 4] + M56 * particles[:, 5]
        return out

    def __init__(self, length: float, name=None):
        '''
        Represents a drift space (empty section) in the beamline.

        Parameters
        ----------
        length : float
            The length of the drift segment in meters.
        '''
        super().__init__(length, name=name)
        self.color = "white"

    def _compute_numeric_matrix(self, length=None, **kwargs):
        '''
        Pure NumPy implementation for drift space transfer matrix.

        Parameters
        ----------
        length : float, optional
            If provided, uses this length instead of self.length.

        Returns
        -------
        np.ndarray
            The 6x6 transfer matrix for the drift segment.
        '''
        l = self.length if length is None else length
        M56 = -(l * self.f / (self.C * self.beta * self.gamma * (self.gamma + 1)))
        mat = np.array([
            [1.0, l, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, l, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, M56],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)
        return mat

    def _compute_symbolic_matrix(self, length=None, **kwargs):
        '''
        SymPy implementation for drift space transfer matrix.

        Parameters
        ----------
        length : float or str, optional
            If string, creates symbolic variable. If float, uses numeric value.
            If None, uses self.length.

        Returns
        -------
        sympy.Matrix
            The 6x6 symbolic transfer matrix for the drift segment.
        '''
        if length is None:
            l = self.length
        else:
            if isinstance(length, str):
                l = symbols(length, real=True)
            else:
                l = length
        M56 = -(l * self.f / (self.C * self.beta * self.gamma * (self.gamma + 1)))
        mat = Matrix([
            [1, l, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, l, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, M56],
            [0, 0, 0, 0, 0, 1]
        ])
        return mat

    def __str__(self):
        return f"Drift beamline segment {self.length} m long"


class qpfLattice(lattice):
    __slots__ = ('current', 'G')
    BORE_RADIUS_MM = 13.5  # 27 mm bore / 2

    def __init__(self, current: float, length: float = 0.0889, fringeType='decay', name=None):
        '''
        Represents a quadrupole focusing magnet. This magnet focuses in the x plane
        and defocuses in the y plane

        Parameters
        ----------
        current : float
            The current supplied to the quadrupole in Amps.
        length : float, optional
            The effective length of the quadrupole magnet in meters.
        fringeType :
        '''
        super().__init__(length, fringeType, name=name)
        self.current = current
        self.color = "cornflowerblue"
        self.G = 2.694  # Quadrupole focusing strength (T/A/m)
        self.aperture_x = self.BORE_RADIUS_MM
        self.aperture_y = self.BORE_RADIUS_MM

    def _compute_numeric_matrix(self, length=None, current=None, **kwargs):
        '''
        Pure NumPy implementation for quadrupole focusing magnet transfer matrix.

        Parameters
        ----------
        length : float, optional
            If provided, uses this length instead of self.length.
        current : float, optional
            If provided, uses this current instead of self.current.

        Returns
        -------
        np.ndarray
            The 6x6 transfer matrix for the quadrupole focusing magnet.
        '''
        l = self.length if length is None else length
        I = self.current if current is None else current
        k = np.abs((self.Q * self.G * I) / (self.M * self.C * self.beta * self.gamma))
        theta = np.sqrt(k) * l
        M11 = np.cos(theta)
        M22 = M11
        M21 = -np.sqrt(k) * np.sin(theta)
        M33 = np.cosh(theta)
        M44 = M33
        M43 = np.sqrt(k) * np.sinh(theta)
        M56 = -(l * self.f / (self.C * self.beta * self.gamma * (self.gamma + 1)))
        if I == 0:
            M12 = l
            M34 = l
        else:
            M12 = np.sin(theta) / np.sqrt(k)
            M34 = np.sinh(theta) / np.sqrt(k)
        mat = np.array([
            [M11, M12, 0.0, 0.0, 0.0, 0.0],
            [M21, M22, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, M33, M34, 0.0, 0.0],
            [0.0, 0.0, M43, M44, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, M56],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)
        return mat

    def _compute_symbolic_matrix(self, length=None, current=None, **kwargs):
        '''
        SymPy implementation for quadrupole focusing magnet transfer matrix.

        Parameters
        ----------
        length : float or str, optional
            If string, creates symbolic variable. If None, uses self.length.
        current : float or str, optional
            If string, creates symbolic variable. If None, uses self.current.

        Returns
        -------
        sympy.Matrix
            The 6x6 symbolic transfer matrix for the quadrupole focusing magnet.
        '''
        if length is None:
            l = self.length
        else:
            if isinstance(length, str):
                l = symbols(length, real=True)
            else:
                l = length
        if current is None:
            I = self.current
        else:
            if isinstance(current, str):
                I = symbols(current, real=True)
            else:
                I = current
        k = sp.Abs((self.Q * self.G * I) / (self.M * self.C * self.beta * self.gamma))
        theta = sp.sqrt(k) * l
        M11 = sp.cos(theta)
        M21 = -(sp.sqrt(k)) * sp.sin(theta)
        M22 = sp.cos(theta)
        M33 = sp.cosh(theta)
        M43 = sp.sqrt(k) * sp.sinh(theta)
        M44 = sp.cosh(theta)
        M56 = -(l * self.f / (self.C * self.beta * self.gamma * (self.gamma + 1)))
        # Use numeric == 0 only for actual numbers; SymPy symbols use general formulas
        if not isinstance(I, sp.Basic) and I == 0:
            M12 = l
            M34 = l
        else:
            M34 = sp.sinh(theta) * (1 / sp.sqrt(k))
            M12 = sp.sin(theta) * (1 / sp.sqrt(k))
        mat = Matrix([
            [M11, M12, 0, 0, 0, 0],
            [M21, M22, 0, 0, 0, 0],
            [0, 0, M33, M34, 0, 0],
            [0, 0, M43, M44, 0, 0],
            [0, 0, 0, 0, 1, M56],
            [0, 0, 0, 0, 0, 1]
        ])
        return mat

    def useMatrice(self, val, **kwargs):
        if not self.chromatic:
            return super().useMatrice(val, **kwargs)
        particles = np.asarray(val, dtype=np.float64)
        l = kwargs.get('length', self.length)
        I = kwargs.get('current', self.current)
        if I == 0:
            return super().useMatrice(val, **kwargs)

        # Per-particle k: k ∝ 1/P (momentum-dependent focusing)
        delta = particles[:, 5]  # ΔK/K₀ × 10³
        bg0 = self.beta * self.gamma
        gamma_p = (self.E * (1 + delta * 1e-3) + self.E0) / self.E0
        bg_p = np.sqrt(np.maximum(gamma_p**2 - 1, 1e-30))
        k0 = np.abs(self.Q * self.G * I / (self.M * self.C * bg0))
        k = k0 * bg0 / bg_p

        sqrtk = np.sqrt(k)
        theta = sqrtk * l
        out = particles.copy()

        # x-plane: focusing (cos/sin)
        C, S = np.cos(theta), np.sin(theta)
        out[:, 0] = C * particles[:, 0] + (S / sqrtk) * particles[:, 1]
        out[:, 1] = -sqrtk * S * particles[:, 0] + C * particles[:, 1]

        # y-plane: defocusing (cosh/sinh)
        Ch, Sh = np.cosh(theta), np.sinh(theta)
        out[:, 2] = Ch * particles[:, 2] + (Sh / sqrtk) * particles[:, 3]
        out[:, 3] = sqrtk * Sh * particles[:, 2] + Ch * particles[:, 3]

        # Longitudinal: per-particle M56
        beta_p = bg_p / gamma_p
        M56 = -(l * self.f / (self.C * beta_p * gamma_p * (gamma_p + 1)))
        out[:, 4] = particles[:, 4] + M56 * particles[:, 5]
        return out

    def __str__(self):
        return f"QPF beamline segment {self.length} m long and a current of {self.current} amps"


class qpdLattice(lattice):
    __slots__ = ('current', 'G')
    BORE_RADIUS_MM = 13.5  # 27 mm bore / 2

    def __init__(self, current: float, length: float = 0.0889, fringeType='decay', name=None):
        '''
        Represents a quadrupole defocusing magnet. This magnet defocuses in the x plane
        and focuses in the y plane

        Parameters
        ----------
        current : float
            The current supplied to the quadrupole in Amps.
        length : float, optional
            The effective length of the quadrupole magnet in meters.
        fringeType :
        '''
        super().__init__(length, fringeType, name=name)
        self.current = current
        self.G = 2.694  # Quadrupole focusing strength (T/A/m)
        self.color = "lightcoral"
        self.aperture_x = self.BORE_RADIUS_MM
        self.aperture_y = self.BORE_RADIUS_MM

    def _compute_numeric_matrix(self, length=None, current=None, **kwargs):
        '''
        Pure NumPy implementation for quadrupole defocusing magnet transfer matrix.

        Parameters
        ----------
        length : float, optional
            If provided, uses this length instead of self.length.
        current : float, optional
            If provided, uses this current instead of self.current.

        Returns
        -------
        np.ndarray
            The 6x6 transfer matrix for the quadrupole defocusing magnet.
        '''
        l = self.length if length is None else length
        I = self.current if current is None else current
        k = np.abs((self.Q * self.G * I) / (self.M * self.C * self.beta * self.gamma))
        theta = np.sqrt(k) * l
        M11 = np.cosh(theta)
        M22 = M11
        M21 = np.sqrt(k) * np.sinh(theta)
        M33 = np.cos(theta)
        M44 = M33
        M43 = -np.sqrt(k) * np.sin(theta)
        M56 = -(l * self.f / (self.C * self.beta * self.gamma * (self.gamma + 1)))
        if I == 0:
            M12 = l
            M34 = l
        else:
            M34 = np.sin(theta) / np.sqrt(k)
            M12 = np.sinh(theta) / np.sqrt(k)
        mat = np.array([
            [M11, M12, 0.0, 0.0, 0.0, 0.0],
            [M21, M22, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, M33, M34, 0.0, 0.0],
            [0.0, 0.0, M43, M44, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, M56],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)
        return mat

    def _compute_symbolic_matrix(self, length=None, current=None, **kwargs):
        '''
        SymPy implementation for quadrupole defocusing magnet transfer matrix.

        Parameters
        ----------
        length : float or str, optional
            If string, creates symbolic variable. If None, uses self.length.
        current : float or str, optional
            If string, creates symbolic variable. If None, uses self.current.

        Returns
        -------
        sympy.Matrix
            The 6x6 symbolic transfer matrix for the quadrupole defocusing magnet.
        '''
        if length is None:
            l = self.length
        else:
            if isinstance(length, str):
                l = symbols(length, real=True)
            else:
                l = length
        if current is None:
            I = self.current
        else:
            if isinstance(current, str):
                I = symbols(current, real=True)
            else:
                I = current
        k = sp.Abs((self.Q * self.G * I) / (self.M * self.C * self.beta * self.gamma))
        theta = sp.sqrt(k) * l
        M11 = sp.cosh(theta)
        M21 = sp.sqrt(k) * sp.sinh(theta)
        M22 = sp.cosh(theta)
        M33 = sp.cos(theta)
        M43 = -(sp.sqrt(k)) * sp.sin(theta)
        M44 = sp.cos(theta)
        M56 = -l * self.f / (self.C * self.beta * self.gamma * (self.gamma + 1))
        # Use numeric == 0 only for actual numbers; SymPy symbols use general formulas
        if not isinstance(I, sp.Basic) and I == 0:
            M12 = l
            M34 = l
        else:
            M34 = sp.sin(theta) * (1 / sp.sqrt(k))
            M12 = sp.sinh(theta) * (1 / sp.sqrt(k))
        mat = Matrix([
            [M11, M12, 0, 0, 0, 0],
            [M21, M22, 0, 0, 0, 0],
            [0, 0, M33, M34, 0, 0],
            [0, 0, M43, M44, 0, 0],
            [0, 0, 0, 0, 1, M56],
            [0, 0, 0, 0, 0, 1]
        ])
        return mat

    def useMatrice(self, val, **kwargs):
        if not self.chromatic:
            return super().useMatrice(val, **kwargs)
        particles = np.asarray(val, dtype=np.float64)
        l = kwargs.get('length', self.length)
        I = kwargs.get('current', self.current)
        if I == 0:
            return super().useMatrice(val, **kwargs)

        # Per-particle k: k ∝ 1/P (momentum-dependent focusing)
        delta = particles[:, 5]  # ΔK/K₀ × 10³
        bg0 = self.beta * self.gamma
        gamma_p = (self.E * (1 + delta * 1e-3) + self.E0) / self.E0
        bg_p = np.sqrt(np.maximum(gamma_p**2 - 1, 1e-30))
        k0 = np.abs(self.Q * self.G * I / (self.M * self.C * bg0))
        k = k0 * bg0 / bg_p

        sqrtk = np.sqrt(k)
        theta = sqrtk * l
        out = particles.copy()

        # x-plane: defocusing (cosh/sinh)
        Ch, Sh = np.cosh(theta), np.sinh(theta)
        out[:, 0] = Ch * particles[:, 0] + (Sh / sqrtk) * particles[:, 1]
        out[:, 1] = sqrtk * Sh * particles[:, 0] + Ch * particles[:, 1]

        # y-plane: focusing (cos/sin)
        C, S = np.cos(theta), np.sin(theta)
        out[:, 2] = C * particles[:, 2] + (S / sqrtk) * particles[:, 3]
        out[:, 3] = -sqrtk * S * particles[:, 2] + C * particles[:, 3]

        # Longitudinal: per-particle M56
        beta_p = bg_p / gamma_p
        M56 = -(l * self.f / (self.C * beta_p * gamma_p * (gamma_p + 1)))
        out[:, 4] = particles[:, 4] + M56 * particles[:, 5]
        return out

    def __str__(self):
        return f"QPD beamline segment {self.length} m long and a current of {self.current} amps"


class dipole(lattice):
    __slots__ = ('angle',)

    def __init__(self, length: float = 0.0889, angle: float = 1.5, fringeType='decay',
                 pole_gap=None, name=None):
        '''
        Represents a dipole bending magnet, which bends the beam horizontally.

        Parameters
        ----------
        length : float, optional
            The effective length of the dipole magnet in meters
        angle : float, optional
            The bending angle of the dipole magnet in degrees.
        fringeType :
        pole_gap : float, optional
            Pole gap in meters. If provided, sets vertical aperture to ±gap/2.
        '''
        super().__init__(length, fringeType, name=name)
        self.color = "forestgreen"
        self.angle = angle
        if pole_gap is not None:
            self.aperture_y = pole_gap * 1000 / 2  # m → mm half-gap

    def _compute_numeric_matrix(self, length=None, angle=None, **kwargs):
        '''
        Pure NumPy implementation for horizontal dipole bending magnet transfer matrix.

        Parameters
        ----------
        length : float, optional
            If provided, uses this length instead of self.length.
        angle : float, optional
            If provided, uses this angle instead of self.angle (in degrees).

        Returns
        -------
        np.ndarray
            The 6x6 transfer matrix for the dipole magnet.
        '''
        l = self.length if length is None else length
        a = self.angle if angle is None else angle
        if a == 0:
            M56 = -(l * self.f / (self.C * self.beta * self.gamma * (self.gamma + 1)))
            return np.array([
                [1.0, l, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, l, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, M56],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            ], dtype=np.float64)
        by = (self.M * self.C * self.beta * self.gamma / self.Q) * (a * np.pi / 180 / self.length)
        rho = self.M * self.C * self.beta * self.gamma / (self.Q * by)
        theta = l / rho
        C = np.cos(theta)
        S = np.sin(theta)
        M16 = rho * (1 - C) * (self.gamma / (self.gamma + 1))
        M26 = S * (self.gamma / (self.gamma + 1))
        M51 = self.f * S / (self.beta * self.C)
        M52 = self.f * rho * (1 - C) / (self.beta * self.C)
        R56 = (l - rho * S) - l / self.gamma**2
        M56 = self.f * R56 * self.gamma / ((self.gamma + 1) * self.beta * self.C)
        mat = np.array([
            [C, rho * S, 0.0, 0.0, 0.0, M16],
            [-S / rho, C, 0.0, 0.0, 0.0, M26],
            [0.0, 0.0, 1.0, l, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [M51, M52, 0.0, 0.0, 1.0, M56],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)
        return mat

    def useMatrice(self, val, **kwargs):
        if not self.chromatic or self.angle == 0:
            return super().useMatrice(val, **kwargs)
        particles = np.asarray(val, dtype=np.float64)
        l = kwargs.get('length', self.length)
        a = self.angle

        # Reference bending (By uses full element length, not sub-step)
        By = (self.M * self.C * self.beta * self.gamma / self.Q) * (a * np.pi / 180 / self.length)
        rho0 = self.M * self.C * self.beta * self.gamma / (self.Q * By)

        # Per-particle momentum
        delta = particles[:, 5]  # ΔK/K₀ × 10³
        bg0 = self.beta * self.gamma
        gamma_p = (self.E * (1 + delta * 1e-3) + self.E0) / self.E0
        bg_p = np.sqrt(np.maximum(gamma_p**2 - 1, 1e-30))
        beta_p = bg_p / gamma_p

        rho = rho0 * bg_p / bg0  # ρ ∝ P
        theta = l / rho
        C_t = np.cos(theta)
        S_t = np.sin(theta)

        x, xp = particles[:, 0], particles[:, 1]
        y, yp = particles[:, 2], particles[:, 3]
        gfac = gamma_p / (gamma_p + 1)

        out = particles.copy()
        out[:, 0] = C_t * x + rho * S_t * xp + rho * (1 - C_t) * gfac * delta
        out[:, 1] = (-S_t / rho) * x + C_t * xp + S_t * gfac * delta
        out[:, 2] = y + l * yp  # y-plane: drift
        # out[:, 3] unchanged
        M51 = self.f * S_t / (beta_p * self.C)
        M52 = self.f * rho * (1 - C_t) / (beta_p * self.C)
        R56 = (l - rho * S_t) - l / gamma_p**2
        M56 = self.f * R56 * gamma_p / ((gamma_p + 1) * beta_p * self.C)
        out[:, 4] = M51 * x + M52 * xp + particles[:, 4] + M56 * delta
        return out

    def _compute_symbolic_matrix(self, length=None, angle=None, **kwargs):
        '''
        SymPy implementation for horizontal dipole bending magnet transfer matrix.

        Parameters
        ----------
        length : float or str, optional
            If string, creates symbolic variable. If None, uses self.length.
        angle : float or str, optional
            If string, creates symbolic variable. If None, uses self.angle.

        Returns
        -------
        sympy.Matrix
            The 6x6 symbolic transfer matrix for the dipole magnet.
        '''
        if length is None:
            l = self.length
        else:
            if isinstance(length, str):
                l = symbols(length, real=True)
            else:
                l = length
        if angle is None:
            a = self.angle
        else:
            if isinstance(angle, str):
                a = symbols(angle, real=True)
            else:
                a = angle
        by = (self.M * self.C * self.beta * self.gamma / self.Q) * (a * sp.pi / 180 / self.length)
        rho = self.M * self.C * self.beta * self.gamma / (self.Q * by)
        theta = l / rho
        C = sp.cos(theta)
        S = sp.sin(theta)
        M16 = rho * (1 - C) * (self.gamma / (self.gamma + 1))
        M26 = S * (self.gamma / (self.gamma + 1))
        M51 = self.f * S / (self.beta * self.C)
        M52 = self.f * rho * (1 - C) / (self.beta * self.C)
        R56 = (l - rho * S) - l / self.gamma**2
        M56 = self.f * R56 * self.gamma / ((self.gamma + 1) * self.beta * self.C)
        mat = Matrix([
            [C, rho * S, 0, 0, 0, M16],
            [-S / rho, C, 0, 0, 0, M26],
            [0, 0, 1, l, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [M51, M52, 0, 0, 1, M56],
            [0, 0, 0, 0, 0, 1]
        ])
        return mat

    def __str__(self):
        return f"Horizontal dipole magnet segment {self.length} m long (curvature) with an angle of {self.angle} degrees"


class dipole_wedge(lattice):
    __slots__ = ('angle', 'dipole_length', 'dipole_angle', 'pole_gap')

    def __init__(self, length, angle: float = 1, dipole_length: float = 0.0889, dipole_angle: float = 1.5,
                 pole_gap=0.014478, enge_fct=0, fringeType='decay', name=None):
        '''
        Represents a dipole magnet with wedge-shaped pole faces at its entrance and/or exit,
        which introduces a vertical focusing or defocusing effect. This class models the
        effect of these wedge angles, often found in spectrometer dipoles.

        Parameters
        ----------
        length : float
            The effective length of the wedge magnet segment in meters.
        angle : float, optional
            The wedge angle (half-angle) of the pole face in degrees. This angle
            contributes to the vertical focusing/defocusing.
        dipole_length : float, optional
            The physical length of the main dipole field region in meters.
            This is used to calculate the magnetic field strength based on the dipole_angle.
        dipole_angle : float, optional
            The total bending angle of the main dipole field region in degrees.
            Used to calculate the magnetic field strength.
        pole_gap : float, optional
            The gap between the dipole poles in meters. Used in the fringe field calculation.
        enge_fct : float, optional
            Placeholder for Enge function parameter, related to fringe field modeling.
        fringeType :
        '''
        super().__init__(length, fringeType, name=name)
        self.color = "lightgreen"
        self.angle = angle
        self.dipole_length = dipole_length
        self.dipole_angle = dipole_angle
        self.pole_gap = pole_gap
        if pole_gap > 0:
            self.aperture_y = pole_gap * 1000 / 2  # m → mm half-gap

    def _compute_numeric_matrix(self, length=None, angle=None, **kwargs):
        '''
        Pure NumPy implementation for dipole magnet with wedge pole faces transfer matrix.

        Parameters
        ----------
        length : float, optional
            If provided, uses this length instead of self.length.
        angle : float, optional
            If provided, uses this angle instead of self.angle (in degrees).

        Returns
        -------
        np.ndarray
            The 6x6 transfer matrix for the wedge dipole magnet.
        '''
        l = self.length if length is None else length
        a = self.angle if angle is None else angle
        dipole_angle = self.dipole_angle
        dipole_length = self.dipole_length
        # Edge kick uses |ρ|: direction depends on pole face geometry, not bending sign
        if abs(dipole_angle) < 1e-14:
            # Zero dipole angle → infinite ρ → zero edge kick (drift-like)
            R = np.inf
        else:
            R = dipole_length / (abs(dipole_angle) * np.pi / 180)
        # Edge kick uses the full wedge angle (thin-lens effect at pole face,
        # not distributed over the magnet length)
        eta = a * np.pi / 180
        Tx = np.tan(eta)
        # Fringe field correction (triangle model): K·g = le/6 (g cancels analytically)
        le = self.length
        h = 1.0 / R
        phi = (le / 6.0) * h * (1 + np.sin(eta) ** 2) / np.cos(eta)
        Ty = np.tan(eta - phi)
        M56 = -self.f * (l / (self.C * self.beta * self.gamma * (self.gamma + 1)))
        mat = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [Tx / R, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -Ty / R, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, M56],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)
        return mat

    def _compute_symbolic_matrix(self, length=None, angle=None, **kwargs):
        '''
        SymPy implementation for dipole magnet with wedge pole faces transfer matrix.

        Parameters
        ----------
        length : float or str, optional
            If string, creates symbolic variable. If None, uses self.length.
        angle : float or str, optional
            If string, creates symbolic variable. If None, uses self.angle.

        Returns
        -------
        sympy.Matrix
            The 6x6 symbolic transfer matrix for the wedge dipole magnet.
        '''
        if length is None:
            l = self.length
        else:
            if isinstance(length, str):
                l = symbols(length, real=True)
            else:
                l = length
        if angle is None:
            a = self.angle
        else:
            if isinstance(angle, str):
                a = symbols(angle, real=True)
            else:
                a = angle
        dipole_angle = self.dipole_angle
        dipole_length = self.dipole_length
        # Edge kick uses |ρ|: direction depends on pole face geometry, not bending sign
        R = dipole_length / (sp.Abs(dipole_angle) * sp.pi / 180)
        # Edge kick uses the full wedge angle (thin-lens, not distributed)
        eta = a * sp.pi / 180
        Tx = sp.tan(eta)
        le = self.length
        # Fringe field correction (triangle model): K·g = le/6 (g cancels analytically)
        h = 1 / R
        phi = sp.simplify((le / 6) * h * (1 + sp.sin(eta) ** 2) / sp.cos(eta))
        Ty = sp.tan(eta - phi)
        M56 = -self.f * (l / (self.C * self.beta * self.gamma * (self.gamma + 1)))
        mat = Matrix([
            [1, 0, 0, 0, 0, 0],
            [Tx / R, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, -Ty / R, 1, 0, 0],
            [0, 0, 0, 0, 1, M56],
            [0, 0, 0, 0, 0, 1]
        ])
        return mat

    def useMatrice(self, val, **kwargs):
        if not self.chromatic:
            return super().useMatrice(val, **kwargs)
        particles = np.asarray(val, dtype=np.float64)

        eta = np.radians(self.angle)
        if abs(self.dipole_angle) < 1e-14:
            return super().useMatrice(val, **kwargs)
        R0 = self.dipole_length / (abs(self.dipole_angle) * np.pi / 180)
        le = self.length

        # Per-particle bending radius: R ∝ P (magnetic rigidity)
        delta = particles[:, 5]  # ΔK/K₀ × 10³
        bg0 = self.beta * self.gamma
        gamma_p = (self.E * (1 + delta * 1e-3) + self.E0) / self.E0
        bg_p = np.sqrt(np.maximum(gamma_p**2 - 1, 1e-30))
        R = R0 * bg_p / bg0  # per-particle R

        Tx = np.tan(eta)
        # Fringe correction (triangle model): K·g = le/6 (g cancels analytically)
        h = 1.0 / R
        phi = (le / 6.0) * h * (1 + np.sin(eta)**2) / np.cos(eta)
        Ty = np.tan(eta - phi)

        out = particles.copy()
        out[:, 1] = particles[:, 1] + (Tx / R) * particles[:, 0]
        out[:, 3] = particles[:, 3] - (Ty / R) * particles[:, 2]

        # Longitudinal: per-particle M56
        beta_p = bg_p / gamma_p
        M56 = -(self.length * self.f / (self.C * beta_p * gamma_p * (gamma_p + 1)))
        out[:, 4] = particles[:, 4] + M56 * particles[:, 5]
        return out

    def __str__(self):
        return f"Horizontal wedge dipole magnet segment {self.length} m long (curvature) with an angle of {self.angle} degrees"


class alphaMagnetLattice(lattice):
    """Alpha magnet in the ideal linear midplane field B_y = g*x, hard edge.

    The first-order map is closed form. In the ideal field the magnet has no
    length scale of its own: the orbit is universal in units of 1/sqrt(k),
    k = g/(B*rho), so every dimensionless map coefficient is a pure number and
    the element is those numbers plus the orbit path length s = S_COEFF/sqrt(k):

        x  = -x0 - (s/2)*a0        exactly -I composed with a drift of s/2,
        a  =       -a0             and achromatic to first order,
        y  =  CC*y0 + UU*s*b0
        b  =  (VV/s)*y0 + CC*b0
        R56 = s*(1/gamma^2 - 1/2)  in momentum coordinates: -s/2 of path-length
                                   dispersion (s ~ sqrt(beta*gamma)) plus the
                                   ordinary velocity slip.

    CC, UU and VV are transcendental constants of the alpha magnet, the
    vertical-plane counterparts of the familiar 0.19165 and 0.07505. They come
    from the stock-COSY INFINITY element in test/transport/alpha_element.fox,
    checked there against an independent lab-frame integration.

    Entrance and exit coincide on the pole edge (the orbit is a closed loop),
    so the magnet occupies no straight-line space and `length` carries the orbit
    path length, which is what the COSY element advances SPOS by. The path
    length depends on the rigidity, so it is recomputed whenever the beam
    energy or particle changes.

    Parameters
    ----------
    current : float
        Coil current in Amps. Only the magnitude matters; the field sign is
        fixed by the geometry.
    gradient_per_amp : float, optional
        Midplane gradient calibration in T/m per A. Defaults to the UH alpha
        magnet value.
    name : str, optional
        Element label.
    """

    __slots__ = ('current', 'gradient_per_amp')

    S_COEFF = 4.642099440404    # s*sqrt(k)
    CC = -0.737113977807        # R33 = R44
    UU = 1.641111845033         # R34/s
    VV = -0.278264388319        # R43*s
    THETA_ALPHA_DEG = 40.70991  # entry angle from the inward normal
    G_PER_AMP = PhysicalConstants.G_alpha_default

    def __init__(self, current: float, gradient_per_amp: float = None, name=None):
        # The path length follows from the rigidity, which the base constructor
        # sets up, so the placeholder below is replaced by _sync_length().
        super().__init__(1.0, name=name)
        self.current = current
        self.gradient_per_amp = (self.G_PER_AMP if gradient_per_amp is None
                                 else gradient_per_amp)
        self.color = "darkorange"
        self._sync_length()

    @property
    def gradient(self):
        '''Midplane gradient magnitude in T/m.'''
        return abs(self.gradient_per_amp * self.current)

    def path_length(self, current=None):
        '''
        Orbit path length s = S_COEFF/sqrt(k), k = g/(B*rho), in meters.

        Parameters
        ----------
        current : float, optional
            If provided, uses this current instead of self.current.
        '''
        I = self.current if current is None else current
        g = abs(self.gradient_per_amp * I)
        if g <= 0:
            raise ValueError("Alpha magnet gradient must be non-zero")
        k = g * self.Q / (self.M * self.C * self.beta * self.gamma)
        return self.S_COEFF / np.sqrt(k)

    def _sync_length(self):
        self.length = self.path_length()

    def setE(self, E):
        super().setE(E)
        self._sync_length()

    def setMQE(self, mass, charge, restE):
        super().setMQE(mass, charge, restE)
        self._sync_length()

    def _compute_numeric_matrix(self, current=None, **kwargs):
        '''
        Pure NumPy implementation for the alpha magnet transfer matrix.

        Parameters
        ----------
        current : float, optional
            If provided, uses this current instead of self.current.

        Returns
        -------
        np.ndarray
            The 6x6 transfer matrix for the alpha magnet.
        '''
        s = self.path_length(current)
        R56 = -s * (1 / self.gamma ** 2 - 0.5)
        M56 = self.f * R56 * self.gamma / ((self.gamma + 1) * self.beta * self.C)
        mat = np.array([
            [-1.0, -0.5 * s, 0.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, self.CC, self.UU * s, 0.0, 0.0],
            [0.0, 0.0, self.VV / s, self.CC, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, M56],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)
        return mat

    def _compute_symbolic_matrix(self, current=None, **kwargs):
        '''
        SymPy implementation for the alpha magnet transfer matrix.

        Parameters
        ----------
        current : float or str, optional
            If string, creates symbolic variable. If None, uses self.current.

        Returns
        -------
        sympy.Matrix
            The 6x6 symbolic transfer matrix for the alpha magnet.
        '''
        if current is None:
            I = self.current
        else:
            if isinstance(current, str):
                I = symbols(current, real=True, nonzero=True)
            else:
                I = current
        k = sp.Abs(self.gradient_per_amp * I) * self.Q / (self.M * self.C * self.beta * self.gamma)
        s = self.S_COEFF / sp.sqrt(k)
        R56 = -s * (1 / self.gamma ** 2 - sp.Rational(1, 2))
        M56 = self.f * R56 * self.gamma / ((self.gamma + 1) * self.beta * self.C)
        mat = Matrix([
            [-1, -s / 2, 0, 0, 0, 0],
            [0, -1, 0, 0, 0, 0],
            [0, 0, self.CC, self.UU * s, 0, 0],
            [0, 0, self.VV / s, self.CC, 0, 0],
            [0, 0, 0, 0, 1, M56],
            [0, 0, 0, 0, 0, 1]
        ])
        return mat

    def __str__(self):
        return (f"Alpha magnet segment at {self.current} A "
                f"({self.gradient:.6f} T/m), orbit path length {self.length} m")


class rfCavityLattice(lattice):
    """RF accelerating cavity (lumped, TW, or SW).

    In FELsim's native tracking this element behaves as a drift: the
    relativistic energy gain is modeled only by downstream adapters
    (RF-Track, elegant). The element stores the parameters needed to
    reconstruct the cavity in those codes.

    Parameters
    ----------
    length : float
        Physical length in metres.
    frequency_hz : float
        RF frequency in Hz.
    phase_deg : float
        RF phase in degrees (adapter-specific convention).
    voltage_mv : float, optional
        Peak total voltage in MV. If omitted, derived from gradient * length.
    gradient_mv_per_m : float, optional
        Peak on-axis accelerating gradient in MV/m. If omitted, derived
        from voltage / length.
    structure_type : {'RFCA', 'TW', 'SW'}
        'RFCA' = lumped cavity, 'TW' = travelling wave (SLAC-style
        constant-gradient), 'SW' = standing wave.
    phase_advance_deg : float
        Cell-to-cell phase advance in degrees (TW/SW only). Default 120°
        (2π/3 mode).
    n_cells : float, optional
        Number of cells in the structure (TW/SW only). If omitted, the
        adapter derives it from length and phase advance assuming
        β_wave = 1.
    name : str, optional
        Element label.
    """

    __slots__ = (
        'frequency_hz', 'phase_deg', 'voltage_mv', 'gradient_mv_per_m',
        'structure_type', 'phase_advance_deg', 'n_cells',
    )

    def __init__(self, length, frequency_hz, phase_deg=0.0,
                 voltage_mv=None, gradient_mv_per_m=None,
                 structure_type='TW', phase_advance_deg=120.0,
                 n_cells=None, name=None):
        super().__init__(length, name=name)
        self.color = 'gold'
        self.frequency_hz = float(frequency_hz)
        self.phase_deg = float(phase_deg)
        stype = str(structure_type).upper()
        if stype not in ('RFCA', 'TW', 'SW'):
            raise ValueError(
                f"rfCavityLattice: structure_type must be 'RFCA', 'TW', or 'SW', got {structure_type!r}"
            )
        self.structure_type = stype
        self.phase_advance_deg = float(phase_advance_deg)
        self.n_cells = n_cells  # may be None; adapter resolves it

        if gradient_mv_per_m is not None:
            self.gradient_mv_per_m = float(gradient_mv_per_m)
            if voltage_mv is not None:
                self.voltage_mv = float(voltage_mv)
                expected_v = float(gradient_mv_per_m) * length
                if abs(float(voltage_mv) - expected_v) / max(abs(float(voltage_mv)), 1e-30) > 0.01:
                    import warnings
                    warnings.warn(
                        f"rfCavityLattice: voltage_mv={voltage_mv} inconsistent with "
                        f"gradient_mv_per_m={gradient_mv_per_m} * L={length} = {expected_v:.4f} MV"
                    )
            else:
                self.voltage_mv = float(gradient_mv_per_m) * length
        elif voltage_mv is not None:
            self.voltage_mv = float(voltage_mv)
            if length <= 0:
                raise ValueError("rfCavityLattice: cannot derive gradient from voltage with zero length")
            self.gradient_mv_per_m = float(voltage_mv) / length
        else:
            raise ValueError(
                "rfCavityLattice: provide voltage_mv or gradient_mv_per_m"
            )

    def useMatrice(self, val, **kwargs):
        if not self.chromatic:
            return super().useMatrice(val, **kwargs)
        particles = np.asarray(val, dtype=np.float64)
        l = kwargs.get('length', self.length)
        delta = particles[:, 5]
        gamma_p = (self.E * (1 + delta * 1e-3) + self.E0) / self.E0
        beta_p = np.sqrt(np.maximum(1 - 1 / gamma_p**2, 1e-30))
        M56 = -(l * self.f / (self.C * beta_p * gamma_p * (gamma_p + 1)))
        out = particles.copy()
        out[:, 0] = particles[:, 0] + l * particles[:, 1]
        out[:, 2] = particles[:, 2] + l * particles[:, 3]
        out[:, 4] = particles[:, 4] + M56 * particles[:, 5]
        return out

    def _compute_numeric_matrix(self, length=None, **kwargs):
        l = self.length if length is None else length
        M56 = -(l * self.f / (self.C * self.beta * self.gamma * (self.gamma + 1)))
        return np.array([
            [1.0, l,   0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, l,   0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, M56],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)

    def _compute_symbolic_matrix(self, length=None, **kwargs):
        if length is None:
            l = self.length
        else:
            if isinstance(length, str):
                l = symbols(length, real=True)
            else:
                l = length
        M56 = -(l * self.f / (self.C * self.beta * self.gamma * (self.gamma + 1)))
        return Matrix([
            [1, l, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, l, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, M56],
            [0, 0, 0, 0, 0, 1],
        ])

    def __str__(self):
        return (f"RF cavity ({self.structure_type}) {self.length} m, "
                f"f={self.frequency_hz/1e9:.3f} GHz, "
                f"E0={self.gradient_mv_per_m:.1f} MV/m, φ={self.phase_deg}°")


class beamline:
    class fringeField(lattice):
        __slots__ = ('B',)

        def __init__(self, length, fieldStrength, current=0):
            super().__init__(length)
            self.B = fieldStrength
            self.color = 'brown'

        def _compute_numeric_matrix(self, length=None, current=None, **kwargs):
            '''
            Pure NumPy implementation for fringe field transfer matrix.
            Currently uses drift space approximation.
            '''
            l = self.length if length is None else length
            M56 = -(l * self.f / (self.C * self.beta * self.gamma * (self.gamma + 1)))
            mat = np.array([
                [1.0, l, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, l, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, M56],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            ], dtype=np.float64)
            return mat

        def _compute_symbolic_matrix(self, length=None, current=None, **kwargs):
            '''
            SymPy implementation for fringe field transfer matrix.
            Currently uses drift space approximation.
            '''
            if length is None:
                l = self.length
            else:
                if isinstance(length, str):
                    l = symbols(length, real=True)
                else:
                    l = length
            M56 = -(l * self.f / (self.C * self.beta * self.gamma * (self.gamma + 1)))
            mat = Matrix([
                [1, l, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, l, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, M56],
                [0, 0, 0, 0, 0, 1]
            ])
            return mat

        def useMatrice(self, val, **kwargs):
            if not self.chromatic:
                return super().useMatrice(val, **kwargs)
            particles = np.asarray(val, dtype=np.float64)
            l = kwargs.get('length', self.length)
            delta = particles[:, 5]
            gamma_p = (self.E * (1 + delta * 1e-3) + self.E0) / self.E0
            beta_p = np.sqrt(np.maximum(1 - 1 / gamma_p**2, 1e-30))
            M56 = -(l * self.f / (self.C * beta_p * gamma_p * (gamma_p + 1)))
            out = particles.copy()
            out[:, 0] = particles[:, 0] + l * particles[:, 1]
            out[:, 2] = particles[:, 2] + l * particles[:, 3]
            out[:, 4] = particles[:, 4] + M56 * particles[:, 5]
            return out

        def __str__(self):
            return f"Fringe field segment {self.length} m long with a magnetic field of {self.B} teslas"

    def __init__(self, line=None):
        self.ORIGINFACTOR = 0.99
        self.FRINGEDELTAZ = 0.01
        self.beamline = line if line is not None else []
        self.totalLen = 0
        self.defineEndFrontPos()
        self._cache_fringe_parameters()

    def defineEndFrontPos(self):
        self.totalLen = 0
        for seg in self.beamline:
            seg.startPos = self.totalLen
            self.totalLen += seg.length
            seg.endPos = self.totalLen

    def _cache_fringe_parameters(self):
        for segment in self.beamline:
            segment._fringe_params_front = None
            segment._fringe_params_end = None
            if isinstance(segment.fringeType, list):
                xData = np.array(segment.fringeType[0], dtype=np.float64)
                yData = np.array(segment.fringeType[1], dtype=np.float64)
                xDataEnd = xData + segment.endPos
                segment._fringe_params_end = self.endFit(xDataEnd, yData, segment.endPos)
                xDataFront = -xData + segment.startPos
                segment._fringe_params_front = self.frontFit(xDataFront, yData, segment.startPos)

    def update_fringe_cache(self):
        self.defineEndFrontPos()
        self._cache_fringe_parameters()

    def interpolateData(self, xData, yData, interval):
        rbf = interpolate.Rbf(xData, yData)
        totalLen = xData[-1] - xData[0]
        xNew = np.linspace(xData[0], xData[-1], math.ceil(totalLen / interval) + 1)
        yNew = rbf(xNew)
        return xNew, yNew

    def _testModeOrder2end(self, x, origin, B0, a1, a2):
        return B0 / (1 + np.exp((a1 * (x - origin)) + (a2 * (x - origin) ** 2)))

    def testFrontFit(self, xData, yData, pos):
        endParams, _ = optimize.curve_fit(self._testModeOrder2front, xData, yData, p0=[pos, 1, 1, 1], maxfev=50000)
        return endParams

    def testendFit(self, xData, yData, pos):
        endParams, _ = optimize.curve_fit(self._testModeOrder2end, xData, yData, p0=[pos, 1, 1, 1], maxfev=50000)
        print(endParams)
        return endParams

    def _testModeOrder2front(self, x, origin, B0, a1, a2):
        return B0 / (1 + np.exp((a1 * (-x - origin)) + (a2 * (-x - origin) ** 2)))

    def _endModel(self, x, origin, B0, strength):
        return (B0 / (1 + np.exp((x - origin) * strength)))

    def _frontModel(self, x, origin, B0, strength):
        return (B0 / (1 + np.exp((-x + origin) * strength)))

    def frontFit(self, xData, yData, pos):
        endParams, _ = optimize.curve_fit(self._frontModel, xData, yData, p0=[pos, 1, 1], maxfev=50000)
        return endParams

    def endFit(self, xData, yData, pos):
        endParams, _ = optimize.curve_fit(self._endModel, xData, yData, p0=[pos, 1, 1], maxfev=50000)
        return endParams

    def _addEnd(self, zList, magnetList, beamline, ind):
        driftLen = 0
        ind2 = ind
        while (ind2 != 0 and isinstance(beamline[ind2 - 1], driftLattice)):
            driftLen = driftLen + beamline[ind2 - 1].length
            ind2 -= 1
        i = 1
        fringeTotalLen = 0
        zList.insert(0, 0)
        while (i < len(zList) and fringeTotalLen <= driftLen):
            fringeLen = zList[i] - zList[i - 1]
            fringeTotalLen += fringeLen
            if fringeTotalLen <= driftLen:
                fringeSeg = self.fringeField(fringeLen, magnetList[i - 1])
                beamline.insert(ind, fringeSeg)
            i += 1
        while (fringeTotalLen > 0 and isinstance(beamline[ind - 1], driftLattice)):
            if (beamline[ind - 1].length <= fringeTotalLen):
                fringeTotalLen -= beamline[ind - 1].length
                beamline.pop(ind - 1)
                ind -= 1
            else:
                beamline[ind - 1].length -= fringeTotalLen
                fringeTotalLen -= fringeTotalLen

    def reconfigureLine(self, interval=None):
        if interval is None:
            interval = self.FRINGEDELTAZ
        beamline = self.beamline
        totalLen = self.totalLen
        zLine = []
        i = 0
        while i <= totalLen:
            zLine.append(i)
            i += interval
        if not interval == (i - totalLen):
            zLine.append(totalLen)
        zLine = np.array(zLine)
        y_values = np.zeros_like(zLine)
        for segment in reversed(beamline):
            if isinstance(segment.fringeType, list):
                if segment._fringe_params_end is None:
                    xData = np.array(segment.fringeType[0], dtype=np.float64) + segment.endPos
                    yData = np.array(segment.fringeType[1], dtype=np.float64)
                    params = self.endFit(xData, yData, segment.endPos)
                else:
                    params = segment._fringe_params_end
                yfield = self._endModel(zLine, *params)
                yfield[zLine < segment.endPos] = 0
                y_values += yfield
            elif (segment.fringeType == 'first order decay'):
                B0 = 1
                strength = 1
                yfield = self._endModel(zLine, segment.endPos - (
                            np.log((1 - self.ORIGINFACTOR) / self.ORIGINFACTOR) / strength), B0, strength)
                yfield[zLine < segment.endPos] = 0
                y_values += yfield
        for segment in beamline:
            if isinstance(segment.fringeType, list):
                if segment._fringe_params_front is None:
                    xData = -np.array(segment.fringeType[0], dtype=np.float64) + segment.startPos
                    yData = np.array(segment.fringeType[1], dtype=np.float64)
                    params = self.frontFit(xData, yData, segment.startPos)
                else:
                    params = segment._fringe_params_front
                yfield = self._frontModel(zLine, *params)
                yfield[zLine > segment.startPos] = 0
                y_values += yfield
            elif (segment.fringeType == 'first order decay'):
                B0 = 1
                strength = 5
                yfield = self._frontModel(zLine, segment.startPos + (
                            np.log((1 - self.ORIGINFACTOR) / self.ORIGINFACTOR) / strength), B0, strength)
                yfield[zLine > segment.startPos] = 0
                y_values += yfield
        i = 0
        while (i < len(beamline)):
            if isinstance(beamline[i], driftLattice):
                index = np.searchsorted(zLine, beamline[i].startPos, side='right')
                totalDriftLen = beamline[i].length
                totalFringeLen = 0
                fringeLen = zLine[index] - beamline[i].startPos
                totalDriftLen -= fringeLen
                while (totalDriftLen >= 0 and index < len(y_values) - 1):
                    totalFringeLen += fringeLen
                    fringe = self.fringeField(fringeLen, y_values[index])
                    beamline.insert(i, fringe)
                    i += 1
                    index += 1
                    fringeLen = zLine[index] - zLine[index - 1]
                    totalDriftLen -= fringeLen
                beamline[i].length -= totalFringeLen
                if (beamline[i].length > 0 and index < len(y_values)):
                    fringe = self.fringeField(beamline[i].length, y_values[index])
                    beamline.insert(i, fringe)
                    i += 1
                beamline.pop(i)
            i += 1
        self.defineEndFrontPos()
        return zLine, y_values