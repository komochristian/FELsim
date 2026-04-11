#   Authors: Christian Komo, Niels Bidault
from sympy import symbols, Matrix
import sympy as sp
import numpy as np
from scipy import interpolate
from scipy import optimize
import math


# IMPORTANT NOTES:
#  by default every beam type is an electron beam type
#  NOTE: default fringe fields for now is noted as [[x list], [y list]],
#  ASSUMES that the measurement begins at 0
#  ex. [[0.01,0.02,0.03,0.95,1],[1.6,0.7,0.2,0.01,1]]
#  getSymbolicMatrice() must use all sympy methods and functions, NOT numpy

class lattice:
    color = 'none'
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
        self.E0 = 0.51099  # Electron rest energy (MeV/c^2)
        self.Q = 1.60217663e-19  # (C)
        self.M = 9.1093837e-31  # (kg)
        self.C = 299792458  # Celerity (m/s)
        self.f = 2856 * (10 ** 6)  # RF frequency (Hz)
        self.M_AMU = 1.66053906892E-27  # Atomic mass unit (kg)
        self.k_MeV = 1e-6 / self.Q  # Conversion factor (MeV / J)
        self.m_p = 1.67262192595e-27  # Proton Mass (kg)
        self.PARTICLES = {"electron": [self.M, self.Q, (self.M * self.C ** 2) * self.k_MeV],
                          "proton": [self.m_p, self.Q, (self.m_p * self.C ** 2) * self.k_MeV]}
        self.gamma = (1 + (self.E / self.E0))
        self.beta = np.sqrt(1 - (1 / (self.gamma ** 2)))
        self.unitsF = 10 ** 6  # Units factor used for conversions from (keV) to (ns)
        self.color = self.__class__.color # Color of beamline element when graphed
        self.fringeType = fringeType  # Each segment has no magnetic fringe by default
        self.startPos = None
        self.endPos = None
        if not length <= 0:
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
            except:
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
        # Vectorized matrix multiplication: (6,6) @ (6,N)^T = (6,N)^T
        transformed = (mat @ particles.T).T
        return transformed.tolist()


class driftLattice(lattice):
    color = "white"
    def __init__(self, length: float, name=None):
        '''
        Represents a drift space (empty section) in the beamline.

        Parameters
        ----------
        length : float
            The length of the drift segment in meters.
        '''
        super().__init__(length, name=name)
        self.color = self.__class__.color

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
    color = "cornflowerblue"
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
        self.color = self.__class__.color
        self.G = 2.694  # Quadrupole focusing strength (T/A/m)

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
        if I == 0:
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

    def __str__(self):
        return f"QPF beamline segment {self.length} m long and a current of {self.current} amps"


class qpdLattice(lattice):
    color = "lightcoral"
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
        self.color = self.__class__.color

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
        if I == 0:
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

    def __str__(self):
        return f"QPD beamline segment {self.length} m long and a current of {self.current} amps"


class dipole(lattice):
    color = "forestgreen"
    def __init__(self, length: float = 0.0889, angle: float = 1.5, fringeType='decay', name=None):
        '''
        Represents a dipole bending magnet, which bends the beam horizontally.

        Parameters
        ----------
        length : float, optional
            The effective length of the dipole magnet in meters
        angle : float, optional
            The bending angle of the dipole magnet in degrees.
        fringeType :
        '''
        super().__init__(length, fringeType, name=name)
        self.color = self.__class__.color
        self.angle = angle

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
        by = (self.M * self.C * self.beta * self.gamma / self.Q) * (a * np.pi / 180 / self.length)
        rho = self.M * self.C * self.beta * self.gamma / (self.Q * by)
        theta = l / rho
        C = np.cos(theta)
        S = np.sin(theta)
        M16 = rho * (1 - C) * (self.gamma / (self.gamma + 1))
        M26 = S * (self.gamma / (self.gamma + 1))
        M51 = -self.f * S / (self.beta * self.C)
        M52 = -self.f * rho * (1 - C) / (self.beta * self.C)
        M56 = -self.f * (l - rho * S) / (self.C * self.beta * self.gamma * (self.gamma + 1))
        mat = np.array([
            [C, rho * S, 0.0, 0.0, 0.0, M16],
            [-S / rho, C, 0.0, 0.0, 0.0, M26],
            [0.0, 0.0, 1.0, l, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [M51, M52, 0.0, 0.0, 1.0, M56],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)
        return mat

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
        M51 = -self.f * S / (self.beta * self.C)
        M52 = -self.f * rho * (1 - C) / (self.beta * self.C)
        M56 = -self.f * (l - rho * S) / (self.C * self.beta * self.gamma * (self.gamma + 1))
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
    color = "lightgreen"
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
        self.color = self.__class__.color
        self.angle = angle
        self.dipole_length = dipole_length
        self.dipole_angle = dipole_angle
        self.pole_gap = pole_gap

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
        R = dipole_length / (abs(dipole_angle) * np.pi / 180)
        eta = (a * np.pi / 180) * l / self.length
        Tx = np.tan(eta)
        # Fringe field contribution using triangle model
        g = self.pole_gap
        le = self.length
        # Analytical integration of K for triangle model: K = le/(6*g)
        K_simplified = le / (6.0 * g)
        h = 1.0 / R
        phi = K_simplified * g * h * (1 + np.sin(eta) ** 2) / np.cos(eta)
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
        eta = (a * sp.pi / 180) * l / self.length
        Tx = sp.tan(eta)
        g = self.pole_gap
        le = self.length
        # Fringe field K integral (triangle model): By not needed since R computed directly
        K_simplified = le / (6 * g)
        h = 1 / R
        phi = sp.simplify(K_simplified * g * h * (1 + sp.sin(eta) ** 2) / sp.cos(eta))
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

    def __str__(self):
        return f"Horizontal wedge dipole magnet segment {self.length} m long (curvature) with an angle of {self.angle} degrees"


class beamline:
    class fringeField(lattice):
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

        def __str__(self):
            return f"Fringe field segment {self.length} m long with a magnetic field of {self.B} teslas"

    def __init__(self, line=[]):
        self.ORIGINFACTOR = 0.99
        self.FRINGEDELTAZ = 0.01
        self.beamline = line
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
    
    def findSegmentAtPos(self, pos):
        for i in range(len(self.beamline)):
            seg = self.beamline[i]
            if (pos >= seg.startPos and pos <= seg.endPos):
                return i
        return -1

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