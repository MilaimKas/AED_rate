"""
Continuum orbital calculations for ejected electrons.

Implements orthogonalized plane wave (OPW) approximation for
the continuum electron wavefunction, as used in the AED theory.
"""

import numpy as np
from typing import Optional, Tuple
from scipy.special import spherical_jn

try:
    from scipy.special import sph_harm
except ImportError:
    from scipy.special import sph_harm_y as sph_harm
from dataclasses import dataclass

from ..utils.constants import CONSTANTS


@dataclass
class ContinuumState:
    """Container for continuum electron state."""

    kinetic_energy: float  # in Hartree
    k_vector: np.ndarray  # Wave vector
    k_magnitude: float  # |k| in a.u.
    de_broglie_wavelength: float  # in Bohr


class ContinuumOrbital:
    """
    Orthogonalized Plane Wave (OPW) continuum orbital.

    The continuum electron is described as a plane wave orthogonalized
    to the occupied orbitals of the neutral molecule:

    |k> = |k_pw> - sum_i |phi_i><phi_i|k_pw>

    Parameters
    ----------
    kinetic_energy : float
        Kinetic energy of ejected electron in Hartree
    mo_coeff : np.ndarray
        MO coefficients of neutral molecule (AO x MO)
    mo_occ : np.ndarray
        MO occupation numbers
    overlap_matrix : np.ndarray
        AO overlap matrix
    """

    def __init__(
        self,
        kinetic_energy: float,
        mo_coeff: Optional[np.ndarray] = None,
        mo_occ: Optional[np.ndarray] = None,
        overlap_matrix: Optional[np.ndarray] = None,
    ):
        self.kinetic_energy = kinetic_energy

        # Wave vector magnitude: E = k^2/2 (in a.u.)
        self.k = np.sqrt(2.0 * kinetic_energy)

        # de Broglie wavelength
        self.wavelength = 2.0 * np.pi / self.k if self.k > 0 else np.inf

        # MO information for orthogonalization
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.overlap = overlap_matrix

        # Precompute occupied orbital indices
        if mo_occ is not None:
            self.occ_idx = np.where(mo_occ > 0.5)[0]
        else:
            self.occ_idx = None

    def plane_wave(self, r: np.ndarray, k_direction: np.ndarray) -> np.ndarray:
        """
        Evaluate plane wave exp(i*k.r) at given points.

        Parameters
        ----------
        r : np.ndarray
            Position vectors (N x 3) in Bohr
        k_direction : np.ndarray
            Unit vector for k direction (normalized)

        Returns
        -------
        np.ndarray
            Complex plane wave values
        """
        k_vec = self.k * k_direction / np.linalg.norm(k_direction)
        phase = r @ k_vec
        return np.exp(1j * phase)

    def partial_wave_expansion(
        self, r: np.ndarray, l_max: int = 10
    ) -> np.ndarray:
        """
        Partial wave expansion of plane wave.

        exp(i*k.r) = sum_l (2l+1) * i^l * j_l(kr) * P_l(cos(theta))

        For spherically symmetric case (k along z-axis).

        Parameters
        ----------
        r : np.ndarray
            Radial distances in Bohr
        l_max : int
            Maximum angular momentum in expansion

        Returns
        -------
        np.ndarray
            Partial wave amplitudes for each l
        """
        kr = self.k * np.atleast_1d(r)
        amplitudes = np.zeros((l_max + 1, len(kr)), dtype=complex)

        for l in range(l_max + 1):
            # Spherical Bessel function
            jl = spherical_jn(l, kr)
            # Coefficient
            coeff = (2 * l + 1) * (1j ** l)
            amplitudes[l] = coeff * jl

        return amplitudes

    def density_of_states(self) -> float:
        """
        Calculate density of translational states for ejected electron.

        rho = k * m / (2 * pi^2 * hbar^2)

        In atomic units: rho = k / (2 * pi^2)

        Returns
        -------
        float
            Density of states per unit energy per unit solid angle
        """
        return self.k / (2.0 * np.pi**2)

    def evaluate_opw(
        self,
        r_points: np.ndarray,
        ao_values: np.ndarray,
        k_direction: np.ndarray = np.array([0, 0, 1]),
    ) -> np.ndarray:
        """
        Evaluate orthogonalized plane wave at grid points.

        |phi_k> = |k> - sum_i <phi_i|k> |phi_i>

        Parameters
        ----------
        r_points : np.ndarray
            Grid points (N x 3) in Bohr
        ao_values : np.ndarray
            AO values at grid points (N x n_ao)
        k_direction : np.ndarray
            Direction of k vector

        Returns
        -------
        np.ndarray
            OPW values at grid points (complex)
        """
        # Pure plane wave
        pw = self.plane_wave(r_points, k_direction)

        if self.mo_coeff is None or self.occ_idx is None:
            return pw

        # Orthogonalize to occupied orbitals
        opw = pw.copy()

        for i in self.occ_idx:
            # Get occupied MO values on grid
            mo_values = ao_values @ self.mo_coeff[:, i]

            # Overlap <phi_i|k_pw> - approximate numerically
            # In practice, this integral over all space needs careful treatment
            # Here we use the fact that for large k, the overlap is small

            # Schmidt orthogonalization
            overlap_ik = np.mean(mo_values.conj() * pw) * (r_points[-1, 2] - r_points[0, 2]) / len(r_points)
            opw -= overlap_ik * mo_values

        return opw

    @staticmethod
    def coulomb_wave_normalization(k: float, Z: float = 0) -> float:
        """
        Coulomb wave normalization factor.

        For neutral final state (Z=0), reduces to plane wave normalization.

        Parameters
        ----------
        k : float
            Wave vector magnitude
        Z : float
            Effective charge (0 for neutral)

        Returns
        -------
        float
            Normalization factor
        """
        if Z == 0:
            return 1.0

        # Sommerfeld parameter
        eta = -Z / k

        # Gamow factor
        gamow = 2.0 * np.pi * eta / (np.exp(2.0 * np.pi * eta) - 1.0)
        return np.sqrt(gamow)


class SphericalContinuum:
    """
    Spherical continuum wave for partial wave analysis.

    Represents the radial part of the continuum wavefunction
    for a specific angular momentum l.
    """

    def __init__(self, kinetic_energy: float, l: int):
        """
        Initialize spherical continuum wave.

        Parameters
        ----------
        kinetic_energy : float
            Kinetic energy in Hartree
        l : int
            Angular momentum quantum number
        """
        self.energy = kinetic_energy
        self.l = l
        self.k = np.sqrt(2.0 * kinetic_energy)

    def regular_solution(self, r: np.ndarray) -> np.ndarray:
        """
        Regular (finite at origin) solution: spherical Bessel j_l(kr).

        Parameters
        ----------
        r : np.ndarray
            Radial distances in Bohr

        Returns
        -------
        np.ndarray
            j_l(kr) values
        """
        r = np.atleast_1d(r)
        return spherical_jn(self.l, self.k * r)

    def phase_shifted_solution(
        self, r: np.ndarray, phase_shift: float
    ) -> np.ndarray:
        """
        Solution with scattering phase shift.

        F_l(r) = cos(delta) * j_l(kr) - sin(delta) * y_l(kr)

        Parameters
        ----------
        r : np.ndarray
            Radial distances in Bohr
        phase_shift : float
            Scattering phase shift in radians

        Returns
        -------
        np.ndarray
            Phase-shifted radial function
        """
        from scipy.special import spherical_yn

        r = np.atleast_1d(r)
        kr = self.k * r

        jl = spherical_jn(self.l, kr)
        yl = spherical_yn(self.l, kr)

        return np.cos(phase_shift) * jl - np.sin(phase_shift) * yl

    def asymptotic_form(self, r: np.ndarray, phase_shift: float = 0.0) -> np.ndarray:
        """
        Asymptotic form of radial wavefunction.

        F_l(r) ~ sin(kr - l*pi/2 + delta) / (kr)   as r -> infinity

        Parameters
        ----------
        r : np.ndarray
            Radial distances in Bohr
        phase_shift : float
            Scattering phase shift

        Returns
        -------
        np.ndarray
            Asymptotic radial function
        """
        r = np.atleast_1d(r)
        kr = self.k * r
        return np.sin(kr - self.l * np.pi / 2.0 + phase_shift) / kr

    def normalization_factor(self) -> float:
        """
        Energy normalization factor.

        For energy-normalized continuum states:
        <E,l|E',l'> = delta(E-E') * delta_{ll'}

        Returns
        -------
        float
            Normalization factor
        """
        # The factor sqrt(2*mu/pi/hbar^2/k) for energy normalization
        # In atomic units with mu=1 (electron): sqrt(2/(pi*k))
        return np.sqrt(2.0 / (np.pi * self.k))


def compute_electron_kinetic_energy(
    collision_energy: float,
    anion_vib_energy: float,
    neutral_vib_energy: float,
    electron_affinity: float,
) -> float:
    """
    Calculate kinetic energy of ejected electron from energy conservation.

    E_collision + E_vib(anion) = E_vib(neutral) + EA + E_electron

    Parameters
    ----------
    collision_energy : float
        Collision energy in Hartree
    anion_vib_energy : float
        Vibrational energy of anion (relative to dissociation) in Hartree
    neutral_vib_energy : float
        Vibrational energy of neutral (relative to its minimum) in Hartree
    electron_affinity : float
        Electron affinity of neutral molecule in Hartree (positive)

    Returns
    -------
    float
        Kinetic energy of ejected electron in Hartree
    """
    # Energy available for electron
    E_electron = collision_energy + anion_vib_energy - neutral_vib_energy - electron_affinity

    return max(0.0, E_electron)
