"""
Continuum orbital calculations for ejected electrons.

Implements orthogonalized plane wave (OPW) approximation for
the continuum electron wavefunction, as used in the AED theory.
"""

import numpy as np
from typing import Optional, Tuple, Callable
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


class DistortedWave:
    """
    Distorted-wave partial-wave continuum in a central model potential.

    The radial function of an electron with kinetic energy E_e and angular
    momentum l moving in a central potential V(r), solving

        u_l'' + [ k² − λ(λ+1)/r² − 2 V(r) ] u_l = 0,   u_l = r · R_l ,

    by outward Numerov integration from the regular origin (u_l ~ r^{λ+1}), then
    matching to free spherical Bessel functions in the asymptotic region to
    extract the phase shift δ_l and normalise

        R_l(r) → cos δ_l · j_l(kr) − sin δ_l · y_l(kr)   (∼ sin(kr−lπ/2+δ_l)/kr).

    With V = 0 (and λ = l) this reduces to j_l(kr), δ_l = 0 — a drop-in
    generalisation of the free partial wave the OPW currently uses.

    The effective angular momentum ``lambda_eff`` (default l) replaces l in the
    *centrifugal* term only; it is the hook for the point-dipole model, where the
    dipole shifts l to a non-integer λ.  The asymptotic match still uses the
    integer l, valid when V(r) is short-range so the wave is free at large r.

    Parameters
    ----------
    kinetic_energy : float
        Electron kinetic energy E_e in Hartree (k = sqrt(2 E_e)).
    l : int
        Partial-wave angular momentum (labels the asymptotic Bessel functions).
    potential : callable, optional
        Central potential V(r) in Hartree, vectorised over r (Bohr).  Must be
        finite at ``r_min``.  None → free particle.
    lambda_eff : float, optional
        Effective angular momentum in the centrifugal term (default l).
    r_min, r_max : float
        Radial integration bounds (Bohr).  r_max must lie beyond the range of V
        and several wavelengths out; defaults to several wavelengths.
    n_grid : int
        Number of integration points.
    """

    def __init__(
        self,
        kinetic_energy: float,
        l: int,
        potential: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        lambda_eff: Optional[float] = None,
        r_min: float = 1.0e-3,
        r_max: Optional[float] = None,
        n_grid: int = 6000,
    ) -> None:
        self.energy = kinetic_energy
        self.l = int(l)
        self.k = float(np.sqrt(2.0 * kinetic_energy))
        self.potential = potential
        self.lambda_eff = float(l) if lambda_eff is None else float(lambda_eff)
        self.r_min = float(r_min)
        if r_max is None:
            # several wavelengths beyond the near region so the match is asymptotic
            wavelength = 2.0 * np.pi / self.k if self.k > 0 else 50.0
            r_max = max(60.0, 8.0 * wavelength)
        self.r_max = float(r_max)
        self.n_grid = int(n_grid)
        self.phase_shift: float = 0.0
        self._solve()

    def _solve(self) -> None:
        """Numerov-integrate u_l, then match asymptotically for δ_l and normalise."""
        from scipy.special import spherical_yn

        r = np.linspace(self.r_min, self.r_max, self.n_grid)
        h = r[1] - r[0]
        lam = self.lambda_eff
        V = self.potential(r) if self.potential is not None else np.zeros_like(r)

        # u'' = f(r) u  with  f = λ(λ+1)/r² + 2V − k²
        f = lam * (lam + 1.0) / r ** 2 + 2.0 * V - self.k ** 2

        # Numerov outward from the regular origin: u ~ r^{λ+1} (overall scale is
        # fixed later by the asymptotic normalisation).
        u = np.zeros_like(r)
        u[0] = r[0] ** (lam + 1.0)
        u[1] = r[1] ** (lam + 1.0)
        c = h * h / 12.0
        w = 1.0 - c * f                      # Numerov weight (1 − h²/12 f)
        for n in range(1, self.n_grid - 1):
            u[n + 1] = (2.0 * u[n] * (1.0 + 5.0 * c * f[n])
                        - u[n - 1] * w[n - 1]) / w[n + 1]

        R = u / r                            # radial function (arbitrary scale)

        # Match R = a·j_l + b·y_l at two asymptotic radii ~quarter-wave apart:
        #   a = N cos δ,  b = −N sin δ.
        i2 = self.n_grid - 1
        di = max(1, int((np.pi / (2.0 * self.k)) / h))   # ~quarter wavelength
        i1 = max(0, i2 - di)
        kr1, kr2 = self.k * r[i1], self.k * r[i2]
        jl1, yl1 = spherical_jn(self.l, kr1), spherical_yn(self.l, kr1)
        jl2, yl2 = spherical_jn(self.l, kr2), spherical_yn(self.l, kr2)
        det = jl1 * yl2 - jl2 * yl1
        a = (R[i1] * yl2 - R[i2] * yl1) / det
        b = (R[i2] * jl1 - R[i1] * jl2) / det
        N = float(np.hypot(a, b))
        self.phase_shift = float(np.arctan2(-b, a))

        self._r = r
        self._R = R / (N if N != 0.0 else 1.0)   # → cos δ j_l − sin δ y_l

    def radial_function(self, r: np.ndarray) -> np.ndarray:
        """
        Distorted radial function R_l^dist(r), normalised like ``j_l(kr)``.

        Evaluated by cubic interpolation of the internal Numerov solution; zero
        outside the integration range.
        """
        from scipy.interpolate import interp1d
        r = np.atleast_1d(r)
        interp = interp1d(
            self._r, self._R, kind="cubic", bounds_error=False, fill_value=0.0,
        )
        return interp(r)


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
