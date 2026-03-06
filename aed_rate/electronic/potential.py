"""
Potential energy curve handling for AED calculations.

Provides classes for:
- Interpolation of discrete potential energy data
- Morse potential analytical form
- Effective potential with centrifugal barrier
"""

import numpy as np
from scipy.interpolate import CubicSpline, UnivariateSpline
from scipy.optimize import minimize_scalar, brentq
from typing import Optional, Tuple, Callable
from dataclasses import dataclass

from ..utils.constants import CONSTANTS, get_reduced_mass


class PotentialEnergyCurve:
    """
    Potential energy curve from discrete data points.

    Handles interpolation and provides effective potential with
    centrifugal barrier for different angular momentum values.

    Parameters
    ----------
    r_points : np.ndarray
        Radial distances in Bohr
    energies : np.ndarray
        Potential energies in Hartree
    dissociation_energy : float, optional
        Asymptotic dissociation energy. If not provided, taken as
        the last point energy.
    spline_smoothing : float, optional
        Smoothing factor for spline interpolation. Default is 0 (exact).
    """

    def __init__(
        self,
        r_points: np.ndarray,
        energies: np.ndarray,
        dissociation_energy: Optional[float] = None,
        spline_smoothing: float = 0.0,
    ):
        self.r_points = np.asarray(r_points)
        self.energies = np.asarray(energies)

        # Sort by radius
        sort_idx = np.argsort(self.r_points)
        self.r_points = self.r_points[sort_idx]
        self.energies = self.energies[sort_idx]

        # Set dissociation energy
        if dissociation_energy is not None:
            self.dissociation_energy = dissociation_energy
        else:
            self.dissociation_energy = self.energies[-1]

        # Create interpolating spline
        if spline_smoothing == 0.0:
            self._spline = CubicSpline(self.r_points, self.energies)
        else:
            self._spline = UnivariateSpline(
                self.r_points, self.energies, s=spline_smoothing
            )

        # Cache for derivatives
        self._derivative_spline = self._spline.derivative(1)
        self._second_derivative_spline = self._spline.derivative(2)

        # Find equilibrium geometry
        self._find_equilibrium()

    def _find_equilibrium(self) -> None:
        """Find equilibrium bond length and energy."""
        r_min = self.r_points[0]
        r_max = self.r_points[-1]

        # Find minimum
        result = minimize_scalar(
            lambda r: self._spline(r), bounds=(r_min, r_max), method="bounded"
        )

        self.r_eq = result.x
        self.e_eq = float(self._spline(self.r_eq))

        # Calculate force constant (second derivative at equilibrium)
        self.force_constant = float(self._second_derivative_spline(self.r_eq))

    def __call__(self, r: np.ndarray) -> np.ndarray:
        """
        Evaluate potential energy at given radii.

        Parameters
        ----------
        r : np.ndarray
            Radial distances in Bohr

        Returns
        -------
        np.ndarray
            Potential energies in Hartree
        """
        r = np.atleast_1d(r)
        result = np.zeros_like(r, dtype=float)

        # Use spline within data range
        in_range = (r >= self.r_points[0]) & (r <= self.r_points[-1])
        result[in_range] = self._spline(r[in_range])

        # Extrapolate to dissociation limit for large r
        result[r > self.r_points[-1]] = self.dissociation_energy

        # Extrapolate repulsively for small r
        if np.any(r < self.r_points[0]):
            # Use exponential repulsion
            r_small = r[r < self.r_points[0]]
            e0 = self.energies[0]
            deriv0 = self._derivative_spline(self.r_points[0])
            result[r < self.r_points[0]] = e0 + deriv0 * (r_small - self.r_points[0])

        return result if result.size > 1 else float(result[0])

    def derivative(self, r: np.ndarray, order: int = 1) -> np.ndarray:
        """
        Evaluate potential energy derivative.

        Parameters
        ----------
        r : np.ndarray
            Radial distances in Bohr
        order : int
            Order of derivative (1 or 2)

        Returns
        -------
        np.ndarray
            Derivative values
        """
        r = np.atleast_1d(r)

        if order == 1:
            result = self._derivative_spline(r)
        elif order == 2:
            result = self._second_derivative_spline(r)
        else:
            result = self._spline.derivative(order)(r)

        return result if result.size > 1 else float(result[0])

    def effective_potential(
        self, r: np.ndarray, J: int, reduced_mass: float
    ) -> np.ndarray:
        """
        Calculate effective potential including centrifugal barrier.

        V_eff(R) = V(R) + hbar^2 * J(J+1) / (2 * mu * R^2)

        Parameters
        ----------
        r : np.ndarray
            Radial distances in Bohr
        J : int
            Angular momentum quantum number
        reduced_mass : float
            Reduced mass in atomic units

        Returns
        -------
        np.ndarray
            Effective potential in Hartree
        """
        r = np.atleast_1d(r)
        v_bare = self(r)
        centrifugal = J * (J + 1) / (2.0 * reduced_mass * r**2)
        return v_bare + centrifugal

    def find_classical_turning_points(
        self, energy: float, J: int, reduced_mass: float
    ) -> Tuple[float, float]:
        """
        Find classical turning points for given energy and angular momentum.

        Parameters
        ----------
        energy : float
            Total energy in Hartree
        J : int
            Angular momentum quantum number
        reduced_mass : float
            Reduced mass in atomic units

        Returns
        -------
        Tuple[float, float]
            Inner and outer turning points in Bohr
        """
        # Define function to find roots
        def f(r):
            return self.effective_potential(r, J, reduced_mass) - energy

        r_min = self.r_points[0]
        r_max = self.r_points[-1] * 2.0

        # Find inner turning point
        try:
            r_inner = brentq(f, r_min, self.r_eq)
        except ValueError:
            r_inner = r_min

        # Find outer turning point
        try:
            r_outer = brentq(f, self.r_eq, r_max)
        except ValueError:
            r_outer = r_max

        return r_inner, r_outer

    def find_barrier_height(self, J: int, reduced_mass: float) -> Tuple[float, float]:
        """
        Find the centrifugal barrier height and position.

        Parameters
        ----------
        J : int
            Angular momentum quantum number
        reduced_mass : float
            Reduced mass in atomic units

        Returns
        -------
        Tuple[float, float]
            (barrier_position, barrier_height) in (Bohr, Hartree)
        """
        if J == 0:
            return self.r_eq, self.e_eq

        # Search for maximum in effective potential
        def neg_v_eff(r):
            return -self.effective_potential(r, J, reduced_mass)

        result = minimize_scalar(
            neg_v_eff, bounds=(self.r_eq, self.r_points[-1] * 2), method="bounded"
        )

        r_barrier = result.x
        v_barrier = self.effective_potential(r_barrier, J, reduced_mass)

        return r_barrier, v_barrier


class MorsePotential(PotentialEnergyCurve):
    """
    Morse potential energy curve.

    V(r) = D_e * (1 - exp(-beta * (r - r_e)))^2 + V_0

    Parameters
    ----------
    D_e : float
        Well depth in Hartree (positive)
    r_e : float
        Equilibrium distance in Bohr
    beta : float
        Width parameter in Bohr^-1
    V_0 : float, optional
        Energy offset (asymptotic energy). Default is 0.
    """

    def __init__(
        self, D_e: float, r_e: float, beta: float, V_0: float = 0.0, n_points: int = 200
    ):
        self.D_e = D_e
        self.r_e = r_e
        self.beta = beta
        self.V_0 = V_0

        # Generate discrete points for parent class
        r_points = np.linspace(0.5 * r_e, 5.0 * r_e, n_points)
        energies = self._morse_function(r_points)

        # Initialize parent with generated points
        super().__init__(
            r_points, energies, dissociation_energy=V_0 + D_e, spline_smoothing=0.0
        )

        # Override equilibrium values with exact Morse values
        self.r_eq = r_e
        self.e_eq = V_0
        self.force_constant = 2.0 * D_e * beta**2

    def _morse_function(self, r: np.ndarray) -> np.ndarray:
        """Evaluate Morse potential."""
        return self.D_e * (1.0 - np.exp(-self.beta * (r - self.r_e))) ** 2 + self.V_0

    def __call__(self, r: np.ndarray) -> np.ndarray:
        """Evaluate Morse potential at given radii."""
        r = np.atleast_1d(r)
        result = self._morse_function(r)
        return result if result.size > 1 else float(result[0])

    def derivative(self, r: np.ndarray, order: int = 1) -> np.ndarray:
        """Analytical derivatives of Morse potential."""
        r = np.atleast_1d(r)
        exp_term = np.exp(-self.beta * (r - self.r_e))

        if order == 1:
            result = 2.0 * self.D_e * self.beta * (1.0 - exp_term) * exp_term
        elif order == 2:
            result = (
                2.0
                * self.D_e
                * self.beta**2
                * exp_term
                * (2.0 * exp_term - 1.0)
            )
        else:
            raise NotImplementedError(f"Order {order} derivative not implemented")

        return result if result.size > 1 else float(result[0])

    def vibrational_energies(self, reduced_mass: float, v_max: int = 20) -> np.ndarray:
        """
        Calculate analytical Morse vibrational energy levels.

        E_v = omega_e * (v + 0.5) - omega_e * x_e * (v + 0.5)^2

        Parameters
        ----------
        reduced_mass : float
            Reduced mass in atomic units
        v_max : int
            Maximum vibrational quantum number to calculate

        Returns
        -------
        np.ndarray
            Vibrational energies in Hartree (relative to minimum)
        """
        omega_e = self.beta * np.sqrt(2.0 * self.D_e / reduced_mass)
        x_e = omega_e / (4.0 * self.D_e)

        v_values = np.arange(v_max + 1)
        energies = omega_e * (v_values + 0.5) - omega_e * x_e * (v_values + 0.5) ** 2

        # Only return bound states
        bound = energies < self.D_e
        return energies[bound]

    @classmethod
    def from_spectroscopic_constants(
        cls,
        omega_e: float,
        omega_e_x_e: float,
        r_e: float,
        reduced_mass: float,
        V_0: float = 0.0,
    ):
        """
        Create Morse potential from spectroscopic constants.

        Parameters
        ----------
        omega_e : float
            Harmonic frequency in Hartree
        omega_e_x_e : float
            Anharmonicity in Hartree
        r_e : float
            Equilibrium distance in Bohr
        reduced_mass : float
            Reduced mass in atomic units
        V_0 : float, optional
            Energy offset

        Returns
        -------
        MorsePotential
            Morse potential instance
        """
        D_e = omega_e**2 / (4.0 * omega_e_x_e)
        beta = omega_e * np.sqrt(reduced_mass / (2.0 * D_e))

        return cls(D_e=D_e, r_e=r_e, beta=beta, V_0=V_0)


def create_oh_system() -> Tuple[MorsePotential, MorsePotential, float]:
    """
    Create OH- and OH potential curves based on literature values.

    Returns Morse potentials fitted to spectroscopic data for
    the O- + H -> OH + e- system.

    Returns
    -------
    Tuple[MorsePotential, MorsePotential, float]
        (anion_potential, neutral_potential, electron_affinity)
    """
    # OH- (1Sigma) parameters from Ref. 2(b) of the paper
    # Values in atomic units
    mu_OH = get_reduced_mass("O", "H")

    # OH- Morse parameters (approximate)
    # D_e ~ 4.5 eV, r_e ~ 1.83 bohr, omega_e ~ 3700 cm^-1
    D_e_anion = CONSTANTS.ev_to_hartree(4.5)
    r_e_anion = 1.83
    omega_e_anion = CONSTANTS.cm1_to_hartree(3700)
    beta_anion = omega_e_anion * np.sqrt(mu_OH / (2.0 * D_e_anion))

    anion_pot = MorsePotential(
        D_e=D_e_anion,
        r_e=r_e_anion,
        beta=beta_anion,
        V_0=0.0,  # Set minimum as zero reference
    )

    # OH (2Pi) parameters
    # D_e ~ 4.4 eV, r_e ~ 1.83 bohr, omega_e ~ 3738 cm^-1
    D_e_neutral = CONSTANTS.ev_to_hartree(4.4)
    r_e_neutral = 1.83
    omega_e_neutral = CONSTANTS.cm1_to_hartree(3738)
    beta_neutral = omega_e_neutral * np.sqrt(mu_OH / (2.0 * D_e_neutral))

    # OH minimum is above OH- by the electron affinity
    EA = CONSTANTS.ev_to_hartree(1.8276)  # OH electron affinity

    neutral_pot = MorsePotential(
        D_e=D_e_neutral,
        r_e=r_e_neutral,
        beta=beta_neutral,
        V_0=EA,  # Shifted up by EA relative to anion
    )

    return anion_pot, neutral_pot, EA


def create_oh_system_acharya() -> Tuple[MorsePotential, MorsePotential, float]:
    """
    Create OH- and OH Morse potentials using the exact parameters from Acharya et al.

    Parameters taken from:
    - Acharya, Kendall, Simons, J. Am. Chem. Soc. 106, 3402 (1984), Table II footnote
    - Acharya, Das, Simons, J. Chem. Phys. 83, 3888 (1985)

    These are the parameters used in the benchmark AED rate calculations.
    Beta is specified directly (not derived from omega_e).

    Returns
    -------
    Tuple[MorsePotential, MorsePotential, float]
        (anion_potential, neutral_potential, electron_affinity_in_Hartree)
    """
    # OH- (1Sigma): D_e = 0.1830 Ha, beta = 1.152 Bohr^-1, R_e = 1.822 Bohr
    anion_pot = MorsePotential(
        D_e=0.1830,
        r_e=1.822,
        beta=1.152,
        V_0=0.0,
    )

    # OH (2Pi): D_e = 0.1698 Ha, beta = 1.214 Bohr^-1, R_e = 1.834 Bohr
    # EA(OH) = 1.83 eV = 0.06723 Hartree (experimental, Schulz 1974)
    EA = CONSTANTS.ev_to_hartree(1.8276)

    neutral_pot = MorsePotential(
        D_e=0.1698,
        r_e=1.834,
        beta=1.214,
        V_0=EA,
    )

    return anion_pot, neutral_pot, EA