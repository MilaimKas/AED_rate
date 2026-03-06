"""
State-to-state AED rate calculation via Fermi's Golden Rule.

Computes the rate for:
    A⁻ + B (E, J) → AB(v', J') + e⁻

using the coupling integrals from nuclear wavefunctions and electronic
coupling matrix elements:

    Rate(v',J'; E,J) = 2π × ρ(E_e) × |V_total|²

where V_total combines radial (ΔJ=0) and rotational (ΔJ=±1) contributions.

References
----------
[1] Acharya, Kendall, Simons, JACS 106, 3402 (1984)
[2] Acharya, Das, Simons, JCP 83, 3888 (1985)
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

from scipy.integrate import simpson

from ..nuclear.nuclear_wavefunction import (
    create_wavefunction_solver,
    BoundState,
    ScatteringState,
)
from ..electronic.continuum import ContinuumOrbital, compute_electron_kinetic_energy
from ..utils.constants import CONSTANTS


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class StateToStateRate:
    """Result of a state-to-state AED rate calculation."""

    v_prime: int          # Final vibrational quantum number
    J: int                # Initial angular momentum
    J_prime: int          # Final angular momentum
    E_collision: float    # Collision energy (Hartree)
    rate: float           # Fermi Golden Rule rate (a.u.)
    electron_energy: float  # Ejected electron KE (Hartree)
    V_rad: complex        # Radial coupling integral (a.u.)
    V_rot: complex        # Rotational coupling integral (a.u.)


# ---------------------------------------------------------------------------
# Angular coupling coefficients
# ---------------------------------------------------------------------------

def angular_coupling_coefficient(J: int, J_prime: int) -> float:
    """
    Angular coupling coefficient C(J, J') for rotational non-BO coupling.

    From ⟨Y_{J'0}|∂/∂θ|Y_{J0}⟩ for M=0 (body-fixed frame of a diatomic):
        J' = J+1:  C = (J+1) / √((2J+1)(2J+3))
        J' = J-1:  C = J / √((2J-1)(2J+1))
        otherwise: C = 0 (selection rule ΔJ = ±1)
    """
    if J_prime == J + 1:
        return (J + 1) / np.sqrt((2 * J + 1) * (2 * J + 3))
    elif J_prime == J - 1 and J > 0:
        return J / np.sqrt((2 * J - 1) * (2 * J + 1))
    else:
        return 0.0


# ---------------------------------------------------------------------------
# Main calculator
# ---------------------------------------------------------------------------

class AEDRateCalculator:
    """
    State-to-state AED rate via Fermi's Golden Rule.

    Ties together nuclear wavefunctions (bound + scattering), electronic
    coupling, and continuum electron density of states to compute rates
    for individual (E, J) → (v', J') transitions.

    Parameters
    ----------
    anion_potential : PotentialEnergyCurve
        Anion PEC (e.g. OH⁻ Morse potential)
    neutral_potential : PotentialEnergyCurve
        Neutral PEC (e.g. OH Morse potential)
    EA : float
        Electron affinity in Hartree
    reduced_mass : float
        Nuclear reduced mass in a.u. (electron masses)
    coupling : ElectronicCoupling or ModelCoupling
        Electronic coupling provider with compute_coupling_at_r(R, E_e)
    solver_method : str
        Wavefunction solver: 'dvr' or 'numerov'
    r_min, r_max : float
        Radial grid bounds (Bohr)
    n_grid : int
        Number of radial grid points
    """

    def __init__(
        self,
        anion_potential,
        neutral_potential,
        EA: float,
        reduced_mass: float,
        coupling,
        solver_method: str = "dvr",
        r_min: float = 0.5,
        r_max: float = 15.0,
        n_grid: int = 500,
    ):
        self.anion_potential = anion_potential
        self.neutral_potential = neutral_potential
        self.EA = EA
        self.mu = reduced_mass
        self.coupling = coupling

        # Create wavefunction solvers with shared grid parameters
        self.anion_solver = create_wavefunction_solver(
            anion_potential, reduced_mass,
            method=solver_method, r_min=r_min, r_max=r_max, n_grid=n_grid,
        )
        self.neutral_solver = create_wavefunction_solver(
            neutral_potential, reduced_mass,
            method=solver_method, r_min=r_min, r_max=r_max, n_grid=n_grid,
        )

        # Cache for coupling curves: electron_energy -> (m_rad, m_rot) arrays
        self._coupling_cache: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}

    # ------------------------------------------------------------------
    # Coupling evaluation on the R grid
    # ------------------------------------------------------------------

    def _evaluate_coupling_on_grid(
        self, electron_energy: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate m_rad(R) and m_rot(R) on the solver's radial grid.

        Caches the result keyed by electron_energy (since the OPW
        depends on k_e = √(2E_e)).
        """
        # Round to avoid floating-point cache misses
        key = round(electron_energy, 12)
        if key in self._coupling_cache:
            return self._coupling_cache[key]

        R_grid = self.anion_solver.r_grid
        m_rad = np.zeros(len(R_grid), dtype=complex)
        m_rot = np.zeros(len(R_grid), dtype=complex)

        for i, R in enumerate(R_grid):
            result = self.coupling.compute_coupling_at_r(R, electron_energy)
            m_rad[i] = result.m_rad
            m_rot[i] = result.m_rot

        self._coupling_cache[key] = (m_rad, m_rot)
        return m_rad, m_rot

    # ------------------------------------------------------------------
    # Coupling integrals
    # ------------------------------------------------------------------

    def compute_coupling_integral_radial(
        self,
        scattering: ScatteringState,
        bound: BoundState,
        m_rad_on_grid: np.ndarray,
    ) -> complex:
        """
        Radial (vibrational) coupling integral, ΔJ = 0.

        V_rad = (1/μ) × ∫ F_{v'}(R) × m_rad(R) × dF_E(R)/dR  dR

        Parameters
        ----------
        scattering : ScatteringState
            Anion scattering wavefunction (box-normalized)
        bound : BoundState
            Neutral bound wavefunction (normalized)
        m_rad_on_grid : np.ndarray
            Radial coupling m_rad(R) evaluated on the solver grid
        """
        # dF_E/dR of the scattering wavefunction
        dF_dR = self.anion_solver.wavefunction_derivative(scattering)

        R_grid = self.anion_solver.r_grid
        integrand = bound.wavefunction * m_rad_on_grid * dF_dR

        V_rad = simpson(integrand, x=R_grid) / self.mu
        return complex(V_rad)

    def compute_coupling_integral_rotational(
        self,
        scattering: ScatteringState,
        bound: BoundState,
        m_rot_on_grid: np.ndarray,
        J: int,
        J_prime: int,
    ) -> complex:
        """
        Rotational coupling integral, ΔJ = ±1.

        V_rot = (1/μ) × C(J,J') × ∫ F_{v'}(R) × m_rot(R) × F_E(R)/R  dR

        Parameters
        ----------
        scattering : ScatteringState
            Anion scattering wavefunction
        bound : BoundState
            Neutral bound wavefunction
        m_rot_on_grid : np.ndarray
            Rotational coupling m_rot(R) on the solver grid
        J : int
            Initial angular momentum
        J_prime : int
            Final angular momentum
        """
        C = angular_coupling_coefficient(J, J_prime)
        if abs(C) < 1e-15:
            return complex(0.0)

        R_grid = self.anion_solver.r_grid
        # F_E(R) / R — the angular derivative acts on Y_{JM}, not on F(R)
        F_over_R = scattering.wavefunction / R_grid
        integrand = bound.wavefunction * m_rot_on_grid * F_over_R

        V_rot = C * simpson(integrand, x=R_grid) / self.mu
        return complex(V_rot)

    # ------------------------------------------------------------------
    # State-to-state rate
    # ------------------------------------------------------------------

    def state_to_state_rate(
        self,
        E_collision: float,
        J: int,
        v_prime: int,
        J_prime: int,
    ) -> StateToStateRate:
        """
        Compute the Fermi Golden Rule rate for a single transition.

        Rate = 2π × ρ(E_e) × |V_total|²

        Parameters
        ----------
        E_collision : float
            Collision energy in Hartree (relative to anion dissociation)
        J : int
            Initial angular momentum quantum number
        v_prime : int
            Final vibrational quantum number
        J_prime : int
            Final angular momentum quantum number

        Returns
        -------
        StateToStateRate
            Rate and all intermediate quantities
        """
        # Check selection rule: only ΔJ = 0, ±1 allowed
        if abs(J_prime - J) > 1:
            return StateToStateRate(
                v_prime=v_prime, J=J, J_prime=J_prime,
                E_collision=E_collision, rate=0.0,
                electron_energy=0.0, V_rad=0j, V_rot=0j,
            )

        # Step 1: solve scattering state on anion surface
        scattering = self.anion_solver.solve_scattering_state(E_collision, J)

        # Step 2: solve bound state on neutral surface
        bound = self.neutral_solver.solve_bound_state(v_prime, J_prime)

        # Step 3: energy conservation → electron kinetic energy
        #
        # Total initial energy (absolute, anion min = 0):
        #   E_total = D_e(anion) + E_collision
        # (the scattering state starts at the anion dissociation limit D_e(anion),
        #  plus E_collision kinetic energy above that limit)
        #
        # Final state: OH(v',J') + e⁻
        #   E_final = bound.energy + E_electron
        # where bound.energy is already on the absolute anion-min=0 scale
        #   (= neutral.V_0 + E_vib_neutral = EA + E_vib(v'))
        #
        # Conservation: E_electron = D_e(anion) + E_collision - bound.energy
        #
        # Using compute_electron_kinetic_energy with the same formula:
        #   E_e = E_coll + anion_vib_energy - E_vib_neutral - EA
        # we pass anion_vib_energy = D_e(anion) so that:
        #   E_e = E_coll + D_e(anion) - E_vib_neutral - EA
        #       = E_coll + D_e(anion) - (bound.energy - neutral.V_0) - neutral.V_0
        #       = D_e(anion) + E_coll - bound.energy  ✓
        E_vib_neutral = bound.energy - self.neutral_potential.V_0
        E_electron = compute_electron_kinetic_energy(
            collision_energy=E_collision,
            anion_vib_energy=self.anion_potential.D_e,
            neutral_vib_energy=E_vib_neutral,
            electron_affinity=self.EA,
        )

        # Step 4: check if transition is energetically allowed
        if E_electron <= 0.0:
            return StateToStateRate(
                v_prime=v_prime, J=J, J_prime=J_prime,
                E_collision=E_collision, rate=0.0,
                electron_energy=0.0, V_rad=0j, V_rot=0j,
            )

        # Step 5: evaluate electronic coupling on the R grid
        m_rad_grid, m_rot_grid = self._evaluate_coupling_on_grid(E_electron)

        # Step 6: compute coupling integrals
        V_rad = 0j
        V_rot = 0j

        if J_prime == J:
            # Radial coupling: ΔJ = 0
            V_rad = self.compute_coupling_integral_radial(
                scattering, bound, m_rad_grid,
            )
        if J_prime == J + 1 or J_prime == J - 1:
            # Rotational coupling: ΔJ = ±1
            V_rot = self.compute_coupling_integral_rotational(
                scattering, bound, m_rot_grid, J, J_prime,
            )

        # Step 7: density of states ρ = k_e / (2π²)
        continuum = ContinuumOrbital(kinetic_energy=E_electron)
        rho = continuum.density_of_states()

        # Step 8: Fermi Golden Rule
        # V_total = V_rad + V_rot (for a given J', at most one is nonzero)
        V_total = V_rad + V_rot
        rate = 2.0 * np.pi * rho * abs(V_total) ** 2

        return StateToStateRate(
            v_prime=v_prime,
            J=J,
            J_prime=J_prime,
            E_collision=E_collision,
            rate=float(rate),
            electron_energy=E_electron,
            V_rad=V_rad,
            V_rot=V_rot,
        )

    def summed_rate(
        self,
        E_collision: float,
        J: int,
        v_prime: int,
    ) -> float:
        """
        Total rate into v' summed over allowed J' = {J-1, J, J+1}.

        The three J' channels contribute incoherently (different final states).
        """
        total = 0.0
        for J_prime in [J - 1, J, J + 1]:
            if J_prime < 0:
                continue
            result = self.state_to_state_rate(E_collision, J, v_prime, J_prime)
            total += result.rate
        return total

    def vibrational_distribution(
        self,
        E_collision: float,
        J: int = 0,
        v_max: Optional[int] = None,
    ) -> Dict[int, float]:
        """
        Compute rate into each v' at fixed (E, J), summed over J'.

        Returns a dict {v': total_rate} for all energetically accessible v'.
        """
        if v_max is None:
            # Use number of neutral bound states as upper limit
            all_states = self.neutral_solver.solve_all_bound_states(J=0)
            v_max = len(all_states) - 1

        distribution: Dict[int, float] = {}
        for v_prime in range(v_max + 1):
            rate = self.summed_rate(E_collision, J, v_prime)
            if rate > 0.0:
                distribution[v_prime] = rate
            else:
                # Once we hit a forbidden transition, higher v' are also forbidden
                break

        return distribution

    def clear_cache(self) -> None:
        """Clear the coupling evaluation cache."""
        self._coupling_cache.clear()
