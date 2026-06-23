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
    V_rad: complex        # Radial coupling integral, l=1 p-wave (a.u.)
    V_rot: complex        # Rotational coupling integral, l=1 p-wave (a.u.)
    V_swave: complex = 0j  # l=0 s-wave coupling integral (A1/σ channel; a.u.)


@dataclass
class CrossSection:
    """
    Result of a state-to-state AED cross-section calculation.

    σ is in atomic units (a₀²).  Uses the Čížek (2001) convention (their
    Eq. 2.7) in the weak-coupling limit, with a unit-amplitude
    (spherical-Bessel) anion scattering state — see cross_section_state_to_state.
    """

    v_prime: int          # Final vibrational quantum number
    J: int                # Initial angular momentum
    J_prime: int          # Final angular momentum
    E_collision: float    # Collision energy (Hartree)
    sigma: float          # Cross section (a.u., a₀²)
    electron_energy: float  # Ejected electron KE (Hartree)
    V_rad: complex        # Radial coupling integral, l=1 p-wave (a.u.)
    V_rot: complex        # Rotational coupling integral, l=1 p-wave (a.u.)
    V_swave: complex = 0j  # l=0 s-wave coupling integral (A1/σ channel; a.u.)


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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate m_rad(R), m_rot(R) and the l=0 s-wave m_swave(R) on the grid.

        m_rad/m_rot are the l=1 (p-wave) radial/rotational couplings; m_swave is
        the l=0 contribution (non-zero only in the A1/σ-symmetry channel — zero
        for couplings that do not provide it, e.g. ModelCoupling).  Cached by
        electron_energy (the OPW depends on k_e = √(2E_e)).
        """
        # Round to avoid floating-point cache misses
        key = round(electron_energy, 12)
        if key in self._coupling_cache:
            return self._coupling_cache[key]

        R_grid = self.anion_solver.r_grid
        m_rad = np.zeros(len(R_grid), dtype=complex)
        m_rot = np.zeros(len(R_grid), dtype=complex)
        m_swave = np.zeros(len(R_grid), dtype=complex)

        for i, R in enumerate(R_grid):
            result = self.coupling.compute_coupling_at_r(R, electron_energy)
            m_rad[i] = result.m_rad
            m_rot[i] = result.m_rot
            m_swave[i] = getattr(result, "m_swave", 0j)

        self._coupling_cache[key] = (m_rad, m_rot, m_swave)
        return m_rad, m_rot, m_swave

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

        V_rot = (1/μ) × C(J,J') × ∫ F_{v'}(R) × m_rot(R) × F_E(R)  dR

        m_rot is defined as ∫ φ_k* ∂φ_HOMO/∂x_B d³r (nuclear coordinate
        derivative), which already contains the 1/R factor via the body-fixed
        frame angular momentum operator.  The nuclear integral therefore uses
        plain F_E(R), not F_E(R)/R.

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
        # Nuclear integral uses F_E(R)/R: the 1/R comes from how ∂/∂θ acts on
        # the nuclear wavefunction F(R)/R × Y(Ω) in spherical coordinates.
        # This is independent of the 1/R already inside m_rot(R) = ⟨φ_k|(1/R)∂φ/∂θ⟩.
        integrand = bound.wavefunction * m_rot_on_grid * scattering.wavefunction / R_grid

        V_rot = C * simpson(integrand, x=R_grid) / self.mu
        return complex(V_rot)

    def _coupling_integrals(
        self,
        scattering: ScatteringState,
        bound: BoundState,
        J: int,
        J_prime: int,
        m_rad_grid: np.ndarray,
        m_rot_grid: np.ndarray,
        m_swave_grid: np.ndarray,
    ) -> Tuple[complex, complex, complex]:
        """
        Coupling integrals for a transition: (V_rad, V_rot, V_swave).

        V_rad (ΔJ=0) and V_rot (ΔJ=±1) are the l=1 p-wave integrals — only one is
        nonzero for a given J'.  V_swave is the l=0 s-wave integral, nonzero only
        when the coupling's A1/σ channel (``swave_channel``) matches this J'
        channel: 'rad' (σ HOMO, ΔJ=0) or 'rot' (π HOMO, ΔJ=±1).  Couplings that
        do not provide an s-wave (e.g. ModelCoupling) give V_swave = 0.
        """
        swave_channel = getattr(self.coupling, "swave_channel", None)
        V_rad = 0j
        V_rot = 0j
        V_swave = 0j
        if J_prime == J:
            V_rad = self.compute_coupling_integral_radial(
                scattering, bound, m_rad_grid,
            )
            if swave_channel == "rad":
                V_swave = self.compute_coupling_integral_radial(
                    scattering, bound, m_swave_grid,
                )
        if J_prime == J + 1 or J_prime == J - 1:
            V_rot = self.compute_coupling_integral_rotational(
                scattering, bound, m_rot_grid, J, J_prime,
            )
            if swave_channel == "rot":
                V_swave = self.compute_coupling_integral_rotational(
                    scattering, bound, m_swave_grid, J, J_prime,
                )
        return V_rad, V_rot, V_swave

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

        # Step 5: evaluate electronic coupling on the R grid (l=1 + l=0 s-wave)
        m_rad_grid, m_rot_grid, m_swave_grid = self._evaluate_coupling_on_grid(E_electron)

        # Step 6: coupling integrals (l=1 radial/rotational, + l=0 s-wave)
        V_rad, V_rot, V_swave = self._coupling_integrals(
            scattering, bound, J, J_prime, m_rad_grid, m_rot_grid, m_swave_grid,
        )

        # Step 7: density of states ρ = k_e / (2π²)
        continuum = ContinuumOrbital(kinetic_energy=E_electron)
        rho = continuum.density_of_states()

        # Step 8: Fermi Golden Rule.  V_rad + V_rot is the l=1 amplitude (only one
        # is nonzero for a given J'); the l=0 s-wave is a distinct final electron
        # state, so it adds incoherently (|·|² sum).
        V_l1 = V_rad + V_rot
        rate = 2.0 * np.pi * rho * (abs(V_l1) ** 2 + abs(V_swave) ** 2)

        return StateToStateRate(
            v_prime=v_prime,
            J=J,
            J_prime=J_prime,
            E_collision=E_collision,
            rate=float(rate),
            electron_energy=E_electron,
            V_rad=V_rad,
            V_rot=V_rot,
            V_swave=V_swave,
        )

    # ------------------------------------------------------------------
    # Cross sections (Čížek 2001 convention, weak-coupling limit)
    # ------------------------------------------------------------------

    def cross_section_state_to_state(
        self,
        E_collision: float,
        J: int,
        v_prime: int,
        J_prime: int,
    ) -> CrossSection:
        """
        State-to-state AED cross section σ_{v',J'}(E; J).

        Uses the Čížek (2001) partial-wave cross-section formula (their Eq. 2.7)
        in the weak-coupling (perturbative) limit appropriate to the
        non-resonant OH⁻ system:

            σ = (2π²/E) (2J+1) ρ(E_e) |V_total|²

        where V_total = V_rad + V_rot is the non-BO coupling integral evaluated
        with a **unit-amplitude** (spherical-Bessel) anion scattering state
        (F_{E,J} → sin(kR − Jπ/2 + δ)).  The factor ρ(E_e) = k_e/(2π²) converts
        our box/L³-stripped OPW coupling to Čížek's energy-normalized
        discrete↔continuum coupling V_dε (since |V_dε|² = ρ |m|², from the width
        identity Γ = 2π|V_dε|² = 2π ρ |m|²).  Combining, σ = (2J+1)(k_e/E)|V|².

        Unlike state_to_state_rate (box-normalized, L-dependent), σ is
        independent of the computational box length L.

        Parameters
        ----------
        E_collision : float
            Collision energy in Hartree (relative to anion dissociation).
        J, v_prime, J_prime : int
            Initial angular momentum, final vibrational and rotational numbers.

        Returns
        -------
        CrossSection
            σ (a.u., a₀²) and intermediate quantities.
        """
        zero = CrossSection(
            v_prime=v_prime, J=J, J_prime=J_prime,
            E_collision=E_collision, sigma=0.0,
            electron_energy=0.0, V_rad=0j, V_rot=0j,
        )

        # Selection rule: only ΔJ = 0, ±1
        if abs(J_prime - J) > 1:
            return zero

        # Step 1: anion scattering state — UNIT AMPLITUDE (Čížek/Bessel).
        # Channel may be closed by the Pekeris centrifugal barrier at high J.
        try:
            scattering = self.anion_solver.solve_scattering_state(
                E_collision, J, normalization="unit_amplitude",
            )
        except ValueError:
            return zero
        except TypeError as exc:
            # The solver does not support normalization='unit_amplitude'.
            raise NotImplementedError(
                "Cross sections require a unit-amplitude scattering state, "
                "available only from the analytical Morse solver. Build the "
                "calculator with solver_method='morse' (the current solver is "
                f"{type(self.anion_solver).__name__})."
            ) from exc

        # Step 2: neutral bound state — may not exist at high J' (the centrifugal
        # term reduces the number of bound vibrational levels).  If so, σ = 0.
        try:
            bound = self.neutral_solver.solve_bound_state(v_prime, J_prime)
        except ValueError:
            return zero

        # Step 3: energy conservation → electron kinetic energy
        # (identical bookkeeping to state_to_state_rate)
        E_vib_neutral = bound.energy - self.neutral_potential.V_0
        E_electron = compute_electron_kinetic_energy(
            collision_energy=E_collision,
            anion_vib_energy=self.anion_potential.D_e,
            neutral_vib_energy=E_vib_neutral,
            electron_affinity=self.EA,
        )
        if E_electron <= 0.0:
            return zero

        # Step 4: coupling on the R grid (depends on E_e via the OPW)
        m_rad_grid, m_rot_grid, m_swave_grid = self._evaluate_coupling_on_grid(E_electron)

        # Step 5: coupling integrals (unit-amplitude scattering state):
        # l=1 radial/rotational + l=0 s-wave.
        V_rad, V_rot, V_swave = self._coupling_integrals(
            scattering, bound, J, J_prime, m_rad_grid, m_rot_grid, m_swave_grid,
        )

        # Step 6: electron density of states ρ = k_e/(2π²)
        continuum = ContinuumOrbital(kinetic_energy=E_electron)
        rho = continuum.density_of_states()

        # Step 7: Čížek Eq. 2.7 (weak-coupling), σ = (2π²/E)(2J+1) ρ |V|².
        # l=1 (V_rad+V_rot) and l=0 (V_swave) are distinct final electron states
        # → incoherent (|·|²) sum of partial-wave channels.
        V_l1 = V_rad + V_rot
        sigma = (
            (2.0 * np.pi ** 2 / E_collision)
            * (2 * J + 1)
            * rho
            * (abs(V_l1) ** 2 + abs(V_swave) ** 2)
        )

        return CrossSection(
            v_prime=v_prime, J=J, J_prime=J_prime,
            E_collision=E_collision, sigma=float(sigma),
            electron_energy=E_electron, V_rad=V_rad, V_rot=V_rot, V_swave=V_swave,
        )

    def total_cross_section(
        self,
        E_collision: float,
        J: int,
        v_max: Optional[int] = None,
    ) -> float:
        """
        Total AD cross section for initial partial wave J at energy E.

        Sums σ over all accessible final vibrational states v' and rotational
        channels J' ∈ {J−1, J, J+1}.  σ_AD(E) is obtained by further summing
        this over initial J (the (2J+1) partial-wave weight is already inside
        each σ via Eq. 2.7).
        """
        if v_max is None:
            all_states = self.neutral_solver.solve_all_bound_states(J=0)
            v_max = len(all_states) - 1

        total = 0.0
        for v_prime in range(v_max + 1):
            for J_prime in (J - 1, J, J + 1):
                if J_prime < 0:
                    continue
                cs = self.cross_section_state_to_state(
                    E_collision, J, v_prime, J_prime,
                )
                total += cs.sigma
        return total

    def total_cross_section_all_J(
        self,
        E_collision: float,
        v_max: Optional[int] = None,
        J_max: Optional[int] = None,
    ) -> float:
        """
        Total AD cross section σ_AD(E), summed over initial partial waves J.

            σ_AD(E) = Σ_J Σ_{v',J'} σ_{v',J→J'}(E)         [Čížek 2001, Eq. 2.8]

        The (2J+1) partial-wave weight is already inside each σ (Eq. 2.7).  The
        sum runs over open channels: it terminates at the first J for which the
        anion scattering channel is closed by the Pekeris centrifugal barrier
        (solve_scattering_state raises ValueError), since the barrier increases
        monotonically with J.

        Parameters
        ----------
        E_collision : float
            Collision energy in Hartree.
        v_max : int, optional
            Maximum final vibrational quantum number (default: all bound).
        J_max : int, optional
            Hard cap on the partial-wave sum (default: until the channel closes).

        Returns
        -------
        float
            σ_AD(E) in atomic units (a₀²).
        """
        sigma_AD = 0.0
        J = 0
        while J_max is None or J <= J_max:
            # Channel-open test: the centrifugal barrier closes all J ≥ this one.
            try:
                self.anion_solver.solve_scattering_state(
                    E_collision, J, normalization="unit_amplitude",
                )
            except ValueError:
                break
            sigma_AD += self.total_cross_section(E_collision, J, v_max=v_max)
            J += 1
        return sigma_AD

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
