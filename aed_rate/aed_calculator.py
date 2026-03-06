"""
High-level interface for Associative Electron Detachment (AED) calculations.

Ties together potentials, electronic coupling, nuclear wavefunctions, and
rate calculators into a single AEDSystem object.

Typical usage
-------------
>>> sys = AEDSystem.oh_system()
>>> k = sys.thermal_rate(300.0)          # cm³/s
>>> dist = sys.vibrational_distribution(E_coll=0.001, J=0)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

from .electronic.potential import create_oh_system_acharya
from .electronic.coupling import ModelCoupling
from .rate.state_to_state import AEDRateCalculator
from .rate.thermal import ThermalRateCalculator
from .utils.constants import CONSTANTS, get_reduced_mass


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class VibDistribution:
    """Vibrational population distribution from an AED process."""

    E_collision: float            # Collision energy (Hartree)
    J: int                        # Initial angular momentum
    rates: Dict[int, float]       # {v': rate (a.u.)}
    total_rate: float             # Sum over all v' (a.u.)

    def normalized(self) -> Dict[int, float]:
        """Return fractional population per v'."""
        if self.total_rate == 0.0:
            return {}
        return {v: r / self.total_rate for v, r in self.rates.items()}

    def peak_v(self) -> int:
        """v' with highest rate."""
        return max(self.rates, key=self.rates.get)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class AEDSystem:
    """
    High-level interface for AED rate calculations.

    Bundles potential energy curves, electronic coupling, nuclear wavefunction
    solvers, and Fermi Golden Rule rate calculators into a single object.
    Once constructed, call :meth:`thermal_rate`, :meth:`vibrational_distribution`,
    or :meth:`state_to_state_rate` to compute observables.

    Parameters
    ----------
    anion_potential : PotentialEnergyCurve
        Anion PEC (e.g. OH⁻ Morse).
    neutral_potential : PotentialEnergyCurve
        Neutral PEC (e.g. OH Morse).
    EA : float
        Electron affinity in Hartree.
    reduced_mass : float
        Nuclear reduced mass in a.u. (electron masses).
    coupling : ModelCoupling or ElectronicCoupling
        Provides compute_coupling_at_r(R, E_e) → CouplingResult.
    solver_method : str
        Wavefunction solver: ``'dvr'`` (default) or ``'numerov'``.
    r_min, r_max : float
        Radial grid bounds in Bohr.
    n_grid : int
        Number of radial grid points.
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
    ) -> None:
        self.anion_potential = anion_potential
        self.neutral_potential = neutral_potential
        self.EA = EA
        self.mu = reduced_mass
        self.coupling = coupling

        self._rate_calc = AEDRateCalculator(
            anion_potential=anion_potential,
            neutral_potential=neutral_potential,
            EA=EA,
            reduced_mass=reduced_mass,
            coupling=coupling,
            solver_method=solver_method,
            r_min=r_min,
            r_max=r_max,
            n_grid=n_grid,
        )
        self._thermal_calc = ThermalRateCalculator(
            rate_calculator=self._rate_calc,
            anion_potential=anion_potential,
            reduced_mass=reduced_mass,
        )

    # ------------------------------------------------------------------
    # Factory constructors
    # ------------------------------------------------------------------

    @classmethod
    def oh_system(
        cls,
        A_rad: float = 0.33,
        alpha_rad: float = 3.0,
        A_rot: float = 2.2,
        alpha_rot: float = 3.0,
        k_power: float = 1.0,
        solver_method: str = "dvr",
        r_min: float = 0.5,
        r_max: float = 15.0,
        n_grid: int = 500,
    ) -> "AEDSystem":
        """
        Factory for the O⁻ + H → OH(v',J') + e⁻ system.

        Uses exact Morse parameters from Acharya, Kendall, Simons (1984/1985)
        and a Gaussian model coupling calibrated against their CPSCF results.

        Parameters
        ----------
        A_rad : float
            Amplitude of radial (non-BO) coupling Gaussian (a.u.).
        alpha_rad : float
            Width of radial coupling Gaussian (Bohr⁻²).
        A_rot : float
            Amplitude of rotational coupling Gaussian (a.u.).
        alpha_rot : float
            Width of rotational coupling Gaussian (Bohr⁻²).
        k_power : float
            Energy scaling exponent: m(R,E_e) × k_e^k_power.
            Use 1.0 for π/σ HOMO (low-k OPW limit), 0.0 for flat coupling.
        solver_method : str
            Nuclear wavefunction solver: ``'dvr'`` (default), ``'numerov'``,
            or ``'morse'`` (analytical Morse + Pekeris; requires mpmath;
            recommended for best phase accuracy in coupling integrals).
        r_min, r_max : float
            Radial grid bounds in Bohr.
        n_grid : int
            Number of radial grid points.

        Returns
        -------
        AEDSystem
            Pre-configured for the OH⁻/OH system.
        """
        anion_pot, neutral_pot, EA = create_oh_system_acharya()
        mu = get_reduced_mass("O", "H")

        # Place the Gaussian coupling centred at the anion equilibrium
        coupling = ModelCoupling(
            R0=anion_pot.r_e,
            A_rad=A_rad,
            alpha_rad=alpha_rad,
            A_rot=A_rot,
            alpha_rot=alpha_rot,
            k_power=k_power,
        )

        return cls(
            anion_potential=anion_pot,
            neutral_potential=neutral_pot,
            EA=EA,
            reduced_mass=mu,
            coupling=coupling,
            solver_method=solver_method,
            r_min=r_min,
            r_max=r_max,
            n_grid=n_grid,
        )

    # ------------------------------------------------------------------
    # State-resolved rates
    # ------------------------------------------------------------------

    def state_to_state_rate(
        self,
        E_collision: float,
        J: int,
        v_prime: int,
        J_prime: Optional[int] = None,
    ) -> float:
        """
        Rate for a single (E, J) → v' transition.

        If J_prime is given, returns the rate for that specific final
        angular momentum channel. Otherwise returns the sum over
        J' = {J-1, J, J+1} (the physically observable quantity).

        Parameters
        ----------
        E_collision : float
            Collision energy above the anion dissociation limit (Hartree).
        J : int
            Initial angular momentum.
        v_prime : int
            Final vibrational quantum number.
        J_prime : int, optional
            Final angular momentum. If None, sums over all allowed J'.

        Returns
        -------
        float
            Rate in atomic units (Hartree / ħ).
        """
        if J_prime is not None:
            result = self._rate_calc.state_to_state_rate(
                E_collision, J, v_prime, J_prime
            )
            return result.rate
        return self._rate_calc.summed_rate(E_collision, J, v_prime)

    def vibrational_distribution(
        self,
        E_collision: float,
        J: int = 0,
        v_max: Optional[int] = None,
    ) -> VibDistribution:
        """
        Rate into each v' at fixed (E, J), summed over J'.

        Parameters
        ----------
        E_collision : float
            Collision energy in Hartree.
        J : int
            Initial angular momentum.
        v_max : int, optional
            Maximum v' to include. Defaults to all neutral bound states.

        Returns
        -------
        VibDistribution
            Contains dict {v': rate}, total rate, and helper methods.
        """
        raw = self._rate_calc.vibrational_distribution(E_collision, J, v_max)
        total = sum(raw.values())
        return VibDistribution(
            E_collision=E_collision,
            J=J,
            rates=raw,
            total_rate=total,
        )

    # ------------------------------------------------------------------
    # Thermal rate constant
    # ------------------------------------------------------------------

    def thermal_rate(
        self,
        T: float,
        n_energy: int = 32,
        v_prime_max: Optional[int] = None,
        J_step: int = 1,
    ) -> float:
        """
        Thermal rate constant k(T) in cm³/s.

        Uses Gauss-Laguerre quadrature for the Boltzmann energy integral
        and a partial-wave sum over J up to J_max(E).

        Parameters
        ----------
        T : float
            Temperature in Kelvin.
        n_energy : int
            Number of Gauss-Laguerre quadrature points (default 32).
        v_prime_max : int, optional
            Maximum v' to include. Defaults to all neutral bound states.
        J_step : int
            Step size for J summation (use > 1 to speed up).

        Returns
        -------
        float
            k(T) in cm³/s.
        """
        return self._thermal_calc.thermal_rate_constant(
            T, n_energy=n_energy, v_prime_max=v_prime_max, J_step=J_step
        )

    def thermal_rate_vs_temperature(
        self,
        T_array: np.ndarray,
        n_energy: int = 32,
        v_prime_max: Optional[int] = None,
        J_step: int = 1,
    ) -> np.ndarray:
        """
        k(T) for an array of temperatures.

        Parameters
        ----------
        T_array : np.ndarray
            Temperatures in Kelvin.

        Returns
        -------
        np.ndarray
            k(T) in cm³/s, same shape as T_array.
        """
        return self._thermal_calc.thermal_rate_vs_temperature(
            T_array, n_energy=n_energy, v_prime_max=v_prime_max, J_step=J_step
        )

    # ------------------------------------------------------------------
    # Bound state info
    # ------------------------------------------------------------------

    def neutral_bound_states(self, J: int = 0) -> list:
        """
        Return all neutral bound state energies at angular momentum J.

        Returns
        -------
        list of BoundState
            Energies on the absolute PEC scale (anion min = 0).
        """
        return self._rate_calc.neutral_solver.solve_all_bound_states(J=J)

    def anion_bound_states(self, J: int = 0) -> list:
        """
        Return all anion bound state energies at angular momentum J.

        Returns
        -------
        list of BoundState
            Energies on the absolute PEC scale (anion min = 0).
        """
        return self._rate_calc.anion_solver.solve_all_bound_states(J=J)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """
        Return a human-readable summary of the system parameters.

        Returns
        -------
        str
            Multi-line system description.
        """
        anion = self.anion_potential
        neutral = self.neutral_potential

        # Count bound states
        anion_states = self.anion_bound_states(J=0)
        neutral_states = self.neutral_bound_states(J=0)

        ea_ev = self.EA * CONSTANTS.hartree_to_ev

        lines = [
            "=" * 60,
            "  AED System Summary",
            "=" * 60,
            "",
            "  Anion PEC:",
            f"    D_e   = {anion.D_e:.4f} Ha  ({anion.D_e * CONSTANTS.hartree_to_ev:.3f} eV)",
            f"    R_e   = {anion.r_e:.4f} Bohr",
            f"    beta  = {anion.beta:.4f} Bohr⁻¹",
            f"    Bound states (J=0): {len(anion_states)}",
            "",
            "  Neutral PEC:",
            f"    D_e   = {neutral.D_e:.4f} Ha  ({neutral.D_e * CONSTANTS.hartree_to_ev:.3f} eV)",
            f"    R_e   = {neutral.r_e:.4f} Bohr",
            f"    beta  = {neutral.beta:.4f} Bohr⁻¹",
            f"    V_0   = {neutral.V_0:.4f} Ha  (= EA)",
            f"    Bound states (J=0): {len(neutral_states)}",
            "",
            f"  Electron affinity: {ea_ev:.4f} eV",
            f"  Reduced mass:      {self.mu:.1f} a.u.",
            "",
            "  Coupling:",
            f"    Type   = {type(self.coupling).__name__}",
        ]

        # Add ModelCoupling-specific info
        if hasattr(self.coupling, "A_rad"):
            lines += [
                f"    R0     = {self.coupling.R0:.4f} Bohr",
                f"    A_rad  = {self.coupling.A_rad:.4f}  alpha_rad = {self.coupling.alpha_rad:.2f} Bohr⁻²",
                f"    A_rot  = {self.coupling.A_rot:.4f}  alpha_rot = {self.coupling.alpha_rot:.2f} Bohr⁻²",
                f"    k_power= {self.coupling.k_power:.1f}",
            ]

        lines.append("=" * 60)
        return "\n".join(lines)

    def __repr__(self) -> str:
        n_anion = len(self.anion_bound_states())
        n_neutral = len(self.neutral_bound_states())
        return (
            f"AEDSystem(EA={self.EA * CONSTANTS.hartree_to_ev:.3f} eV, "
            f"mu={self.mu:.0f} a.u., "
            f"coupling={type(self.coupling).__name__}, "
            f"anion_v={n_anion}, neutral_v={n_neutral})"
        )
