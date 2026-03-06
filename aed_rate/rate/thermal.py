"""
Thermal AED rate constant via Boltzmann-weighted partial-wave summation.

Computes:
    k(T) = prefactor × Σ_J (2J+1) × ∫₀^∞ √E × e^{-E/k_BT} × Σ_{v'} Rate(v',J; E,J) dE

using Gauss-Laguerre quadrature for the energy integral (natural for the
exponential weight) and a J summation truncated at J_max(E) where the
centrifugal barrier exceeds the collision energy.

References
----------
[1] Acharya, Kendall, Simons, JACS 106, 3402 (1984)
[2] Acharya, Das, Simons, JCP 83, 3888 (1985)
"""

import numpy as np
from typing import Optional

from scipy.special import roots_laguerre

from .state_to_state import AEDRateCalculator
from ..utils.constants import CONSTANTS


class ThermalRateCalculator:
    """
    Thermal AED rate constant by averaging over collision energy and J.

    Parameters
    ----------
    rate_calculator : AEDRateCalculator
        State-to-state rate calculator
    anion_potential : PotentialEnergyCurve
        Anion potential (for centrifugal barrier heights)
    reduced_mass : float
        Nuclear reduced mass in a.u.
    """

    def __init__(
        self,
        rate_calculator: AEDRateCalculator,
        anion_potential,
        reduced_mass: float,
    ):
        self.rate_calc = rate_calculator
        self.anion_potential = anion_potential
        self.mu = reduced_mass

    def max_angular_momentum(self, E_collision: float) -> int:
        """
        Find J_max where the centrifugal barrier height equals E_collision.

        All partial waves with J ≤ J_max can classically reach the
        interaction region. Uses binary search on barrier heights.
        """
        if E_collision <= 0:
            return 0

        # Binary search: find largest J where barrier_height < E_collision
        J_low, J_high = 0, 200
        while J_low < J_high:
            J_mid = (J_low + J_high + 1) // 2
            try:
                _r_bar, V_bar = self.anion_potential.find_barrier_height(
                    J_mid, self.mu,
                )
                # find_barrier_height may return arrays
                V_bar = float(np.asarray(V_bar).flat[0])
            except (ValueError, IndexError):
                # No barrier found — J is too high
                J_high = J_mid - 1
                continue

            if V_bar <= E_collision:
                J_low = J_mid
            else:
                J_high = J_mid - 1

        return J_low

    def _summed_rate_at_E_J(
        self,
        E_collision: float,
        J: int,
        v_prime_max: int,
    ) -> float:
        """Sum rate over all energetically accessible v' at fixed (E, J)."""
        total = 0.0
        for v_prime in range(v_prime_max + 1):
            rate = self.rate_calc.summed_rate(E_collision, J, v_prime)
            if rate <= 0.0:
                break  # higher v' are also forbidden
            total += rate
        return total

    def thermal_rate_constant(
        self,
        T: float,
        n_energy: int = 32,
        v_prime_max: Optional[int] = None,
        J_step: int = 1,
    ) -> float:
        """
        Compute thermal rate constant k(T) in cm³/s.

        Uses Gauss-Laguerre quadrature for the energy integral:
            ∫₀^∞ f(E) × exp(-E/k_BT) dE = (k_BT) ∫₀^∞ f(k_BT × x) × exp(-x) dx

        The partial-wave sum runs over J = 0, J_step, 2*J_step, ..., J_max(E)
        with (2J+1) weighting.

        Parameters
        ----------
        T : float
            Temperature in Kelvin
        n_energy : int
            Number of Gauss-Laguerre quadrature points
        v_prime_max : int, optional
            Maximum v' to include. If None, uses all neutral bound states.
        J_step : int
            Step size for J summation (use >1 to speed up, with interpolation)

        Returns
        -------
        float
            k(T) in cm³/s
        """
        kBT = CONSTANTS.k_B * T

        # Determine v_max from neutral bound states
        if v_prime_max is None:
            all_states = self.rate_calc.neutral_solver.solve_all_bound_states(J=0)
            v_prime_max = len(all_states) - 1

        # Gauss-Laguerre nodes and weights: ∫₀^∞ f(x) e^{-x} dx ≈ Σ w_i f(x_i)
        x_nodes, w_laguerre = roots_laguerre(n_energy)
        E_nodes = kBT * x_nodes  # transform: E = kBT × x

        # Prefactor for the thermal rate constant:
        # k(T) = (8πμ)^{-1/2} × (kBT)^{-3/2} × (kBT) × Σ_i w_i × √E_i × (...)
        # The (kBT) comes from the Gauss-Laguerre variable change dE = kBT dx
        # Combined: prefactor = (kBT)^{-1/2} / √(8πμ)
        prefactor = 1.0 / np.sqrt(8.0 * np.pi * self.mu * kBT)

        rate_integral = 0.0

        for i, (E, w) in enumerate(zip(E_nodes, w_laguerre)):
            if E <= 0:
                continue

            J_max = self.max_angular_momentum(E)
            if J_max <= 0:
                continue

            # Partial-wave sum: Σ_J (2J+1) × Rate(E, J) / J_max²
            # The 1/J_max² normalization comes from the cross section → rate
            # conversion (Acharya 1985 Eq. 11)
            J_sum = 0.0
            for J in range(0, J_max + 1, J_step):
                rate_J = self._summed_rate_at_E_J(E, J, v_prime_max)
                weight_J = (2 * J + 1) * J_step  # J_step accounts for skipping
                J_sum += weight_J * rate_J

            # Integrand: √E × J_sum / J_max²
            rate_integral += w * np.sqrt(E) * J_sum / max(J_max ** 2, 1)

            # Clear coupling cache periodically to manage memory
            if i % 8 == 0:
                self.rate_calc.clear_cache()

        # Convert from a.u. to cm³/s
        k_au = prefactor * rate_integral
        return CONSTANTS.rate_au_to_cm3s(k_au)

    def thermal_rate_vs_temperature(
        self,
        T_array: np.ndarray,
        n_energy: int = 32,
        v_prime_max: Optional[int] = None,
        J_step: int = 1,
    ) -> np.ndarray:
        """Compute k(T) for an array of temperatures. Returns array in cm³/s."""
        return np.array([
            self.thermal_rate_constant(T, n_energy, v_prime_max, J_step)
            for T in T_array
        ])
