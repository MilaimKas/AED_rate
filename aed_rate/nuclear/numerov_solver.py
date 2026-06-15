"""
Numerov shooting method nuclear wavefunction solver.

Solves the radial Schrödinger equation using the 4th-order Numerov
algorithm.  Bound states are found by outward+inward shooting matched
at the outer classical turning point; scattering states by outward-only
shooting from the inner wall.

Strengths
---------
- Works for any PotentialEnergyCurve (not restricted to Morse).
- Scattering states have the correct inner-wall phase (F(R_min) = 0,
  no box quantization at R_max).

Limitations / status
--------------------
- Bound-state eigenvalue search is less robust than DVR: the bisection
  on number of nodes can miss or double-count states near degeneracies.
- Untested/unvalidated against reference data — use DVR or MorseAnalyticSolver
  for production calculations until a validation suite is in place.
"""

from __future__ import annotations

import warnings

import numpy as np
from typing import List, Optional

from ..electronic.potential import PotentialEnergyCurve
from ..utils.constants import AEDValidationWarning
from ._base_solver import BoundState, ScatteringState, WavefunctionSolver


class NumerovWavefunctionSolver(WavefunctionSolver):
    """
    Numerov shooting method solver for nuclear radial wavefunctions.

    Parameters
    ----------
    potential : PotentialEnergyCurve
        Potential energy curve for the diatomic system.
    reduced_mass : float
        Reduced mass in atomic units (electron masses).
    r_min, r_max : float
        Radial grid bounds (Bohr).
    n_grid : int
        Number of grid points.
    """

    def __init__(
        self,
        potential: PotentialEnergyCurve,
        reduced_mass: float,
        r_min: float = 0.5,
        r_max: float = 20.0,
        n_grid: int = 2000,
    ) -> None:
        warnings.warn(
            "NumerovWavefunctionSolver is not validated against reference data. "
            "Use solver_method='morse' (analytical Morse, recommended for coupling "
            "integrals) or 'dvr' for production calculations.",
            AEDValidationWarning,
            stacklevel=2,
        )
        super().__init__(potential, reduced_mass, r_min, r_max, n_grid)

    # ------------------------------------------------------------------
    # Core Numerov propagators
    # ------------------------------------------------------------------

    def _g_array(self, energy: float, J: int) -> np.ndarray:
        """g[n] = 2μ(E - V_eff[n]) — positive in classically allowed region."""
        V_eff = self.potential.effective_potential(self.r_grid, J, self.mu)
        return 2.0 * self.mu * (energy - V_eff)

    def _numerov_outward(
        self, g: np.ndarray, f0: float = 0.0, f1: float = 1e-10
    ) -> np.ndarray:
        """
        Numerov outward propagation from index 0.

        Renormalises periodically to prevent overflow in the classically
        forbidden inner region.
        """
        h2 = self.dr ** 2
        n = len(g)
        f = np.zeros(n)
        f[0] = f0
        f[1] = f1

        for i in range(1, n - 1):
            num = (2.0 * (1.0 - 5.0 * h2 * g[i] / 12.0) * f[i]
                   - (1.0 + h2 * g[i - 1] / 12.0) * f[i - 1])
            den = 1.0 + h2 * g[i + 1] / 12.0
            f[i + 1] = num / den

            if i % 100 == 0 and np.abs(f[i + 1]) > 1e10:
                scale = np.max(np.abs(f[:i + 2]))
                if scale > 1e-30:
                    f[:i + 2] /= scale

        return f

    def _numerov_inward(
        self, g: np.ndarray, fN: float = 0.0, fN1: float = 1e-10
    ) -> np.ndarray:
        """Numerov inward propagation from the last index."""
        h2 = self.dr ** 2
        n = len(g)
        f = np.zeros(n)
        f[-1] = fN
        f[-2] = fN1

        for i in range(n - 2, 0, -1):
            num = (2.0 * (1.0 - 5.0 * h2 * g[i] / 12.0) * f[i]
                   - (1.0 + h2 * g[i + 1] / 12.0) * f[i + 1])
            den = 1.0 + h2 * g[i - 1] / 12.0
            f[i - 1] = num / den

            if i % 100 == 0 and np.abs(f[i - 1]) > 1e10:
                scale = np.max(np.abs(f[i - 1:]))
                if scale > 1e-30:
                    f[i - 1:] /= scale

        return f

    def _find_matching_point(self, energy: float, g: np.ndarray) -> int:
        """Index of the outer classical turning point for outward/inward matching."""
        allowed = g > 0  # classically allowed: E > V_eff
        if np.any(~allowed):
            forbidden = np.where(~allowed)[0]
            if len(forbidden) > 0 and forbidden[-1] > len(g) // 2:
                idx = forbidden[-1] - 10
            else:
                idx = len(g) // 2
        else:
            idx = len(g) // 2
        return max(10, min(idx, len(g) - 10))

    # ------------------------------------------------------------------
    # Eigenvalue search
    # ------------------------------------------------------------------

    def _shooting_method(
        self,
        g_func,
        target_nodes: int,
        E_min: float,
        E_max: float,
        tol: float = 1e-10,
        max_iter: int = 100,
    ) -> float:
        """
        Bisection shooting method: find E such that outward solution has
        target_nodes nodes and the log-derivative matches at the turning point.

        Parameters
        ----------
        g_func : callable
            Function E -> g array (= 2μ(E - V_eff)).
        target_nodes : int
            Required number of nodes (= v).
        E_min, E_max : float
            Energy bracket.
        """
        E_low, E_high = E_min, E_max
        energy = (E_low + E_high) / 2.0

        for _ in range(max_iter):
            g = g_func(energy)
            match_idx = self._find_matching_point(energy, g)

            f_out = self._numerov_outward(g[:match_idx + 2])
            f_in = self._numerov_inward(g[match_idx - 1:])

            scale = f_out[-2] / f_in[1] if abs(f_in[1]) > 1e-20 else 1.0
            f_in_s = f_in * scale

            df_out = (f_out[-1] - f_out[-3]) / (2.0 * self.dr)
            df_in = (f_in_s[2] - f_in_s[0]) / (2.0 * self.dr)

            if abs(f_out[-2]) > 1e-20:
                mismatch = df_out / f_out[-2] - df_in / f_in_s[1]
            else:
                mismatch = 0.0

            nodes = int(np.sum(np.diff(np.sign(f_out[f_out != 0])) != 0))

            if nodes > target_nodes:
                E_high = energy
            elif nodes < target_nodes:
                E_low = energy
            else:
                if abs(mismatch) < tol:
                    break
                if mismatch > 0:
                    E_low = energy
                else:
                    E_high = energy

            energy = (E_low + E_high) / 2.0
            if abs(E_high - E_low) < tol:
                break

        return energy

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve_bound_state(
        self, v: int, J: int = 0, energy_guess: Optional[float] = None
    ) -> BoundState:
        """
        Bound-state wavefunction for quantum number (v, J).

        Uses outward+inward shooting matched at the outer classical
        turning point.

        .. warning::
            Eigenvalue search has not been fully validated. Use DVR or
            MorseAnalyticSolver for production calculations.
        """
        V_eff = self.potential.effective_potential(self.r_grid, J, self.mu)

        if energy_guess is None:
            omega = np.sqrt(self.potential.force_constant / self.mu)
            energy_guess = self.potential.e_eq + omega * (v + 0.5)

        E_min = self.potential.e_eq
        E_max = self.potential.dissociation_energy

        def g_func(E: float) -> np.ndarray:
            return 2.0 * self.mu * (E - V_eff)

        energy = self._shooting_method(g_func, v, E_min, E_max)

        g = g_func(energy)
        match_idx = self._find_matching_point(energy, g)

        f_out = self._numerov_outward(g[:match_idx + 2])
        f_in = self._numerov_inward(g[match_idx - 1:])
        scale = f_out[match_idx] / f_in[1]
        f_in *= scale

        f = np.concatenate([f_out[:match_idx], f_in[1:]])
        if len(f) > self.n_grid:
            f = f[:self.n_grid]
        elif len(f) < self.n_grid:
            f = np.pad(f, (0, self.n_grid - len(f)))

        f_norm, norm = self._box_normalize(f)

        return BoundState(
            v=v, J=J, energy=energy,
            r_grid=self.r_grid.copy(),
            wavefunction=f_norm,
            normalization=norm,
        )

    def solve_scattering_state(
        self, E_collision: float, J: int = 0
    ) -> ScatteringState:
        """
        Scattering wavefunction at collision energy E_collision.

        Outward Numerov shooting from the inner wall (F(R_min) = 0),
        giving the physical standing-wave phase without box quantization.
        """
        E_total = self.potential.dissociation_energy + E_collision
        g = self._g_array(E_total, J)

        f = self._numerov_outward(g, f0=0.0, f1=1e-10)

        f_norm, _ = self._box_normalize(f)

        k = np.sqrt(2.0 * self.mu * E_collision)
        phase_shift = self._extract_phase_shift(
            self.r_grid[-100:], f_norm[-100:], k, J
        )

        return ScatteringState(
            E=E_collision, J=J,
            r_grid=self.r_grid.copy(),
            wavefunction=f_norm,
            phase_shift=phase_shift,
        )

    def solve_all_bound_states(self, J: int = 0) -> List[BoundState]:
        """All bound vibrational states for given J."""
        V_eff = self.potential.effective_potential(self.r_grid, J, self.mu)
        E_max = min(self.potential.dissociation_energy, np.max(V_eff))
        omega = np.sqrt(self.potential.force_constant / self.mu)

        states = []
        v = 0
        while True:
            E_guess = self.potential.e_eq + omega * (v + 0.5)
            if E_guess >= E_max or v > 50:
                break
            try:
                state = self.solve_bound_state(v, J, energy_guess=E_guess)
                if state.energy < E_max:
                    states.append(state)
                    v += 1
                else:
                    break
            except Exception:
                break

        return states
