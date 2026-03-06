"""
Base class and state containers for nuclear wavefunction solvers.

Defines the shared interface (BoundState, ScatteringState, WavefunctionSolver)
that all concrete solvers (DVR, Numerov, Morse) implement.
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from scipy.integrate import simpson
from scipy.optimize import brentq
from typing import List, Optional

from ..electronic.potential import PotentialEnergyCurve


# ---------------------------------------------------------------------------
# State containers
# ---------------------------------------------------------------------------


@dataclass
class BoundState:
    """Container for a bound vibrational state wavefunction."""

    v: int             # Vibrational quantum number
    J: int             # Rotational quantum number
    energy: float      # Energy (Hartree)
    r_grid: np.ndarray
    wavefunction: np.ndarray
    normalization: float


@dataclass
class ScatteringState:
    """Container for a continuum scattering state wavefunction."""

    E: float           # Collision energy (Hartree)
    J: int             # Angular momentum quantum number
    r_grid: np.ndarray
    wavefunction: np.ndarray
    phase_shift: float


# ---------------------------------------------------------------------------
# Abstract base solver
# ---------------------------------------------------------------------------


class WavefunctionSolver(ABC):
    """
    Abstract base class for nuclear radial wavefunction solvers.

    Sets up the uniform radial grid and provides helper methods shared by
    all concrete implementations: normalisation, 3-point derivative, and
    asymptotic phase-shift extraction.

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
        n_grid: int = 500,
    ) -> None:
        self.potential = potential
        self.mu = reduced_mass
        self.r_min = r_min
        self.r_max = r_max
        self.n_grid = n_grid
        self.r_grid = np.linspace(r_min, r_max, n_grid)
        self.dr = float(self.r_grid[1] - self.r_grid[0])

    # ------------------------------------------------------------------
    # Abstract interface — every solver must implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def solve_bound_state(
        self, v: int, J: int = 0, energy_guess: Optional[float] = None
    ) -> BoundState:
        """Bound-state wavefunction for quantum numbers (v, J)."""

    @abstractmethod
    def solve_scattering_state(
        self, E_collision: float, J: int = 0
    ) -> ScatteringState:
        """Scattering wavefunction at collision energy E_collision (Hartree)."""

    @abstractmethod
    def solve_all_bound_states(self, J: int = 0) -> List[BoundState]:
        """All bound vibrational states for given J."""

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _normalize_wavefunction(
        self, f: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """
        Normalize so that ∫|F(R)|² dR = 1.

        Returns (f_normalized, norm).
        """
        f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
        norm_sq = simpson(f ** 2, x=self.r_grid)
        if norm_sq <= 0 or not np.isfinite(norm_sq):
            norm_sq = np.sum(f ** 2) * self.dr
        norm = float(np.sqrt(max(norm_sq, 1e-30)))
        return f / norm, norm

    def wavefunction_derivative(self, state) -> np.ndarray:
        """
        Radial derivative dF/dR via 3-point central differences, O(dr²).

        Subclasses may override with a more accurate method (e.g. the
        TISE-based cumulative integral in MorseAnalyticSolver).
        """
        f = state.wavefunction
        deriv = np.zeros_like(f)
        deriv[1:-1] = (f[2:] - f[:-2]) / (2.0 * self.dr)
        deriv[0] = (f[1] - f[0]) / self.dr
        deriv[-1] = (f[-1] - f[-2]) / self.dr
        return deriv

    def _extract_phase_shift(
        self, r: np.ndarray, f: np.ndarray, k: float, J: int = 0
    ) -> float:
        """
        Asymptotic phase shift δ from wavefunction tail.

        Fits  F(R) ~ A sin(kR − Jπ/2 + δ)  using two widely separated
        points via Brent root-finding on the two-point consistency equation.
        """
        r1, r2 = r[-50], r[-10]
        f1, f2 = f[-50], f[-10]
        arg1 = k * r1 - J * np.pi / 2.0
        arg2 = k * r2 - J * np.pi / 2.0

        def eq(delta: float) -> float:
            return f1 * np.sin(arg2 + delta) - f2 * np.sin(arg1 + delta)

        try:
            return float(brentq(eq, -np.pi, np.pi))
        except ValueError:
            return 0.0

    def get_interpolated_wavefunction(self, state, kind: str = "cubic"):
        """Return a scipy interpolant for the wavefunction."""
        from scipy.interpolate import interp1d
        return interp1d(
            state.r_grid, state.wavefunction,
            kind=kind, bounds_error=False, fill_value=0.0,
        )
