"""
DVR (Discrete Variable Representation) nuclear wavefunction solver.

Solves the radial Schrödinger equation by matrix diagonalization of the
finite-difference Hamiltonian on a uniform grid:

    H = T + V_eff(R)

where T is the tridiagonal second-derivative matrix and V_eff includes
the centrifugal term J(J+1)/(2μR²).

Strengths
---------
- Gives all bound eigenvalues/eigenvectors simultaneously.
- Bound-state wavefunctions are numerically clean and well-tested.

Limitations
-----------
- Scattering states suffer from box quantization: both F(R_min) = 0
  and F(R_max) = 0 are enforced, selecting an arbitrary standing-wave
  phase instead of the physical one.
- The 3-point derivative underestimates dF/dR by sin(kΔ)/(kΔ) ≈ 0.91
  in the inner Morse well where k_local ≈ 25 Bohr⁻¹.

For coupling integrals requiring the correct scattering phase and an
accurate dF/dR, use MorseAnalyticSolver instead.
"""

from __future__ import annotations

import warnings

import numpy as np
from typing import List, Optional

from ..electronic.potential import PotentialEnergyCurve
from ..utils.constants import AEDValidationWarning
from ._base_solver import BoundState, ScatteringState, WavefunctionSolver


class DVRWavefunctionSolver(WavefunctionSolver):
    """
    DVR solver for nuclear radial wavefunctions.

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
        super().__init__(potential, reduced_mass, r_min, r_max, n_grid)
        self._build_kinetic_matrix()
        self._eigenvalue_cache: dict = {}
        self._eigenvector_cache: dict = {}

    # ------------------------------------------------------------------
    # Hamiltonian construction
    # ------------------------------------------------------------------

    def _build_kinetic_matrix(self) -> None:
        """
        Build tridiagonal kinetic-energy matrix T = −ℏ²/(2μ) d²/dR².

        In finite-difference form on a uniform grid with spacing dr:
            T_ii  =  1 / (μ dr²)
            T_i,i±1 = −1 / (2μ dr²)
        """
        n = self.n_grid
        dr2 = self.dr ** 2
        diag_main = np.full(n, 2.0)
        diag_off = np.full(n - 1, -1.0)
        self._T = (1.0 / (2.0 * self.mu * dr2)) * (
            np.diag(diag_main) + np.diag(diag_off, -1) + np.diag(diag_off, 1)
        )

    def _solve_eigensystem(self, J: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Diagonalize the Hamiltonian for angular momentum J.

        Results are cached: repeated calls for the same J are free.

        Returns (eigenvalues, eigenvectors) where eigenvectors[i] is the
        i-th wavefunction (row vector).
        """
        if J in self._eigenvalue_cache:
            return self._eigenvalue_cache[J], self._eigenvector_cache[J]

        V_eff = self.potential.effective_potential(self.r_grid, J, self.mu)
        V_matrix = np.diag(V_eff)
        # Hard-wall BCs: large diagonal values at the grid edges
        V_matrix[0, 0] = 10.0
        V_matrix[-1, -1] = 10.0

        eigenvalues, eigenvectors = np.linalg.eigh(self._T + V_matrix)
        eigenvectors = eigenvectors.T  # rows → individual wavefunctions

        self._eigenvalue_cache[J] = eigenvalues
        self._eigenvector_cache[J] = eigenvectors
        return eigenvalues, eigenvectors

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve_bound_state(
        self, v: int, J: int = 0, energy_guess: Optional[float] = None
    ) -> BoundState:
        """
        Bound-state wavefunction for vibrational quantum number v at J.

        Parameters
        ----------
        v : int
            Vibrational quantum number (0-based).
        J : int
            Rotational quantum number.
        energy_guess : float, optional
            Ignored (kept for API compatibility with other solvers).
        """
        eigenvalues, eigenvectors = self._solve_eigensystem(J)

        E_dissoc = self.potential.dissociation_energy
        bound_indices = np.where(eigenvalues < E_dissoc)[0]

        if v >= len(bound_indices):
            raise ValueError(
                f"Vibrational state v={v} not bound at J={J}. "
                f"Only {len(bound_indices)} bound states exist."
            )

        idx = bound_indices[v]
        wf, norm = self._box_normalize(eigenvectors[idx])

        return BoundState(
            v=v, J=J,
            energy=float(eigenvalues[idx]),
            r_grid=self.r_grid.copy(),
            wavefunction=wf,
            normalization=norm,
        )

    def solve_scattering_state(
        self, E_collision: float, J: int = 0
    ) -> ScatteringState:
        """
        Scattering wavefunction at collision energy E_collision.

        Selects the box eigenstate whose energy is closest to
        E_dissoc + E_collision, then box-normalizes it (∫|F|² dR = 1).

        Note: the box enforces F(R_max) = 0, which selects an arbitrary
        standing-wave phase rather than the physical scattering phase.  This
        biases coupling integrals.  Use the analytical Morse solver
        (solver_method='morse') for coupling integrals / cross sections, which
        gives the correct phase, an exact derivative, and a unit-amplitude
        normalization option.
        """
        warnings.warn(
            "DVR scattering states enforce F(R_max)=0, giving an arbitrary "
            "standing-wave phase that biases coupling integrals. Use "
            "solver_method='morse' for coupling integrals / cross sections.",
            AEDValidationWarning,
            stacklevel=2,
        )
        eigenvalues, eigenvectors = self._solve_eigensystem(J)

        E_total = self.potential.dissociation_energy + E_collision
        continuum_indices = np.where(eigenvalues >= self.potential.dissociation_energy)[0]

        if len(continuum_indices) == 0:
            raise ValueError(
                f"No continuum states found. Try increasing r_max or n_grid."
            )

        closest = continuum_indices[
            np.argmin(np.abs(eigenvalues[continuum_indices] - E_total))
        ]

        wf, _ = self._box_normalize(eigenvectors[closest])

        k = np.sqrt(2.0 * self.mu * E_collision)

        phase_shift = self._extract_phase_shift(
            self.r_grid[-100:], wf[-100:], k, J
        )

        return ScatteringState(
            E=E_collision, J=J,
            r_grid=self.r_grid.copy(),
            wavefunction=wf,
            phase_shift=phase_shift,
        )

    def solve_all_bound_states(self, J: int = 0) -> List[BoundState]:
        """All bound vibrational states for given J."""
        eigenvalues, eigenvectors = self._solve_eigensystem(J)
        E_dissoc = self.potential.dissociation_energy
        bound_indices = np.where(eigenvalues < E_dissoc)[0]

        states = []
        for v, idx in enumerate(bound_indices):
            wf, norm = self._box_normalize(eigenvectors[idx])
            states.append(BoundState(
                v=v, J=J,
                energy=float(eigenvalues[idx]),
                r_grid=self.r_grid.copy(),
                wavefunction=wf,
                normalization=norm,
            ))
        return states
