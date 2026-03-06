"""
High-level interface for the nuclear wavefunction subpackage.

Re-exports the state containers and solver classes, and provides the
``create_wavefunction_solver`` factory for selecting a solver by name.

Typical usage
-------------
>>> from aed_rate.nuclear import create_wavefunction_solver, BoundState
>>> solver = create_wavefunction_solver(potential, mu, method="morse")
>>> state  = solver.solve_bound_state(v=3, J=0)
"""

from ._base_solver import BoundState, ScatteringState, WavefunctionSolver
from .dvr_solver import DVRWavefunctionSolver
from .numerov_solver import NumerovWavefunctionSolver

from ..electronic.potential import PotentialEnergyCurve
from typing import Literal


def create_wavefunction_solver(
    potential: PotentialEnergyCurve,
    reduced_mass: float,
    method: Literal["dvr", "numerov", "morse"] = "dvr",
    r_min: float = 0.5,
    r_max: float = 20.0,
    n_grid: int = 500,
) -> WavefunctionSolver:
    """
    Factory for nuclear wavefunction solvers.

    Parameters
    ----------
    potential : PotentialEnergyCurve
        Potential energy curve for the diatomic system.
    reduced_mass : float
        Reduced mass in atomic units.
    method : {'dvr', 'numerov', 'morse'}
        Solver algorithm:

        ``'dvr'``
            Discrete Variable Representation — matrix diagonalization.
            Reliable for bound states; scattering states have box-phase
            and derivative-amplitude errors.
        ``'numerov'``
            Numerov shooting method. Works for any PEC; scattering states
            have correct inner-wall phase. Bound-state search not yet
            fully validated.
        ``'morse'``
            Analytical Laguerre bound states + Numerov scattering states
            on the Pekeris-corrected Morse potential. Requires a
            ``MorsePotential`` instance. Best choice for coupling integrals.
    r_min, r_max : float
        Radial grid bounds (Bohr).
    n_grid : int
        Number of grid points.

    Returns
    -------
    WavefunctionSolver
        Concrete solver instance.
    """
    method = method.lower()
    if method == "dvr":
        return DVRWavefunctionSolver(potential, reduced_mass, r_min, r_max, n_grid)
    elif method == "numerov":
        return NumerovWavefunctionSolver(potential, reduced_mass, r_min, r_max, n_grid)
    elif method == "morse":
        from .morse_solver import MorseAnalyticSolver
        return MorseAnalyticSolver(potential, reduced_mass, r_min, r_max, n_grid)
    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose 'dvr', 'numerov', or 'morse'."
        )
