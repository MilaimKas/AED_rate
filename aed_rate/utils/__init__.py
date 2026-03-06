"""Shared utilities: physical constants, atomic data, unit conversions."""

from .constants import (
    CONSTANTS,
    ATOMIC_MASSES,
    ELECTRON_AFFINITIES,
    get_reduced_mass,
    get_total_mass,
)

# Plotting is optional (requires matplotlib)
try:
    from .plotting import (
        plot_potential_curves,
        plot_bound_states,
        plot_scattering_state,
        plot_energy_levels,
        plot_coupling_curve,
    )
except ImportError:
    pass
