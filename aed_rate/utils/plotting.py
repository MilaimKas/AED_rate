"""
Plotting utilities for AED rate calculation ingredients.

Provides publication-quality visualization of potential energy curves,
nuclear wavefunctions (bound and scattering), and energy level diagrams.

All functions return (fig, ax) or (fig, axes) for further customization.
"""

import numpy as np
from typing import Optional, List, Tuple, Union, Literal

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from ..utils.constants import CONSTANTS


# ======================================================================
# Helpers
# ======================================================================

EnergyUnit = Literal["eV", "cm-1", "hartree"]


def _require_matplotlib() -> None:
    """Raise an informative error if matplotlib is not installed."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install matplotlib  "
            "or: pip install aed_rate[plot]"
        )


def _energy_converter(unit: EnergyUnit):
    """Return (scale_factor, label_string) for the chosen energy unit."""
    if unit == "eV":
        return CONSTANTS.hartree_to_ev, "Energy (eV)"
    elif unit == "cm-1":
        return CONSTANTS.hartree_to_cm1, r"Energy (cm$^{-1}$)"
    elif unit == "hartree":
        return 1.0, "Energy (Hartree)"
    else:
        raise ValueError(f"Unknown unit '{unit}'. Use 'eV', 'cm-1', or 'hartree'.")


def _get_colors(n: int) -> list:
    """Return n distinct colors from a colormap."""
    cmap = plt.cm.viridis
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


# ======================================================================
# 1. Potential energy curves
# ======================================================================

def plot_potential_curves(
    anion,
    neutral,
    EA: float,
    reduced_mass: Optional[float] = None,
    J_values: Optional[List[int]] = None,
    R_range: Tuple[float, float] = (0.8, 8.0),
    n_points: int = 500,
    unit: EnergyUnit = "eV",
    E_ref: str = "anion_min",
    figsize: Tuple[float, float] = (8, 6),
):
    """
    Plot anion and neutral potential energy curves.

    Parameters
    ----------
    anion : PotentialEnergyCurve
        Anion potential (e.g. OH⁻)
    neutral : PotentialEnergyCurve
        Neutral potential (e.g. OH)
    EA : float
        Electron affinity in Hartree
    reduced_mass : float, optional
        Reduced mass (a.u.), required if J_values is given
    J_values : list of int, optional
        Angular momenta for centrifugal barrier overlay
    R_range : tuple
        (R_min, R_max) in Bohr
    n_points : int
        Number of R grid points for plotting
    unit : str
        Energy axis unit: 'eV', 'cm-1', or 'hartree'
    E_ref : str
        Energy reference: 'anion_min' (default) shifts so anion minimum = 0
    figsize : tuple
        Figure size in inches

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    _require_matplotlib()
    scale, ylabel = _energy_converter(unit)

    R = np.linspace(R_range[0], R_range[1], n_points)
    V_an = np.asarray(anion(R), dtype=float)
    V_ne = np.asarray(neutral(R), dtype=float)

    # Reference energy
    E0 = anion.e_eq if E_ref == "anion_min" else 0.0

    fig, ax = plt.subplots(figsize=figsize)

    # Main PECs
    ax.plot(R, (V_an - E0) * scale, "b-", lw=2, label=r"Anion (OH$^-$)")
    ax.plot(R, (V_ne - E0) * scale, "r-", lw=2, label="Neutral (OH)")

    # Dissociation limits
    D_an = (anion.dissociation_energy - E0) * scale
    D_ne = (neutral.dissociation_energy - E0) * scale
    ax.axhline(D_an, color="b", ls="--", lw=0.8, alpha=0.5)
    ax.axhline(D_ne, color="r", ls="--", lw=0.8, alpha=0.5)

    # EA annotation
    R_mid = 0.5 * (R_range[0] + R_range[1])
    V_an_mid = float(anion(np.array([R_mid]))) if hasattr(anion(np.array([R_mid])), '__float__') else anion(np.array([R_mid]))[0]
    V_ne_mid = float(neutral(np.array([R_mid]))) if hasattr(neutral(np.array([R_mid])), '__float__') else neutral(np.array([R_mid]))[0]
    # Place EA arrow at dissociation limits
    ax.annotate(
        "", xy=(R_range[1] * 0.85, D_an), xytext=(R_range[1] * 0.85, D_ne),
        arrowprops=dict(arrowstyle="<->", color="green", lw=1.5),
    )
    ax.text(
        R_range[1] * 0.87, 0.5 * (D_an + D_ne),
        f"EA = {EA * scale:.2f} {unit}", color="green", fontsize=10,
        va="center",
    )

    # Equilibrium markers
    ax.plot(anion.r_eq, (anion.e_eq - E0) * scale, "bv", ms=8)
    ax.plot(neutral.r_eq, (neutral.e_eq - E0) * scale, "rv", ms=8)

    # Centrifugal barriers
    if J_values and reduced_mass is not None:
        for J in J_values:
            V_eff_an = np.asarray(
                anion.effective_potential(R, J, reduced_mass), dtype=float
            )
            ax.plot(
                R, (V_eff_an - E0) * scale, "b:",
                lw=1, alpha=0.6, label=f"Anion J={J}",
            )

    ax.set_xlabel("R (Bohr)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(r"Potential Energy Curves: O$^-$ + H System", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(R_range)

    # Set sensible y-limits: focus on the well region
    y_min = -0.5
    y_max = max(D_an, D_ne) * 1.15
    ax.set_ylim(y_min, y_max)

    fig.tight_layout()
    return fig, ax


# ======================================================================
# 2. Bound state wavefunctions overlaid on PEC
# ======================================================================

def plot_bound_states(
    states: list,
    potential,
    n_states: int = 6,
    wf_scale: Optional[float] = None,
    plot_density: bool = False,
    R_range: Optional[Tuple[float, float]] = None,
    unit: EnergyUnit = "eV",
    E_ref: Optional[float] = None,
    figsize: Tuple[float, float] = (8, 7),
):
    """
    Plot bound state wavefunctions overlaid on PEC (textbook style).

    Each wavefunction is shifted vertically to its energy level and
    scaled for visibility.

    Parameters
    ----------
    states : list of BoundState
        Bound states to plot (from solve_all_bound_states)
    potential : PotentialEnergyCurve
        Potential energy curve
    n_states : int
        Number of states to plot (from v=0)
    wf_scale : float, optional
        Scaling factor for wavefunction amplitude. If None, auto-scaled.
    plot_density : bool
        If True, plot |F(R)|² instead of F(R)
    R_range : tuple, optional
        (R_min, R_max) in Bohr. If None, auto from potential.
    unit : str
        Energy axis unit
    E_ref : float, optional
        Reference energy (Hartree). If None, uses potential minimum.
    figsize : tuple
        Figure size

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    _require_matplotlib()
    scale, ylabel = _energy_converter(unit)

    n_plot = min(n_states, len(states))
    if n_plot == 0:
        raise ValueError("No states to plot.")

    # Reference energy
    if E_ref is None:
        E_ref = potential.e_eq

    # R range
    if R_range is None:
        R_range = (
            max(0.5, potential.r_eq - 2.0),
            min(potential.r_eq + 6.0, states[0].r_grid[-1]),
        )

    # Dense R grid for potential curve
    R_fine = np.linspace(R_range[0], R_range[1], 500)
    V = np.asarray(potential(R_fine), dtype=float)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot PEC
    ax.plot(R_fine, (V - E_ref) * scale, "k-", lw=2, label="V(R)")

    # Dissociation limit
    D = (potential.dissociation_energy - E_ref) * scale
    ax.axhline(D, color="k", ls="--", lw=0.8, alpha=0.4, label="Dissoc. limit")

    # Auto-scale: make wavefunction amplitude ~15% of energy range
    if wf_scale is None:
        E_range = (states[min(n_plot - 1, len(states) - 1)].energy - states[0].energy)
        if E_range > 0:
            wf_scale = 0.3 * E_range * scale
        else:
            wf_scale = 0.5 * scale

    colors = _get_colors(n_plot)

    for i in range(n_plot):
        state = states[i]
        E_v = (state.energy - E_ref) * scale

        # Energy level line
        r_in_range = (state.r_grid >= R_range[0]) & (state.r_grid <= R_range[1])
        R_wf = state.r_grid[r_in_range]
        wf = state.wavefunction[r_in_range]

        if plot_density:
            wf_plot = wf ** 2
        else:
            wf_plot = wf

        # Draw energy level
        ax.axhline(E_v, color=colors[i], ls="-", lw=0.5, alpha=0.4)

        # Draw wavefunction shifted to energy level
        ax.plot(R_wf, E_v + wf_scale * wf_plot, color=colors[i], lw=1.2)

        # Fill between for visual clarity
        ax.fill_between(
            R_wf, E_v, E_v + wf_scale * wf_plot,
            alpha=0.15, color=colors[i],
        )

        # Label
        ax.text(
            R_range[1] * 0.95, E_v,
            f"v={state.v}", fontsize=9, va="center", color=colors[i],
        )

    ax.set_xlabel("R (Bohr)", fontsize=12)
    wf_label = r"$|\psi|^2$" if plot_density else r"$\psi(R)$"
    ax.set_ylabel(f"{ylabel}  (+scaled {wf_label})", fontsize=12)
    ax.set_title("Bound Vibrational States", fontsize=13)
    ax.set_xlim(R_range)

    # Zoom y-axis into the well: from just below minimum to ~20% above
    # the highest plotted state (or dissociation, whichever is lower)
    E_min_plot = (potential.e_eq - E_ref) * scale
    E_top = (states[n_plot - 1].energy - E_ref) * scale
    y_margin = 0.2 * (E_top - E_min_plot + wf_scale)
    ax.set_ylim(E_min_plot - y_margin, E_top + wf_scale + y_margin)

    fig.tight_layout()
    return fig, ax


# ======================================================================
# 3. Scattering state with effective potential
# ======================================================================

def plot_scattering_state(
    scattering_state,
    potential,
    reduced_mass: float,
    J: int = 0,
    R_range: Optional[Tuple[float, float]] = None,
    unit: EnergyUnit = "eV",
    E_ref: Optional[float] = None,
    figsize: Tuple[float, float] = (8, 8),
):
    """
    Plot scattering state: effective potential + wavefunction.

    Two-panel figure:
    - Top: effective potential V_eff(R,J) with collision energy line
    - Bottom: scattering wavefunction F(R)

    Parameters
    ----------
    scattering_state : ScatteringState
        Scattering state from solve_scattering_state
    potential : PotentialEnergyCurve
        Potential (typically anion)
    reduced_mass : float
        Reduced mass in a.u.
    J : int
        Angular momentum (for labeling; V_eff computed from potential)
    R_range : tuple, optional
        (R_min, R_max) in Bohr
    unit : str
        Energy axis unit for top panel
    E_ref : float, optional
        Reference energy. If None, uses potential minimum.
    figsize : tuple
        Figure size

    Returns
    -------
    fig, axes : matplotlib Figure and array of Axes (2 panels)
    """
    _require_matplotlib()
    scale, ylabel = _energy_converter(unit)

    if E_ref is None:
        E_ref = potential.e_eq

    if R_range is None:
        R_range = (scattering_state.r_grid[0], scattering_state.r_grid[-1])

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True,
                             gridspec_kw={"height_ratios": [1, 1]})

    R_fine = np.linspace(R_range[0], R_range[1], 500)

    # --- Top panel: effective potential ---
    ax_pot = axes[0]
    V_eff = np.asarray(
        potential.effective_potential(R_fine, J, reduced_mass), dtype=float
    )
    V_bare = np.asarray(potential(R_fine), dtype=float)

    ax_pot.plot(R_fine, (V_bare - E_ref) * scale, "b-", lw=1.5,
                label="V(R)", alpha=0.5)
    if J > 0:
        ax_pot.plot(R_fine, (V_eff - E_ref) * scale, "b-", lw=2,
                    label=f"V_eff(R, J={J})")
    else:
        ax_pot.plot(R_fine, (V_eff - E_ref) * scale, "b-", lw=2,
                    label="V_eff(R, J=0)")

    # Collision energy line
    E_total = potential.dissociation_energy + scattering_state.E
    ax_pot.axhline(
        (E_total - E_ref) * scale, color="r", ls="--", lw=1.5,
        label=f"E_coll = {scattering_state.E * scale:.4f} {unit}",
    )

    # Dissociation limit
    ax_pot.axhline(
        (potential.dissociation_energy - E_ref) * scale,
        color="gray", ls=":", lw=1, alpha=0.5,
    )

    # Zoom into well region: from minimum to a bit above collision energy
    D_plot = (potential.dissociation_energy - E_ref) * scale
    E_coll_plot = (E_total - E_ref) * scale
    ax_pot.set_ylim(-0.5, max(D_plot, E_coll_plot) * 1.3)

    ax_pot.set_ylabel(ylabel, fontsize=12)
    ax_pot.legend(fontsize=9)
    ax_pot.set_title(
        f"Scattering State: E = {scattering_state.E * CONSTANTS.hartree_to_cm1:.0f} cm⁻¹, "
        f"J = {J}, δ = {scattering_state.phase_shift:.3f} rad",
        fontsize=12,
    )

    # --- Bottom panel: wavefunction ---
    ax_wf = axes[1]
    r_grid = scattering_state.r_grid
    wf = scattering_state.wavefunction

    mask = (r_grid >= R_range[0]) & (r_grid <= R_range[1])
    ax_wf.plot(r_grid[mask], wf[mask], "b-", lw=1.2)
    ax_wf.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax_wf.set_xlabel("R (Bohr)", fontsize=12)
    ax_wf.set_ylabel("F(R)", fontsize=12)
    ax_wf.set_xlim(R_range)

    fig.tight_layout()
    return fig, axes


# ======================================================================
# 4. Energy level diagram
# ======================================================================

def plot_energy_levels(
    anion,
    neutral,
    reduced_mass: float,
    EA: float,
    v_max: int = 12,
    unit: EnergyUnit = "eV",
    figsize: Tuple[float, float] = (7, 8),
):
    """
    Plot vibrational energy level diagram for anion and neutral.

    Horizontal lines for each vibrational level, arranged in two columns
    (anion on left, neutral on right), with the EA gap visible.

    Parameters
    ----------
    anion : MorsePotential
        Anion potential
    neutral : MorsePotential
        Neutral potential
    reduced_mass : float
        Reduced mass in a.u.
    EA : float
        Electron affinity in Hartree
    v_max : int
        Maximum vibrational quantum number to show
    unit : str
        Energy axis unit
    figsize : tuple
        Figure size

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    _require_matplotlib()
    scale, ylabel = _energy_converter(unit)

    # Get vibrational energies (relative to each minimum)
    E_anion = anion.vibrational_energies(reduced_mass, v_max=v_max)
    E_neutral = neutral.vibrational_energies(reduced_mass, v_max=v_max)

    # Absolute energies (anion min = 0 reference)
    E_anion_abs = E_anion + anion.V_0   # anion.V_0 = 0 typically
    E_neutral_abs = E_neutral + neutral.V_0  # neutral.V_0 = EA

    fig, ax = plt.subplots(figsize=figsize)

    # Anion levels (left column, x = 0.1 to 0.45)
    for v, E in enumerate(E_anion_abs):
        ax.plot([0.1, 0.45], [E * scale, E * scale], "b-", lw=1.5)
        ax.text(0.07, E * scale, f"{v}", fontsize=8, va="center",
                ha="right", color="b")

    # Neutral levels (right column, x = 0.55 to 0.9)
    for v, E in enumerate(E_neutral_abs):
        ax.plot([0.55, 0.9], [E * scale, E * scale], "r-", lw=1.5)
        ax.text(0.93, E * scale, f"{v}", fontsize=8, va="center",
                ha="left", color="r")

    # Dissociation limits
    D_an = anion.dissociation_energy * scale
    D_ne = neutral.dissociation_energy * scale
    ax.plot([0.1, 0.45], [D_an, D_an], "b--", lw=1, alpha=0.5)
    ax.plot([0.55, 0.9], [D_ne, D_ne], "r--", lw=1, alpha=0.5)

    # EA gap annotation
    ax.annotate(
        "", xy=(0.5, anion.V_0 * scale), xytext=(0.5, neutral.V_0 * scale),
        arrowprops=dict(arrowstyle="<->", color="green", lw=2),
    )
    mid_EA = 0.5 * (anion.V_0 + neutral.V_0) * scale
    ax.text(0.52, mid_EA, f"EA = {EA * scale:.3f} {unit}",
            fontsize=10, color="green", va="center")

    # Labels
    ax.text(0.275, D_an * 1.05, r"OH$^-$ (anion)", fontsize=12,
            ha="center", color="b", fontweight="bold")
    ax.text(0.725, D_ne * 1.05, "OH (neutral)", fontsize=12,
            ha="center", color="r", fontweight="bold")

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title("Vibrational Energy Levels", fontsize=13)

    # Remove x-axis (it's just a layout axis)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig, ax


# ======================================================================
# 5. Electronic coupling vs R (with optional HF PECs)
# ======================================================================

def plot_coupling_curve(
    results: list,
    E_anion: Optional[np.ndarray] = None,
    E_neutral: Optional[np.ndarray] = None,
    E_homo: Optional[np.ndarray] = None,
    unit: EnergyUnit = "eV",
    log_scale: bool = False,
    figsize: Tuple[float, float] = (9, 10),
):
    """
    Plot electronic coupling |m_rad(R)|, |m_rot(R)| with optional HF diagnostics.

    Creates a multi-panel figure:
    - Panel 1 (if HF energies provided): HF PECs for anion and neutral
    - Panel 2: coupling magnitudes |m_rad| and |m_rot|
    - Panel 3 (if HOMO energy provided): HOMO orbital energy vs R
    - Bottom of panel 2: ratio |m_rot|/|m_rad| where both are meaningful

    Parameters
    ----------
    results : list of CouplingResult
        Coupling results at different R (from compute_coupling_curve)
    E_anion : np.ndarray, optional
        HF total energy of anion at each R (Hartree)
    E_neutral : np.ndarray, optional
        HF total energy of neutral at each R (Hartree)
    E_homo : np.ndarray, optional
        HOMO orbital energy at each R (Hartree)
    unit : str
        Energy unit for PEC panel
    log_scale : bool
        If True, use log scale on coupling y-axis
    figsize : tuple
        Figure size

    Returns
    -------
    fig, axes : matplotlib Figure and array of Axes
    """
    _require_matplotlib()
    scale, ylabel = _energy_converter(unit)

    R_vals = np.array([r.R for r in results])
    m_rad = np.array([abs(r.m_rad) for r in results])
    m_rot = np.array([abs(r.m_rot) for r in results])

    # Determine number of panels
    has_pec = E_anion is not None and E_neutral is not None
    has_homo = E_homo is not None
    n_panels = 1 + int(has_pec) + int(has_homo)
    ratios = []
    if has_pec:
        ratios.append(1.0)
    ratios.append(1.0)  # coupling panel always present
    if has_homo:
        ratios.append(0.7)

    fig, axes = plt.subplots(
        n_panels, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": ratios},
    )
    if n_panels == 1:
        axes = [axes]

    panel_idx = 0

    # --- Panel: HF PECs ---
    if has_pec:
        ax_pec = axes[panel_idx]
        # Plot relative to anion energy at first R (arbitrary reference)
        E_ref = E_anion[0]
        ax_pec.plot(R_vals, (E_anion - E_ref) * scale, "b-o", lw=2, ms=4,
                    label=r"$E_\mathrm{anion}$ (RHF)")
        ax_pec.plot(R_vals, (E_neutral - E_ref) * scale, "r-s", lw=2, ms=4,
                    label=r"$E_\mathrm{neutral}$ (UHF)")
        # Also show the gap
        gap = (E_neutral - E_anion) * scale
        ax_gap = ax_pec.twinx()
        ax_gap.plot(R_vals, gap, "g--", lw=1.5, alpha=0.6, label="Gap")
        ax_gap.set_ylabel(f"Gap ({unit})", fontsize=10, color="green")
        ax_gap.tick_params(axis="y", labelcolor="green")
        ax_gap.axhline(0, color="green", ls=":", lw=0.8, alpha=0.4)

        ax_pec.set_ylabel(f"HF Energy ({unit})", fontsize=12)
        ax_pec.set_title("Ab Initio Coupling: HF PECs + Coupling vs R", fontsize=13)
        ax_pec.legend(fontsize=9, loc="upper left")
        ax_gap.legend(fontsize=9, loc="upper right")
        ax_pec.grid(True, alpha=0.3)
        panel_idx += 1

    # --- Panel: Coupling magnitudes ---
    ax_coup = axes[panel_idx]
    ax_coup.plot(R_vals, m_rad, "bo-", lw=2, ms=5,
                 label=r"$|m_\mathrm{rad}|$ (radial)")
    ax_coup.plot(R_vals, m_rot, "rs-", lw=2, ms=5,
                 label=r"$|m_\mathrm{rot}|$ (rotational)")
    if log_scale:
        ax_coup.set_yscale("log")
    ax_coup.set_ylabel("Coupling (a.u.)", fontsize=12)
    ax_coup.legend(fontsize=10)
    ax_coup.grid(True, alpha=0.3)
    if not has_pec:
        ax_coup.set_title("Electronic Non-BO Coupling vs R", fontsize=13)
    panel_idx += 1

    # --- Panel: HOMO orbital energy ---
    if has_homo:
        ax_homo = axes[panel_idx]
        ax_homo.plot(R_vals, E_homo * scale, "m-D", lw=2, ms=4,
                     label=r"$\epsilon_\mathrm{HOMO}$")
        ax_homo.set_ylabel(f"HOMO energy ({unit})", fontsize=12)
        ax_homo.set_xlabel("R (Bohr)", fontsize=12)
        ax_homo.legend(fontsize=10)
        ax_homo.grid(True, alpha=0.3)
    else:
        ax_coup.set_xlabel("R (Bohr)", fontsize=12)

    fig.tight_layout()
    return fig, axes


# ======================================================================
# 6. Scattering-state derivative dF_E/dR
# ======================================================================

def plot_scattering_derivative(
    scattering_state,
    derivative: Optional[np.ndarray] = None,
    R_range: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (8, 6),
):
    """
    Plot the scattering wavefunction F_E(R) and its derivative dF_E/dR(R).

    The radial coupling integral V_rad = (1/μ)∫ χ_{v'} m_rad (dF_E/dR) dR uses
    dF_E/dR — the fast oscillator that drives the cancellation (its local
    wavenumber is large in the inner Morse well).  This figure shows F_E and
    dF_E/dR on a shared R axis so that oscillation can be inspected directly.

    Parameters
    ----------
    scattering_state : ScatteringState
        From solve_scattering_state.  If ``derivative`` is None, the function
        uses ``scattering_state._analytical_derivative`` when present
        (the exact TISE derivative from the Morse solver), else np.gradient.
    derivative : np.ndarray, optional
        dF_E/dR on the state's r_grid.  Pass the solver's
        ``wavefunction_derivative(state)`` for the exact value.
    R_range : (float, float), optional
        Plot range in Bohr (default: full grid).
    figsize : tuple
        Figure size.

    Returns
    -------
    fig, axes : matplotlib Figure and the two Axes (F, dF/dR).
    """
    _require_matplotlib()

    r = scattering_state.r_grid
    F = scattering_state.wavefunction
    if derivative is None:
        derivative = getattr(scattering_state, "_analytical_derivative", None)
        if derivative is None:
            derivative = np.gradient(F, r)

    if R_range is None:
        R_range = (r[0], r[-1])
    mask = (r >= R_range[0]) & (r <= R_range[1])

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    axes[0].plot(r[mask], F[mask], "b-", lw=1.2)
    axes[0].axhline(0, color="k", lw=0.5, alpha=0.3)
    axes[0].set_ylabel(r"$F_E(R)$", fontsize=12)
    axes[0].set_title(
        f"Scattering state: E = "
        f"{scattering_state.E * CONSTANTS.hartree_to_cm1:.0f} cm⁻¹, "
        f"J = {scattering_state.J}",
        fontsize=12,
    )
    axes[1].plot(r[mask], np.asarray(derivative)[mask], "r-", lw=1.2)
    axes[1].axhline(0, color="k", lw=0.5, alpha=0.3)
    axes[1].set_ylabel(r"$dF_E/dR$", fontsize=12)
    axes[1].set_xlabel("R (Bohr)", fontsize=12)
    axes[1].set_xlim(R_range)

    fig.tight_layout()
    return fig, axes


# ======================================================================
# 7. Electronic intermediates: ∂φ_HOMO/∂R and the OPW φ_k (molecular plane)
# ======================================================================

def _dominant_real(field: np.ndarray) -> Tuple[np.ndarray, str]:
    """Return (real or imaginary part, label) — whichever dominates in norm."""
    field = np.asarray(field)
    if np.iscomplexobj(field):
        if np.linalg.norm(field.imag) > np.linalg.norm(field.real):
            return field.imag, "Im"
        return field.real, "Re"
    return field, ""


def plot_electronic_intermediates(
    intermediates,
    slab: float = 0.4,
    figsize: Tuple[float, float] = (11, 9),
    n_levels: int = 30,
):
    """
    Contour the electronic coupling ingredients in the molecular (x–z) plane.

    Four panels show, on a thin slab |y| < ``slab`` through the bond axis:
    ∂φ_HOMO/∂R, ∂φ_HOMO/∂θ, and the two OPW continuum functions φ_k (radial /
    rotational channels).  m_rad = Σ w φ_k_rad* ∂φ/∂R is the overlap of the
    top-left and bottom-left panels (similarly m_rot, right column).

    Built from a :class:`CouplingIntermediates` object
    (ElectronicCoupling.compute_coupling_intermediates).  The Becke grid is
    scattered, so panels use triangulated contours of points within the slab.

    Parameters
    ----------
    intermediates : CouplingIntermediates
        From ElectronicCoupling.compute_coupling_intermediates.
    slab : float
        Half-thickness (Bohr) of the |y| slab selected around the molecular plane.
    figsize : tuple
        Figure size.
    n_levels : int
        Number of filled-contour levels.

    Returns
    -------
    fig, axes : matplotlib Figure and the 2×2 Axes array.
    """
    _require_matplotlib()

    c = intermediates.coords
    z, x, y = c[:, 2], c[:, 0], c[:, 1]            # bond axis = z, π direction = x
    in_slab = np.abs(y) < slab
    zz, xx = z[in_slab], x[in_slab]

    fields = [
        (intermediates.dphi_dR,      r"$\partial\varphi_{\rm HOMO}/\partial R$"),
        (intermediates.dphi_dtheta,  r"$\partial\varphi_{\rm HOMO}/\partial\theta$"),
        (intermediates.phi_k_rad,    r"OPW $\varphi_k$ (radial channel)"),
        (intermediates.phi_k_rot,    r"OPW $\varphi_k$ (rotational channel)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)
    for ax, (field, title) in zip(axes.flat, fields):
        vals, part = _dominant_real(field)
        vals = vals[in_slab]
        vmax = np.max(np.abs(vals)) or 1.0
        levels = np.linspace(-vmax, vmax, n_levels)
        tcf = ax.tricontourf(zz, xx, vals, levels=levels, cmap="RdBu_r", extend="both")
        fig.colorbar(tcf, ax=ax, shrink=0.85)
        # atom markers: O at origin, H at z = R
        ax.plot([0.0], [0.0], "ko", ms=7)
        ax.plot([intermediates.R], [0.0], "k^", ms=7)
        lbl = f"{part} {title}" if part else title
        ax.set_title(lbl, fontsize=10)
        ax.set_aspect("equal")

    for ax in axes[-1, :]:
        ax.set_xlabel("z — bond axis (Bohr)", fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel("x (Bohr)", fontsize=10)
    fig.suptitle(
        f"Electronic coupling ingredients at R = {intermediates.R:.3f} Bohr, "
        f"E_e = {intermediates.electron_energy * CONSTANTS.hartree_to_ev:.3f} eV "
        f"(slab |y| < {slab} Bohr)",
        fontsize=12,
    )
    fig.tight_layout()
    return fig, axes


# ======================================================================
# 8. Coupling integrand and its phase cancellation
# ======================================================================

def plot_coupling_integrand(
    r_grid: np.ndarray,
    bound_wavefunction: np.ndarray,
    m_rad_on_grid: np.ndarray,
    scattering_derivative: np.ndarray,
    label: str = "",
    R_range: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (8, 8),
):
    """
    Visualize the radial coupling integrand and its near-cancellation.

    The (un-normalized) radial coupling integral is

        ∫ g(R) dR,   g(R) = χ_{v'}(R) · m_rad(R) · dF_E/dR

    (V_rad = that integral divided by μ).  Two panels:
      - the integrand g(R) — large +/- lobes;
      - the running integral ∫^R g dR' normalized to its final value, which
        overshoots far beyond 1 before settling, revealing why the result is a
        small residual of heavy cancellation (and hence grid-sensitive).

    All three inputs must be sampled on the same ``r_grid``.

    Parameters
    ----------
    r_grid : np.ndarray
        Radial grid (Bohr).
    bound_wavefunction : np.ndarray
        Neutral bound state χ_{v'}(R) on r_grid.
    m_rad_on_grid : np.ndarray
        Radial electronic coupling m_rad(R) on r_grid (real part used).
    scattering_derivative : np.ndarray
        dF_E/dR on r_grid (e.g. solver.wavefunction_derivative(state)).
    label : str
        Annotation for the curve (e.g. "v'=6, k_e=0.22").
    R_range : (float, float), optional
        Plot range in Bohr (default: full grid).
    figsize : tuple
        Figure size.

    Returns
    -------
    fig, axes : matplotlib Figure and the two Axes (integrand, running integral).
    """
    _require_matplotlib()
    from scipy.integrate import cumulative_trapezoid

    g = (np.asarray(bound_wavefunction)
         * np.asarray(m_rad_on_grid).real
         * np.asarray(scattering_derivative))
    C = cumulative_trapezoid(g, r_grid, initial=0.0)
    C_final = C[-1]

    if R_range is None:
        R_range = (r_grid[0], r_grid[-1])
    mask = (r_grid >= R_range[0]) & (r_grid <= R_range[1])

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    axes[0].plot(r_grid[mask], g[mask], "b-", lw=1.2, label=label or None)
    axes[0].axhline(0, color="k", lw=0.5, alpha=0.3)
    axes[0].set_ylabel(r"integrand $g(R)=\chi\, m_{\rm rad}\, dF_E/dR$", fontsize=11)
    axes[0].set_title("Coupling integrand and its phase cancellation", fontsize=12)
    if label:
        axes[0].legend(fontsize=9)

    if abs(C_final) > 0:
        axes[1].plot(r_grid[mask], (C / C_final)[mask], "g-", lw=1.2)
    axes[1].axhline(1.0, color="k", ls="--", lw=0.6)
    axes[1].axhline(0, color="k", lw=0.5, alpha=0.3)
    axes[1].set_ylabel(r"$\int^R\! g\,dR' / \int^\infty\! g\,dR'$", fontsize=11)
    axes[1].set_xlabel("R (Bohr)", fontsize=12)
    axes[1].set_xlim(R_range)
    # annotate the cancellation severity
    overshoot = float(np.max(np.abs(C)) / abs(C_final)) if abs(C_final) > 0 else np.inf
    axes[1].text(0.02, 0.95,
                 f"final ∫g dR = {C_final:.2e}\novershoot ×{overshoot:.0f} → cancellation",
                 transform=axes[1].transAxes, va="top", fontsize=9, style="italic")

    fig.tight_layout()
    return fig, axes
