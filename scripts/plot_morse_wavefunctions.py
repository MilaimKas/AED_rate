"""
Visualization of analytical Morse vs DVR wavefunctions.

Uses the existing aed_rate.utils.plotting utilities for standard panels
(bound states on PEC, scattering state with effective potential) and
adds custom comparison panels highlighting where the two methods differ:

  Row 0 — bound states v'=3..8 on neutral OH PEC (Morse analytic)
  Row 1 — scattering state on anion OH⁻ PEC (Morse analytic)
  Row 2 — direct overlay: DVR vs Morse scattering wavefunction + derivative
           in the coupling region

The derivative panel is the key diagnostic: the DVR central-difference
underestimates |dF/dR| by sin(kΔ)/(kΔ) ≈ 0.89 in the inner well, while
the Morse analytical derivative is exact.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from aed_rate.electronic.potential import create_oh_system_acharya
from aed_rate.nuclear.morse_solver import MorseAnalyticSolver
from aed_rate.nuclear.nuclear_wavefunction import DVRWavefunctionSolver
from aed_rate.utils.constants import get_reduced_mass, CONSTANTS
from aed_rate.utils.plotting import plot_bound_states, plot_scattering_state

# -----------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------
anion_pot, neutral_pot, EA = create_oh_system_acharya()
mu = get_reduced_mass("O", "H")
E_coll_cm1 = 66.0
E_ha = CONSTANTS.cm1_to_hartree(E_coll_cm1)
N = 500    # grid points
R_COUPLING = (1.2, 3.0)   # zoom window for comparison panels

# -----------------------------------------------------------------------
# Build solvers
# -----------------------------------------------------------------------
print("Building solvers …")
sol_morse_an = MorseAnalyticSolver(anion_pot,  mu, r_min=0.5, r_max=15.0, n_grid=N)
sol_morse_ne = MorseAnalyticSolver(neutral_pot, mu, r_min=0.5, r_max=15.0, n_grid=N)
sol_dvr_an   = DVRWavefunctionSolver(anion_pot,  mu, r_min=0.5, r_max=15.0, n_grid=N)

# -----------------------------------------------------------------------
# Compute wavefunctions
# -----------------------------------------------------------------------
print("Computing neutral bound states (Morse) …")
neutral_states_morse = [sol_morse_ne.solve_bound_state(v, J=0) for v in range(3, 9)]

print("Computing neutral bound states (DVR) …")
neutral_states_dvr   = [sol_dvr_an.solve_bound_state(v, J=0)
                        for v in range(3, 9)]   # on anion grid for comparibility
# For bound states we want neutral, so use the Morse solver's neutral grid
# The DVR neutral solver shares the same grid, use sol_dvr_ne below
sol_dvr_ne = DVRWavefunctionSolver(neutral_pot, mu, r_min=0.5, r_max=15.0, n_grid=N)
neutral_states_dvr = [sol_dvr_ne.solve_bound_state(v, J=0) for v in range(3, 9)]

print(f"Computing Morse scattering state at E={E_coll_cm1:.0f} cm⁻¹ …")
scat_morse = sol_morse_an.solve_scattering_state(E_ha, J=0)
dF_morse   = sol_morse_an.wavefunction_derivative(scat_morse)   # analytical

print("Computing DVR scattering state …")
scat_dvr   = sol_dvr_an.solve_scattering_state(E_ha, J=0)
dF_dvr     = sol_dvr_an.wavefunction_derivative(scat_dvr)       # central difference

R = sol_morse_an.r_grid

# -----------------------------------------------------------------------
# Figure 1 – Standard plots from plotting utilities
# -----------------------------------------------------------------------

# --- Bound states on neutral PEC (Morse) ---
print("Plotting bound states …")
fig1, ax1 = plot_bound_states(
    neutral_states_morse,
    neutral_pot,
    n_states=6,
    unit="eV",
    R_range=(0.8, 5.0),
    figsize=(8, 7),
)
ax1.set_title(
    r"Neutral OH bound states  v'=3…8  [analytical Morse, J=0]",
    fontsize=12,
)
fig1.tight_layout()
fig1.savefig("plots/morse_bound_states.png", dpi=140, bbox_inches="tight")
print("Saved  plots/morse_bound_states.png")

# --- Scattering state (Morse) ---
print("Plotting scattering state (Morse) …")
fig2, axes2 = plot_scattering_state(
    scat_morse,
    anion_pot,
    reduced_mass=mu,
    J=0,
    R_range=(0.5, 10.0),
    unit="eV",
    figsize=(9, 7),
)
axes2[0].set_title(
    rf"Scattering state on OH⁻ PEC  [E = {E_coll_cm1:.0f} cm⁻¹, analytical Morse+U]",
    fontsize=11,
)
fig2.tight_layout()
fig2.savefig("plots/morse_scattering_state.png", dpi=140, bbox_inches="tight")
print("Saved  plots/morse_scattering_state.png")

# -----------------------------------------------------------------------
# Figure 2 – DVR vs Morse comparison in the coupling region
# -----------------------------------------------------------------------
print("Plotting DVR vs Morse comparison …")

fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))
fig3.suptitle(
    rf"DVR vs Analytical Morse  |  E_coll = {E_coll_cm1:.0f} cm⁻¹, J=0",
    fontsize=13, fontweight="bold",
)

mask = (R >= R_COUPLING[0]) & (R <= R_COUPLING[1])
k_asym = np.sqrt(2 * mu * E_ha)

# ── Panel (0,0): scattering wavefunctions ──────────────────────────────
ax = axes3[0, 0]
ax.plot(R[mask], scat_dvr.wavefunction[mask],   "C0",  lw=1.8,
        label="DVR  (box eigenstate)")
ax.plot(R[mask], scat_morse.wavefunction[mask],  "C1",  lw=1.8, ls="--",
        label="Morse + U  (physical phase)")
ax.axhline(0, color="k", lw=0.5)
ax.set_title(r"Scattering wavefunction  $F_E(R)$", fontsize=11)
ax.set_xlabel("R (Bohr)")
ax.set_ylabel("Amplitude (a.u.)")
ax.set_xlim(R_COUPLING)
ax.legend(fontsize=9)

# ── Panel (0,1): derivatives dF/dR ─────────────────────────────────────
ax = axes3[0, 1]
ax.plot(R[mask], dF_dvr[mask],   "C0",  lw=1.8,
        label="DVR  (central diff, ×sin(kΔ)/kΔ error)")
ax.plot(R[mask], dF_morse[mask], "C1",  lw=1.8, ls="--",
        label="Morse + U  (analytical, exact)")
ax.axhline(0, color="k", lw=0.5)
ax.set_title(r"Scattering wavefunction derivative  $dF_E/dR$", fontsize=11)
ax.set_xlabel("R (Bohr)")
ax.set_ylabel("Derivative (a.u.)")
ax.set_xlim(R_COUPLING)
ax.legend(fontsize=9)

# Add annotation about the amplitude ratio in the inner well
inner_mask = (R >= 1.3) & (R <= 2.0)
if inner_mask.any():
    ratio = (np.max(np.abs(dF_dvr[inner_mask])) /
             np.max(np.abs(dF_morse[inner_mask])))
    ax.text(0.03, 0.97,
            f"Peak amplitude ratio DVR/Morse = {ratio:.3f}\n"
            f"Expected sin(kΔ)/(kΔ) ≈ {np.sin(k_asym*sol_dvr_an.dr)/(k_asym*sol_dvr_an.dr):.3f}",
            transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(fc="white", alpha=0.7, ec="gray", boxstyle="round"))

# ── Panel (1,0): DVR vs Morse – neutral bound states v'=6,7 ───────────
ax = axes3[1, 0]
dvr_bnd6   = sol_dvr_ne.solve_bound_state(6, J=0)
dvr_bnd7   = sol_dvr_ne.solve_bound_state(7, J=0)
morse_bnd6 = sol_morse_ne.solve_bound_state(6, J=0)
morse_bnd7 = sol_morse_ne.solve_bound_state(7, J=0)

ax.plot(R[mask], dvr_bnd6.wavefunction[mask],   "C2",       lw=1.8, label="DVR v'=6")
ax.plot(R[mask], morse_bnd6.wavefunction[mask],  "C2",       lw=1.8, ls="--",
        label="Morse v'=6")
ax.plot(R[mask], dvr_bnd7.wavefunction[mask],   "C3",       lw=1.8, label="DVR v'=7")
ax.plot(R[mask], morse_bnd7.wavefunction[mask],  "C3",       lw=1.8, ls="--",
        label="Morse v'=7")
ax.axhline(0, color="k", lw=0.5)
ax.set_title(r"Neutral bound states  $F_{v'}(R)$  (coupling region)", fontsize=11)
ax.set_xlabel("R (Bohr)")
ax.set_ylabel("Amplitude (a.u.)")
ax.set_xlim(R_COUPLING)
ax.legend(fontsize=9, ncol=2)

# ── Panel (1,1): coupling integrands ───────────────────────────────────
ax = axes3[1, 1]
R0, alpha_r, A_rad = 1.822, 3.0, 0.33
m_R = A_rad * np.exp(-alpha_r * (R - R0) ** 2)   # Gaussian coupling

for (dvr_bnd, morse_bnd, col, label) in [
    (dvr_bnd6, morse_bnd6, "C2", "v'=6"),
    (dvr_bnd7, morse_bnd7, "C3", "v'=7"),
]:
    intgd_dvr   = dvr_bnd.wavefunction   * m_R * dF_dvr
    intgd_morse = morse_bnd.wavefunction * m_R * dF_morse
    ax.plot(R[mask], intgd_dvr[mask],   col,       lw=1.5, label=f"DVR {label}")
    ax.plot(R[mask], intgd_morse[mask], col,       lw=1.5, ls="--",
            label=f"Morse {label}")

ax.axhline(0, color="k", lw=0.5)
ax.set_title(r"Coupling integrand  $F_{v'}\,m(R)\,dF_E/dR$", fontsize=11)
ax.set_xlabel("R (Bohr)")
ax.set_ylabel("Integrand (a.u.)")
ax.set_xlim(R_COUPLING)
ax.legend(fontsize=9, ncol=2)

fig3.tight_layout()
fig3.savefig("plots/dvr_vs_morse_comparison.png", dpi=140, bbox_inches="tight")
print("Saved  plots/dvr_vs_morse_comparison.png")

# -----------------------------------------------------------------------
# Figure 3 – Pekeris: scattering states at J=0, 5, 10
# -----------------------------------------------------------------------
print("Plotting Pekeris J-dependence …")

fig4, axes4 = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
fig4.suptitle(
    rf"Pekeris-corrected scattering states: E_coll = {E_coll_cm1:.0f} cm⁻¹, J = 0, 5, 10",
    fontsize=13, fontweight="bold",
)
# At E_coll=66 cm⁻¹ only J=0..3 are accessible (B_J·c₀ < E_coll)
for ax, J in zip(axes4, [0, 2, 3]):
    print(f"  J={J} …", end=" ", flush=True)
    try:
        scat_J = sol_morse_an.solve_scattering_state(E_ha, J=J)
        dF_J   = sol_morse_an.wavefunction_derivative(scat_J)
        ax.plot(R[mask], scat_J.wavefunction[mask], "C0", lw=1.5,
                label=r"$F_E$")
        scale = 0.04
        ax.plot(R[mask], dF_J[mask] * scale, "C1", lw=1.2, alpha=0.85,
                label=rf"$dF_E/dR \times {scale}$")
        p = sol_morse_an._pekeris(J)
        ax.set_title(
            f"J={J}  |  D_e_eff={p['D_e_eff']:.4f} Ha\n"
            f"R_e_eff={p['R_e_eff']:.4f} Bohr  λ_eff={p['lam_eff']:.2f}",
            fontsize=9,
        )
        print("OK")
    except ValueError as e:
        ax.text(0.5, 0.5, str(e), transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color="red")
        print(f"skipped: {e}")
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xlabel("R (Bohr)")
    ax.set_xlim(R_COUPLING)
    ax.legend(fontsize=8)

axes4[0].set_ylabel("Amplitude (a.u.)")
fig4.tight_layout()
fig4.savefig("plots/morse_pekeris_J_series.png", dpi=140, bbox_inches="tight")
print("Saved  plots/morse_pekeris_J_series.png")
print("\nDone.")
