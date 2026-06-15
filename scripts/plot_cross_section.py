"""
Compute and plot the total AD cross section σ_AD(E) for OH⁻ (O⁻ + H → OH + e⁻).

Uses the Čížek (2001) cross-section convention (Eq. 2.7–2.8) in the
weak-coupling limit, with a unit-amplitude anion scattering state:

    σ_AD(E) = Σ_J Σ_{v',J'} (2π²/E)(2J+1) ρ(E_e) |V_{v',J→J'}|²

evaluated by AEDRateCalculator.total_cross_section_all_J.  The partial-wave
sum runs until the Pekeris centrifugal barrier closes the channel.

CAVEAT: the precomputed coupling spline covers k_e ∈ [0.01, 0.40] a.u.
(E_e ≲ 2.2 eV).  Low-v' channels emit faster electrons (E_e up to ~3 eV) whose
k_e is clipped to 0.40; those channels are minor (phase-cancellation
suppressed), but the high-E_e tail of σ_AD is therefore only approximate.

Run:  python scripts/plot_cross_section.py
"""

from __future__ import annotations

import os

import numpy as np
from scipy.interpolate import RectBivariateSpline

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aed_rate.electronic.coupling import CouplingResult
from aed_rate.electronic.potential import create_oh_system_acharya
from aed_rate.rate.state_to_state import AEDRateCalculator
from aed_rate.utils.constants import CONSTANTS, get_reduced_mass


NPZ_PATH = "oh_minus_coupling_6311pgss.npz"
N_GRID = 4000                      # converged to ~1 %; 6000 for production
R_MIN, R_MAX = 0.5, 15.0
A0_ANG = 0.529177                  # Bohr → Å
N_ENERGY = 14                      # collision energies in the scan
E_LO_CM, E_HI_CM = 40.0, 2000.0    # collision-energy range (cm⁻¹)
ACHARYA_E_CM = (66.0, 256.0, 732.0)


def _npz_coupling(npz_path: str):
    """Spline-backed coupling provider from a precomputed NPZ (no PySCF)."""
    d = np.load(npz_path)
    R, ke = d["R_grid"], d["k_e_grid"]
    mr, mo = d["m_rad_2d"], d["m_rot_2d"]
    lo, hi = float(d["R_min"][0]), float(d["R_cutoff"][0])
    sr = RectBivariateSpline(R, ke, mr, kx=3, ky=3)
    so = RectBivariateSpline(R, ke, mo, kx=3, ky=3)

    class _Coupling:
        """Interpolate m_rad, m_rot from the precomputed grid."""

        def compute_coupling_at_r(self, R: float, electron_energy: float, **_):
            """Bicubic-spline coupling at (R, k_e); zero outside the R grid."""
            k = float(np.sqrt(max(2.0 * electron_energy, 0.0)))
            if R < lo or R > hi:
                return CouplingResult(R=R, m_rad=0j, m_rot=0j,
                                      electron_energy=electron_energy, k_electron=k)
            kc = float(np.clip(k, ke[0], ke[-1]))
            return CouplingResult(
                R=R,
                m_rad=complex(float(sr(R, kc, grid=False))),
                m_rot=complex(float(so(R, kc, grid=False))),
                electron_energy=electron_energy, k_electron=k,
            )

    return _Coupling()


def main() -> None:
    """Scan collision energy, compute σ_AD(E), print a table and save a plot."""
    if not os.path.exists(NPZ_PATH):
        raise SystemExit(f"{NPZ_PATH} not found — run precompute_coupling.py first.")

    anion, neutral, EA = create_oh_system_acharya()
    mu = get_reduced_mass("O", "H")
    calc = AEDRateCalculator(
        anion, neutral, EA, mu,
        coupling=_npz_coupling(NPZ_PATH),
        solver_method="morse", r_min=R_MIN, r_max=R_MAX, n_grid=N_GRID,
    )

    # Log-spaced collision-energy grid, plus the three Acharya energies
    E_cm = np.unique(np.concatenate([
        np.geomspace(E_LO_CM, E_HI_CM, N_ENERGY),
        np.array(ACHARYA_E_CM),
    ]))
    E_ha = np.array([CONSTANTS.cm1_to_hartree(e) for e in E_cm])
    E_eV = E_ha * CONSTANTS.hartree_to_ev

    sigma_a0 = np.empty_like(E_ha)
    print(f"{'E(cm⁻¹)':>9} {'E(eV)':>9} {'σ_AD(a₀²)':>13} {'σ_AD(Å²)':>13}")
    for i, E in enumerate(E_ha):
        s = calc.total_cross_section_all_J(E)
        sigma_a0[i] = s
        calc.clear_cache()  # coupling cache is keyed by E_e; reset per energy
        print(f"{E_cm[i]:9.1f} {E_eV[i]:9.4f} {s:13.4e} {s*A0_ANG**2:13.4e}")

    sigma_Ang2 = sigma_a0 * A0_ANG ** 2

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    ax.loglog(E_eV, sigma_Ang2, "o-", lw=1.6, ms=4, label="σ_AD (this work)")
    # mark the Acharya benchmark energies
    for ecm in ACHARYA_E_CM:
        eev = CONSTANTS.cm1_to_hartree(ecm) * CONSTANTS.hartree_to_ev
        ax.axvline(eev, color="grey", ls=":", lw=0.8)
        ax.text(eev, ax.get_ylim()[0], f" {ecm:.0f} cm⁻¹",
                rotation=90, va="bottom", ha="right", fontsize=7, color="grey")
    ax.set_xlabel("collision energy $E$ (eV)")
    ax.set_ylabel(r"$\sigma_{\rm AD}(E)$ (Å$^2$)")
    ax.set_title("OH$^-$ associative-detachment cross section (non-resonant non-BO)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    # context annotation: resonant halides are ~10 Å² (Čížek 2001)
    ax.text(0.03, 0.05,
            "for scale: resonant H+X⁻ halides are ~10 Å² (Čížek 2001)\n"
            "→ non-resonant OH⁻ is ~10 orders smaller",
            transform=ax.transAxes, fontsize=8, style="italic", va="bottom")
    fig.tight_layout()

    out = os.path.join("plots", "cross_section_vs_E.png")
    os.makedirs("plots", exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
