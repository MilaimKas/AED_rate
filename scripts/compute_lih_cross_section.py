#!/usr/bin/env python3
"""
Compute the LiH⁻ + → LiH(v',J') + e⁻ associative-detachment cross section.

This is the first end-to-end use of the σ-HOMO machinery: open-shell (²Σ⁺)
CPSCF coupling + the l=0 s-wave channel.  Because the detaching electron sits
in the diffuse 3σ, the s-wave (σ ∝ k near threshold) is expected to dominate
the l=1 p-wave (σ ∝ k³) at low collision energy — qualitatively unlike the
π-HOMO OH⁻ case.

Two stages
----------
1. Precompute the open-shell coupling on an (R, k_e) grid (CPSCF + s-wave OPW)
   and cache it to an .npz.  Skipped if the cache already exists.
2. Build an AEDSystem with that coupling and trace σ_AD(E) across threshold,
   then plot it on log-log axes to expose the threshold power law.

Usage
-----
    python scripts/compute_lih_cross_section.py [--basis BASIS] [--force]
"""

import sys
import argparse
import numpy as np

sys.path.insert(0, ".")

from aed_rate.electronic.potential import create_lih_system
from aed_rate.electronic.wavefunctions import ElectronicStructure
from aed_rate.electronic.coupling import InterpolatedCoupling
from aed_rate.aed_calculator import AEDSystem
from aed_rate.utils.constants import get_reduced_mass, CONSTANTS


def precompute_lih_coupling(basis: str, out: str) -> None:
    """Precompute and cache the open-shell LiH⁻ CPSCF coupling (with s-wave)."""
    anion_pot, _neutral, _EA = create_lih_system()
    print(f"LiH⁻ Morse: R_e={anion_pot.r_e:.3f} Bohr, β={anion_pot.beta:.3f} Bohr⁻¹")

    # R grid: dense around the diffuse equilibrium, out to ~90 % of D_e.
    R_cutoff = anion_pot.r_e + np.log(1.0 / (1.0 - 0.9 ** 0.5)) / anion_pot.beta
    R_grid = np.linspace(2.0, R_cutoff, 36)
    print(f"R grid: {len(R_grid)} points, [{R_grid[0]:.3f}, {R_grid[-1]:.3f}] Bohr")

    # k_e grid: from just above threshold (k≈0.02) to a few-tenths a.u.
    k_e_grid = np.array([0.02, 0.05, 0.10, 0.20, 0.35, 0.55])

    # aug-cc-pVDZ: diffuse functions are essential — they set the s-wave
    # magnitude (the orthogonalised monopole leaks through the diffuse virtuals).
    es = ElectronicStructure("Li", "H", basis=basis)

    coupling = InterpolatedCoupling(
        electronic_structure=es,
        anion_potential=anion_pot,
        R_min=float(R_grid[0]),
        R_cutoff=float(R_grid[-1]),
        n_points=len(R_grid),
        k_e_grid=k_e_grid,
        homo_symmetry="sigma",   # LiH⁻ detaches from the 3σ
        charge=-1,
        spin=1,                  # open-shell ²Σ⁺ anion → UHF/CPSCF
        grid_level=3,
    )
    coupling.R_grid = R_grid
    coupling.precompute(verbose=True)
    coupling.save(out)
    print(f"\nSaved coupling to '{out}'  (swave_channel={coupling.swave_channel})")


def main() -> None:
    """Precompute (if needed) then trace and plot σ_AD(E) for LiH⁻."""
    parser = argparse.ArgumentParser(description="LiH⁻ AED cross section")
    parser.add_argument("--basis", default="aug-cc-pvdz")
    parser.add_argument("--npz", default="lih_minus_coupling_swave.npz")
    parser.add_argument("--force", action="store_true",
                        help="Recompute the coupling even if the cache exists")
    args = parser.parse_args()

    import os
    if args.force or not os.path.exists(args.npz):
        precompute_lih_coupling(args.basis, args.npz)
    else:
        print(f"Using cached coupling '{args.npz}'")

    # Load the coupling without needing PySCF again.
    coupling = InterpolatedCoupling.from_npz(args.npz)
    print(f"Loaded coupling: swave_channel={coupling.swave_channel}")

    anion_pot, neutral_pot, EA = create_lih_system()
    mu = get_reduced_mass("Li", "H")

    system = AEDSystem(
        anion_potential=anion_pot,
        neutral_potential=neutral_pot,
        EA=EA,
        reduced_mass=mu,
        coupling=coupling,
        solver_method="morse",
        n_grid=6000,
    )

    # Collision-energy sweep across threshold (Hartree).  1 meV ≈ 3.7e-5 Ha.
    eV = CONSTANTS.hartree_to_ev
    E_meV = np.array([0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500])  # meV
    E_ha = E_meV * 1e-3 / eV

    print(f"\n{'E (meV)':>9s}  {'k_e (a.u.)':>11s}  {'σ_AD (a0²)':>13s}  {'σ_AD (Å²)':>12s}")
    sig_a0 = []
    for E in E_ha:
        s = system.sigma_AD(E, unit="a0^2")
        sig_a0.append(s)
        s_ang = system._convert_sigma(s, "Angstrom^2")
        # k_e at this energy for the v'=0 J=0 channel (rough, for the table only)
        print(f"  {E*eV*1e3:7.1f}  {'':>11s}  {s:13.4e}  {s_ang:12.4e}")
    sig_a0 = np.array(sig_a0)

    # Empirical low-energy slope of σ_AD vs the NUCLEAR collision energy.
    # NOTE: this is NOT a clean Wigner partial-wave law.  The Wigner powers
    # (σ_l ∝ k_e^{2l+1}) refer to the *electron* momentum k_e; this curve is
    # vs E_coll and convolves the (2π²/E) flux prefactor, the growing sum over
    # initial nuclear partial waves J, and the per-v' electron thresholds.
    lo = slice(0, 4)  # lowest few points
    p = np.polyfit(np.log(E_ha[lo]), np.log(sig_a0[lo]), 1)[0]
    print(f"\nLow-energy slope d(log σ_AD)/d(log E_coll) ≈ {p:.2f}  "
          f"(empirical; mixes flux prefactor + J-sum + electron thresholds)")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.loglog(E_meV, sig_a0, "o-", color="C3", label="σ_AD (l=0 + l=1)")
        ax.set_xlabel("collision energy  E  (meV)")
        ax.set_ylabel(r"$\sigma_{AD}$  ($a_0^2$)")
        ax.set_title(f"LiH⁻ associative detachment  (slope ≈ {p:.2f})")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        out_png = "plots/lih_cross_section.png"
        fig.savefig(out_png, dpi=130)
        print(f"\nSaved plot to '{out_png}'")
    except Exception as exc:  # pragma: no cover - plotting is best-effort
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":
    main()
