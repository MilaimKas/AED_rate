#!/usr/bin/env python3
"""
Precompute CPSCF electronic coupling curves for the OH⁻ system.

Generates m_rad(R, k_e) and m_rot(R, k_e) on a 2-D grid using PySCF's
CPSCF infrastructure, then saves to .npz files that
InterpolatedCoupling.load() can read.

Grid strategy
-------------
- Inner region R ∈ [0.8, 4.4] Bohr: uniform, 50 points (~0.07 Bohr spacing).
- Outer region R ∈ (4.4, 10.0] Bohr: uniform, 28 points (~0.2 Bohr spacing).
- Combined: 78 points total covering the full physically relevant range.

The outer region is needed to determine whether the coupling decays to zero
(physical, as the HOMO localises on O⁻ at dissociation) or plateaus
(orbital-reordering artefact where the MOC fails to track the HOMO).

Usage
-----
    cd /path/to/AED_rate
    python scripts/precompute_coupling.py [--basis BASIS] [--out OUT]

    BASIS  : PySCF basis string, default '6-31g'
    OUT    : output .npz filename, default 'oh_minus_coupling_6-31g.npz'
"""

import sys
import argparse
import numpy as np

sys.path.insert(0, ".")

from aed_rate.electronic.potential import create_oh_system_acharya
from aed_rate.electronic.wavefunctions import ElectronicStructure
from aed_rate.electronic.coupling import InterpolatedCoupling


# ---------------------------------------------------------------------------
# R grid construction
# ---------------------------------------------------------------------------

def build_r_grid(R_inner_max: float = 4.4,
                 R_outer_max: float = 10.0,
                 n_inner: int = 50,
                 n_outer: int = 28,
                 R_min: float = 0.8) -> np.ndarray:
    """
    Non-uniform R grid: dense near equilibrium, coarse at large R.

    Parameters
    ----------
    R_inner_max : float
        Boundary between dense and sparse regions (Bohr).
    R_outer_max : float
        Maximum R to include (Bohr).
    n_inner, n_outer : int
        Number of points in each region.
    R_min : float
        Minimum R (Bohr).

    Returns
    -------
    np.ndarray
        Sorted, unique R values (Bohr).
    """
    inner = np.linspace(R_min, R_inner_max, n_inner)
    # Skip the first point of outer to avoid duplicate at R_inner_max
    outer = np.linspace(R_inner_max, R_outer_max, n_outer + 1)[1:]
    return np.concatenate([inner, outer])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Precompute and save the CPSCF coupling on the extended R grid."""
    parser = argparse.ArgumentParser(description="Precompute OH⁻ CPSCF coupling")
    parser.add_argument("--basis", default="6-31g",
                        help="PySCF basis set (default: 6-31g)")
    parser.add_argument("--out",   default=None,
                        help="Output .npz filename (default: auto from basis)")
    parser.add_argument("--r-max", type=float, default=10.0,
                        help="Maximum R in Bohr (default: 10.0)")
    parser.add_argument("--n-outer", type=int, default=28,
                        help="Grid points in outer region R∈(4.4, R_max] (default: 28)")
    args = parser.parse_args()

    # Output filename: derive from basis if not given
    if args.out is None:
        safe_basis = args.basis.replace("-", "").replace("*", "s").replace("+", "p")
        args.out = f"oh_minus_coupling_{safe_basis}.npz"

    anion_pot, _neutral_pot, _EA = create_oh_system_acharya()
    print(f"OH⁻ Morse: R_e={anion_pot.r_e:.3f} Bohr, β={anion_pot.beta:.3f} Bohr⁻¹")

    R_inner_max = anion_pot.r_e + np.log(1.0 / (1.0 - 0.9 ** 0.5)) / anion_pot.beta
    print(f"Inner region boundary (90 % of D_e): {R_inner_max:.3f} Bohr")

    R_grid = build_r_grid(
        R_inner_max=R_inner_max,
        R_outer_max=args.r_max,
        n_inner=50,
        n_outer=args.n_outer,
    )
    print(f"R grid: {len(R_grid)} points,  "
          f"[{R_grid[0]:.3f}, {R_grid[-1]:.3f}] Bohr")

    # k_e grid: covers v'=8 at 66 cm⁻¹ (k_e≈0.07) up to ~3 eV electrons
    k_e_grid = np.array([0.01, 0.03, 0.07, 0.15, 0.25, 0.40])

    es = ElectronicStructure("O", "H", basis=args.basis)

    # InterpolatedCoupling requires a contiguous linspace R_grid for the
    # RectBivariateSpline.  We supply our custom grid by overriding R_grid
    # after construction.
    coupling = InterpolatedCoupling(
        electronic_structure=es,
        anion_potential=anion_pot,
        R_min=float(R_grid[0]),
        R_cutoff=float(R_grid[-1]),
        n_points=len(R_grid),
        k_e_grid=k_e_grid,
        homo_symmetry="pi",
        grid_level=3,
    )
    # Override the linspace grid with our non-uniform one
    coupling.R_grid = R_grid

    print(f"k_e grid: {k_e_grid}")
    print(f"Basis:    {args.basis}")
    print()

    coupling.precompute(verbose=True)

    coupling.save(args.out)
    print(f"\nSaved to '{args.out}'")

    # Summary: print every 10th point + last 5 at the representative k_e
    mid_ke_idx = len(k_e_grid) // 2
    k_e_mid    = k_e_grid[mid_ke_idx]
    print(f"\nSample of m_rad at k_e = {k_e_mid:.3f} a.u. "
          f"(every 10th point + last 5):")
    print(f"  {'R (Bohr)':>10s}  {'m_rad':>12s}  {'m_rot':>12s}")
    indices = list(range(0, len(R_grid), 10)) + list(range(-5, 0))
    seen = set()
    for idx in indices:
        if idx in seen:
            continue
        seen.add(idx)
        R   = coupling.R_grid[idx]
        m_r = coupling._m_rad_2d[idx, mid_ke_idx]
        m_t = coupling._m_rot_2d[idx, mid_ke_idx]
        print(f"  {R:10.3f}  {m_r:+12.4e}  {m_t:+12.4e}")


if __name__ == "__main__":
    main()
