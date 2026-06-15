"""
Compute LiH and LiH⁻ potential energy curves at CCSD(T)/aug-cc-pVTZ and fit
each to a Morse potential, for use as an AED system.

LiH   : closed-shell ¹Σ⁺ (4 e⁻)  → RHF + RCCSD(T)
LiH⁻  : open-shell  ²Σ⁺ (5 e⁻)   → UHF + UCCSD(T), extra e⁻ in the diffuse 3σ

The Morse parameters (D_e, R_e, β) and the adiabatic electron affinity
EA = E_min(LiH) − E_min(LiH⁻) are printed in the convention used by the
package (anion minimum = 0, neutral minimum = EA).

Run:  python scripts/compute_lih_pecs.py
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from pyscf import gto, scf, cc

HARTREE_EV = 27.211386245988
BASIS = "aug-cc-pvtz"


def ccsdt_energy(R: float, charge: int, spin: int) -> float:
    """Total CCSD(T) energy of LiH at bond length R (Bohr), given charge/spin."""
    mol = gto.M(atom=f"Li 0 0 0; H 0 0 {R}", basis=BASIS,
                charge=charge, spin=spin, unit="Bohr", verbose=0)
    if spin == 0:
        mf = scf.RHF(mol).run()
        mycc = cc.CCSD(mf).run()
    else:
        mf = scf.UHF(mol).run()
        mycc = cc.UCCSD(mf).run()
    return float(mycc.e_tot + mycc.ccsd_t())


def scan(charge: int, spin: int, R_grid: np.ndarray) -> np.ndarray:
    """CCSD(T) energy on a grid of bond lengths."""
    return np.array([ccsdt_energy(R, charge, spin) for R in R_grid])


def _morse(R: np.ndarray, D_e: float, R_e: float, beta: float, E_min: float) -> np.ndarray:
    """Morse form V(R) = D_e (1 − e^{−β(R−R_e)})² + E_min."""
    return D_e * (1.0 - np.exp(-beta * (R - R_e))) ** 2 + E_min


def fit_morse(R: np.ndarray, E: np.ndarray) -> dict:
    """Least-squares Morse fit; returns D_e, R_e, beta (a.u.) and E_min (Ha)."""
    i_min = int(np.argmin(E))
    p0 = (E[-1] - E[i_min], R[i_min], 1.0, E[i_min])  # D_e, R_e, beta, E_min
    popt, _ = curve_fit(_morse, R, E, p0=p0, maxfev=20000)
    D_e, R_e, beta, E_min = popt
    rms = float(np.sqrt(np.mean((_morse(R, *popt) - E) ** 2)))
    return dict(D_e=float(D_e), R_e=float(R_e), beta=float(beta),
                E_min=float(E_min), rms=rms)


def main() -> None:
    """Scan both PECs, fit Morse, report parameters and the adiabatic EA."""
    # Denser sampling through the well, sparser toward dissociation.
    R_grid = np.unique(np.concatenate([
        np.linspace(2.2, 5.0, 18),
        np.linspace(5.0, 9.0, 9),
    ]))

    print(f"CCSD(T)/{BASIS} scan over {len(R_grid)} points "
          f"[{R_grid[0]:.2f}, {R_grid[-1]:.2f}] Bohr ...")
    E_neu = scan(charge=0, spin=0, R_grid=R_grid)   # LiH   ¹Σ⁺
    E_ani = scan(charge=-1, spin=1, R_grid=R_grid)  # LiH⁻  ²Σ⁺

    fn = fit_morse(R_grid, E_neu)
    fa = fit_morse(R_grid, E_ani)
    EA = fn["E_min"] - fa["E_min"]   # adiabatic EA (anion lower → EA > 0)

    print("\n  R(Bohr)   E(LiH)        E(LiH⁻)       vert.EA(eV)")
    for R, en, ea in zip(R_grid, E_neu, E_ani):
        print(f"  {R:6.3f}  {en:13.6f}  {ea:13.6f}  {(en-ea)*HARTREE_EV:8.3f}")

    print("\n=== Morse fits (a.u.) ===")
    for name, f in (("LiH⁻ (anion)", fa), ("LiH (neutral)", fn)):
        print(f"  {name:14s}  D_e={f['D_e']:.5f} Ha ({f['D_e']*HARTREE_EV:.3f} eV)  "
              f"R_e={f['R_e']:.4f}  beta={f['beta']:.4f}  rmsfit={f['rms']:.2e}")
    print(f"\n  Adiabatic EA(LiH) = {EA:.5f} Ha = {EA*HARTREE_EV:.4f} eV")

    print("\n=== Package convention (anion min = 0, neutral min = EA) ===")
    print(f"  anion   = MorsePotential(D_e={fa['D_e']:.4f}, r_e={fa['R_e']:.4f}, "
          f"beta={fa['beta']:.4f}, V_0=0.0)")
    print(f"  neutral = MorsePotential(D_e={fn['D_e']:.4f}, r_e={fn['R_e']:.4f}, "
          f"beta={fn['beta']:.4f}, V_0={EA:.5f})")

    np.savez("lih_pec_data.npz", R=R_grid, E_neutral=E_neu, E_anion=E_ani,
             EA=np.array([EA]))
    print("\nSaved raw PEC data to lih_pec_data.npz")


if __name__ == "__main__":
    main()
