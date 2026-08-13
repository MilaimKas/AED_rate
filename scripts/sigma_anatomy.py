#!/usr/bin/env python3
"""
Anatomy of the AED cross section: decompose sigma(E) into Simons' (1998) factors.

Simons' semiquantum rate (his eq. 42) factorizes the non-BO ejection rate as

    R_T  ~  rho(E_e) . |(P/mu) chi(Q0)|^2 . |m*|^2 . (1/v0)

i.e. electron density-of-states x the SQUARE OF THE NUCLEAR-WAVEFUNCTION
DERIVATIVE (not its amplitude) x electronic coupling strength x passage speed.
This script tests two of his predictions numerically for LiH-:

  (A) Threshold law.  For a fixed final channel, track sigma(E_coll) and its
      pieces vs collision energy.  The claim: the nuclear-derivative factor
      |dF_E/dR|^2 at R_e scales as k_in^2, cancelling the (2 pi^2 / E) ~ 1/k_in^2
      flux prefactor -> sigma -> const at threshold (NOT Wigner 1/v).

  (B) Electron-energy propensity (Simons condition 2).  A fast ejected electron
      oscillates rapidly -> poor overlap with d(psi)/dQ -> small coupling.
      The claim: |m(R_e; E_e)|^2 falls with electron energy E_e.

Usage
-----
    python scripts/sigma_anatomy.py
"""

import sys
import numpy as np

sys.path.insert(0, ".")

from aed_rate.electronic.potential import create_lih_system
from aed_rate.electronic.coupling import InterpolatedCoupling
from aed_rate.aed_calculator import AEDSystem
from aed_rate.nuclear.nuclear_wavefunction import create_wavefunction_solver
from aed_rate.utils.constants import CONSTANTS, get_reduced_mass


def _slope(x: np.ndarray, y: np.ndarray, lo: int = 0, hi: int = 6) -> float:
    """Log-log slope of y(x) over the index window [lo:hi]."""
    return float(np.polyfit(np.log(x[lo:hi]), np.log(y[lo:hi]), 1)[0])


def main() -> None:
    """Decompose LiH- sigma(E) into Simons' factors and plot."""
    import warnings
    warnings.filterwarnings("ignore")

    anion, neutral, EA = create_lih_system()
    mu = get_reduced_mass("Li", "H")
    R_e = anion.r_e
    eV = CONSTANTS.hartree_to_ev

    coupling = InterpolatedCoupling.from_npz("lih_minus_coupling_swave.npz")
    system = AEDSystem(anion, neutral, EA, mu, coupling=coupling,
                       solver_method="morse", n_grid=6000)
    rc = system._rate_calc
    solver = rc.anion_solver              # analytic Morse solver (unit-amplitude)

    ie = int(np.argmin(np.abs(solver.r_grid - R_e)))   # grid index nearest R_e

    # ---- Part A: fixed channel (J=0 -> v'=6, J'=1, the l=1 rotational route) --
    J, v_prime, J_prime = 0, 6, 1
    E_meV = np.geomspace(1e-3, 500, 22)

    sigma, Ee, rho, prefac, V2, dFdR2, k_in = ([] for _ in range(7))
    for e in E_meV:
        E = e * 1e-3 / eV
        cs = rc.cross_section_state_to_state(E, J, v_prime, J_prime)
        scatt = solver.solve_scattering_state(E, J, normalization="unit_amplitude")
        dF = solver.wavefunction_derivative(scatt)      # dF_E/dR on r_grid
        sigma.append(cs.sigma)
        Ee.append(cs.electron_energy)
        rho.append(np.sqrt(2 * cs.electron_energy) / (2 * np.pi ** 2))  # k_e/2pi^2
        prefac.append(2 * np.pi ** 2 / E)
        V2.append(abs(cs.V_rot) ** 2)
        dFdR2.append(dF[ie] ** 2)                       # |dF_E/dR|^2 at R_e
        k_in.append(np.sqrt(2 * mu * E))

    E_meV, sigma, Ee, rho, prefac, V2, dFdR2, k_in = map(
        np.asarray, (E_meV, sigma, Ee, rho, prefac, V2, dFdR2, k_in))

    print("=== Part A: threshold anatomy (channel J=0->v'6,J'1) ===")
    print(f"  slope d(log sigma)/d(log E_coll)      = {_slope(E_meV, sigma):+.2f}   (flat ~ 0)")
    print(f"  slope d(log |V_rot|^2)/d(log k_in)    = {_slope(k_in, V2):+.2f}   (Simons deriv -> +2)")
    print(f"  slope d(log |dF/dR(R_e)|^2)/d(log k)  = {_slope(k_in, dFdR2):+.2f}   (deriv factor -> +2)")
    print(f"  E_e over the sweep: {Ee.min()*eV:.3f}..{Ee.max()*eV:.3f} eV (should be ~const)")

    # ---- Part B: electron-energy propensity |m(R_e;E_e)|^2 vs E_e -------------
    k_e = np.linspace(coupling.k_e_grid[0], coupling.k_e_grid[-1], 30)
    E_e_scan = 0.5 * k_e ** 2
    m2 = np.array([abs(coupling.compute_coupling_at_r(R_e, ee).m_rot) ** 2
                   for ee in E_e_scan])
    print("\n=== Part B: electron-energy propensity ===")
    print(f"  slope d(log |m_rot(R_e)|^2)/d(log E_e) = "
          f"{_slope(E_e_scan, m2, 0, len(E_e_scan)):+.2f}   (Simons: should be NEGATIVE)")

    # ---- Plot ----------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))

    # (1) sigma and its reconstruction from the factors
    recon = prefac * (2 * J + 1) * rho * V2
    ax[0].loglog(E_meV, sigma, "o-", label=r"$\sigma$ (code)")
    ax[0].loglog(E_meV, recon, "x--", label=r"$(2\pi^2/E)(2J{+}1)\rho|V|^2$")
    ax[0].set_xlabel("collision energy (meV)")
    ax[0].set_ylabel(r"$\sigma$  ($a_0^2$)")
    ax[0].set_title(f"(A) flat threshold  (slope {_slope(E_meV, sigma):+.2f})")
    ax[0].grid(True, which="both", alpha=0.3); ax[0].legend(fontsize=8)

    # (2) the competing factors vs k_in: 1/E falls, |V|^2 rises as k^2, product flat
    ax[1].loglog(k_in, prefac / prefac[0], label=r"$2\pi^2/E \propto k_{in}^{-2}$")
    ax[1].loglog(k_in, V2 / V2[0], label=r"$|V_{rot}|^2$")
    ax[1].loglog(k_in, dFdR2 / dFdR2[0], ":", label=r"$|dF_E/dR|^2_{R_e}$")
    ax[1].loglog(k_in, (prefac * V2) / (prefac * V2)[0], "k-", lw=2,
                 label="product (flat)")
    ax[1].set_xlabel(r"$k_{in}$ (a.u.)")
    ax[1].set_ylabel("factor (normalised)")
    ax[1].set_title("(A) 1/E flux vs derivative coupling")
    ax[1].grid(True, which="both", alpha=0.3); ax[1].legend(fontsize=8)

    # (3) Simons electron-energy propensity
    ax[2].loglog(E_e_scan * eV, m2, "o-", color="C3")
    ax[2].set_xlabel(r"ejected-electron energy $E_e$ (eV)")
    ax[2].set_ylabel(r"$|m_{rot}(R_e)|^2$")
    ax[2].set_title("(B) electron-energy propensity")
    ax[2].grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    out = "plots/lih_sigma_anatomy.png"
    fig.savefig(out, dpi=130)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
