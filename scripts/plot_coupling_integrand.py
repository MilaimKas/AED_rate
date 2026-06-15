"""
Visualize the radial AED coupling integrand and its phase cancellation.

The state-to-state coupling integral is

    V_rad(v') = (1/μ) ∫ χ_{v'}(R) · m_rad(R; k_e) · dF_E/dR  dR
              = (1/μ) ∫ g_{v'}(R) dR

where χ_{v'} is the neutral bound state, m_rad the electronic coupling, and
F_E the (unit-amplitude) anion scattering state.  The point of this figure is
to show *why* the integral is numerically delicate: the integrand g_{v'}(R)
has large positive and negative lobes that nearly cancel, so the net value is
a small residual.  The cumulative integral overshoots its final value by
orders of magnitude before settling — which is exactly what makes the result
sensitive to the radial grid spacing.

Three panels:
  (a) the three factors (χ_{v'}, m_rad, dF_E/dR), each scaled to unit max;
  (b) the integrand g_{v'}(R) for several v';
  (c) the running integral normalized to its final value, |C(R)/C(∞)|.

Run:  python scripts/plot_coupling_integrand.py
"""

from __future__ import annotations

import os

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import RectBivariateSpline

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aed_rate.electronic.coupling import CouplingResult
from aed_rate.electronic.potential import create_oh_system_acharya
from aed_rate.rate.state_to_state import AEDRateCalculator
from aed_rate.utils.constants import CONSTANTS, get_reduced_mass


NPZ_PATH = "oh_minus_coupling_6311pgss.npz"
E_COLL_CM = 66.0
N_GRID = 6000          # fine grid (cross sections converge by ~6000)
R_MIN, R_MAX = 0.5, 15.0
V_PRIMES = (6, 7, 8)   # well-behaved, under-shoot, over-shoot
PLOT_R_MAX = 4.5       # the integral is essentially complete by R ≈ 4 Bohr


def _npz_coupling(npz_path: str):
    """Build a spline-backed coupling provider from a precomputed NPZ."""
    d = np.load(npz_path)
    R, ke = d["R_grid"], d["k_e_grid"]
    mr, mo = d["m_rad_2d"], d["m_rot_2d"]
    lo, hi = float(d["R_min"][0]), float(d["R_cutoff"][0])
    sr = RectBivariateSpline(R, ke, mr, kx=3, ky=3)
    so = RectBivariateSpline(R, ke, mo, kx=3, ky=3)

    class _Coupling:
        """Minimal coupling provider backed by precomputed 2D splines."""

        def compute_coupling_at_r(self, R: float, electron_energy: float, **_):
            """Interpolate m_rad, m_rot at (R, k_e); zero outside the grid."""
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
    """Compute the integrand for several v' and render the diagnostic figure."""
    if not os.path.exists(NPZ_PATH):
        raise SystemExit(f"{NPZ_PATH} not found — run precompute_coupling.py first.")

    anion, neutral, EA = create_oh_system_acharya()
    mu = get_reduced_mass("O", "H")
    calc = AEDRateCalculator(
        anion, neutral, EA, mu,
        coupling=_npz_coupling(NPZ_PATH),
        solver_method="morse", r_min=R_MIN, r_max=R_MAX, n_grid=N_GRID,
    )

    E = CONSTANTS.cm1_to_hartree(E_COLL_CM)
    Rg = calc.anion_solver.r_grid

    # Anion scattering state (unit amplitude) and its exact derivative — these
    # are the SAME for every v' (they depend only on the collision energy).
    scat = calc.anion_solver.solve_scattering_state(E, 0, normalization="unit_amplitude")
    dFE = calc.anion_solver.wavefunction_derivative(scat)

    # Per-v' pieces
    data = {}
    for v in V_PRIMES:
        cs = calc.cross_section_state_to_state(E, 0, v, 0)
        bound = calc.neutral_solver.solve_bound_state(v, 0)
        m_rad, _ = calc._evaluate_coupling_on_grid(cs.electron_energy)
        g = bound.wavefunction * m_rad.real * dFE          # integrand g_{v'}(R)
        C = cumulative_trapezoid(g, Rg, initial=0.0)       # running integral
        data[v] = dict(
            bound=bound.wavefunction, m_rad=m_rad.real, g=g, C=C,
            C_final=C[-1], k_e=float(np.sqrt(2.0 * cs.electron_energy)),
            sigma=cs.sigma,
        )

    mask = Rg <= PLOT_R_MAX
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 10.5), sharex=True)

    # (a) the three factors, each scaled to unit peak for shape comparison
    ax = axes[0]
    b6 = data[6]["bound"]
    ax.plot(Rg[mask], (b6 / np.max(np.abs(b6)))[mask], label=r"$\chi_{v'=6}(R)$", lw=1.6)
    mr6 = data[6]["m_rad"]
    ax.plot(Rg[mask], (mr6 / np.max(np.abs(mr6)))[mask],
            label=r"$m_{\rm rad}(R)$", lw=1.6)
    ax.plot(Rg[mask], (dFE / np.max(np.abs(dFE)))[mask],
            label=r"$dF_E/dR$ (anion scatt.)", lw=1.0, alpha=0.8)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("scaled to unit peak")
    ax.set_title(
        f"AED radial coupling integrand  (OH$^-$, $E_{{\\rm coll}}$ = {E_COLL_CM:.0f} cm$^{{-1}}$, "
        f"J=0, n_grid={N_GRID})"
    )
    ax.legend(loc="upper right", fontsize=9)

    # (b) the actual integrand g_{v'}(R) = χ·m_rad·dF/dR
    ax = axes[1]
    for v in V_PRIMES:
        ax.plot(Rg[mask], data[v]["g"][mask],
                label=fr"$v'={v}$ ($k_e$={data[v]['k_e']:.3f})", lw=1.3)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel(r"integrand $g_{v'}(R)$")
    ax.legend(loc="upper right", fontsize=9)
    ax.text(0.02, 0.95,
            "large +/- lobes → the net integral is a small residual",
            transform=ax.transAxes, va="top", fontsize=9, style="italic")

    # (c) running integral normalized to its final value (the cancellation story)
    ax = axes[2]
    for v in V_PRIMES:
        C = data[v]["C"]
        Cf = data[v]["C_final"]
        ax.plot(Rg[mask], (C / Cf)[mask],
                label=fr"$v'={v}$  (final $\mu V$={Cf:.2e})", lw=1.3)
    ax.axhline(1.0, color="k", lw=0.6, ls="--", label="final value")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel(r"running $\int^R g\,dR'\ /\ \int^\infty g\,dR'$")
    ax.set_xlabel(r"$R$ (Bohr)")
    ax.set_ylim(-6, 8)
    ax.legend(loc="upper right", fontsize=9)
    ax.text(0.02, 0.05,
            "overshoot far beyond 1 then settle ⇒ heavy cancellation ⇒ grid-sensitive",
            transform=ax.transAxes, va="bottom", fontsize=9, style="italic")

    ax.set_xlim(Rg[mask][0], PLOT_R_MAX)
    fig.tight_layout()

    out = os.path.join("plots", "coupling_integrand_vprime.png")
    os.makedirs("plots", exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")

    # Console summary of the cancellation severity
    print(f"\n{'v':>3} {'k_e':>7} {'final muV':>12} {'max|run|/|final|':>16} "
          f"{'sigma(a0^2)':>13}")
    for v in V_PRIMES:
        C = data[v]["C"]; Cf = data[v]["C_final"]
        overshoot = float(np.max(np.abs(C)) / abs(Cf))
        print(f"{v:>3} {data[v]['k_e']:7.4f} {Cf:12.3e} {overshoot:16.1f} "
              f"{data[v]['sigma']:13.3e}")


if __name__ == "__main__":
    main()
