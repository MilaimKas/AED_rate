"""
Diagnostic plot: scattering state phase sensitivity and coupling integrands.

Shows:
  Row 0 — Scattering state F_E(R) and dF_E/dR in the coupling region
          for 3-pt FD (n=500 and n=2000) and sinc-DVR (n=500).
  Row 1 — Neutral bound states v'=6 and v'=7, and the coupling integrands
          F_v'(R) * m(R) * dF_E/dR for both.
  Row 2 — Fourier spectrum of the coupling integrand (beat frequencies)
          and cumulative (running) integral to show cancellation.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import simpson

from aed_rate import AEDSystem
from aed_rate.electronic.continuum import compute_electron_kinetic_energy
from aed_rate.utils.constants import CONSTANTS as C

E_ha = C.cm1_to_hartree(66)
R0, alpha_r, A_rad = 1.822, 3.0, 0.33
zoom = (1.3, 2.8)

# -----------------------------------------------------------------------
# 1. Build systems
# -----------------------------------------------------------------------
sys500  = AEDSystem.oh_system(n_grid=500)
sys2000 = AEDSystem.oh_system(n_grid=2000)

# Inline sinc-DVR: patch a copy of the 500-pt anion solver
import copy

sol_sinc = copy.copy(sys500._rate_calc.anion_solver)
sol_sinc._eigenvalue_cache = {}
sol_sinc._eigenvector_cache = {}

n  = sol_sinc.n_grid
dr = sol_sinc.dr
mu = sol_sinc.mu
i_idx, j_idx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
d = i_idx - j_idx
with np.errstate(divide="ignore", invalid="ignore"):
    T_sinc = np.where(d == 0, np.pi**2 / 3.0, (-1.0)**d * 2.0 / d**2)
np.fill_diagonal(T_sinc, np.pi**2 / 3.0)
sol_sinc._T = T_sinc / (2.0 * mu * dr**2)

# -----------------------------------------------------------------------
# 2. Collect wavefunctions
# -----------------------------------------------------------------------
scat_fd500  = sys500._rate_calc.anion_solver.solve_scattering_state(E_ha, J=0)
scat_fd2000 = sys2000._rate_calc.anion_solver.solve_scattering_state(E_ha, J=0)
scat_sinc   = sol_sinc.solve_scattering_state(E_ha, J=0)

dF_fd500  = sys500._rate_calc.anion_solver.wavefunction_derivative(scat_fd500)
dF_fd2000 = sys2000._rate_calc.anion_solver.wavefunction_derivative(scat_fd2000)
dF_sinc   = sys500._rate_calc.anion_solver.wavefunction_derivative(scat_sinc)

bnd6 = sys500._rate_calc.neutral_solver.solve_bound_state(6, J=0)
bnd7 = sys500._rate_calc.neutral_solver.solve_bound_state(7, J=0)

R500  = scat_fd500.r_grid
R2000 = scat_fd2000.r_grid

gaus500 = A_rad * np.exp(-alpha_r * (R500 - R0)**2)

mask500  = (R500  >= zoom[0]) & (R500  <= zoom[1])
mask2000 = (R2000 >= zoom[0]) & (R2000 <= zoom[1])

intgd6 = bnd6.wavefunction * gaus500 * dF_fd500
intgd7 = bnd7.wavefunction * gaus500 * dF_fd500

# Local k in coupling region
k_scatt = np.sqrt(2 * mu * 0.1833)   # scattering state in well (≈25 Bohr^-1)
k_v6    = np.sqrt(2 * mu * max(bnd6.energy - sys500.neutral_potential.V_0, 1e-6))
k_v7    = np.sqrt(2 * mu * max(bnd7.energy - sys500.neutral_potential.V_0, 1e-6))

print(f"k_scatt (inner) = {k_scatt:.1f} Bohr^-1")
print(f"k_v6 (inner)    = {k_v6:.1f} Bohr^-1  beat = {abs(k_scatt-k_v6):.1f}")
print(f"k_v7 (inner)    = {k_v7:.1f} Bohr^-1  beat = {abs(k_scatt-k_v7):.1f}")

# -----------------------------------------------------------------------
# 3. Figure
# -----------------------------------------------------------------------
fig = plt.figure(figsize=(15, 13))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.35)

# ---- Row 0: scattering states ----------------------------------------
panels = [
    (R500,  scat_fd500,  dF_fd500,  mask500,  "3-pt FD,  n=500"),
    (R2000, scat_fd2000, dF_fd2000, mask2000, "3-pt FD,  n=2000"),
    (R500,  scat_sinc,   dF_sinc,   mask500,  "Sinc-DVR, n=500"),
]
scale = 0.04   # scale dF/dR to overlay on F

for col, (Rv, scat, dF, mask, title) in enumerate(panels):
    ax = fig.add_subplot(gs[0, col])
    ax.plot(Rv[mask], scat.wavefunction[mask], "C0", lw=1.3, label=r"$F_E$")
    ax.plot(Rv[mask], dF[mask] * scale, "C1", lw=1.0, alpha=0.85,
            label=r"$dF_E/dR \times 0.04$")
    g = A_rad * np.exp(-alpha_r * (Rv[mask] - R0)**2)
    ax.fill_between(Rv[mask], -g * scale, g * scale,
                    alpha=0.18, color="C2", label="coupling env.")
    ax.axvline(R0, color="k", ls=":", lw=0.8)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("R (Bohr)")
    ax.set_xlim(zoom)
    ax.legend(fontsize=7, loc="upper right")
    if col == 0:
        ax.set_ylabel("Amplitude (a.u.)")

# Shared title for row 0
fig.text(0.5, 0.975,
         "Scattering state in coupling region  [E_coll = 66 cm⁻¹, J = 0]",
         ha="center", va="top", fontsize=12, fontweight="bold")

# Add max dF/dR annotation
for col, (Rv, scat, dF, mask, _) in enumerate(panels):
    ax = fig.axes[col]
    max_dF = np.max(np.abs(dF[mask]))
    k_local_max = np.max(np.abs(scat.wavefunction[mask]))
    predicted = k_scatt * k_local_max
    ax.text(0.03, 0.97,
            f"|dF/dR|max = {max_dF:.3f}\n"
            f"k·|F|max   = {predicted:.3f}\n"
            f"ratio = {max_dF/predicted:.3f}",
            transform=ax.transAxes, fontsize=7, va="top",
            bbox=dict(fc="white", alpha=0.6, ec="none"))

# ---- Row 1: bound states and coupling integrand ----------------------
ax1a = fig.add_subplot(gs[1, 0])
ax1b = fig.add_subplot(gs[1, 1])
ax1c = fig.add_subplot(gs[1, 2])

for ax, bnd, col, vprime in [(ax1a, bnd6, "C2", 6), (ax1b, bnd7, "C3", 7)]:
    ax.plot(R500[mask500], bnd.wavefunction[mask500], col, lw=1.5,
            label=rf"$F_{{v'={vprime}}}$")
    ax.plot(R500[mask500], dF_fd500[mask500] * scale, "C1", lw=1.0, alpha=0.7,
            label=r"$dF_E/dR \times 0.04$")
    ax.fill_between(R500[mask500],
                    -gaus500[mask500] * scale, gaus500[mask500] * scale,
                    alpha=0.15, color="gray", label="coupling env.")
    ax.axvline(R0, color="k", ls=":", lw=0.8)
    ax.set_title(f"Neutral bound state v'={vprime}  (k_local ≈ {(k_v6 if vprime==6 else k_v7):.0f} Bohr⁻¹)",
                 fontsize=9)
    ax.set_xlabel("R (Bohr)")
    ax.set_xlim(zoom)
    ax.legend(fontsize=7)
ax1a.set_ylabel("Amplitude (a.u.)")

I6 = simpson(intgd6, x=R500)
I7 = simpson(intgd7, x=R500)
ax1c.plot(R500[mask500], intgd6[mask500], "C2", lw=1.3,
          label=f"v'=6   ∫={I6:.2e}")
ax1c.plot(R500[mask500], intgd7[mask500], "C3", lw=1.3,
          label=f"v'=7   ∫={I7:.2e}")
ax1c.axhline(0, color="k", lw=0.5)
ax1c.axvline(R0, color="k", ls=":", lw=0.8)
ax1c.set_title(r"Coupling integrand  $F_{v'} \cdot m(R) \cdot dF_E/dR$", fontsize=10)
ax1c.set_xlabel("R (Bohr)")
ax1c.set_xlim(zoom)
ax1c.legend(fontsize=7)
ax1c.set_ylabel("Integrand (a.u.)")

# ---- Row 2: FFT and cumulative integral ------------------------------
ax2a = fig.add_subplot(gs[2, :2])
ax2b = fig.add_subplot(gs[2, 2])

def fft_spectrum(signal, r):
    dr = r[1] - r[0]
    freqs = np.fft.rfftfreq(len(signal), dr) * 2 * np.pi
    amp   = np.abs(np.fft.rfft(signal)) * dr
    return freqs, amp

mask_fft = (R500 >= 0.8) & (R500 <= 3.5)
f6, a6 = fft_spectrum(intgd6[mask_fft], R500[mask_fft])
f7, a7 = fft_spectrum(intgd7[mask_fft], R500[mask_fft])

ax2a.semilogy(f6, a6 + 1e-15, "C2", lw=1.3, label="v'=6 integrand")
ax2a.semilogy(f7, a7 + 1e-15, "C3", lw=1.3, label="v'=7 integrand")

markers = [
    (abs(k_scatt - k_v6), "beat v'=6", "C2"),
    (abs(k_scatt - k_v7), "beat v'=7", "C3"),
    (k_scatt + k_v6,      "sum v'=6",  "C2"),
    (k_scatt + k_v7,      "sum v'=7",  "C3"),
    (k_scatt,             "k_scatt",   "gray"),
]
ymin = 1e-12
for k, lab, col in markers:
    ax2a.axvline(k, color=col, ls="--", lw=0.8, alpha=0.6)
    ax2a.text(k + 0.3, ymin * 3, lab, fontsize=7, color=col, alpha=0.85, rotation=90)

ax2a.set_xlabel("Wavenumber  k (Bohr⁻¹)")
ax2a.set_ylabel("FFT amplitude  |ℱ{integrand}|")
ax2a.set_title("Fourier spectrum of coupling integrands"
               " — low-k peak = slowly-varying beat component we integrate", fontsize=10)
ax2a.set_xlim(0, 55)
ax2a.legend(fontsize=8)

# Cumulative (running) integral — visualises cancellation
cum6 = np.cumsum(intgd6[mask500] * np.gradient(R500[mask500]))
cum7 = np.cumsum(intgd7[mask500] * np.gradient(R500[mask500]))
ax2b.plot(R500[mask500], cum6, "C2", lw=1.5, label=f"v'=6  final={cum6[-1]:.2e}")
ax2b.plot(R500[mask500], cum7, "C3", lw=1.5, label=f"v'=7  final={cum7[-1]:.2e}")
ax2b.axhline(0, color="k", lw=0.5)
ax2b.axvline(R0, color="k", ls=":", lw=0.8)
ax2b.set_title("Running (cumulative) coupling integral\n"
               "— final value = tiny residual of many cancellations", fontsize=9)
ax2b.set_xlabel("R (Bohr)")
ax2b.set_ylabel("Partial integral")
ax2b.legend(fontsize=7)
ax2b.set_xlim(zoom)

# -----------------------------------------------------------------------
# 4. Save
# -----------------------------------------------------------------------
plt.savefig("plots/wavefunction_phase_analysis.png", dpi=140, bbox_inches="tight")
print("Saved  plots/wavefunction_phase_analysis.png")
