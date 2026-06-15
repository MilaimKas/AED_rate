# aed_rate

**Associative electron detachment (AED) cross sections and rates from
non-Born–Oppenheimer coupling.**

For a diatomic process

```
A⁻ + B  →  AB(v', J')  +  e⁻
```

where the anion is electronically bound at every bond length (no resonance), the
electron is ejected only through non-Born–Oppenheimer coupling — nuclear motion
driving an electronic transition into the continuum. The theory follows Acharya,
Kendall & Simons (1984/1985); the absolute cross-section normalization follows
Čížek *et al.* (2001).

### Foreworlds

I used to work on electronic structure and dynamics of molecular anions and I always wanted to implement the non-BO-detachment process for associative detachment reaction described in *Acharya et al.* However, due to lack of time and coding knowledge back in the days, the project was abounded. The rise of AI coding agent allowed me to pursue the goal as a side project.

This package has been almost entirely vibe-coded (using Claude Sonnet and Opus). I have a deep understanding of the underlying theory and needed to guide the AI assistant but the code is entriely AI generated. **It has not been fully reviewed, neither the physics nor the code. Only sanity checked.**     

---

## Installation

```bash
pip install -e .                 # core: numpy, scipy
pip install -e ".[pyscf]"        # + PySCF, to compute the ab initio coupling
pip install -e ".[plot]"         # + matplotlib, for the plotting helpers
```

Python ≥ 3.10. PySCF is needed only to *precompute* the coupling; everything
afterwards (loading it, cross sections, rates) runs on numpy/scipy alone.

---

## Quick start

The electronic coupling is computed ab initio (CPSCF) once and cached to disk;
all observables are then evaluated from the cached file.

```python
# 1. Precompute the coupling once (needs PySCF) ----------------------------
from aed_rate.electronic.potential import create_oh_system_acharya
from aed_rate.electronic.wavefunctions import ElectronicStructure
from aed_rate.electronic.coupling import InterpolatedCoupling

anion, neutral, EA = create_oh_system_acharya()
es = ElectronicStructure("O", "H", basis="6-311+G**")     # diffuse-augmented
coupling = InterpolatedCoupling(es, anion, n_points=40)
coupling.precompute()
coupling.save("oh_coupling.npz")

# 2. Use it (no PySCF required from here on) -------------------------------
from aed_rate import AEDSystem
from aed_rate.electronic.coupling import InterpolatedCoupling
from aed_rate.utils.constants import CONSTANTS

sys = AEDSystem.oh_system(
    coupling=InterpolatedCoupling.from_npz("oh_coupling.npz"),
    n_grid=6000,
)
E = CONSTANTS.cm1_to_hartree(66.0)                         # collision energy

sys.sigma_AD(E, unit="Angstrom^2")                        # total σ_AD(E)
sys.cross_section(E, J=0, v_prime=6, J_prime=0, unit="Angstrom^2")  # state-resolved
sys.thermal_rate(300.0)                                   # k(T), cm³/s
sys.diagnostic(E, J=0, save_dir="plots/diag")             # every step, as figures
```

For a quick smoke test without PySCF, `AEDSystem.oh_system()` (no `coupling`
argument) falls back to a Gaussian `ModelCoupling` stand-in — convenient but not
physical.

---

## Computing each step separately

Every ingredient of the cross section is its own object, with a `compute_*`
method and a matching `plotting` helper.

```python
import numpy as np
from aed_rate.electronic.potential import create_oh_system_acharya
from aed_rate.electronic.wavefunctions import ElectronicStructure
from aed_rate.electronic.coupling import ElectronicCoupling
from aed_rate.nuclear.nuclear_wavefunction import create_wavefunction_solver
from aed_rate.utils.constants import CONSTANTS, get_reduced_mass
from aed_rate.utils import plotting

anion, neutral, EA = create_oh_system_acharya()
mu = get_reduced_mass("O", "H")
E  = CONSTANTS.cm1_to_hartree(66.0)

# 1. Potential energy curves
plotting.plot_potential_curves(anion, neutral, EA)

# 2. Nuclear wavefunctions (bound χ_{v'}, scattering F_E, and dF_E/dR)
solver = create_wavefunction_solver(anion, mu, method="morse", n_grid=6000)
bound  = solver.solve_bound_state(v=0, J=0)
scatt  = solver.solve_scattering_state(E, J=0, normalization="unit_amplitude")
dF_dR  = solver.wavefunction_derivative(scatt)
plotting.plot_bound_states(solver.solve_all_bound_states(J=0), anion)
plotting.plot_scattering_derivative(scatt, dF_dR)

# 3. Electronic coupling m_rad(R), m_rot(R) and its real-space ingredients
coupling = ElectronicCoupling(ElectronicStructure("O", "H", basis="6-311+G**"),
                              homo_symmetry="pi")
m   = coupling.compute_coupling_at_r(R=1.822, electron_energy=0.01)   # m_rad, m_rot
ing = coupling.compute_coupling_intermediates(R=1.822, electron_energy=0.01)
plotting.plot_electronic_intermediates(ing)        # ∂φ_HOMO/∂R and the OPW φ_k

# 4. The coupling integrand χ·m_rad·dF_E/dR (and its cancellation)
m_rad_R = np.array([coupling.compute_coupling_at_r(R, 0.01).m_rad.real
                    for R in solver.r_grid])
plotting.plot_coupling_integrand(solver.r_grid, bound.wavefunction, m_rad_R, dF_dR)
```

Each `plotting.*` function returns `(fig, axes)` for further customization.

---

## Studying a new system A⁻ + B

Provide two Morse curves and the reduced mass, then precompute the coupling for
your atoms — the precompute is system-agnostic (atoms and basis come from
`ElectronicStructure`).

```python
from aed_rate.electronic.potential import MorsePotential
from aed_rate.electronic.wavefunctions import ElectronicStructure
from aed_rate.electronic.coupling import InterpolatedCoupling
from aed_rate.rate.state_to_state import AEDRateCalculator
from aed_rate.utils.constants import CONSTANTS, get_reduced_mass

EA = CONSTANTS.ev_to_hartree(1.0)                     # electron affinity of AB
anion   = MorsePotential(D_e=..., r_e=..., beta=..., V_0=0.0)   # anion min = 0
neutral = MorsePotential(D_e=..., r_e=..., beta=..., V_0=EA)    # neutral min = EA
mu = get_reduced_mass("A", "B")

# precompute once (closed-shell anion; π HOMO tracking, σ via a generic fallback)
es = ElectronicStructure("A", "B", basis="aug-cc-pVDZ")
coupling = InterpolatedCoupling(es, anion, charge=-1, spin=0,
                                homo_symmetry="pi", n_points=40)
coupling.precompute()
coupling.save("AB_coupling.npz")

calc = AEDRateCalculator(
    anion, neutral, EA, mu,
    coupling=InterpolatedCoupling.from_npz("AB_coupling.npz"),
    solver_method="morse", n_grid=6000,
)
calc.total_cross_section_all_J(CONSTANTS.cm1_to_hartree(66.0))    # σ_AD(E), a₀²
```

---

## Notes

- **Use a diffuse-augmented basis** (e.g. `6-311+G**`, `aug-cc-pVDZ`): the anion
  HOMO's diffuse tail dominates the coupling to the long-wavelength continuum.
- **Cross sections need the Morse solver** (`solver_method="morse"`) and
  `n_grid ≳ 6000` to resolve the near-cancelling coupling integrand.
- `ModelCoupling` is a Gaussian stand-in, not a physical coupling.

---

## Limitations & approximations

**Applicability**
- Diatomic molecules only.
- Non-resonant regime: the anion is electronically bound at all bond lengths.
  Resonant systems (e.g. H + halogen⁻) require a nonlocal resonance model instead.
- First-order, weak-coupling (Fermi golden rule): no resonance back-coupling;
  valid when the detachment probability ≪ 1.

**Electronic structure**
- Koopmans / frozen-orbital, single-determinant HF (CPSCF) — no electron correlation.
- Closed-shell anion only (RHF, spin = 0); orbital tracking tuned for a π HOMO.

**Continuum electron**
- Orthogonalized plane wave: no electron–molecule interaction beyond
  orthogonalization (no polarization or exchange potential).
- Low-k (k·r ≲ 1) OPW, single l = 1 (p-wave) channel; the l = 0 s-wave is dropped
  — exact for π HOMOs, an approximation for σ / diffuse orbitals.

**Nuclear motion**
- Morse potential curves; Pekeris approximation for the centrifugal term (J > 0).
- The second-derivative (∇²) non-BO term is neglected (≈ 0.1–0.3 %).

Research code — internally consistent but not benchmarked against experiment.

---

## Package layout

| Module | Contents |
|---|---|
| `electronic/potential.py` | `MorsePotential`, `create_oh_system_acharya()` |
| `electronic/coupling.py` | `InterpolatedCoupling` (precompute / `from_npz`), `ElectronicCoupling` (CPSCF), `ModelCoupling` |
| `electronic/continuum.py` | `ContinuumOrbital`, OPW, electron density of states |
| `nuclear/` | DVR / Numerov / Morse solvers; `create_wavefunction_solver()` |
| `rate/state_to_state.py` | `AEDRateCalculator`: cross sections and rates |
| `rate/thermal.py` | `ThermalRateCalculator`: k(T) |
| `aed_calculator.py` | `AEDSystem` high-level facade + `diagnostic()` |
| `utils/plotting.py` | per-step plotting helpers |

---

## References

- Acharya, Kendall, Simons, *J. Am. Chem. Soc.* **106**, 3402 (1984).
- Acharya, Das, Simons, *J. Chem. Phys.* **83**, 3888 (1985).
- Čížek, Horáček, Thiel, Hotop, *J. Phys. B* **34**, 983 (2001).
- Simons, *J. Phys. Chem. A* **102**, 6035 (1998).
