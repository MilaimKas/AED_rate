# Theory of Associative Electron Detachment (AED)

## References

- **[1]** Acharya, Kendall, Simons — *J. Am. Chem. Soc.* **106**, 3402 (1984)
  "Vibration-Induced Electron Detachment in Molecular Anions"
- **[2]** Acharya, Das, Simons — *J. Chem. Phys.* **83**, 3888 (1985)
  "Associative Electron Detachment: O⁻ + H → OH + e⁻"
- **[3]** Simons — *J. Phys. Chem. A* **102**, 6035 (1998)
  "Semiquantum Expressions for Electronically Nonadiabatic Electron Ejection Rates"

All equations below are in **atomic units** (ℏ = mₑ = e = a₀ = 1).

> **Verification status (2026-06-11).** The rate equation, prefactor, OPW
> normalization, and density-of-states factor have been checked against
> Acharya 1983 (ref [1], Eqs. 1–7) and Acharya 1985 (ref [2], Eq. 3). Two
> things remain **open**: (i) the normalization convention of the nuclear
> *collisional scattering* state F_{E,J} (box- vs energy- vs flux-normalized —
> see §7.3), which is **not stated in any obtainable Acharya source** (the 1983
> paper is bound→bound and structurally has no such state; the 1985 paper does
> not specify it) and must be derived from the per-collision-rate requirement;
> and (ii) the exact R-power bookkeeping of the rotational coupling (§4.2–4.3).
> Note: the *outgoing-electron* OPW normalization is **settled** (box-norm, L³
> cancels — §3.2, §6.1) and is not at issue. These are flagged inline.

---

## 1. Physical Picture

In associative electron detachment (AED), an atomic anion A⁻ collides with
a neutral atom B. The collision forms a transient molecular anion AB⁻, which
can eject an electron to produce a stable neutral molecule AB plus a free
electron:

```
A⁻ + B  →  AB(v', J')  +  e⁻
```

For systems like O⁻ + H, the anion potential energy surface (¹Σ for OH⁻)
lies **below** the neutral surface (²Π for OH) at all bond lengths R.
Within the Born-Oppenheimer approximation, the electron is always bound and
detachment is forbidden. Detachment becomes possible only through
**non-Born-Oppenheimer coupling** — the exchange of energy and momentum
between nuclear motion and electronic degrees of freedom.

This is fundamentally different from dissociative electron attachment (DEA)
where potential curve crossings allow the process within the BO picture.

---

## 2. Born-Oppenheimer Framework

### 2.1 Electronic Schrödinger equation

```
hₑ(r|R) ψₖ(r|R) = Eₖ(R) ψₖ(r|R)                              [3, Eq. 1]
```

where r = electronic coordinates, R = nuclear coordinates, ψₖ = BO electronic
wavefunctions, Eₖ(R) = potential energy surfaces.

### 2.2 Full Schrödinger equation and BO expansion

The total wavefunction is expanded in the BO basis:

```
Ψ(r,R) = Σₖ ψₖ(r|R) χₖ(R)                                     [3, Eq. 4]
```

where χₖ(R) are nuclear (vibration-rotation) wavefunctions.

### 2.3 Non-BO coupling terms

Substituting the BO expansion into the full Schrödinger equation gives
coupled equations. The coupling between electronic states i and f arises
from the nuclear kinetic energy operator acting on the electronic
wavefunctions:

```
Coupling = Σₐ (1/mₐ) ⟨ψf| (-i∂/∂Rₐ) |ψi⟩ · (-i∂χi/∂Rₐ)     [3, Eq. 7]
         + Σₐ (1/2mₐ) ⟨ψf| (-∂²/∂Rₐ²) |ψi⟩ χi
```

The second-derivative term is smaller by a factor of (mₑ/μ)^{1/2} and is
neglected. Only the first-derivative (momentum) coupling is retained.

---

## 3. State-to-State AED Rate (Fermi Golden Rule)

### 3.1 The rate expression

The rate for a specific transition from an initial scattering state
(collision energy E, angular momentum J) to a final bound state
(vibrational quantum number v', rotational quantum number J') is given by
Fermi's Golden Rule:

```
Rate(v',J'; E,J) = 2π × ρ(Eₑ) × |V_{v,v'}|²                   [1, Eq. 2]
```

where:
- ρ(Eₑ) is the density of translational states of the ejected electron
- V_{v,v'} is the non-BO coupling matrix element
- Eₑ is the kinetic energy of the ejected electron

The matrix element V_{v,v'} carries a factor of 1/μ from the nuclear kinetic
energy operator (see §4). Writing it out, Acharya's full prefactor is
**2πℏ³/μ²** [1, Eq. 2; 2, Eq. 3]: with V_{v,v'} = (1/μ) ∫ … the rate becomes
2π ρ (1/μ²) |∫ …|², i.e. 2πℏ³/μ² with ℏ = 1. Our implementation matches this
(the 1/μ lives inside V_rad / V_rot).

### 3.2 Density of states

The translational density of states for a free electron confined to a box
of volume L³ is, in full [1, Eq. 3]:

```
ρ(Eₑ) = mₑ L³ kₑ / (2π² ℏ²)
```

**The L³ cancels exactly.** The OPW φₖ is box-normalized with a prefactor
1/L^{3/2} (§6.1, [1, Eq. 4]), so the coupling matrix element — which contains
φₖ once — carries |φₖ|² ∝ 1/L³. The L³ in ρ cancels this L⁻³, and the rate is
independent of the (fictitious) box size. Acharya states this explicitly:
"the L³ factor appearing in ρ … is cancelled by a factor of L⁻³ that arises
from the normalization of φₖ" [1, p. 3404].

In practice we therefore use the **per-unit-volume** density of states with an
**un-normalized** OPW (no 1/L^{3/2} factor):

```
ρ(Eₑ) = kₑ / (2π²)     (in atomic units, per unit volume)      [1, Eq. 3]
```

where kₑ = √(2Eₑ) is the electron's wave-vector magnitude. **(Verified
against Acharya 1983 Eqs. 3–4 and 7.)**

### 3.3 Energy conservation

The electron kinetic energy is determined by conservation of energy:

```
Eₑ = E_collision + E_anion_dissoc - E_{v',J'}(neutral) - EA
```

where:
- E_collision = kinetic energy of the A⁻ + B collision
- E_anion_dissoc = dissociation energy of the anion (energy of its dissociation limit relative to its minimum)
- E_{v',J'} = rovibrational energy of the neutral product (relative to neutral minimum)
- EA = electron affinity of the neutral molecule (energy gap between surfaces)

With the zero energy at the minimum of the anion's potential well. For the process to occur, Eₑ > 0 is required.

---

## 4. The Coupling Matrix Element

### 4.1 Full coupling integral

The non-BO coupling matrix element has the general form [2, Eq. 3]:

```
V = ⟨χ_{v',J'} | ⟨ψf| ∇ |ψi⟩ · (∇/μ) | χ_{E,J}⟩
```

where 
- ∇ is the gradient with respect to nuclear coordinates
- χ_{v',J'} is the nuclear WF of the neutral product with vibrational state v' and rotational state J'
- ψf is the final electronic WF (in the Koopman approximation taken to be the orthogonlized plane wave of the outgoing electron: OPW -> φₖ(r))
- ψi is the initial electronic WF of the anion complex (in the Koopman approximation taken to be the HOMO).
- χ_{E,J} is  the scattering nuclear WF of the anion.
- μ is the reduced mass of the system.

### 4.2 Radial (vibrational) and angular (rotational) decomposition

For a diatomic molecule, the nuclear gradient in spherical coordinates
(R, θ, φ) decomposes into:

```
∇ = R̂ ∂/∂R  +  θ̂ (1/R)∂/∂θ  +  φ̂ (1/R sinθ)∂/∂φ
```

This gives two types of coupling:

**Radial (vibrational) coupling** — from ∂/∂R acting on both electronic
and nuclear wavefunctions:

```
V_rad(J,J') = (1/μ) × δ_{J,J'} × ∫ F_{v',J'}(R) × m_rad(R) × dF_{E,J}(R)/dR × dR
```

where m_rad(R) is the electronic part of the radial coupling (see next section 4.3)


**Angular (rotational) coupling** — from (1/R)∂/∂θ acting on electronic
and nuclear wavefunctions:

```
V_rot(J,J') = (1/μ) × C(J,J') × ∫ F_{v',J'}(R) × m_rot(R) × F_{E,J}(R)/R × dR
```

where m_rot(R) is the electronic part of the rotational coupling (see next section 4.3)

Key differences:
- Radial coupling involves dF_{E,J}/dR (derivative of scattering wavefunction)
  and preserves J (ΔJ = 0)
- Rotational coupling involves F_{E,J}/R (not the derivative) and changes J
  by ±1 (from angular momentum algebra of spherical harmonics)
- C(J,J') are angular coupling coefficients from ⟨Y_{J'M'}|∂/∂θ|Y_{JM}⟩ (see last equation in [2], not numbered in the paper), described in section 4.4.

> **⚠ Open: rotational R-factor bookkeeping.** There is a convention clash
> between the m_rot definition in §4.3 (written *with* a 1/R) and the V_rot
> integrand above (which *also* divides by F_{E,J} by R). Taken literally that
> double-counts to 1/R². The implementation is in fact self-consistent because
> it defines the *electronic* factor as the θ-derivative m_rot ≡ ⟨φₖ|∂ψᵢ/∂θ⟩ =
> R⟨φₖ|∂ψᵢ/∂x_B⟩ (carrying an explicit R via ∂/∂θ = R ∂/∂x_B at the reference
> geometry), and the 1/R in the integrand then leaves a single net x-derivative
> coupling. The exact R-power should be checked against Acharya 1985 p. 3890,
> where the angular integrals Y*_{J'M'}(1/sinθ)(dY_{JM}/dθ) and
> Y*_{J'M'}(1/sin²θ)Y_{JM} are given. Do **not** treat §4.2–4.3 as settled
> until this is reconciled with the code's m_rot = R·∂C/∂x convention
> (`coupling.py precompute`, `mo1[1,0,:,homo]*R`).

### 4.3 Electronic coupling matrix elements

The electronic parts of the coupling are:

```
m_rad(R) = ⟨φₖ(r) | ∂ψᵢ(r;R)/∂R⟩_r                            [1, Eq. 7]

m_rot(R) = ⟨φₖ(r) | (1/R) ∂ψᵢ(r;R)/∂θ⟩_r
```

where:
- ψᵢ(r;R) = anion electronic wavefunction (specifically, the HOMO)
- φₖ(r) = continuum electron wavefunction (OPW approximation)
- The subscript r means integration over electronic coordinates only

### 4.4 Angular coupling coefficients C(J,J')

The angular integrals over spherical harmonics give:

```
⟨Y_{J'M'} | ∂/∂θ | Y_{JM}⟩
```

These are non-zero only for ΔJ = ±1 (and ΔM = 0 for the relevant
components). The explicit forms involve Clebsch-Gordan coefficients.

For the total rate at given (E,J), one sums incoherently over
J' = J-1, J, J+1:

```
Rate_total(E,J) = Σ_{v'} Σ_{J'=J-1,J,J+1} 2π ρ(Eₑ) |V_rad(J,J') + V_rot(J,J')|²
```

Note: V_rad contributes only to J' = J, while V_rot contributes to
J' = J ± 1. At J' = J there is no interference between radial and
rotational terms (they contribute to different J' channels).

---

## 5. Electronic Structure Ingredients

### 5.1 Koopmans' theorem approximation

The neutral+free-electron wavefunction is approximated as [1]:

```
ψf(r;R) ≈ Â[ψ_neutral(r';R) × φₖ(rₙ)]
```

where Â is the antisymmetrizer, ψ_neutral is obtained from the anion
wavefunction by removing the HOMO electron, and φₖ is the continuum
orbital. This is the Koopmans' theorem (frozen orbital) approximation.

### 5.2 LCAO-MO expansion and derivatives

The anion HOMO is expanded in atomic orbitals:

```
φ_HOMO(r; R) = Σⱼ Cⱼ(R) gⱼ(r)
```

where Cⱼ(R) are the MO coefficients (R-dependent) and gⱼ(r) are
Gaussian-type atomic orbitals.

The derivative with respect to R has two physically distinct terms:

```
∂φ_HOMO/∂R = Σⱼ (dCⱼ/dR) gⱼ(r)   +   Σⱼ Cⱼ(R) (∂gⱼ/∂R)
             └─ coefficient term ─┘   └─ AO-motion (Pulay) term ─┘
```

- **Coefficient term** — the response of the MO coefficients as the SCF
  solution changes with geometry. Computed analytically via PySCF's CPSCF
  (coupled-perturbed SCF), `hessian.rhf.solve_mo1()`. CPSCF's h1 *does*
  include overlap-derivative (Sˣ) terms, so the moving basis influences the
  *coefficient* response; but this is still only the first term — it is
  ∂C/∂R evaluated in the **fixed** AO basis at R₀.
- **AO-motion (Pulay) term** — the explicit derivative of the basis
  functions as the H atom moves. This is a **separate** term, *not* contained
  in CPSCF's mo1. We evaluate ∂φ/∂R on a fixed grid as `ao_values @ dC/dR`,
  which omits it by construction.

**Why omitting the AO-motion term is justified here.** For the OH⁻ π HOMO
detaching into a πₓ-symmetry OPW (∝ x), the AO-motion contribution to the
*coupling* is ∫ (OPW ∝ x) × ∂χ_H/∂z_H d³r, which vanishes by C∞v symmetry for
an s-type H basis (E1x × A1 angular integral = 0). We verified this
numerically: CPSCF and a full finite-difference calculation that *includes*
AO motion agree to machine precision, and the AO-motion term is ≈ 10⁻¹⁵
(`test_plausibility.py` Section 10). So for this system the coefficient term
is not merely dominant — it is the *only* nonzero contribution to the coupling.

**Relation to Acharya.** Acharya did not use CPSCF: they evaluated the LCAO
coefficients Cⱼ(R) at many bond lengths by HF-SCF, least-squares fit them to
polynomials in R, and differentiated the polynomials [1, p. 3404]. This yields
the same coefficient derivative dCⱼ/dR; CPSCF simply obtains it analytically.

### 5.3 Computing MO derivatives with PySCF

For a diatomic A-B along the z-axis (A at origin, B at z=R):

```python
mf = scf.RHF(mol).run()
hess = mf.Hessian()
h1ao = hess.make_h1(mf.mo_coeff, mf.mo_occ)
mo1, mo_e1 = hess.solve_mo1(mf.mo_energy, mf.mo_coeff, mf.mo_occ, h1ao)
# mo1 shape: [natm][3, nao, nocc]
```

Extracting derivatives:
- **Bond stretch**: `dC/dR = mo1[1][2, :, :]` (atom B, z-direction)
- **Rotation**: `dC/dθ = R × mo1[1][0, :, :]` (atom B, x-direction, scaled by R)

The rotational derivative follows because for a diatomic along z,
an infinitesimal rotation by dθ displaces atom B by dx_B = R dθ in the
x-direction, so ∂/∂θ = R × ∂/∂x_B.

Evaluating on a real-space grid:

```python
grids = dft.gen_grid.Grids(mol)
grids.build()
ao_values = mol.eval_gto("GTOval_sph", grids.coords)   # (N_grid, N_ao)
d_phi_dR     = ao_values @ dC_dR[:, homo_idx]           # radial
d_phi_dtheta = ao_values @ dC_dtheta[:, homo_idx]       # rotational
```

---

## 6. Continuum Electron: Orthogonalized Plane Wave (OPW)

### 6.1 General OPW

The continuum electron is described by a box-normalized plane wave
orthogonalized to the occupied orbitals [1, Eq. 4]:

```
φₖ = (1/L^{3/2}) [ exp(ik·r) − Σⱼ ⟨φⱼ|exp(ik·r)⟩ φⱼ ]
```

where the sum runs over the occupied orbitals **φⱼ of the anion** (OH⁻) —
i.e. the orbitals retained in the Koopmans neutral-plus-electron determinant
(all anion occupied orbitals *except* the detaching HOMO). The ejected
electron must be orthogonal to these. The L^{-3/2} normalization factor cancels
with the L³ in the density of states ρ (§3.2), so the box size never appears
in the rate. **(Verified against Acharya 1983 Eq. 4 — note: the orthogonalization
is to the anion's orbitals, not the neutral's.)**

### 6.2 Low-k (long wavelength) limit

When kR_mol ≪ 1 (low electron kinetic energy), the plane wave can be
expanded and only the leading angular momentum component retained [1, Eq. 5-6]:

**σ symmetry** (e.g., 3σ HOMO of LiH⁻):
```
φₖ^σ ≈ kz × z                                                   [1, Eq. 5]
```

**π symmetry** (e.g., 1π HOMO of OH⁻):
```
φₖ^π ≈ kx × x    (for πₓ component)                            [1, Eq. 6]
```

These are the m=0 and m=±1 components of the partial wave expansion,
respectively. The low-k limit is valid when kₑ × r_e < 0.5.

### 6.3 Switchover criterion

For higher electron energies (k × r_e > 0.5), the full numerical OPW
on a Becke grid should be used instead of the analytical low-k forms.

---

## 7. Nuclear Wavefunctions

### 7.1 Scattering state (anion surface)

The scattering wavefunction χ_{E,J}(R,θ,φ) = F_{E,J}(R)/R × Y_{JM}(θ,φ)
where the radial part F_{E,J}(R) satisfies:

```
[-1/(2μ) d²/dR² + V_eff(R)] F_{E,J}(R) = E_total × F_{E,J}(R)
```

with V_eff(R) = V_anion(R) + J(J+1)/(2μR²) and
E_total = V_anion(∞) + E_collision.

Boundary conditions: F(0) = 0, F(R→∞) ~ sin(kR - Jπ/2 + δ).

### 7.2 Bound state (neutral surface)

The bound vibrational wavefunction χ_{v',J'} = F_{v',J'}(R)/R × Y_{J'M'}(θ,φ)
where F_{v',J'}(R) satisfies the same radial equation with V_neutral(R).

### 7.3 Normalization

> **⚠ Open issue — this is the leading suspect for the absolute-rate
> discrepancy with Acharya.**

A crucial distinction: the density-of-states factor ρ in the rate formula is
the **electron** DOS (§3.2). It does **not** set the normalization of the
**nuclear** scattering state F_{E,J} — that is a separate convention, and the
two must not be conflated.

- **Bound states** (final neutral F_{v',J'}, and the anion vibrational states
  in the vibration-induced 1983 work) are normalized to ∫|F|² dR = 1. This is
  unambiguous and matches Acharya.
- **The scattering state** F_{E,J} (initial O⁻ + H collision state in the 1985
  AED work) has **no universally fixed normalization**, and Acharya 1985 does
  not state theirs explicitly. It is *not* recoverable from the 1983 paper
  either: that work is bound→bound (vibration-induced) and structurally has no
  collisional scattering state. So the convention must be **derived** from the
  requirement that the rate W be a per-collision-complex rate in s⁻¹.
  The standard choice for a *rate* is **energy normalization**,
  ⟨E|E'⟩ = δ(E − E'), giving the asymptotic form

  ```
  F_{E,J}(R) → √(2μ / (π kₙᵤ𝒸)) · sin(kₙᵤ𝒸 R − Jπ/2 + δ)
  ```

  with kₙᵤ𝒸 = √(2μE_collision).

- **What our code currently does:** box normalization, ∫|F|² dR = 1 over
  [R_min, R_max]. This is **L-dependent** (a larger box forces a smaller
  amplitude) and is therefore *not* physical on its own. Converting box → energy
  normalization multiplies the amplitude by √(μL / (π kₙᵤ𝒸)) and the rate by
  μL / (π kₙᵤ𝒸). For OH⁻ at E_collision = 66 cm⁻¹ (μ ≈ 1728, L ≈ 14.5 Bohr,
  kₙᵤ𝒸 ≈ 1.02) this factor is ≈ 88.5 in amplitude, **≈ 7800× in rate**.

**Implication.** Because the box-normalized rate is L-dependent and unphysical,
absolute rates from the current pipeline cannot be compared to Acharya's
tabulated s⁻¹ values until this convention is pinned down. (Empirically the
box-normalized J=0 rate happens to land near Acharya's *J-summed* Table IV
values rather than the J=0 Table I values — possibly coincidental, possibly a
clue.) **Do not patch this with a single empirical factor.** The convention is genuinely
undocumented in every obtainable Acharya source (confirmed: the 1985 paper states
only that F_{E,J} is "obtained by numerically integrating the radial Schrödinger
equation"). Note that flux- and energy-normalization differ by only √(2/π) ≈ 0.8 —
they are *not* intermediate between box and energy; both are ~5000–7800× larger
than box-norm at OH⁻/66 cm⁻¹.

**Status of the investigation (2026-06-11).**
- The diffuse basis (6-311+G**) fixed the *coupling*: the v′ distribution now
  matches Acharya Table I in shape (peaks at v′=6). Because E_collision is fixed
  within a table, the F_{E,J} normalization is a single v′-independent constant, so
  the shape agreement validates the coupling and is **blind** to the normalization.
- Residual factor (our box-norm / Acharya, J=0, v′=4–6 geomean): **15× at 66 cm⁻¹,
  34× at 256, 78× at 732** — i.e. it **grows with collision energy** (≈ E^0.5–0.65).
- This is degenerate between two readings and the present data cannot separate them:
  (A) Acharya used energy/flux normalization (residual ∝ √E) — but the *direction*
  is backwards (energy-norm would make Acharya ~7800× larger, not 14× smaller);
  (B) our coupling's k_e-dependence is inaccurate at higher E (k_e moves into the
  extrapolated end of the [0.01, 0.40] grid where the low-k OPW + spline degrade),
  with box-norm roughly correct.
- **Decisive resolution (deferred):** an independent absolute calculation that does
  not normalize a quantum scattering state — the Simons 1998 semiclassical rate
  (§9) — would settle whether the true rate is ~10² s⁻¹ (Acharya) or ~10³ (ours).

### 7.4 Solver architecture

Three solvers are available, all sharing the interface defined by the
abstract base class `WavefunctionSolver` (`_base_solver.py`):

| Solver | Bound states | Scattering states | PEC type |
|--------|-------------|-------------------|----------|
| `DVRWavefunctionSolver` | Matrix diagonalization | Box eigenstate nearest E | Any |
| `NumerovWavefunctionSolver` | Outward+inward shooting | Outward-only shooting | Any |
| `MorseAnalyticSolver` | Exact Laguerre polynomials | Numerov + TISE derivative | Morse |

**Recommended solver for AED coupling integrals: `MorseAnalyticSolver`**
(see rationale in §7.5–7.8 below).

### 7.5 DVR solver

The Hamiltonian is discretised on a uniform grid with spacing Δr:

```
T_ij = 1/(2μ Δr²) × { +2   if i = j
                       −1   if |i−j| = 1
                        0   otherwise }
```

Diagonalisation of H = T + diag(V_eff) gives all eigenvalues and
eigenvectors simultaneously. Bound eigenstates are identified as those
with eigenvalue below the dissociation limit; scattering states as
those above.

**Strengths**: robust, gives the entire spectrum in one shot, bound-state
wavefunctions are clean.

**Limitations for scattering states**:

1. *Phase error*. Hard-wall boundary conditions enforce F(R_min) = 0 and
   F(R_max) = 0. The physical scattering state only satisfies the first;
   at large R it is an oscillatory standing wave. The DVR eigenstate
   closest to the desired energy has a different standing-wave phase,
   which can change the coupling integral.

2. *Derivative amplitude error*. The 3-point central-difference formula
   F'[n] = (F[n+1] − F[n−1]) / (2 Δr) applied to a locally oscillatory
   function F = A sin(k_local R + φ) gives F'_CD = A k cos(···) ×
   sin(k Δr) / (k Δr). For k_local ≈ 25 Bohr⁻¹ (inner Morse well at
   R ≈ R_e for OH) and Δr ≈ 0.029 Bohr, the damping factor
   sin(kΔr)/(kΔr) ≈ 0.91, i.e. the derivative amplitude is
   **underestimated by ~9%**.

### 7.6 Numerov solver

The Numerov algorithm solves F'' = -g F (where g = 2μ(E − V_eff) > 0
in the classically allowed region) via a 4th-order finite-difference
recursion:

```
F[n+1] = (2 F[n] (1 − 5 c g[n]) − F[n−1] (1 + c g[n−1])) / (1 + c g[n+1])

c = Δr² / 12
```

**Bound states**: outward propagation from R_min and inward from R_max,
matched at the outer classical turning point. The eigenvalue is found
by bisection on the number of nodes plus the log-derivative mismatch
at the matching point.

**Scattering states**: outward-only propagation from the inner wall
with F[0] = 0, F[1] = Δr. This correctly enforces the physical boundary
condition at R_min without constraining R_max, giving the correct
standing-wave phase.

> **Status**: the Numerov solver works for any PEC but has not been
> systematically validated. DVR is preferred for bound states; the
> Morse solver is preferred for scattering states in AED coupling
> integrals.

### 7.7 Morse analytical solver

For Morse potentials V(R) = D_e (1 − e^{−β(R−R_e)})² + V_0, the
radial Schrödinger equation can be solved in closed form.

#### 7.7.1 Bound states — generalised Laguerre polynomials

The substitution z = 2λ e^{−β(R−R_e)} with λ = √(2μD_e)/β transforms
the equation into a confluent hypergeometric form with polynomial
solutions:

```
ψ_v(R) = z^{λ−v−½} exp(−z/2) L_v^{2λ−2v−1}(z)
```

where L_v^α is the generalised Laguerre polynomial.

The energies follow exactly:

```
E_v = V_0 + ω_e (v + ½) − ω_e x_e (v + ½)²

ω_e = β √(2D_e / μ),    ω_e x_e = ω_e² / (4D_e)
```

The maximum bound vibrational quantum number is v_max = ⌊λ − ½⌋.

#### 7.7.2 Pekeris approximation for J > 0

For J > 0 the centrifugal term J(J+1)/(2μR²) breaks the exact Morse
form. The Pekeris (1934) approximation replaces 1/R² by a three-term
expansion in the Morse basis variable ξ = e^{−β(R−R_e)}:

```
1/R² ≈ (1/R_e²) [c₀ + c₁ ξ + c₂ ξ²]

c₀ = 1 − 3p + 3p²
c₁ = 4p − 6p²
c₂ = 3p² − p         (p = 1/(β R_e))
```

This restores the potential to exact Morse form with effective parameters:

```
B_J = J(J+1)/(2μR_e²)

a₀ = D_e + B_J c₀,    a₁ = 2D_e − B_J c₁,    a₂ = D_e + B_J c₂

D_e_eff = a₁²/(4a₂),   R_e_eff = R_e + ln(2a₂/a₁)/β,   V_0_eff = V_0 + a₀ − D_e_eff
```

The bound-state energies and wavefunctions then use the same Laguerre
formula with λ_eff = √(2μD_e_eff)/β.

For scattering states, the effective collision energy is:

```
E_coll_eff = E_coll − B_J c₀
```

The channel is inaccessible (energetically closed) when E_coll_eff ≤ 0.
For the OH⁻ system at E = 66 cm⁻¹, only J = 0–3 are accessible.

#### 7.7.3 Scattering states — Numerov + TISE derivative

The exact analytical scattering solution uses the Tricomi confluent
hypergeometric function U(a, b, z) with complex parameters
a ≈ −λ + ik/β, b = 1 + 2ik/β. However, for small z (large R, the
asymptotic region), computing U(a, b, z) involves catastrophic
cancellation:

```
U(a,b,z) = Γ(1−b)/Γ(a−b+1) × M(a,b,z) + Γ(b−1)/Γ(a) × z^{1−b} × M(...)
```

With λ ≈ 22 (OH system), a − b + 1 ≈ −22, and the Gamma functions
Γ(−22 ± ik/β) have magnitude ~10⁻²¹ (near the 22nd pole). Both terms
are individually ~10⁻²¹ and their sum is O(1) — **21 digits of
cancellation**. Even arbitrary-precision arithmetic (mpmath at 200+
decimal places) is unreliable for generic λ ∈ [10, 50].

The implemented solution is a **hybrid**: Numerov outward shooting for
the wavefunction F(R), combined with the TISE for its derivative.

**Wavefunction F(R)**: standard Numerov outward propagation from the
inner wall (F(R_min) = 0) on the Pekeris-corrected effective Morse
potential. This is unconditionally stable, gives the correct phase, and
runs entirely in numpy (no mpmath loop over grid points).

**Derivative dF/dR via TISE integration**: instead of central differences,
the Schrödinger equation itself provides the exact second derivative:

```
F''(R) = 2μ [V_eff(R) − E_total] F(R)      (exact from TISE)
```

The first derivative is recovered by cumulative trapezoid integration:

```
F'[n+1] = F'[n] + Δr/2 × (F''[n] + F''[n+1])
```

This avoids the **sin(kΔr)/(kΔr) amplitude damping** of finite-difference
formulas. The physical reason: differencing the oscillatory wavefunction
F samples it at discrete points separated by Δr, and when k Δr is not
small the interpolation error is systematic. Integrating F'' does not
suffer from this because F'' = −g F is a smooth product of two known
functions (the potential and the wavefunction), and the trapezoid
quadrature error is O(Δr²) without amplitude bias.

Quantitative comparison (OH at E_coll = 66 cm⁻¹):
- 3-point central difference: |dF/dR|_max = 2.20 (inner well)
- TISE cumulative integral:   |dF/dR|_max = 2.42
- Ratio: 0.908 ≈ sin(kΔr)/(kΔr) with k_local ≈ 25, Δr = 0.029 Bohr

#### 7.7.4 Why the hybrid is the best choice for AED coupling integrals

The AED coupling matrix element (Section 4) involves:

```
∫ F_{v'}(R) × m(R) × dF_E/dR × dR     (radial coupling)
```

This integral is dominated by the inner Morse well (R ≈ 1.3–2.5 Bohr)
where:
- The bound-state wavefunction F_{v'} is largest,
- The coupling function m(R) peaks,
- The scattering wavefunction oscillates rapidly (k_local ≈ 25 Bohr⁻¹).

The ~9% amplitude error in the derivative from central differences
translates directly into a ~9% error in the radial coupling integral,
which squares to ~17% in the rate. The TISE-based derivative removes
this systematic bias.

### 7.8 Module structure

```
aed_rate/nuclear/
├── _base_solver.py          BoundState, ScatteringState, WavefunctionSolver (ABC)
├── dvr_solver.py            DVRWavefunctionSolver
├── numerov_solver.py        NumerovWavefunctionSolver
├── morse_solver.py          MorseAnalyticSolver + Pekeris parameter computation
└── nuclear_wavefunction.py  create_wavefunction_solver() factory + re-exports
```

The factory function `create_wavefunction_solver(potential, mu, method=...)`
accepts `method = "dvr"`, `"numerov"`, or `"morse"` and returns the
appropriate solver. All solvers implement the same interface:
`solve_bound_state()`, `solve_scattering_state()`, `solve_all_bound_states()`,
`wavefunction_derivative()`.

---

## 8. Thermal Rate Constant

### 8.1 Averaging over collision conditions

The thermal rate constant requires averaging over the Maxwell-Boltzmann
distribution of collision energies and summing over angular momenta [2, Eq. 3-6]:

```
k(T) = Σ_J P(J) × ∫ P(E) × Σ_{v',J'} Rate(v',J'; E,J) dE
```

### 8.2 Angular momentum weighting

```
P(J) dJ = (2J+1) dJ / [J_M (J_M + 1)]     for J ≤ J_M          [2, Eq. 4]
P(J)    = 0                                for J > J_M
```

i.e. P(J) = (2J+1) / [J_M(J_M+1)] — the exact normalization is J_M(J_M+1),
not J_M² (they coincide only for large J_M). Here J_M(E) is the maximum
angular momentum for which the centrifugal barrier on the anion surface does
not exceed E_collision; collisions with J > J_M do not sample the short-R
region where the coupling is significant and are effectively unweighted. The
(2J+1) weighting comes from converting the impact-parameter disk area 2πb db
to angular momentum via ℏ²π(2J+1)dJ/(2μE) [2, Eq. 4].

### 8.3 Energy distribution

```
P(E) = (2/√π) × (k_BT)^{-3/2} × √E × exp(-E/k_BT)            [2, Eq. 6]
```

This is the Maxwell-Boltzmann distribution converted from velocity to
energy variables.

### 8.4 Numerical integration

The energy integral has the form ∫₀^∞ √E × exp(-E/kT) × f(E) dE.
With the substitution x = E/(k_BT), this becomes:

```
(k_BT)^{3/2} × ∫₀^∞ √x × exp(-x) × f(x k_BT) dx
```

This is naturally suited for **generalized Gauss-Laguerre quadrature**
with weight function x^{1/2} exp(-x) (α = 1/2).

### 8.5 Maximum angular momentum J_M

J_M(E) is found by requiring that the centrifugal barrier height equals
the collision energy. For each J, the effective potential
V_eff(R) = V_anion(R) + J(J+1)/(2μR²) has a local maximum (the
centrifugal barrier). J_M is the largest J for which this maximum
does not exceed V_anion(∞) + E_collision.

### 8.6 Units

The thermal rate constant k(T) has units of cm³/s in conventional
notation. Conversion from atomic units:

```
k(T) [cm³/s] = k(T) [a.u.] × a₀³ / t_au
             = k(T) [a.u.] × (0.529177 × 10⁻⁸)³ / (2.4189 × 10⁻¹⁷)
```

---

## 9. Simons (1998) Semiquantum Approximation

### 9.1 Motivation

The full quantum rate expression (Section 3) requires computing nuclear
wavefunctions at every energy. Simons [3] derived a semiclassical
approximation that provides physical insight.

### 9.2 Key result

The total electron ejection rate from an initial state with energy εᵢ is:

```
R_T = 2π × Σ_f F(εᵢ - εf) × dEf × |(P/μ) χᵢ(Q₀)|² × |m*|² × 1/(πV₀)
                                                                 [3, Eq. 42]
```

where:
- F(εᵢ - εf) = density of states of ejected electron at energy εᵢ - εf
- dEf = spacing between neighboring final-state levels
- (P/μ)χᵢ(Q₀) = **derivative** of the initial vibrational wavefunction
  evaluated at Q₀ (not the wavefunction itself!)
- |m*|² = integrated electronic coupling strength
- V₀ = √(2μ(εf - Vf(Q₀)))/μ = classical speed on the neutral surface at Q₀
- Q₀ = geometry where anion-neutral surfaces approach most closely

### 9.3 Physical interpretation

The rate depends on:
1. **|(P/μ)χᵢ(Q₀)|²** — the nuclear momentum density at the coupling region.
   Unlike photon absorption (which depends on |χ(Q₀)|²), non-BO transitions
   require momentum exchange, so the **derivative** enters.
2. **1/V₀** — slower passage through the coupling region gives more time
   for the transition, enhancing the rate.
3. **|m*|²** — the integrated electronic coupling strength.
4. **F × dEf** — the density of final states, favoring transitions where
   more states are available.

### 9.4 Why the simplest classical approximation fails

A naive classical treatment (replacing quantum propagators with classical
trajectories) gives zero rate because the delta function
δ(Vf + Eₑ - Vᵢ) never fires: the anion surface is always below the
neutral, so Vf - Vᵢ > 0 everywhere, and with Eₑ > 0, the argument
is always positive. The semiquantum treatment succeeds because it
retains quantum nuclear kinetic energy, allowing the system to
"tunnel" in the energy-transfer sense.

---

## 10. Selection Rules and Propensities

### 10.1 Symmetry selection rules

The electronic coupling ⟨ψf|∂/∂Q|ψᵢ⟩ is non-zero only if the direct
product of symmetries Γ(ψf) × Γ(∂/∂Q) × Γ(ψᵢ) contains the totally
symmetric representation.

### 10.2 Vibrational propensity rules

Transitions to the energetically closest neutral vibrational level are
generally favored (small Eₑ → large ρ, and favorable overlap). However,
the branching ratios are controlled by quantum mechanical phase factors
in the nuclear overlap integral, not by classical momentum conservation.

### 10.3 Orbital character and coupling strength

- **σ bonding/antibonding HOMOs** (e.g., LiH⁻): strongly modulated by
  bond stretching → large m_rad → vibrational coupling dominates →
  fast detachment (10⁸–10¹¹ s⁻¹).
- **π nonbonding HOMOs** (e.g., OH⁻): weakly affected by stretching →
  small m_rad → rotational coupling becomes important → slow detachment
  (10⁰–10⁵ s⁻¹ for bound states).
- Large electron affinity (OH: 1.83 eV) → large energy gap → small ρ →
  slower rates.

### 10.4 J-dependence

The rate as a function of J has a characteristic structure:

```
Rate(J) ~ |A + B(2J+1)|² × (2J+1)
```

where A = radial contribution (J-independent) and B = rotational
contribution. This produces oscillations: destructive interference
between A and B at intermediate J, with the rate rising again at high J
before the J_M cutoff.
