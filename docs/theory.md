# Theory of Associative Electron Detachment (AED)

## References

- **[1]** Acharya, Kendall, Simons — *J. Am. Chem. Soc.* **106**, 3402 (1984)
  "Vibration-Induced Electron Detachment in Molecular Anions"
- **[2]** Acharya, Das, Simons — *J. Chem. Phys.* **83**, 3888 (1985)
  "Associative Electron Detachment: O⁻ + H → OH + e⁻"
- **[3]** Simons — *J. Phys. Chem. A* **102**, 6035 (1998)
  "Semiquantum Expressions for Electronically Nonadiabatic Electron Ejection Rates"

All equations below are in **atomic units** (ℏ = mₑ = e = a₀ = 1).

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

### 3.2 Density of states

The translational density of states per unit energy for a free electron
in a box of volume L³:

```
ρ(Eₑ) = kₑ / (2π²)     (in atomic units)                       [1, Eq. 3]
```

where kₑ = √(2Eₑ) is the electron's wave vector magnitude.

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

For the process to occur, Eₑ > 0 is required.

---

## 4. The Coupling Matrix Element

### 4.1 Full coupling integral

The non-BO coupling matrix element has the general form [2, Eq. 3]:

```
V = ⟨χ_{v',J'} | ⟨ψf| ∇ |ψi⟩ · (∇/μ) | χ_{E,J}⟩
```

where ∇ is the gradient with respect to nuclear coordinates.

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

**Angular (rotational) coupling** — from (1/R)∂/∂θ acting on electronic
and nuclear wavefunctions:

```
V_rot(J,J') = (1/μ) × C(J,J') × ∫ F_{v',J'}(R) × m_rot(R) × F_{E,J}(R)/R × dR
```

Key differences:
- Radial coupling involves dF_{E,J}/dR (derivative of scattering wavefunction)
  and preserves J (ΔJ = 0)
- Rotational coupling involves F_{E,J}/R (not the derivative) and changes J
  by ±1 (from angular momentum algebra of spherical harmonics)
- C(J,J') are angular coupling coefficients from ⟨Y_{J'M'}|∂/∂θ|Y_{JM}⟩

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

The derivative with respect to R:

```
∂φ_HOMO/∂R = Σⱼ (dCⱼ/dR) gⱼ(r) + Σⱼ Cⱼ(R) (∂gⱼ/∂R)
```

The first term (coefficient derivative) typically dominates. The second
term (basis function derivative) is significant for basis functions
centered on atoms that move with R. In our implementation:

- dCⱼ/dR is computed analytically via PySCF's CPSCF (coupled-perturbed SCF)
  using `hessian.rhf.solve_mo1()`
- The basis function derivative ∂gⱼ/∂R is implicitly included in the
  CPSCF solution through the overlap response terms

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

The continuum electron is described by a plane wave orthogonalized
to the occupied orbitals [1, Eq. 4]:

```
φₖ = exp(ik·r) - Σⱼ ⟨χⱼ|exp(ik·r)⟩ χⱼ
```

where the sum runs over all occupied orbitals χⱼ of the neutral molecule.
The L^{-3/2} normalization factor from the box-normalized plane wave
cancels with L³ from the density of states ρ.

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

We use **box normalization**: ∫|F(R)|² dR = 1 over the computational
domain [R_min, R_max]. The density of states factor ρ in the rate formula
accounts for the continuous spectrum.

### 7.4 Solver methods

- **DVR (Discrete Variable Representation)**: Matrix diagonalization on a
  uniform grid. More robust, gives all eigenvalues simultaneously. Preferred.
- **Numerov**: Shooting method with outward/inward integration and matching.
  Useful for specific states but less robust.

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
P(J) = (2J+1) / J_M²     for J ≤ J_M                           [2, Eq. 4]
P(J) = 0                  for J > J_M
```

where J_M(E) is the maximum angular momentum for which the centrifugal
barrier on the anion surface does not exceed E_collision. This follows
from the classical cross section σ = πb² with b = √(J(J+1))/(√(2μE)).

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
