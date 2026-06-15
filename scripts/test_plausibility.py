#!/usr/bin/env python3
"""
Plausibility checks for all AED rate ingredients.

Tests each module independently against known physical values
for the O⁻ + H → OH + e⁻ system before combining them in the
master rate equation.

Reference values:
- OH bond length:     0.9697 Å = 1.8324 Bohr
- OH D_e:             4.392 eV
- OH ω_e:             3738 cm⁻¹
- OH⁻ D_e:            ~4.5 eV
- OH⁻ ω_e:            ~3700 cm⁻¹
- OH electron affinity: 1.8276 eV
- OH⁻ Koopmans' IP:   should be close to EA (~1.83 eV)
- OH neutral vib levels: ~20-25 bound states
"""

import numpy as np
import sys

# ======================================================================
# Helpers
# ======================================================================

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"


def check(
    name: str,
    value: float,
    expected: float,
    tolerance: float,
    unit: str = "",
) -> str:
    """Check a value against expected, return status string."""
    diff = abs(value - expected)
    rel = diff / abs(expected) if expected != 0 else diff
    status = PASS if rel < tolerance else (WARN if rel < 2 * tolerance else FAIL)
    print(f"  [{status}] {name}: {value:.6f} {unit}  "
          f"(expected {expected:.6f}, rel err {rel:.2%})")
    return status


def section(title: str) -> None:
    """Print section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ======================================================================
# 1. Potential energy curves (potential.py)
# ======================================================================

def test_potentials() -> bool:
    """Test Morse PECs for OH and OH⁻."""
    section("1. Potential Energy Curves")

    from aed_rate.electronic.potential import create_oh_system, MorsePotential
    from aed_rate.utils.constants import CONSTANTS, get_reduced_mass

    anion, neutral, EA = create_oh_system()
    mu = get_reduced_mass("O", "H")

    print(f"\n  Reduced mass μ(OH) = {mu:.2f} a.u. "
          f"({mu / CONSTANTS.amu_to_me:.4f} amu)")

    # 1a. Equilibrium geometries
    print("\n  --- OH⁻ (anion) ---")
    check("r_e (Bohr)", anion.r_eq, 1.83, 0.02)
    check("D_e (eV)", CONSTANTS.hartree_to_ev * (anion.D_e), 4.5, 0.05)

    # Vibrational frequency: ω = β√(2D_e/μ), convert to cm⁻¹
    omega_anion = anion.beta * np.sqrt(2.0 * anion.D_e / mu)
    freq_anion = CONSTANTS.hartree_to_cm1 * (omega_anion)
    check("ω_e (cm⁻¹)", freq_anion, 3700, 0.05, "cm⁻¹")

    print("\n  --- OH (neutral) ---")
    check("r_e (Bohr)", neutral.r_eq, 1.83, 0.02)
    check("D_e (eV)", CONSTANTS.hartree_to_ev * (neutral.D_e), 4.4, 0.05)

    omega_neutral = neutral.beta * np.sqrt(2.0 * neutral.D_e / mu)
    freq_neutral = CONSTANTS.hartree_to_cm1 * (omega_neutral)
    check("ω_e (cm⁻¹)", freq_neutral, 3738, 0.05, "cm⁻¹")

    # 1b. Electron affinity
    check("EA (eV)", CONSTANTS.hartree_to_ev * (EA), 1.8276, 0.01)

    # 1c. Anion curve below neutral at all R
    R_test = np.linspace(1.0, 8.0, 100)
    V_anion = anion(R_test)
    V_neutral = neutral(R_test)
    all_below = np.all(V_anion < V_neutral)
    status = PASS if all_below else FAIL
    print(f"\n  [{status}] Anion curve below neutral at all R: {all_below}")

    # 1d. Morse vibrational energies (analytical)
    vib_neutral = neutral.vibrational_energies(mu)
    n_bound = len(vib_neutral)
    status = PASS if 15 < n_bound < 30 else WARN
    print(f"  [{status}] Neutral OH bound states: {n_bound} (expected ~20-25)")

    if n_bound >= 2:
        # Fundamental frequency: E(v=1) - E(v=0)
        fundamental = CONSTANTS.hartree_to_cm1 * (vib_neutral[1] - vib_neutral[0])
        check("Fundamental ν₀₁ (cm⁻¹)", fundamental, 3570, 0.05, "cm⁻¹")

    # 1e. Centrifugal barrier
    r_bar, V_bar = anion.find_barrier_height(J=10, reduced_mass=mu)
    r_bar = float(np.asarray(r_bar).flat[0])
    V_bar = float(np.asarray(V_bar).flat[0])
    barrier_height_cm1 = CONSTANTS.hartree_to_cm1 * (V_bar - anion.dissociation_energy)
    status = PASS if barrier_height_cm1 > 0 else FAIL
    print(f"  [{status}] J=10 barrier height: {barrier_height_cm1:.1f} cm⁻¹ "
          f"at R={r_bar:.2f} Bohr")

    return True


# ======================================================================
# 2. Nuclear wavefunctions (nuclear_wavefunction.py)
# ======================================================================

def test_nuclear_wavefunctions() -> bool:
    """Test DVR bound and scattering states."""
    section("2. Nuclear Wavefunctions (DVR)")

    from aed_rate.electronic.potential import create_oh_system
    from aed_rate.nuclear.nuclear_wavefunction import create_wavefunction_solver
    from aed_rate.utils.constants import CONSTANTS, get_reduced_mass

    anion, neutral, EA = create_oh_system()
    mu = get_reduced_mass("O", "H")

    # Use DVR with fine grid
    solver_neutral = create_wavefunction_solver(
        neutral, mu, method="dvr", r_min=0.5, r_max=15.0, n_grid=500,
    )
    solver_anion = create_wavefunction_solver(
        anion, mu, method="dvr", r_min=0.5, r_max=15.0, n_grid=500,
    )

    # 2a. Neutral bound states
    print("\n  --- Neutral OH bound states ---")
    states = solver_neutral.solve_all_bound_states(J=0)
    n_bound = len(states)
    status = PASS if 15 < n_bound < 30 else WARN
    print(f"  [{status}] Number of bound states (DVR): {n_bound}")

    if n_bound >= 2:
        fundamental = CONSTANTS.hartree_to_cm1 * (
            states[1].energy - states[0].energy
        )
        check("DVR fundamental ν₀₁ (cm⁻¹)", fundamental, 3570, 0.10, "cm⁻¹")

    # Print first few vibrational energies
    print("\n  Vibrational energies (relative to minimum, cm⁻¹):")
    for s in states[:6]:
        E_rel = CONSTANTS.hartree_to_cm1 * (s.energy - neutral.V_0)
        print(f"    v={s.v}: {E_rel:.1f} cm⁻¹")

    # 2b. Normalization check
    print("\n  --- Normalization checks ---")
    from scipy.integrate import simpson
    for v in [0, 3]:
        if v < n_bound:
            state = states[v]
            norm = simpson(state.wavefunction**2, x=state.r_grid)
            status = PASS if abs(norm - 1.0) < 0.01 else FAIL
            print(f"  [{status}] ∫|F_{{v={v}}}|² dR = {norm:.6f}")

    # 2c. Node counting
    print("\n  --- Node counting ---")
    for v in [0, 1, 2, 5]:
        if v < n_bound:
            state = states[v]
            # Count zero crossings (ignoring edges)
            wf = state.wavefunction[10:-10]  # trim edges
            nodes = np.sum(np.diff(np.sign(wf[wf != 0])) != 0)
            status = PASS if nodes == v else WARN
            print(f"  [{status}] v={v}: {nodes} nodes (expected {v})")

    # 2d. Scattering state
    print("\n  --- Anion scattering state ---")
    E_coll = CONSTANTS.cm1_to_hartree(500)  # 500 cm⁻¹ collision energy
    scat = solver_anion.solve_scattering_state(E_coll, J=0)
    norm_scat = simpson(scat.wavefunction**2, x=scat.r_grid)
    status = PASS if abs(norm_scat - 1.0) < 0.05 else WARN
    print(f"  [{status}] Scattering state norm: {norm_scat:.6f}")
    print(f"  Phase shift δ = {scat.phase_shift:.4f} rad")

    # Check scattering wf oscillates at large R
    wf_tail = scat.wavefunction[-200:-50]
    n_oscillations = np.sum(np.diff(np.sign(wf_tail)) != 0) // 2
    status = PASS if n_oscillations >= 3 else WARN
    print(f"  [{status}] Oscillations in asymptotic region: {n_oscillations}")

    # 2e. Wavefunction derivative
    print("\n  --- Wavefunction derivative ---")
    state_v0 = states[0]
    deriv = solver_neutral.wavefunction_derivative(state_v0)
    # Derivative should be zero near peak (turning point)
    peak_idx = np.argmax(np.abs(state_v0.wavefunction))
    status = PASS if abs(deriv[peak_idx]) < 0.5 * np.max(np.abs(deriv)) else WARN
    print(f"  [{status}] dF/dR near peak is small: {abs(deriv[peak_idx]):.4f} "
          f"vs max {np.max(np.abs(deriv)):.4f}")

    return True


# ======================================================================
# 3. Continuum orbital (continuum.py)
# ======================================================================

def test_continuum() -> bool:
    """Test OPW and density of states."""
    section("3. Continuum Orbital (OPW)")

    from aed_rate.electronic.continuum import (
        ContinuumOrbital, compute_electron_kinetic_energy,
    )
    from aed_rate.utils.constants import CONSTANTS

    # 3a. Density of states: ρ = k/(2π²)
    print("\n  --- Density of states ---")
    for E_eV in [0.1, 0.5, 1.0, 2.0]:
        E_ha = CONSTANTS.ev_to_hartree(E_eV)
        opw = ContinuumOrbital(kinetic_energy=E_ha)
        rho = opw.density_of_states()
        k = np.sqrt(2.0 * E_ha)
        rho_expected = k / (2.0 * np.pi**2)
        status = PASS if abs(rho - rho_expected) < 1e-10 else FAIL
        print(f"  [{status}] E={E_eV:.1f} eV: ρ={rho:.6f} "
              f"(k={k:.4f}, λ={opw.wavelength:.2f} Bohr)")

    # 3b. Energy conservation
    print("\n  --- Energy conservation ---")
    EA = CONSTANTS.ev_to_hartree(1.8276)
    E_coll = CONSTANTS.cm1_to_hartree(500)  # 500 cm⁻¹

    # For v'=0, approximate neutral vib energy ~ 0.008 Hartree
    # Anion dissoc energy ~ 4.5 eV
    D_anion = CONSTANTS.ev_to_hartree(4.5)
    E_v0_neutral = CONSTANTS.cm1_to_hartree(1850)  # ~half of ω_e

    E_electron = compute_electron_kinetic_energy(
        collision_energy=E_coll,
        anion_vib_energy=D_anion,
        neutral_vib_energy=E_v0_neutral,
        electron_affinity=EA,
    )
    E_electron_eV = CONSTANTS.hartree_to_ev * (E_electron)
    status = PASS if E_electron > 0 else FAIL
    print(f"  [{status}] E_electron = {E_electron_eV:.4f} eV "
          f"(should be > 0 for allowed transition)")

    # For very high v', electron energy should become negative (forbidden)
    E_v_high = CONSTANTS.cm1_to_hartree(30000)  # above D_e
    E_electron_high = compute_electron_kinetic_energy(
        collision_energy=E_coll,
        anion_vib_energy=D_anion,
        neutral_vib_energy=E_v_high,
        electron_affinity=EA,
    )
    status = PASS if E_electron_high == 0.0 else FAIL
    print(f"  [{status}] High v' gives E_electron = {E_electron_high:.6f} "
          f"(should be 0, forbidden)")

    # 3c. Low-k regime check
    print("\n  --- Low-k OPW check ---")
    E_low = CONSTANTS.cm1_to_hartree(100)  # very low energy
    opw_low = ContinuumOrbital(kinetic_energy=E_low)
    k_low = opw_low.k
    r_e = 1.83
    kr_product = k_low * r_e
    status = PASS if kr_product < 0.5 else WARN
    print(f"  [{status}] k*r_e = {kr_product:.4f} at 100 cm⁻¹ "
          f"(low-k valid if < 0.5)")

    E_high = CONSTANTS.ev_to_hartree(2.0)
    opw_high = ContinuumOrbital(kinetic_energy=E_high)
    kr_high = opw_high.k * r_e
    status = PASS if kr_high > 0.5 else WARN
    print(f"  [{status}] k*r_e = {kr_high:.4f} at 2.0 eV "
          f"(full OPW needed if > 0.5)")

    return True


# ======================================================================
# 4. Electronic coupling (coupling.py) — model only
# ======================================================================

def test_coupling_model() -> bool:
    """Test model coupling (no PySCF needed)."""
    section("4. Electronic Coupling (Model)")

    from aed_rate.electronic.coupling import ModelCoupling
    from aed_rate.utils.constants import CONSTANTS

    # OH⁻ parameters
    R0 = 1.83  # Bohr

    # For OH⁻ with π HOMO:
    # m_rot should be larger than m_rad (π orbital is weakly
    # modulated by stretch but strongly mixed by rotation)
    model = ModelCoupling(
        R0=R0,
        A_rad=0.01,   # small radial (π HOMO)
        alpha_rad=1.0,
        A_rot=0.05,    # larger rotational
        alpha_rot=1.0,
    )

    E_electron = CONSTANTS.cm1_to_hartree(5000)  # ~0.6 eV
    result = model.compute_coupling_at_r(R0, E_electron)

    print(f"\n  At R = R₀ = {R0:.2f} Bohr:")
    print(f"  |m_rad| = {abs(result.m_rad):.6f}")
    print(f"  |m_rot| = {abs(result.m_rot):.6f}")

    # 4a. m_rot > m_rad for π HOMO
    status = PASS if abs(result.m_rot) > abs(result.m_rad) else WARN
    print(f"  [{status}] |m_rot| > |m_rad| (expected for π HOMO)")

    # 4b. Coupling decays at large R
    result_far = model.compute_coupling_at_r(5.0, E_electron)
    status = PASS if abs(result_far.m_rad) < 0.1 * abs(result.m_rad) else FAIL
    print(f"  [{status}] Coupling decays at large R: "
          f"|m_rad(5.0)|/|m_rad(R₀)| = {abs(result_far.m_rad)/abs(result.m_rad):.4f}")

    # 4c. Coupling curve shape
    print("\n  --- Coupling curve ---")
    R_grid = np.linspace(1.0, 5.0, 9)
    results = model.compute_coupling_curve(R_grid, E_electron)
    print(f"  {'R (Bohr)':>10s}  {'|m_rad|':>10s}  {'|m_rot|':>10s}")
    for r in results:
        print(f"  {r.R:10.2f}  {abs(r.m_rad):10.6f}  {abs(r.m_rot):10.6f}")

    # Peak should be at R0
    m_rad_vals = [abs(r.m_rad) for r in results]
    peak_idx = np.argmax(m_rad_vals)
    peak_R = results[peak_idx].R
    status = PASS if abs(peak_R - R0) < 0.6 else FAIL
    print(f"\n  [{status}] Coupling peaks near R₀: peak at R={peak_R:.2f}")

    return True


# ======================================================================
# 5. Electronic structure (wavefunctions.py) — requires PySCF
# ======================================================================

def test_electronic_structure() -> bool:
    """Test SCF calculations (requires PySCF)."""
    section("5. Electronic Structure (PySCF)")

    try:
        from aed_rate.electronic.wavefunctions import ElectronicStructure
    except ImportError:
        print("  [SKIP] PySCF not available")
        return True

    from aed_rate.utils.constants import CONSTANTS

    es = ElectronicStructure("O", "H", basis="aug-cc-pVDZ")

    # 5a. Anion SCF at equilibrium
    print("\n  --- OH⁻ anion (RHF, aug-cc-pVDZ) ---")
    anion_mo = es.compute_anion(r=1.83, spin=0)
    print(f"  Total energy: {anion_mo.total_energy:.6f} Hartree")

    # HOMO energy → Koopmans' IP
    homo_idx = np.where(anion_mo.mo_occ > 0.5)[0][-1]
    homo_energy = anion_mo.mo_energy[homo_idx]
    ip_koopmans = -homo_energy
    ip_eV = CONSTANTS.hartree_to_ev * (ip_koopmans)
    # Koopmans' IP at HF level overestimates EA significantly
    # (missing orbital relaxation + correlation). ~2-4 eV is typical.
    check("Koopmans' IP (eV)", ip_eV, 1.83, 1.0)
    print(f"  HOMO index: {homo_idx}, HOMO energy: {homo_energy:.6f} Hartree")

    # 5b. Neutral SCF
    print("\n  --- OH neutral (UHF, aug-cc-pVDZ) ---")
    neutral_mo = es.compute_neutral(r=1.83, spin=1)
    print(f"  Total energy: {neutral_mo.total_energy:.6f} Hartree")

    # Energy gap: E(neutral) - E(anion) should be positive (anion is more stable)
    # However at HF level with small basis, UHF neutral can artifactually
    # come out lower than RHF anion — this is a known deficiency of HF
    # for electron affinities of open-shell systems.
    delta_E = neutral_mo.total_energy - anion_mo.total_energy
    delta_E_eV = CONSTANTS.hartree_to_ev * (delta_E)
    status = PASS if delta_E > 0 else WARN
    print(f"  [{status}] E(neutral) - E(anion) = {delta_E_eV:.4f} eV "
          f"(should be > 0 ideally, HF often gets this wrong)")
    print(f"  Note: ΔSCF EA at HF/aug-cc-pVDZ is unreliable — "
          f"need correlated methods for quantitative EA")

    # 5c. PEC along R
    print("\n  --- PEC scan ---")
    R_points = np.array([1.5, 1.83, 2.2, 3.0])
    for R in R_points:
        anion_data = es.compute_anion(R, spin=0)
        neutral_data = es.compute_neutral(R, spin=1)
        gap = neutral_data.total_energy - anion_data.total_energy
        print(f"  R={R:.2f}: E_anion={anion_data.total_energy:.6f}, "
              f"gap={CONSTANTS.hartree_to_ev * (gap):.3f} eV")

    return True


# ======================================================================
# 6. Ab initio coupling (coupling.py) — requires PySCF
# ======================================================================

def test_coupling_ab_initio() -> bool:
    """Test CPSCF-based coupling (requires PySCF)."""
    section("6. Ab Initio Coupling (CPSCF)")

    try:
        from aed_rate.electronic.wavefunctions import ElectronicStructure
        from aed_rate.electronic.coupling import ElectronicCoupling
    except ImportError:
        print("  [SKIP] PySCF not available")
        return True

    from aed_rate.utils.constants import CONSTANTS

    # Use small basis for speed
    es = ElectronicStructure("O", "H", basis="aug-cc-pVDZ")
    coupling = ElectronicCoupling(
        es, homo_symmetry="pi", grid_level=3,
    )

    # 6a. Coupling at equilibrium
    print("\n  --- Coupling at R = 1.83 Bohr ---")
    E_electron = CONSTANTS.cm1_to_hartree(5000)  # ~0.6 eV
    result = coupling.compute_coupling_at_r(1.83, E_electron)

    print(f"  m_rad = {result.m_rad:.6e}")
    print(f"  m_rot = {result.m_rot:.6e}")
    print(f"  |m_rad| = {abs(result.m_rad):.6e}")
    print(f"  |m_rot| = {abs(result.m_rot):.6e}")
    print(f"  k_e = {result.k_electron:.4f}")

    # For OH⁻ π HOMO, we expect |m_rot| >= |m_rad|
    ratio = abs(result.m_rot) / abs(result.m_rad) if abs(result.m_rad) > 1e-15 else float('inf')
    print(f"  |m_rot|/|m_rad| = {ratio:.2f}")

    # Both should be non-zero
    status = PASS if abs(result.m_rad) > 1e-10 else FAIL
    print(f"  [{status}] m_rad is non-zero")
    status = PASS if abs(result.m_rot) > 1e-10 else FAIL
    print(f"  [{status}] m_rot is non-zero")

    # 6b. Coupling vs R (should peak near equilibrium, decay at large R)
    print("\n  --- Coupling curve ---")
    R_grid = np.array([1.4, 1.6, 1.83, 2.1, 2.5, 3.0, 4.0])
    results = coupling.compute_coupling_curve(R_grid, E_electron)

    print(f"  {'R':>6s}  {'|m_rad|':>12s}  {'|m_rot|':>12s}  {'ratio':>8s}")
    for r in results:
        rat = abs(r.m_rot) / abs(r.m_rad) if abs(r.m_rad) > 1e-15 else float('inf')
        print(f"  {r.R:6.2f}  {abs(r.m_rad):12.6e}  {abs(r.m_rot):12.6e}  {rat:8.2f}")

    # Coupling at R=4 should be much smaller than at R=1.83
    m_rad_eq = abs([r for r in results if r.R == 1.83][0].m_rad)
    m_rad_far = abs([r for r in results if r.R == 4.0][0].m_rad)
    if m_rad_eq > 1e-15:
        decay = m_rad_far / m_rad_eq
        status = PASS if decay < 0.5 else WARN
        print(f"\n  [{status}] Radial coupling decay R=4/R=1.83: {decay:.4f}")

    return True


# ======================================================================
# 7. Electronic coupling convergence vs basis set (uses precomputed NPZ)
# ======================================================================

# Maps a short label to the NPZ file produced by precompute_coupling.py.
# The 6-31G entry now points to the extended grid (R up to 10 Bohr).
_COUPLING_FILES = {
    "6-31G (R≤10)":    "oh_minus_coupling_6-31g.npz",
    "aug-cc-pVDZ":     "oh_minus_coupling_aug-cc-pvdz.npz",
    "aug-cc-pVDZ+OPW": "oh_minus_coupling_aug-cc-pvdz_opw.npz",
}


def _load_npz_coupling(path: str) -> dict:
    """Load a 2D (R, k_e) coupling NPZ and return a plain dict."""
    import os
    if not os.path.exists(path):
        return {}
    d = np.load(path)
    return {
        "R":     d["R_grid"],
        "k_e":   d["k_e_grid"],
        "m_rad": d["m_rad_2d"],   # shape (n_R, n_ke)
        "m_rot": d["m_rot_2d"],
    }


def test_coupling_basis_convergence() -> bool:
    """
    Check basis-set convergence of the electronic coupling.

    For each available NPZ, prints:
    - Sign-change position of m_rad and m_rot vs R
    - m_rad / k_e at R_e (should be ~constant across k_e — low-k scaling)
    - Whether m_rad decays at large R (physical) or plateaus (orbital-reordering artefact)
    """
    section("7. Electronic Coupling — Basis Convergence")

    available = {
        label: _load_npz_coupling(path)
        for label, path in _COUPLING_FILES.items()
    }
    available = {k: v for k, v in available.items() if v}

    if not available:
        print("  [SKIP] No precomputed NPZ files found. "
              "Run scripts/precompute_coupling.py first.")
        return True

    print(f"\n  {'Basis':>18s}  {'R_c(m_rad)':>12s}  {'R_c(m_rot)':>12s}  "
          f"{'m_rad/k_e @ R_e':>16s}  {'decays?':>8s}")
    print("  " + "-" * 75)

    R_e_anion = 1.822  # Bohr (Acharya OH⁻ equilibrium)

    for label, d in available.items():
        R, k_e, m_rad, m_rot = d["R"], d["k_e"], d["m_rad"], d["m_rot"]
        mid_k = len(k_e) // 2  # representative k_e column

        # Sign-change position in m_rad
        m_r = m_rad[:, mid_k]
        sign_changes_r = [
            (R[i] + R[i+1]) / 2
            for i in range(len(R) - 1)
            if m_r[i] * m_r[i+1] < 0
        ]
        Rc_rad = f"{sign_changes_r[0]:.3f}" if sign_changes_r else "none"

        # Sign-change position in m_rot
        m_ro = m_rot[:, mid_k]
        sign_changes_ro = [
            (R[i] + R[i+1]) / 2
            for i in range(len(R) - 1)
            if m_ro[i] * m_ro[i+1] < 0
        ]
        Rc_rot = f"{sign_changes_ro[0]:.3f}" if sign_changes_ro else "none"

        # m_rad / k_e at R closest to R_e — should be approximately k_e-independent
        i_Re = int(np.argmin(np.abs(R - R_e_anion)))
        ratio_at_Re = m_rad[i_Re, :] / k_e  # shape (n_ke,)
        ratio_str = f"{ratio_at_Re[mid_k]:.4f}"

        # Decay check: meaningful only if grid extends well beyond 6 Bohr.
        # For short grids we simply report that we cannot assess it.
        # Reference amplitude: max of |m_rad| in R ∈ [2.0, 4.0] Bohr — past
        # the sign change and before any large-R artefact.
        if R[-1] < 7.0:
            decays = f"NEED MORE R (R_max={R[-1]:.1f})"
        else:
            mask_ref = (R >= 2.0) & (R <= 4.0)
            ref_abs  = np.max(np.abs(m_r[mask_ref])) if mask_ref.any() else 1e-30
            last_abs = np.abs(m_r[-1])
            decay_ratio = last_abs / (ref_abs + 1e-30)
            if decay_ratio < 0.05:
                decays = f"YES ({decay_ratio:.3f} of peak)"
            elif decay_ratio < 0.3:
                decays = f"PARTIAL ({decay_ratio:.3f})"
            else:
                decays = f"NO/PLATEAU ({decay_ratio:.3f})"

        print(f"  {label:>18s}  {Rc_rad:>12s}  {Rc_rot:>12s}  "
              f"{ratio_str:>16s}  {decays:>8s}")

        # Detailed k_e scaling check
        cv = np.std(ratio_at_Re) / (np.abs(np.mean(ratio_at_Re)) + 1e-30)
        status = PASS if cv < 0.10 else (WARN if cv < 0.30 else FAIL)
        print(f"  [{status}]   m_rad/k_e coefficient of variation at R_e: {cv:.3f} "
              f"(expect < 0.10 for pure k_e^1 scaling)")

    return True


# ======================================================================
# 8. Orbital-tracking health check (MOC overlap along R)
# ======================================================================

def test_orbital_tracking() -> bool:
    """
    Qualitative orbital-tracking diagnostic for the HOMO across geometries.

    Physical background
    -------------------
    m_rad = <HOMO|∂/∂R|OPW>  preserves orbital symmetry under bond-stretch:
    the ∂/∂R of a π HOMO remains π-like, so m_rad is robust to HOMO-index
    changes.  m_rot = <HOMO|∂/∂θ|OPW> mixes σ and π character (∂/∂θ rotates
    the orbital frame), so it is highly sensitive to which specific orbital is
    tracked.  Orbital reordering therefore shows up in m_rot long before it
    shows up in m_rad.

    Generates a derivative diagnostic plot (dm/dR vs R): abrupt orbital
    reassignments appear as large isolated spikes compared to the smooth
    background.  Inspect the saved PNG rather than trusting hard-coded
    thresholds.
    """
    section("8. Orbital Tracking — Derivative Diagnostic")

    available = {
        label: _load_npz_coupling(path)
        for label, path in _COUPLING_FILES.items()
    }
    available = {k: v for k, v in available.items() if v}

    if not available:
        print("  [SKIP] No NPZ files found.")
        return True

    for label, d in available.items():
        R = d["R"]
        print(f"\n  {label}: R ∈ [{R[0]:.2f}, {R[-1]:.2f}] Bohr, "
              f"{len(R)} points")

    print("\n  → Derivative diagnostic plot generated in plots/orbital_tracking.png")
    print("    Inspect the plot: spikes in dm/dR mark orbital-reordering events.")
    print("    Physical sign-change gradients are smooth; artefacts are sharp spikes.")

    return True


# ======================================================================
# Plot helpers called from generate_plots()
# ======================================================================

def _plot_coupling_vs_r(save_dir: str, plt) -> None:
    """
    Plot m_rad(R) and m_rot(R)/k_e for each available basis at a mid-range k_e.

    Two panels:
      Left  — raw m_rad(R): shows sign change and amplitude vs basis
      Right — m_rad(R)/k_e: removes k_e scaling to isolate the geometric factor;
              curves should overlap if k_e^1 is the correct scaling
    """
    import os

    available = {
        label: _load_npz_coupling(path)
        for label, path in _COUPLING_FILES.items()
    }
    available = {k: v for k, v in available.items() if v}
    if not available:
        print("  [SKIP] No NPZ files — skipping coupling vs R plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.flatten()
    titles = ["m_rad(R) — raw", "m_rad(R) / k_e — geometric factor",
              "m_rot(R) — raw", "m_rot(R) / k_e — geometric factor"]
    for ax, t in zip(axes, titles):
        ax.set_title(t)
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.axvline(1.822, color="gray", lw=0.7, ls=":", label="R_e(OH⁻)")

    colors = ["C0", "C1", "C2", "C3"]
    for (label, d), color in zip(available.items(), colors):
        R, k_e, m_rad, m_rot = d["R"], d["k_e"], d["m_rad"], d["m_rot"]
        mid_k = len(k_e) // 2
        k_mid  = k_e[mid_k]

        axes[0].plot(R, m_rad[:, mid_k], color=color, label=f"{label} (k_e={k_mid:.2f})")
        axes[1].plot(R, m_rad[:, mid_k] / k_mid, color=color, label=label)
        axes[2].plot(R, m_rot[:, mid_k], color=color, label=f"{label} (k_e={k_mid:.2f})")
        axes[3].plot(R, m_rot[:, mid_k] / k_mid, color=color, label=label)

        # Also overlay two k_e extremes (dashed) to visualise k_e dependence
        for k_idx, ls in [(0, ":"), (-1, "--")]:
            k_val = k_e[k_idx]
            axes[0].plot(R, m_rad[:, k_idx], color=color, ls=ls, alpha=0.4)
            axes[1].plot(R, m_rad[:, k_idx] / k_val, color=color, ls=ls, alpha=0.4)
            axes[2].plot(R, m_rot[:, k_idx], color=color, ls=ls, alpha=0.4)
            axes[3].plot(R, m_rot[:, k_idx] / k_val, color=color, ls=ls, alpha=0.4)

    for ax in axes:
        ax.set_xlabel("R (Bohr)")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("a.u.")
    axes[2].set_ylabel("a.u.")

    fig.suptitle("Electronic coupling vs R\n"
                 "solid = mid k_e, dotted/dashed = low/high k_e extremes")
    fig.tight_layout()
    path = os.path.join(save_dir, "coupling_vs_R_basis.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_coupling_integrand(
    save_dir: str, plt,
    anion_pot, neutral_pot, EA: float, mu: float,
) -> None:
    """
    Plot the radial coupling integrand  F_v'(R) × m_rad(R) × dF_E/dR  for v'=4..8.

    Uses the best available NPZ (aug-cc-pVDZ+OPW > aug-cc-pVDZ > 6-31G).
    Also shows the cumulative integral (running sum) to reveal where cancellation
    occurs and which R region dominates the final coupling value.

    Two rows of panels:
      Top    — integrand shape for each v'
      Bottom — cumulative integral ∫₀ᴿ integrand dR′
    """
    import os
    from scipy.integrate import cumulative_trapezoid
    from aed_rate.electronic.potential import create_oh_system_acharya
    from aed_rate.nuclear.morse_solver import MorseAnalyticSolver
    from aed_rate.utils.constants import CONSTANTS
    from scipy.interpolate import RectBivariateSpline

    # Pick best available NPZ
    npz_priority = [
        "oh_minus_coupling_aug-cc-pvdz_opw.npz",
        "oh_minus_coupling_aug-cc-pvdz.npz",
        "oh_minus_coupling_6-31g.npz",
    ]
    npz_path = None
    for p in npz_priority:
        import os as _os
        if _os.path.exists(p):
            npz_path = p
            break
    if npz_path is None:
        print("  [SKIP] No NPZ file for integrand plot")
        return

    d = np.load(npz_path)
    R_npz = d["R_grid"]
    k_e_npz = d["k_e_grid"]
    m_rad_2d = d["m_rad_2d"]
    spl = RectBivariateSpline(R_npz, k_e_npz, m_rad_2d, kx=3, ky=3)

    # Nuclear wavefunctions on a fine grid
    anion_p, neutral_p, EA_ach = create_oh_system_acharya()
    solver_anion   = MorseAnalyticSolver(anion_p,   mu, r_min=0.5, r_max=15.0, n_grid=1000)
    solver_neutral = MorseAnalyticSolver(neutral_p, mu, r_min=0.5, r_max=15.0, n_grid=1000)

    E_coll = CONSTANTS.cm1_to_hartree(66.0)
    scat   = solver_anion.solve_scattering_state(E_coll, J=0)
    dFdR   = solver_anion.wavefunction_derivative(scat)
    R_grid = scat.r_grid

    v_list   = [4, 5, 6, 7, 8]
    colors   = ["C4", "C0", "C2", "C1", "C3"]
    # Acharya Table I reference for annotation
    ref_rates = {8: 0.113, 7: 68.8, 6: 125.0, 5: 61.1, 4: 12.5}

    fig, axes = plt.subplots(2, len(v_list), figsize=(14, 7), sharex=True)

    for col, (v, color) in enumerate(zip(v_list, colors)):
        try:
            bs = solver_neutral.solve_bound_state(v, J=0)
        except ValueError:
            continue

        # Electron energy for this v'
        E_vib = bs.energy - neutral_p.V_0
        E_e = anion_p.D_e + E_coll - E_vib - EA_ach
        if E_e <= 0:
            axes[0, col].set_title(f"v'={v}\n(forbidden)")
            continue
        k_e = float(np.sqrt(2.0 * E_e))

        # m_rad interpolated at the correct k_e on the nuclear grid
        # zero outside the NPZ R range (no extrapolation)
        m_rad_on_grid = np.zeros(len(R_grid))
        mask = (R_grid >= R_npz[0]) & (R_grid <= R_npz[-1])
        k_e_clamped = float(np.clip(k_e, k_e_npz[0], k_e_npz[-1]))
        m_rad_on_grid[mask] = spl(R_grid[mask], k_e_clamped, grid=False)

        integrand = bs.wavefunction * m_rad_on_grid * dFdR
        cumint    = cumulative_trapezoid(integrand, x=R_grid, initial=0.0)

        # Restrict plot to the region with significant coupling (R_npz range + a bit)
        R_plot_max = min(R_npz[-1] + 0.5, 5.0)
        mask_plot  = R_grid <= R_plot_max

        ref = ref_rates.get(v, "?")
        axes[0, col].plot(R_grid[mask_plot], integrand[mask_plot], color=color, lw=1)
        axes[0, col].axhline(0, color="k", lw=0.5, ls="--")
        axes[0, col].axvline(1.822, color="gray", lw=0.7, ls=":")
        axes[0, col].set_title(f"v'={v}  (ref: {ref} s⁻¹)")

        axes[1, col].plot(R_grid[mask_plot], cumint[mask_plot], color=color, lw=1)
        axes[1, col].axhline(0, color="k", lw=0.5, ls="--")
        axes[1, col].axvline(1.822, color="gray", lw=0.7, ls=":")

        # Mark the final integral value
        final_val = cumint[mask_plot][-1]
        axes[1, col].annotate(f"  I={final_val:.2e}", xy=(R_plot_max, final_val),
                              fontsize=7, va="center")

    axes[0, 0].set_ylabel("F_v' × m_rad × dF_E/dR  (a.u.)")
    axes[1, 0].set_ylabel("Cumulative integral  (a.u.)")
    for ax in axes[1]:
        ax.set_xlabel("R (Bohr)")

    fig.suptitle(
        f"Radial coupling integrand & cumulative sum\n"
        f"Coupling from: {os.path.basename(npz_path)}  |  "
        f"E_coll=66 cm⁻¹, J=0"
    )
    fig.tight_layout()
    path = os.path.join(save_dir, "coupling_integrand.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_homo_tracking(save_dir: str, plt) -> None:
    """
    Orbital correlation diagram from a fast SCF-only HOMO tracking scan.

    Two rows × 2 columns (one column per unique electronic-structure basis):
      Row 1 — Orbital energy correlation diagram.
               Each occupied MO is plotted as small coloured dots at each R,
               coloured by symmetry irrep.  The MOC-tracked HOMO is drawn as
               a thick connected line so that index jumps (orbital reordering)
               are immediately visible as the thick line "jumping" from one dot
               cluster to another.  The first virtual MO is shown dashed.
      Row 2 — HOMO MO index (step function, left axis) with the MOC overlap
               quality overlaid on a twin right axis.  A drop in overlap
               coincides exactly with the transition region in row 1.

    Physical interpretation guide
    ------------------------------
    - An index jump with overlap ≈ 1: the SCF reordered two MOs in energy
      but the physical wavefunction is continuous — MOC handled it correctly.
    - A sudden overlap drop (< 0.8): a genuine ambiguity between two orbitals
      of the same symmetry; the coupling near that R may be unreliable.
    """
    import os
    from aed_rate.electronic.coupling import scan_homo_tracking

    # Map each unique electronic-structure basis to a scan R grid.
    # aug-cc-pVDZ and aug-cc-pVDZ+OPW differ only in OPW treatment,
    # not in the SCF, so we scan only once.
    _SCAN_PARAMS = {
        "6-31G (R≤10)": ("6-31g",      np.linspace(0.8, 10.0, 60)),
        "aug-cc-pVDZ":  ("aug-cc-pvdz", np.linspace(0.8,  4.4, 40)),
    }

    # Skip bases whose NPZ file doesn't exist (no point scanning if not used)
    params_to_run = {
        label: params
        for label, params in _SCAN_PARAMS.items()
        if os.path.exists(_COUPLING_FILES.get(label, ""))
           or label == "6-31G (R≤10)"  # always attempt if file exists
    }
    params_to_run = {
        label: params
        for label, params in _SCAN_PARAMS.items()
    }

    # Colour scheme for symmetry irrep labels
    IRREP_COLOR = {
        "A1": "#4878CF", "a1": "#4878CF",
        "A2": "#6ACC65", "a2": "#6ACC65",
        "E1x": "#D65F5F", "e1x": "#D65F5F", "B1": "#D65F5F", "b1": "#D65F5F",
        "E1y": "#B47CC7", "e1y": "#B47CC7", "B2": "#B47CC7", "b2": "#B47CC7",
        "E2x": "#C4AD66", "e2x": "#C4AD66",
        "E2y": "#77BEDB", "e2y": "#77BEDB",
    }
    default_color = "#AAAAAA"

    n_cols = len(params_to_run)
    fig, axes = plt.subplots(2, n_cols, figsize=(6 * n_cols, 9))
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    for col, (label, (basis, R_scan)) in enumerate(params_to_run.items()):
        print(f"  Running HOMO tracking scan: {label}  "
              f"({len(R_scan)} points, basis={basis}) ...", flush=True)
        try:
            log = scan_homo_tracking(basis, R_scan, n_virt=2, verbose=False)
        except Exception as exc:
            print(f"    [SKIP] scan failed: {exc}")
            continue

        R          = log["R"]
        occ_e      = log["occ_energies"]       # (n_R, n_occ)
        occ_irr    = log["occ_irreps"]          # list[list[str]]
        virt_e     = log["virt_energies"]       # (n_R, n_virt)
        homo_e     = log["homo_energy"]
        homo_idx   = log["homo_idx"]
        overlap    = log["moc_overlap"]
        overlap2nd = log["moc_overlap_2nd"]

        ax_corr   = axes[0, col]
        ax_idx    = axes[1, col]
        ax_ovlp   = ax_idx.twinx()

        # ---- Row 1: orbital correlation diagram ----
        n_occ = occ_e.shape[1]
        legend_irreps: set = set()
        for i, Ri in enumerate(R):
            for j in range(n_occ):
                e = occ_e[i, j]
                if np.isnan(e):
                    continue
                irr = occ_irr[i][j] if j < len(occ_irr[i]) else "?"
                color = IRREP_COLOR.get(irr, default_color)
                ax_corr.scatter(Ri, e, color=color, s=8, zorder=2,
                                label=irr if irr not in legend_irreps else "")
                legend_irreps.add(irr)

            # First virtual
            if virt_e.shape[1] >= 1 and not np.isnan(virt_e[i, 0]):
                ax_corr.scatter(Ri, virt_e[i, 0], color="grey",
                                marker="^", s=8, alpha=0.5, zorder=2)

        # Tracked HOMO as thick connected line
        ax_corr.plot(R, homo_e, color="black", lw=2.0, zorder=3,
                     label="tracked HOMO")

        # Shade R regions where HOMO index changes
        idx_changes = np.where(np.diff(homo_idx) != 0)[0]
        for ic in idx_changes:
            ax_corr.axvspan(R[ic], R[ic + 1], color="gold",
                            alpha=0.35, zorder=1)

        ax_corr.axvline(1.822, color="grey", lw=0.8, ls=":", label="R_e(OH⁻)")
        ax_corr.set_title(f"{label}\nOrbital energy correlation")
        ax_corr.set_ylabel("MO energy (Hartree)")
        ax_corr.set_xlabel("R (Bohr)")
        # Restrict to valence region — O 1s core at ~-20 Ha is not relevant
        # and collapses the vertical scale.  Show orbitals above -2 Ha.
        ax_corr.set_ylim(-2.0, 0.5)
        # Deduplicated legend
        handles, lbls = ax_corr.get_legend_handles_labels()
        seen: dict = {}
        for h, l in zip(handles, lbls):
            if l not in seen:
                seen[l] = h
        ax_corr.legend(seen.values(), seen.keys(), fontsize=7,
                       loc="lower left", markerscale=1.5)

        # ---- Row 2: HOMO index + MOC overlap ----
        ax_idx.step(R, homo_idx, where="post", color="steelblue",
                    lw=1.5, label="HOMO index")
        ax_idx.set_ylabel("HOMO MO index", color="steelblue")
        ax_idx.tick_params(axis="y", labelcolor="steelblue")
        ax_idx.set_xlabel("R (Bohr)")
        ax_idx.set_title("HOMO index  &  MOC overlap quality")

        # MOC overlap on twin axis
        valid = ~np.isnan(overlap)
        ax_ovlp.plot(R[valid], overlap[valid], color="darkorange",
                     lw=1.5, label="best overlap")
        valid2 = ~np.isnan(overlap2nd)
        if valid2.any():
            ax_ovlp.plot(R[valid2], overlap2nd[valid2], color="darkorange",
                         lw=1.0, ls="--", label="2nd-best overlap")
        ax_ovlp.axhline(0.9, color="darkorange", lw=0.7, ls=":",
                        alpha=0.6)
        ax_ovlp.set_ylabel("MOC overlap |⟨prev|S|j⟩|", color="darkorange")
        ax_ovlp.tick_params(axis="y", labelcolor="darkorange")
        ax_ovlp.set_ylim(0, 1.05)

        for ax in [ax_idx, ax_corr]:
            ax.axvline(1.822, color="grey", lw=0.8, ls=":")
            for ic in idx_changes:
                ax.axvspan(R[ic], R[ic + 1], color="gold", alpha=0.25)

        lines1, labs1 = ax_idx.get_legend_handles_labels()
        lines2, labs2 = ax_ovlp.get_legend_handles_labels()
        ax_idx.legend(lines1 + lines2, labs1 + labs2, fontsize=7)

    fig.suptitle(
        "HOMO tracking diagnostic\n"
        "Gold bands = orbital reordering events  |  "
        "Black line = MOC-tracked HOMO"
    )
    fig.tight_layout()
    path = os.path.join(save_dir, "homo_tracking.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_orbital_tracking(save_dir: str, plt) -> None:
    """
    Derivative diagnostic for orbital-tracking health.

    Two rows × N_basis columns:
      Row 1 — dm_rad/dR(R): radial coupling derivative
      Row 2 — dm_rot/dR(R): rotational coupling derivative

    Discontinuous orbital reassignments appear as isolated large-amplitude
    spikes on an otherwise smooth baseline.  Physical sign-change gradients
    appear as broad, rounded bumps.  A vertical grey band marks the expected
    sign-change region (R ≈ 1.4–1.8 Bohr).
    """
    import os

    available = {
        label: _load_npz_coupling(path)
        for label, path in _COUPLING_FILES.items()
    }
    available = {k: v for k, v in available.items() if v}
    if not available:
        print("  [SKIP] No NPZ files — skipping orbital tracking plot")
        return

    n_bases = len(available)
    fig, axes = plt.subplots(2, n_bases, figsize=(5 * n_bases, 8), sharey="row")
    if n_bases == 1:
        axes = axes.reshape(2, 1)

    row_labels = ["dm_rad / dR  (a.u./Bohr)", "dm_rot / dR  (a.u./Bohr)"]

    for col, (label, d) in enumerate(available.items()):
        R, k_e, m_rad, m_rot = d["R"], d["k_e"], d["m_rad"], d["m_rot"]
        mid_k  = len(k_e) // 2
        dR     = np.diff(R)
        R_mid  = (R[:-1] + R[1:]) / 2

        for row, (coupling, comp_name) in enumerate([(m_rad, "m_rad"),
                                                      (m_rot, "m_rot")]):
            ax = axes[row, col]
            # Plot derivative for all k_e values (faint) and mid k_e (solid)
            for k_idx in range(len(k_e)):
                m     = coupling[:, k_idx]
                dm_dR = np.diff(m) / dR
                lw    = 1.4 if k_idx == mid_k else 0.5
                alpha = 1.0 if k_idx == mid_k else 0.35
                label_curve = f"k_e={k_e[k_idx]:.2f}" if k_idx == mid_k else None
                ax.plot(R_mid, dm_dR, lw=lw, alpha=alpha, label=label_curve)

            # Grey band: expected sign-change region
            ax.axvspan(1.3, 2.0, color="grey", alpha=0.12, label="sign-change zone")
            ax.axhline(0, color="k", lw=0.5, ls="--")
            ax.axvline(1.822, color="gray", lw=0.7, ls=":", label="R_e(OH⁻)")
            ax.set_xlabel("R (Bohr)")
            if col == 0:
                ax.set_ylabel(row_labels[row])
            ax.set_title(f"{label}\n{comp_name}")
            ax.legend(fontsize=6)

    fig.suptitle(
        "Orbital tracking diagnostic — dm/dR vs R\n"
        "Smooth curves: physical variation.  "
        "Sharp isolated spikes: orbital reordering."
    )
    fig.tight_layout()
    path = os.path.join(save_dir, "orbital_tracking.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ======================================================================
# 9. State-to-state rates vs Acharya Table I
# ======================================================================

def test_state_to_state_rates() -> bool:
    """
    Compare state-to-state AED rates with Acharya 1984 Table I.

    Uses the 6-31G CPSCF coupling (NPZ) and Acharya Morse potentials with
    the analytic Morse solver.  Evaluates J=0 → J'=0 (ΔJ=0, radial
    coupling) at E_coll = 66 cm⁻¹ for v' = 3..8.

    Also prints the box→energy normalisation factor (μL/πk_nuc):
        F_energy = F_box × √(μL/πk_nuc)
        rate_physical = rate_code × μL/(πk_nuc)

    This factor is the key source of discrepancy between the box-normalised
    code and Acharya's energy-normalised convention.
    """
    section("9. State-to-State Rates vs Acharya Table I")

    import os
    NPZ_PATH = "oh_minus_coupling_6-31g.npz"

    if not os.path.exists(NPZ_PATH):
        print(f"  [SKIP] {NPZ_PATH} not found. "
              "Run scripts/precompute_coupling.py first.")
        return True

    from scipy.interpolate import RectBivariateSpline
    from aed_rate.electronic.coupling import CouplingResult
    from aed_rate.electronic.potential import create_oh_system_acharya
    from aed_rate.rate.state_to_state import AEDRateCalculator
    from aed_rate.utils.constants import CONSTANTS, get_reduced_mass

    anion, neutral, EA = create_oh_system_acharya()
    mu = get_reduced_mass("O", "H")

    # Build spline coupling from NPZ — no PySCF required
    d       = np.load(NPZ_PATH)
    R_npz   = d["R_grid"]
    k_e_npz = d["k_e_grid"]
    m_rad2  = d["m_rad_2d"]
    m_rot2  = d["m_rot_2d"]
    _R_lo   = float(d["R_min"][0])
    _R_hi   = float(d["R_cutoff"][0])
    _spl_r  = RectBivariateSpline(R_npz, k_e_npz, m_rad2, kx=3, ky=3)
    _spl_o  = RectBivariateSpline(R_npz, k_e_npz, m_rot2, kx=3, ky=3)

    class _NpzCoupling:
        """Minimal coupling provider backed by precomputed 2D splines."""

        def compute_coupling_at_r(
            self, R: float, electron_energy: float, **_
        ) -> CouplingResult:
            """Interpolate m_rad and m_rot from NPZ; return zero outside grid."""
            k_e = float(np.sqrt(max(2.0 * electron_energy, 0.0)))
            if R < _R_lo or R > _R_hi:
                return CouplingResult(
                    R=R, m_rad=0j, m_rot=0j,
                    electron_energy=electron_energy, k_electron=k_e,
                )
            ke_c = float(np.clip(k_e, k_e_npz[0], k_e_npz[-1]))
            return CouplingResult(
                R=R,
                m_rad=complex(float(_spl_r(R, ke_c, grid=False))),
                m_rot=complex(float(_spl_o(R, ke_c, grid=False))),
                electron_energy=electron_energy,
                k_electron=k_e,
            )

    calc = AEDRateCalculator(
        anion, neutral, EA, mu,
        coupling=_NpzCoupling(),
        solver_method="morse",
        r_min=0.5, r_max=15.0, n_grid=1000,
    )

    E_coll = CONSTANTS.cm1_to_hartree(66.0)
    k_nuc  = float(np.sqrt(2.0 * mu * E_coll))
    L      = calc.anion_solver.r_max - calc.anion_solver.r_min

    # F_energy = F_box × √(μL/πk_nuc)  → rate_phys = rate_code × μL/(πk_nuc)
    norm_fac  = float(np.sqrt(mu * L / (np.pi * k_nuc)))
    rate_fac  = norm_fac ** 2
    au_to_s1  = 1.0 / CONSTANTS.au_time_to_s

    print(f"\n  E_coll = {CONSTANTS.hartree_to_cm1 * E_coll:.1f} cm⁻¹   "
          f"k_nuc = {k_nuc:.4f} a.u.   L = {L:.1f} Bohr")
    print(f"  Box→energy norm factor: √(μL/πk) = {norm_fac:.1f}   "
          f"rate correction = {rate_fac:.0f}×")
    print(f"  [NOTE] Scattering state is box-normalised (∫|F|²dR=1).  "
          f"Physical rate = code rate × {rate_fac:.0f}.")

    # Acharya 1984 Table I, J=0, E_coll=66 cm⁻¹ (ΔJ=0), s⁻¹
    ACHARYA = {8: 0.113, 7: 68.8, 6: 125.0, 5: 61.1, 4: 12.5, 3: 0.975}

    print(f"\n  {'v':>3}  {'E_e(eV)':>8}  {'k_e':>6}  "
          f"{'|I_rad|':>10}  "
          f"{'rate_code':>12}  {'rate_phys':>12}  "
          f"{'ref(Ach)':>10}  {'ratio':>7}")
    print("  " + "-" * 84)

    for v in range(3, 9):
        r = calc.state_to_state_rate(E_coll, J=0, v_prime=v, J_prime=0)
        if r.rate == 0.0:
            print(f"  {v:>3}  {'-- forbidden --':>77}")
            continue

        E_e_eV  = CONSTANTS.hartree_to_ev * r.electron_energy
        k_e     = float(np.sqrt(2.0 * r.electron_energy))
        I_rad   = abs(r.V_rad) * mu    # nuclear integral  ∫ F_v' m_rad dF/dR dR
        rate_s  = r.rate * au_to_s1
        rate_p  = rate_s * rate_fac
        ref     = ACHARYA.get(v)
        ratio   = rate_p / ref if ref else float("nan")
        ref_s   = f"{ref:10.3f}" if ref else f"{'---':>10}"
        rat_s   = f"{ratio:7.2f}" if ref else "    ---"

        print(f"  {v:>3}  {E_e_eV:8.4f}  {k_e:6.3f}  "
              f"{I_rad:10.4e}  "
              f"{rate_s:12.5e}  {rate_p:12.5e}  "
              f"{ref_s}  {rat_s}")

    # Coupling amplitude sanity: m_rad/k_e at R_e should be k_e-independent
    print("\n  --- m_rad/k_e at R_e=1.822 Bohr from NPZ ---")
    i_Re = int(np.argmin(np.abs(R_npz - 1.822)))
    print(f"  {'k_e':>6}   {'m_rad':>12}   {'m_rad/k_e':>10}")
    for j, ke in enumerate(k_e_npz):
        mr = m_rad2[i_Re, j]
        print(f"  {ke:6.3f}   {mr:12.5e}   {mr / ke:10.5f}")

    return True


# ======================================================================
# 10. CPSCF vs finite-difference coupling check
# ======================================================================

def test_cpscf_vs_fd() -> bool:
    """
    Verify the CPSCF MO-coefficient derivative against central finite differences.

    At R_e = 1.822 Bohr, three estimates of ∂φ_HOMO/∂R are compared:

    (a) CPSCF      — PySCF analytical mo1 response (coefficient derivative
                     in the fixed AO basis at R_e)
    (b) FD-coeff   — (C(R+δ)−C(R−δ))/(2δ) with MOC phase-alignment,
                     contracted with AO values at R_e (no AO-basis motion)
    (c) FD-full    — (φ(r;R+δ)−φ(r;R−δ))/(2δ) on a fixed Becke grid,
                     includes AO-basis motion as H atom shifts

    For each, m_rad = ∫ w × OPW_x × ∂φ/∂R d³r is evaluated at k_e=0.15 a.u.
    The contribution of AO-basis motion to m_rad is also printed separately.
    """
    section("10. CPSCF vs Finite-Difference Coupling Check")

    try:
        from pyscf import gto, scf, dft, symm as pyscf_symm
        from scipy.special import spherical_jn
        from scipy.interpolate import RectBivariateSpline
    except ImportError:
        print("  [SKIP] PySCF not available")
        return True

    import os
    from aed_rate.electronic.coupling import InterpolatedCoupling

    R_E   = 1.822   # Bohr (Acharya OH⁻ equilibrium)
    DELTA = 0.005   # Bohr — small for accuracy, large vs SCF noise
    BASIS = "6-31g"
    K_E   = 0.15    # a.u. — representative k_e in the Acharya regime

    def _scf(R: float):
        """Build and run RHF for OH⁻ at bond length R (Bohr)."""
        mol = gto.Mole()
        mol.atom     = f"O 0 0 0; H 0 0 {R}"
        mol.basis    = BASIS
        mol.charge   = -1
        mol.spin     = 0
        mol.unit     = "Bohr"
        mol.symmetry = True
        mol.verbose  = 0
        mol.build()
        mf = scf.RHF(mol)
        mf.verbose = 0
        mf.kernel()
        return mol, mf

    def _align_homo(mol_ref, C_ref, S_ref, mol_new, mf_new):
        """
        Return (overlap, phase-aligned HOMO coefficients) at mol_new,
        using the MOC relative to C_ref / S_ref.
        """
        try:
            orbsym = list(pyscf_symm.label_orb_symm(
                mol_new, mol_new.irrep_name, mol_new.symm_orb, mf_new.mo_coeff))
        except Exception:
            orbsym = ["?"] * mf_new.mo_coeff.shape[1]
        occ   = np.where(np.asarray(mf_new.mo_occ) > 0.5)[0]
        pi_x  = {"E1x", "e1x", "B1", "b1"}
        cands = [j for j in occ if orbsym[j] in pi_x] or list(occ)
        ovlps = {j: float(C_ref @ S_ref @ mf_new.mo_coeff[:, j]) for j in cands}
        best  = max(ovlps, key=lambda j: abs(ovlps[j]))
        sign  = float(np.sign(ovlps[best])) or 1.0
        return abs(ovlps[best]), sign * mf_new.mo_coeff[:, best]

    print(f"\n  R_e={R_E} Bohr  δ={DELTA} Bohr  k_e={K_E} a.u.  basis={BASIS}")
    print("  Running SCF at R_e and R_e±δ ...", flush=True)

    mol_0, mf_0 = _scf(R_E)
    mol_p, mf_p = _scf(R_E + DELTA)
    mol_m, mf_m = _scf(R_E - DELTA)

    S_0        = mol_0.intor("int1e_ovlp")
    homo_idx_0 = InterpolatedCoupling._find_e1x_homo(mol_0, mf_0)
    C_0        = mf_0.mo_coeff[:, homo_idx_0].copy()

    ov_p, C_p = _align_homo(mol_0, C_0, S_0, mol_p, mf_p)
    ov_m, C_m = _align_homo(mol_0, C_0, S_0, mol_m, mf_m)
    print(f"  MOC overlap: R+δ = {ov_p:.4f}   R−δ = {ov_m:.4f}")

    # FD of MO coefficients evaluated in fixed AO basis at R_e
    dC_FD_coeff = (C_p - C_m) / (2.0 * DELTA)

    # CPSCF at R_e
    print("  Running CPSCF at R_e ...", flush=True)
    hess   = mf_0.Hessian()
    h1ao   = hess.make_h1(mf_0.mo_coeff, mf_0.mo_occ)
    mo1, _ = hess.solve_mo1(mf_0.mo_energy, mf_0.mo_coeff, mf_0.mo_occ, h1ao)
    mo1    = np.array(mo1)  # (natm, 3, nao, nocc)
    # atom H (idx 1), z-direction (bond axis, idx 2)
    dC_CPSCF = mo1[1, 2, :, homo_idx_0]

    # Coefficient-level comparison
    n_cp  = np.linalg.norm(dC_CPSCF)
    n_fd  = np.linalg.norm(dC_FD_coeff)
    cosim = float(dC_CPSCF @ dC_FD_coeff) / max(n_cp * n_fd, 1e-30)
    rdiff = np.linalg.norm(dC_CPSCF - dC_FD_coeff) / max(n_cp, 1e-30)

    print(f"\n  --- MO coefficient derivative dC/dR ---")
    print(f"  |dC_CPSCF|            = {n_cp:.6e}")
    print(f"  |dC_FD|               = {n_fd:.6e}")
    print(f"  cosine(CPSCF, FD)     = {cosim:.6f}   (1 = identical direction)")
    print(f"  |CPSCF−FD| / |CPSCF| = {rdiff:.4f}")
    st_c = PASS if abs(cosim - 1.0) < 0.05 and rdiff < 0.15 else FAIL
    print(f"  [{st_c}] CPSCF ≈ FD-coeff  (cosim > 0.95, reldiff < 0.15)")

    # Becke grid at R_e for orbital evaluation
    grids = dft.gen_grid.Grids(mol_0)
    grids.level = 3
    grids.build()

    # AO values at the SAME grid points for three geometries
    # (grid is centred on atoms at R_e; we evaluate AOs at R±δ on this grid)
    ao_0 = mol_0.eval_gto("GTOval_sph", grids.coords)
    ao_p = mol_p.eval_gto("GTOval_sph", grids.coords)
    ao_m = mol_m.eval_gto("GTOval_sph", grids.coords)

    # Three estimates of ∂φ_HOMO/∂R on the Becke grid
    dphi_CPSCF    = ao_0 @ dC_CPSCF         # CPSCF (fixed-AO coefficient response)
    dphi_FD_coeff = ao_0 @ dC_FD_coeff      # FD coefficients, same fixed AO basis
    dphi_FD_full  = (ao_p @ C_p - ao_m @ C_m) / (2.0 * DELTA)   # full physical FD

    # OPW:  3 j₁(k_e r)/r × x   (E1x/π_x symmetry; → k_e × x as k_e r → 0)
    r_mag  = np.linalg.norm(grids.coords, axis=1)
    r_safe = np.maximum(r_mag, 1e-10)
    j1_r   = spherical_jn(1, K_E * r_safe) / r_safe
    opw_x  = 3.0 * j1_r * grids.coords[:, 0]

    m_CPSCF    = float(np.sum(grids.weights * opw_x * dphi_CPSCF))
    m_FD_coeff = float(np.sum(grids.weights * opw_x * dphi_FD_coeff))
    m_FD_full  = float(np.sum(grids.weights * opw_x * dphi_FD_full))
    ao_motion  = m_FD_full - m_FD_coeff

    print(f"\n  --- m_rad at R_e={R_E} Bohr, k_e={K_E} a.u. ---")
    print(f"  m_rad(CPSCF)      = {m_CPSCF:+.6e}   (CPSCF coefficient response)")
    print(f"  m_rad(FD-coeff)   = {m_FD_coeff:+.6e}   (FD coefficients, fixed AO)")
    print(f"  m_rad(FD-full)    = {m_FD_full:+.6e}   (full ∂φ/∂R incl. AO motion)")
    print(f"  AO-motion term    = {ao_motion:+.6e}   "
          f"({abs(ao_motion / max(abs(m_FD_full), 1e-20)):.1%} of full FD)")

    NPZ_PATH = "oh_minus_coupling_6-31g.npz"
    if os.path.exists(NPZ_PATH):
        dn   = np.load(NPZ_PATH)
        spl  = RectBivariateSpline(
            dn["R_grid"], dn["k_e_grid"], dn["m_rad_2d"], kx=3, ky=3
        )
        ke_c  = float(np.clip(K_E, dn["k_e_grid"][0], dn["k_e_grid"][-1]))
        m_npz = float(spl(R_E, ke_c, grid=False))
        print(f"  m_rad(NPZ spline) = {m_npz:+.6e}   (precomputed 6-31G)")

    # Summary ratios
    ref_m = m_FD_full if abs(m_FD_full) > 1e-15 else m_CPSCF
    print(f"\n  Ratios vs m_rad(FD-full):")
    for label, val in [
        ("CPSCF",     m_CPSCF),
        ("FD-coeff",  m_FD_coeff),
    ]:
        r = val / ref_m if abs(ref_m) > 1e-15 else float("nan")
        print(f"    {label:12s}: {r:.4f}")

    st_m = PASS if abs(m_CPSCF / max(abs(m_FD_full), 1e-20) - 1.0) < 0.20 else WARN
    print(f"\n  [{st_m}] m_rad(CPSCF) ≈ m_rad(FD-full)  (tolerance 20%)")

    return True


# ======================================================================
# 11. Visual plausibility plots
# ======================================================================

def generate_plots(save_dir: str = "plots") -> None:
    """Generate all plausibility plots and save to disk."""
    section("11. Generating Plots")

    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend for saving
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [SKIP] matplotlib not installed — skipping plots")
        return

    from aed_rate.utils.plotting import (
        plot_potential_curves,
        plot_bound_states,
        plot_scattering_state,
        plot_energy_levels,
    )
    from aed_rate.electronic.potential import create_oh_system
    from aed_rate.nuclear.nuclear_wavefunction import create_wavefunction_solver
    from aed_rate.utils.constants import CONSTANTS, get_reduced_mass

    import os
    os.makedirs(save_dir, exist_ok=True)

    anion, neutral, EA = create_oh_system()
    mu = get_reduced_mass("O", "H")

    # --- Plot 1: Potential energy curves ---
    print("\n  Plotting potential energy curves...")
    fig, ax = plot_potential_curves(
        anion, neutral, EA, reduced_mass=mu,
        J_values=[5, 15], unit="eV",
    )
    fig.savefig(os.path.join(save_dir, "potential_curves.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_dir}/potential_curves.png")

    # --- Plot 2: Neutral OH bound states ---
    print("\n  Plotting bound states...")
    solver_neutral = create_wavefunction_solver(
        neutral, mu, method="dvr", r_min=0.5, r_max=15.0, n_grid=500,
    )
    states = solver_neutral.solve_all_bound_states(J=0)

    fig, ax = plot_bound_states(
        states, neutral, n_states=8, unit="eV",
        R_range=(0.8, 6.0),
    )
    fig.savefig(os.path.join(save_dir, "bound_states.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_dir}/bound_states.png")

    # Also plot probability density
    fig, ax = plot_bound_states(
        states, neutral, n_states=6, unit="eV",
        R_range=(0.8, 5.0), plot_density=True,
    )
    ax.set_title("Bound States — Probability Density")
    fig.savefig(os.path.join(save_dir, "bound_states_density.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_dir}/bound_states_density.png")

    # --- Plot 3: Scattering state on anion surface ---
    print("\n  Plotting scattering state...")
    solver_anion = create_wavefunction_solver(
        anion, mu, method="dvr", r_min=0.5, r_max=15.0, n_grid=500,
    )
    E_coll = CONSTANTS.cm1_to_hartree(500)
    scat = solver_anion.solve_scattering_state(E_coll, J=0)

    fig, axes = plot_scattering_state(
        scat, anion, mu, J=0, unit="eV",
    )
    fig.savefig(os.path.join(save_dir, "scattering_state.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_dir}/scattering_state.png")

    # --- Plot 4: Energy level diagram ---
    print("\n  Plotting energy levels...")
    fig, ax = plot_energy_levels(
        anion, neutral, mu, EA, v_max=15, unit="eV",
    )
    fig.savefig(os.path.join(save_dir, "energy_levels.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_dir}/energy_levels.png")

    # --- Plot 5: Electronic coupling vs R — multi-basis convergence ---
    _plot_coupling_vs_r(save_dir, plt)

    # --- Plot 6: Coupling integrand for v' = 4..8 ---
    _plot_coupling_integrand(save_dir, plt, anion, neutral, EA, mu)

    # --- Plot 7: Orbital tracking derivative diagnostic ---
    _plot_orbital_tracking(save_dir, plt)

    # --- Plot 8: HOMO tracking correlation diagram ---
    _plot_homo_tracking(save_dir, plt)

    print(f"\n  All plots saved to {save_dir}/")


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    """Run all plausibility checks."""
    print("\n" + "#" * 60)
    print("  AED Rate Package — Plausibility Checks")
    print("#" * 60)

    # Modules that don't need PySCF
    test_potentials()
    test_nuclear_wavefunctions()
    test_continuum()
    test_coupling_model()

    # Modules that need PySCF
    has_pyscf = False
    try:
        import pyscf
        has_pyscf = True
    except ImportError:
        pass

    if has_pyscf:
        test_electronic_structure()
        test_coupling_ab_initio()
    else:
        section("5. Electronic Structure (PySCF)")
        print("  [SKIP] PySCF not installed — skipping SCF tests")
        section("6. Ab Initio Coupling (CPSCF)")
        print("  [SKIP] PySCF not installed — skipping CPSCF tests")

    # Coupling convergence & orbital-tracking (use precomputed NPZ, no PySCF needed)
    test_coupling_basis_convergence()
    test_orbital_tracking()

    # State-to-state rates (no PySCF needed — uses precomputed NPZ)
    test_state_to_state_rates()

    # CPSCF vs finite-difference coupling check (requires PySCF)
    if has_pyscf:
        test_cpscf_vs_fd()
    else:
        section("10. CPSCF vs FD Coupling Check")
        print("  [SKIP] PySCF not installed — skipping CPSCF/FD check")

    # Generate plots
    generate_plots()

    print(f"\n{'='*60}")
    print("  All plausibility checks completed.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
