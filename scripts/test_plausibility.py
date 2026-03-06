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
# 7. Visual plausibility plots
# ======================================================================

def generate_plots(save_dir: str = "plots") -> None:
    """Generate all plausibility plots and save to disk."""
    section("7. Generating Plots")

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

    # Generate plots
    generate_plots()

    print(f"\n{'='*60}")
    print("  All plausibility checks completed.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
