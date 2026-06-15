#!/usr/bin/env python3
"""
Validation against Acharya, Das & Simons, JCP 83, 3888 (1985).

Compares our computed AED rates with their Tables I–III for the
O⁻ + H → OH(v', J') + e⁻ system at specific collision energies.

Reference data
--------------
Table I:   J=0, E_coll = 66 cm⁻¹, v' = 3–8 (radial coupling only)
Table II:  J=1, E_coll = 66 cm⁻¹, v' = 3–8 (radial + rotational)
Table III: J=2, E_coll = 66 cm⁻¹, v' = 3–8 (radial + rotational)

All rates in s⁻¹.
"""

import sys
import numpy as np

sys.path.insert(0, ".")

from aed_rate.electronic.potential import create_oh_system_acharya
from aed_rate.electronic.coupling import ModelCoupling, InterpolatedCoupling
from aed_rate.electronic.wavefunctions import ElectronicStructure
from aed_rate.rate.state_to_state import AEDRateCalculator, angular_coupling_coefficient
from aed_rate.utils.constants import CONSTANTS, get_reduced_mass


# ======================================================================
# Setup
# ======================================================================

def setup_system(
    A_rad: float = 0.145,
    alpha_rad: float = 3.0,
    A_rot: float = 3.12,
    alpha_rot: float = 3.0,
    k_power: float = 1.0,
    solver_method: str = "morse",
    n_grid: int = 500,
    ab_initio: bool = False,
    cpscf_npz: str = "oh_minus_coupling_abinitio.npz",
) -> AEDRateCalculator:
    """
    Create the OH system with Acharya parameters.

    Parameters
    ----------
    ab_initio : bool
        If True, load precomputed CPSCF coupling from ``cpscf_npz``.
        If False (default), use the Gaussian ModelCoupling.
    cpscf_npz : str
        Path to the .npz file written by InterpolatedCoupling.save().
    """
    anion_pot, neutral_pot, EA = create_oh_system_acharya()
    mu = get_reduced_mass("O", "H")

    if ab_initio:
        es = ElectronicStructure("O", "H", basis="6-31g")
        coupling = InterpolatedCoupling(es, anion_pot)
        coupling.load(cpscf_npz)
        print(f"Using ab initio CPSCF coupling from '{cpscf_npz}'")
        print(f"  R range: [{coupling.R_min:.3f}, {coupling.R_cutoff:.3f}] Bohr")
        print(f"  k_e range: [{coupling.k_e_grid[0]:.3f}, {coupling.k_e_grid[-1]:.3f}] a.u.")
    else:
        coupling = ModelCoupling(
            R0=anion_pot.r_e,
            A_rad=A_rad,
            alpha_rad=alpha_rad,
            A_rot=A_rot,
            alpha_rot=alpha_rot,
            k_power=k_power,
        )

    calc = AEDRateCalculator(
        anion_potential=anion_pot,
        neutral_potential=neutral_pot,
        EA=EA,
        reduced_mass=mu,
        coupling=coupling,
        solver_method=solver_method,
        r_min=0.5,
        r_max=15.0,
        n_grid=n_grid,
    )
    return calc


# ======================================================================
# Acharya reference data (Tables I–III)
# ======================================================================

# Table I: J=0, E_coll = 66 cm⁻¹, ΔJ=0 only (radial)
ACHARYA_TABLE_I = {
    8: 0.113,
    7: 68.8,
    6: 125.0,
    5: 61.1,
    4: 12.5,
    3: 0.975,
}

# Table II: J=1, E_coll = 66 cm⁻¹
# Columns: v', ΔJ=-1 (J'=0), ΔJ=0 (J'=1), ΔJ=+1 (J'=2), Total
ACHARYA_TABLE_II = {
    8: {"dJ-1": 0.172, "dJ0": 0.0971, "dJ+1": 0.0401, "total": 0.309},
    7: {"dJ-1": 105.0, "dJ0": 59.0,   "dJ+1": 24.3,   "total": 188.0},
    6: {"dJ-1": 191.0, "dJ0": 107.0,  "dJ+1": 44.2,   "total": 342.0},
    5: {"dJ-1": 93.3,  "dJ0": 52.4,   "dJ+1": 21.6,   "total": 167.0},
    4: {"dJ-1": 19.1,  "dJ0": 10.7,   "dJ+1": 4.42,   "total": 34.2},
    3: {"dJ-1": 1.49,  "dJ0": 0.837,  "dJ+1": 0.345,  "total": 2.67},
}

# Table III: J=2, E_coll = 66 cm⁻¹
ACHARYA_TABLE_III = {
    8: {"dJ-1": 0.296, "dJ0": 0.0857, "dJ+1": 0.0268, "total": 0.409},
    7: {"dJ-1": 180.0, "dJ0": 52.1,   "dJ+1": 16.3,   "total": 249.0},
    6: {"dJ-1": 328.0, "dJ0": 94.7,   "dJ+1": 29.6,   "total": 452.0},
    5: {"dJ-1": 160.0, "dJ0": 46.3,   "dJ+1": 14.5,   "total": 221.0},
    4: {"dJ-1": 32.8,  "dJ0": 9.49,   "dJ+1": 2.96,   "total": 45.3},
    3: {"dJ-1": 2.56,  "dJ0": 0.740,  "dJ+1": 0.231,  "total": 3.53},
}


# ======================================================================
# Tests
# ======================================================================

def validate_table_I(calc: AEDRateCalculator, E_coll_Ha: float) -> dict:
    """Validate against Table I: J=0, radial coupling only."""
    print("\n" + "=" * 70)
    print("TABLE I: J=0, E_coll = 66 cm⁻¹, radial coupling (ΔJ=0)")
    print("=" * 70)
    print(f"{'v_prime':>7}  {'Acharya (s⁻¹)':>14}  {'Computed (s⁻¹)':>14}  {'Ratio':>8}")
    print("-" * 50)

    results = {}
    for v_prime in sorted(ACHARYA_TABLE_I.keys(), reverse=True):
        ref = ACHARYA_TABLE_I[v_prime]
        res = calc.state_to_state_rate(E_coll_Ha, J=0, v_prime=v_prime, J_prime=0)
        rate_s1 = CONSTANTS.rate_au_to_s1(res.rate)
        ratio = rate_s1 / ref if ref > 0 else float("inf")
        results[v_prime] = {"computed": rate_s1, "reference": ref, "ratio": ratio}
        print(f"{v_prime:>7d}  {ref:>14.3f}  {rate_s1:>14.3f}  {ratio:>8.3f}")

    return results


def validate_table_II(calc: AEDRateCalculator, E_coll_Ha: float) -> dict:
    """Validate against Table II: J=1."""
    print("\n" + "=" * 70)
    print("TABLE II: J=1, E_coll = 66 cm⁻¹")
    print("=" * 70)
    print(f"{'v_prime':>7}  {'ΔJ':>4}  {'Acharya':>10}  {'Computed':>10}  {'Ratio':>8}")
    print("-" * 50)

    results = {}
    for v_prime in sorted(ACHARYA_TABLE_II.keys(), reverse=True):
        ref_data = ACHARYA_TABLE_II[v_prime]
        computed_total = 0.0

        for dJ_label, J_prime in [("dJ-1", 0), ("dJ0", 1), ("dJ+1", 2)]:
            ref = ref_data[dJ_label]
            res = calc.state_to_state_rate(E_coll_Ha, J=1, v_prime=v_prime, J_prime=J_prime)
            rate_s1 = CONSTANTS.rate_au_to_s1(res.rate)
            computed_total += rate_s1
            ratio = rate_s1 / ref if ref > 0 else float("inf")
            dJ = J_prime - 1
            print(f"{v_prime:>7d}  {dJ:>+4d}  {ref:>10.3f}  {rate_s1:>10.3f}  {ratio:>8.3f}")

        ref_total = ref_data["total"]
        ratio_total = computed_total / ref_total if ref_total > 0 else float("inf")
        print(f"{'':>7s}  {'tot':>4s}  {ref_total:>10.3f}  {computed_total:>10.3f}  {ratio_total:>8.3f}")
        print()
        results[v_prime] = {"computed_total": computed_total, "reference_total": ref_total}

    return results


def validate_table_III(calc: AEDRateCalculator, E_coll_Ha: float) -> dict:
    """Validate against Table III: J=2."""
    print("\n" + "=" * 70)
    print("TABLE III: J=2, E_coll = 66 cm⁻¹")
    print("=" * 70)
    print(f"{'v_prime':>7}  {'ΔJ':>4}  {'Acharya':>10}  {'Computed':>10}  {'Ratio':>8}")
    print("-" * 50)

    results = {}
    for v_prime in sorted(ACHARYA_TABLE_III.keys(), reverse=True):
        ref_data = ACHARYA_TABLE_III[v_prime]
        computed_total = 0.0

        for dJ_label, J_prime in [("dJ-1", 1), ("dJ0", 2), ("dJ+1", 3)]:
            ref = ref_data[dJ_label]
            res = calc.state_to_state_rate(E_coll_Ha, J=2, v_prime=v_prime, J_prime=J_prime)
            rate_s1 = CONSTANTS.rate_au_to_s1(res.rate)
            computed_total += rate_s1
            ratio = rate_s1 / ref if ref > 0 else float("inf")
            dJ = J_prime - 2
            print(f"{v_prime:>7d}  {dJ:>+4d}  {ref:>10.3f}  {rate_s1:>10.3f}  {ratio:>8.3f}")

        ref_total = ref_data["total"]
        ratio_total = computed_total / ref_total if ref_total > 0 else float("inf")
        print(f"{'':>7s}  {'tot':>4s}  {ref_total:>10.3f}  {computed_total:>10.3f}  {ratio_total:>8.3f}")
        print()
        results[v_prime] = {"computed_total": computed_total, "reference_total": ref_total}

    return results


def validate_selection_rules(calc: AEDRateCalculator, E_coll_Ha: float) -> None:
    """Verify ΔJ = ±1 selection rule."""
    print("\n" + "=" * 70)
    print("SELECTION RULES CHECK")
    print("=" * 70)

    # ΔJ = 2 should be forbidden
    res = calc.state_to_state_rate(E_coll_Ha, J=5, v_prime=5, J_prime=7)
    assert res.rate == 0.0, f"ΔJ=2 should be forbidden, got rate={res.rate}"
    print("✓ ΔJ=2 (J=5→J'=7): rate = 0  (forbidden)")

    # ΔJ = -2 should be forbidden
    res = calc.state_to_state_rate(E_coll_Ha, J=5, v_prime=5, J_prime=3)
    assert res.rate == 0.0, f"ΔJ=-2 should be forbidden, got rate={res.rate}"
    print("✓ ΔJ=-2 (J=5→J'=3): rate = 0  (forbidden)")

    # ΔJ = 0 should be allowed
    res = calc.state_to_state_rate(E_coll_Ha, J=0, v_prime=5, J_prime=0)
    assert res.rate > 0.0, f"ΔJ=0 should be allowed, got rate={res.rate}"
    print(f"✓ ΔJ=0 (J=0→J'=0): rate = {CONSTANTS.rate_au_to_s1(res.rate):.3f} s⁻¹  (allowed)")

    print("All selection rule checks passed.")


def validate_energy_conservation(calc: AEDRateCalculator, E_coll_Ha: float) -> None:
    """Verify that high v' transitions are forbidden by energy conservation."""
    print("\n" + "=" * 70)
    print("ENERGY CONSERVATION CHECK")
    print("=" * 70)

    # At E=66 cm⁻¹, very high v' should be energetically forbidden
    # v'=8 is barely accessible; v'=9+ should be forbidden
    for v_prime in [8, 9, 10]:
        try:
            res = calc.state_to_state_rate(E_coll_Ha, J=0, v_prime=v_prime, J_prime=0)
            rate_s1 = CONSTANTS.rate_au_to_s1(res.rate)
            status = "accessible" if res.rate > 0 else "FORBIDDEN"
            print(f"  v'={v_prime}: E_e = {res.electron_energy:.6f} Ha, "
                  f"rate = {rate_s1:.4f} s⁻¹ [{status}]")
        except (ValueError, IndexError) as e:
            print(f"  v'={v_prime}: {e}")

    print("Energy conservation checks done.")


def validate_angular_coefficients() -> None:
    """Verify angular coupling coefficients match known values."""
    print("\n" + "=" * 70)
    print("ANGULAR COUPLING COEFFICIENTS")
    print("=" * 70)

    # C(0, 1) = 1/√3 ≈ 0.5774
    c01 = angular_coupling_coefficient(0, 1)
    expected_01 = 1.0 / np.sqrt(3.0)
    print(f"  C(0,1) = {c01:.6f}  (expected {expected_01:.6f})")
    assert abs(c01 - expected_01) < 1e-10

    # C(1, 0) = 1/√3 ≈ 0.5774
    c10 = angular_coupling_coefficient(1, 0)
    expected_10 = 1.0 / np.sqrt(3.0)
    print(f"  C(1,0) = {c10:.6f}  (expected {expected_10:.6f})")
    assert abs(c10 - expected_10) < 1e-10

    # C(1, 2) = 2/√(3×5) = 2/√15 ≈ 0.5164
    c12 = angular_coupling_coefficient(1, 2)
    expected_12 = 2.0 / np.sqrt(15.0)
    print(f"  C(1,2) = {c12:.6f}  (expected {expected_12:.6f})")
    assert abs(c12 - expected_12) < 1e-10

    # C(2, 1) = 2/√(3×5) = 2/√15 ≈ 0.5164
    c21 = angular_coupling_coefficient(2, 1)
    expected_21 = 2.0 / np.sqrt(15.0)
    print(f"  C(2,1) = {c21:.6f}  (expected {expected_21:.6f})")
    assert abs(c21 - expected_21) < 1e-10

    # C(0, 0) = 0 (selection rule)
    c00 = angular_coupling_coefficient(0, 0)
    print(f"  C(0,0) = {c00:.6f}  (expected 0.0)")
    assert c00 == 0.0

    # C(2, 4) = 0 (|ΔJ| > 1)
    c24 = angular_coupling_coefficient(2, 4)
    print(f"  C(2,4) = {c24:.6f}  (expected 0.0)")
    assert c24 == 0.0

    print("All angular coefficient checks passed.")


def validate_vibrational_distribution(calc: AEDRateCalculator, E_coll_Ha: float) -> None:
    """Check that rates decrease for high and low v' (bell-shaped distribution)."""
    print("\n" + "=" * 70)
    print("VIBRATIONAL DISTRIBUTION (J=0)")
    print("=" * 70)

    dist = calc.vibrational_distribution(E_coll_Ha, J=0, v_max=10)
    print(f"{'v_prime':>7s}  {'Rate (s⁻¹)':>12s}")
    print("-" * 25)
    for v_prime in sorted(dist.keys()):
        rate_s1 = CONSTANTS.rate_au_to_s1(dist[v_prime])
        print(f"{v_prime:>7d}  {rate_s1:>12.4f}")

    # The distribution should peak around v'=6 (Acharya Table I)
    if dist:
        peak_v = max(dist, key=dist.get)
        print(f"\nPeak at v' = {peak_v} (Acharya: v' = 6)")


def print_diagnostic_info(calc: AEDRateCalculator, E_coll_Ha: float) -> None:
    """Print diagnostic information about the wavefunctions and coupling."""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC INFORMATION")
    print("=" * 70)

    # Reduced mass
    mu = calc.mu
    print(f"Reduced mass: {mu:.2f} m_e = {mu / CONSTANTS.amu_to_me:.6f} amu")

    # Anion potential info
    ap = calc.anion_potential
    print(f"\nAnion (OH⁻): D_e={ap.D_e:.4f} Ha, R_e={ap.r_e:.3f} Bohr, "
          f"β={ap.beta:.3f} Bohr⁻¹")
    lam_anion = np.sqrt(2 * mu * ap.D_e) / ap.beta
    print(f"  λ = √(2μD_e)/β = {lam_anion:.2f}  →  v_max = {int(lam_anion - 0.5)}")

    # Neutral potential info
    np_ = calc.neutral_potential
    print(f"Neutral (OH):  D_e={np_.D_e:.4f} Ha, R_e={np_.r_e:.3f} Bohr, "
          f"β={np_.beta:.3f} Bohr⁻¹, V_0={np_.V_0:.5f} Ha")
    lam_neutral = np.sqrt(2 * mu * np_.D_e) / np_.beta
    print(f"  λ = √(2μD_e)/β = {lam_neutral:.2f}  →  v_max = {int(lam_neutral - 0.5)}")

    # EA and collision energy
    print(f"\nEA = {calc.EA:.5f} Ha = {calc.EA * CONSTANTS.hartree_to_ev:.4f} eV")
    print(f"E_coll = {E_coll_Ha:.6f} Ha = {E_coll_Ha * CONSTANTS.hartree_to_cm1:.1f} cm⁻¹")

    # Scattering state at J=0
    scat = calc.anion_solver.solve_scattering_state(E_coll_Ha, J=0)
    from scipy.integrate import simpson as _simp
    print(f"\nScattering state (J=0): norm check = "
          f"{_simp(scat.wavefunction**2, x=scat.r_grid):.6f}")
    print(f"  Phase shift: δ = {scat.phase_shift:.4f} rad")

    # Bound states on neutral surface
    print("\nNeutral bound state energies (J=0):")
    for v in range(9):
        try:
            bs = calc.neutral_solver.solve_bound_state(v, J=0)
            E_vib = bs.energy - np_.V_0
            print(f"  v={v}: E = {bs.energy:.6f} Ha, "
                  f"E_vib = {E_vib:.6f} Ha = {E_vib * CONSTANTS.hartree_to_cm1:.1f} cm⁻¹")
        except (ValueError, IndexError):
            print(f"  v={v}: not bound")
            break

    # Electron energy for v'=6 (dominant channel)
    bs6 = calc.neutral_solver.solve_bound_state(6, J=0)
    E_e = ap.D_e + E_coll_Ha - bs6.energy
    k_e = np.sqrt(max(2.0 * E_e, 0.0))
    print(f"\nFor v'=6, J'=0:")
    print(f"  E_electron = {E_e:.6f} Ha = {E_e * CONSTANTS.hartree_to_cm1:.1f} cm⁻¹")
    print(f"  k_e = {k_e:.6f} a.u.")

    # Coupling at R_e
    coupling_at_Re = calc.coupling.compute_coupling_at_r(ap.r_e, E_e)
    print(f"  m_rad(R_e) = {coupling_at_Re.m_rad:.6f}")
    print(f"  m_rot(R_e) = {coupling_at_Re.m_rot:.6f}")


# ======================================================================
# Main
# ======================================================================

def main():
    print("=" * 70)
    print("AED RATE VALIDATION: Acharya, Das & Simons, JCP 83, 3888 (1985)")
    print("=" * 70)

    E_coll_cm1 = 66.0
    E_coll_Ha = CONSTANTS.cm1_to_hartree(E_coll_cm1)

    print(f"\nCollision energy: {E_coll_cm1} cm⁻¹ = {E_coll_Ha:.6f} Ha")

    # Use precomputed CPSCF ab initio coupling (6-31G, 30 R points)
    calc = setup_system(
        solver_method="morse",
        n_grid=500,
        ab_initio=True,
        cpscf_npz="oh_minus_coupling_6-31g.npz",
    )

    # Diagnostics first
    print_diagnostic_info(calc, E_coll_Ha)

    # Angular coefficient check (independent of coupling parameters)
    validate_angular_coefficients()

    # Selection rules
    validate_selection_rules(calc, E_coll_Ha)

    # Energy conservation
    validate_energy_conservation(calc, E_coll_Ha)

    # Table I: J=0
    results_I = validate_table_I(calc, E_coll_Ha)

    # Table II: J=1
    results_II = validate_table_II(calc, E_coll_Ha)

    # Table III: J=2
    results_III = validate_table_III(calc, E_coll_Ha)

    # Vibrational distribution
    validate_vibrational_distribution(calc, E_coll_Ha)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Table I ratios
    print("\nTable I (J=0) — ratio computed/Acharya:")
    ratios_I = [r["ratio"] for r in results_I.values() if r["reference"] > 1.0]
    for v, r in sorted(results_I.items(), reverse=True):
        marker = "  ←peak" if v == 6 else ""
        print(f"  v'={v}: ratio = {r['ratio']:.3f}{marker}")
    if ratios_I:
        print(f"  Mean ratio (v' with rate > 1 s⁻¹): {np.mean(ratios_I):.3f}")

    # Table II totals
    print("\nTable II (J=1) — total rate ratio:")
    for v, r in sorted(results_II.items(), reverse=True):
        ratio = r["computed_total"] / r["reference_total"]
        print(f"  v'={v}: ratio = {ratio:.3f}")

    # Table III totals
    print("\nTable III (J=2) — total rate ratio:")
    for v, r in sorted(results_III.items(), reverse=True):
        ratio = r["computed_total"] / r["reference_total"]
        print(f"  v'={v}: ratio = {ratio:.3f}")


if __name__ == "__main__":
    main()
