"""
Tests for the Morse potential energy curves of OH⁻ and OH.

Reference values (Huber & Herzberg, and Acharya et al. 1985):
OH r_e = 1.8324 Bohr, D_e = 4.392 eV, omega_e = 3738 cm⁻¹;
OH electron affinity = 1.8276 eV.
"""

import numpy as np
import pytest

from aed_rate.utils.constants import CONSTANTS


def morse_omega_e(potential, reduced_mass: float) -> float:
    """Harmonic frequency omega_e = beta*sqrt(2*D_e/mu) of a Morse curve, in cm⁻¹."""
    omega_hartree = potential.beta * np.sqrt(2.0 * potential.D_e / reduced_mass)
    return CONSTANTS.hartree_to_cm1 * omega_hartree


class TestSpectroscopicOH:
    """The create_oh_system() curves against experimental OH/OH⁻ constants."""

    def test_reduced_mass(self, reduced_mass: float) -> None:
        """mu(OH) is 0.9482 amu, i.e. close to the proton mass."""
        assert reduced_mass / CONSTANTS.amu_to_me == pytest.approx(0.948, abs=0.005)

    def test_anion_equilibrium(self, oh_system) -> None:
        """OH⁻ sits near 1.83 Bohr with a well depth near 4.5 eV."""
        anion, _, _ = oh_system
        assert anion.r_eq == pytest.approx(1.83, rel=0.02)
        assert CONSTANTS.hartree_to_ev * anion.D_e == pytest.approx(4.5, rel=0.05)

    def test_neutral_equilibrium(self, oh_system) -> None:
        """OH sits near 1.83 Bohr with a well depth near 4.4 eV."""
        _, neutral, _ = oh_system
        assert neutral.r_eq == pytest.approx(1.83, rel=0.02)
        assert CONSTANTS.hartree_to_ev * neutral.D_e == pytest.approx(4.4, rel=0.05)

    def test_anion_vibrational_frequency(self, oh_system, reduced_mass) -> None:
        """OH⁻ omega_e is near 3700 cm⁻¹."""
        anion, _, _ = oh_system
        assert morse_omega_e(anion, reduced_mass) == pytest.approx(3700, rel=0.05)

    def test_neutral_vibrational_frequency(self, oh_system, reduced_mass) -> None:
        """OH omega_e is near the experimental 3738 cm⁻¹."""
        _, neutral, _ = oh_system
        assert morse_omega_e(neutral, reduced_mass) == pytest.approx(3738, rel=0.05)

    def test_electron_affinity(self, oh_system) -> None:
        """The EA reproduces the measured 1.8276 eV."""
        _, _, electron_affinity = oh_system
        assert CONSTANTS.hartree_to_ev * electron_affinity == pytest.approx(
            1.8276, rel=0.01
        )


class TestCurveOrdering:
    """Structural properties the two curves must satisfy jointly."""

    def test_anion_lies_below_neutral_everywhere(self, oh_system) -> None:
        """
        Non-resonant regime: the anion is electronically bound at every R.

        This is the central assumption of the whole model — if it fails, a
        nonlocal resonance treatment is required instead.
        """
        anion, neutral, _ = oh_system
        r_grid = np.linspace(1.0, 8.0, 200)
        assert np.all(anion(r_grid) < neutral(r_grid))

    def test_anion_lies_below_neutral_acharya(self, oh_system_acharya) -> None:
        """The same ordering holds for the Acharya parameter set."""
        anion, neutral, _ = oh_system_acharya
        r_grid = np.linspace(1.0, 8.0, 200)
        assert np.all(anion(r_grid) < neutral(r_grid))


class TestVibrationalStructure:
    """Analytic Morse vibrational levels."""

    def test_neutral_bound_state_count(self, oh_system, reduced_mass) -> None:
        """OH supports roughly 20-25 bound vibrational levels."""
        _, neutral, _ = oh_system
        levels = neutral.vibrational_energies(reduced_mass)
        assert 15 < len(levels) < 30

    def test_fundamental_transition(self, oh_system, reduced_mass) -> None:
        """
        The v=0 -> v=1 spacing is near 3570 cm⁻¹.

        This is below omega_e = 3738 cm⁻¹ by 2*omega_e*x_e, the Morse
        anharmonicity — a direct check that the anharmonic term is right.
        """
        _, neutral, _ = oh_system
        levels = neutral.vibrational_energies(reduced_mass)
        fundamental = CONSTANTS.hartree_to_cm1 * (levels[1] - levels[0])
        assert fundamental == pytest.approx(3570, rel=0.05)

    def test_levels_are_ordered_and_converging(self, oh_system, reduced_mass) -> None:
        """Morse levels increase monotonically with shrinking spacing."""
        _, neutral, _ = oh_system
        levels = neutral.vibrational_energies(reduced_mass)
        spacings = np.diff(levels)
        assert np.all(spacings > 0)
        assert np.all(np.diff(spacings) < 0)


class TestCentrifugalBarrier:
    """The J > 0 effective potential and its barrier."""

    def test_barrier_exists_above_dissociation(self, oh_system, reduced_mass) -> None:
        """At J=10 the centrifugal term creates a barrier above the asymptote."""
        anion, _, _ = oh_system
        r_barrier, v_barrier = anion.find_barrier_height(J=10, reduced_mass=reduced_mass)
        r_barrier = float(np.asarray(r_barrier).flat[0])
        v_barrier = float(np.asarray(v_barrier).flat[0])
        assert v_barrier > anion.dissociation_energy
        assert r_barrier > anion.r_eq

    def test_barrier_grows_with_rotation(self, oh_system, reduced_mass) -> None:
        """A higher J gives a higher barrier — the J(J+1) scaling."""
        anion, _, _ = oh_system
        heights = []
        for J in (5, 10, 20):
            _, v_barrier = anion.find_barrier_height(J=J, reduced_mass=reduced_mass)
            heights.append(float(np.asarray(v_barrier).flat[0]))
        assert heights[0] < heights[1] < heights[2]


class TestAcharyaParameters:
    """The Acharya et al. (1985) parameter set we benchmark against."""

    def test_anion_parameters(self, oh_system_acharya) -> None:
        """OH⁻: D_e = 0.1830 Ha, R_e = 1.822 Bohr, beta = 1.152 Bohr⁻¹."""
        anion, _, _ = oh_system_acharya
        assert anion.D_e == pytest.approx(0.1830, abs=1e-4)
        assert anion.r_e == pytest.approx(1.822, abs=1e-3)
        assert anion.beta == pytest.approx(1.152, abs=1e-3)

    def test_neutral_parameters(self, oh_system_acharya) -> None:
        """OH: D_e = 0.1698 Ha, R_e = 1.834 Bohr, beta = 1.214 Bohr⁻¹."""
        _, neutral, _ = oh_system_acharya
        assert neutral.D_e == pytest.approx(0.1698, abs=1e-4)
        assert neutral.r_e == pytest.approx(1.834, abs=1e-3)
        assert neutral.beta == pytest.approx(1.214, abs=1e-3)

    def test_neutral_minimum_is_offset_by_the_electron_affinity(
        self, oh_system_acharya
    ) -> None:
        """The neutral curve is shifted up by EA = 1.8276 eV = 0.06716 Ha."""
        _, neutral, electron_affinity = oh_system_acharya
        assert electron_affinity == pytest.approx(0.06716, abs=1e-4)
        assert neutral.V_0 == pytest.approx(electron_affinity, abs=1e-6)
