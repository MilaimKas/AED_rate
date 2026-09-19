"""
Tests for the continuum electron description (OPW) and energy conservation.

The ejected electron is modelled as an orthogonalized plane wave; what is
checked here is the free-electron kinematics, the density of states that
enters the golden-rule prefactor, and the energy balance that decides
which final vibrational states are open.
"""

import numpy as np
import pytest

from aed_rate.electronic.continuum import (
    ContinuumOrbital,
    SphericalContinuum,
    compute_electron_kinetic_energy,
)
from aed_rate.utils.constants import CONSTANTS

# OH equilibrium bond length, the length scale the k*r_e criterion uses.
R_EQUILIBRIUM = 1.83


class TestFreeElectronKinematics:
    """Wave vector and wavelength of the ejected electron."""

    @pytest.mark.parametrize("energy_ev", [0.1, 0.5, 1.0, 2.0])
    def test_wave_vector_from_energy(self, energy_ev: float) -> None:
        """E = k^2/2 in atomic units."""
        energy = CONSTANTS.ev_to_hartree(energy_ev)
        orbital = ContinuumOrbital(kinetic_energy=energy)
        assert orbital.k == pytest.approx(np.sqrt(2.0 * energy))
        assert 0.5 * orbital.k**2 == pytest.approx(energy)

    def test_wavelength_matches_wave_vector(self) -> None:
        """lambda = 2*pi/k."""
        orbital = ContinuumOrbital(kinetic_energy=CONSTANTS.ev_to_hartree(1.0))
        assert orbital.wavelength == pytest.approx(2.0 * np.pi / orbital.k)

    def test_zero_energy_gives_infinite_wavelength(self) -> None:
        """A threshold electron has zero k and an infinite de Broglie wavelength."""
        orbital = ContinuumOrbital(kinetic_energy=0.0)
        assert orbital.k == 0.0
        assert np.isinf(orbital.wavelength)

    def test_negative_energy_is_rejected(self) -> None:
        """A negative kinetic energy is unphysical and must raise."""
        with pytest.raises(ValueError, match="non-negative"):
            ContinuumOrbital(kinetic_energy=-1e-3)


class TestDensityOfStates:
    """The per-volume density of continuum states, rho = k/(2*pi^2)."""

    @pytest.mark.parametrize("energy_ev", [0.1, 0.5, 1.0, 2.0])
    def test_formula(self, energy_ev: float) -> None:
        """rho equals k/(2*pi^2) exactly, in atomic units."""
        orbital = ContinuumOrbital(
            kinetic_energy=CONSTANTS.ev_to_hartree(energy_ev)
        )
        assert orbital.density_of_states() == pytest.approx(
            orbital.k / (2.0 * np.pi**2), abs=1e-12
        )

    def test_scales_as_square_root_of_energy(self) -> None:
        """
        rho grows as sqrt(E): quadrupling the energy doubles it.

        This sqrt(E) factor is what suppresses detachment at threshold.
        """
        low = ContinuumOrbital(kinetic_energy=CONSTANTS.ev_to_hartree(0.5))
        high = ContinuumOrbital(kinetic_energy=CONSTANTS.ev_to_hartree(2.0))
        assert high.density_of_states() / low.density_of_states() == pytest.approx(
            2.0, rel=1e-10
        )

    def test_vanishes_at_threshold(self) -> None:
        """No continuum states are available at exactly zero energy."""
        assert ContinuumOrbital(kinetic_energy=0.0).density_of_states() == 0.0


class TestEnergyConservation:
    """Which final vibrational states are energetically open."""

    # Representative OH⁻ + H numbers: anion dissociation energy and EA.
    ANION_DISSOCIATION = CONSTANTS.ev_to_hartree(4.5)
    ELECTRON_AFFINITY = CONSTANTS.ev_to_hartree(1.8276)
    COLLISION_ENERGY = CONSTANTS.cm1_to_hartree(500.0)

    def electron_energy(self, neutral_vib_cm1: float) -> float:
        """Electron kinetic energy for a neutral vibrational level, in Hartree."""
        return compute_electron_kinetic_energy(
            collision_energy=self.COLLISION_ENERGY,
            anion_vib_energy=self.ANION_DISSOCIATION,
            neutral_vib_energy=CONSTANTS.cm1_to_hartree(neutral_vib_cm1),
            electron_affinity=self.ELECTRON_AFFINITY,
        )

    def test_low_lying_final_state_is_open(self) -> None:
        """Detaching into v'=0 leaves the electron with positive energy."""
        assert self.electron_energy(1850.0) > 0.0

    def test_high_final_state_is_closed(self) -> None:
        """
        A final vibrational energy above the available budget is forbidden,
        and the function clamps to zero rather than returning a negative
        energy that would produce an imaginary k.
        """
        assert self.electron_energy(30000.0) == 0.0

    def test_budget_is_shared_with_the_final_vibration(self) -> None:
        """
        Every cm⁻¹ put into vibration is taken from the electron.

        This is the origin of the inverted v' distribution: the highest
        open v' has the slowest electron and the largest Franck-Condon
        overlap with the incoming continuum state.
        """
        difference = self.electron_energy(1000.0) - self.electron_energy(3000.0)
        assert difference == pytest.approx(CONSTANTS.cm1_to_hartree(2000.0))

    def test_collision_energy_adds_to_the_budget(self) -> None:
        """Raising the collision energy raises the electron energy one for one."""
        extra = CONSTANTS.cm1_to_hartree(100.0)
        base = self.electron_energy(1850.0)
        raised = compute_electron_kinetic_energy(
            collision_energy=self.COLLISION_ENERGY + extra,
            anion_vib_energy=self.ANION_DISSOCIATION,
            neutral_vib_energy=CONSTANTS.cm1_to_hartree(1850.0),
            electron_affinity=self.ELECTRON_AFFINITY,
        )
        assert raised - base == pytest.approx(extra)


class TestLowKRegime:
    """The k*r_e criterion that justifies the low-k OPW approximation."""

    def test_near_threshold_electron_is_long_wavelength(self) -> None:
        """At 100 cm⁻¹ the electron wavelength dwarfs the bond, k*r_e << 1."""
        orbital = ContinuumOrbital(kinetic_energy=CONSTANTS.cm1_to_hartree(100.0))
        assert orbital.k * R_EQUILIBRIUM < 0.5

    def test_electron_volt_electron_needs_the_full_opw(self) -> None:
        """By 2 eV the wavelength is comparable to the bond, k*r_e > 0.5."""
        orbital = ContinuumOrbital(kinetic_energy=CONSTANTS.ev_to_hartree(2.0))
        assert orbital.k * R_EQUILIBRIUM > 0.5


class TestPlaneWave:
    """The plane-wave factor before orthogonalization."""

    def test_unit_modulus(self) -> None:
        """exp(i*k.r) has modulus one everywhere."""
        orbital = ContinuumOrbital(kinetic_energy=CONSTANTS.ev_to_hartree(1.0))
        points = np.random.default_rng(0).normal(size=(20, 3))
        values = orbital.plane_wave(points, np.array([0.0, 0.0, 1.0]))
        assert np.allclose(np.abs(values), 1.0)

    def test_direction_is_normalised_internally(self) -> None:
        """An unnormalised direction vector gives the same wave as a unit one."""
        orbital = ContinuumOrbital(kinetic_energy=CONSTANTS.ev_to_hartree(1.0))
        points = np.random.default_rng(1).normal(size=(10, 3))
        unit = orbital.plane_wave(points, np.array([0.0, 0.0, 1.0]))
        scaled = orbital.plane_wave(points, np.array([0.0, 0.0, 7.0]))
        assert np.allclose(unit, scaled)

    def test_zero_direction_is_rejected(self) -> None:
        """A zero direction vector has no defined propagation axis."""
        orbital = ContinuumOrbital(kinetic_energy=CONSTANTS.ev_to_hartree(1.0))
        with pytest.raises(ValueError, match="zero vector"):
            orbital.plane_wave(np.zeros((3, 3)), np.zeros(3))


class TestSphericalContinuum:
    """Partial-wave radial functions used for the l=0 and l=1 channels."""

    @pytest.mark.parametrize("l, value_at_origin", [(0, 1.0), (1, 0.0)])
    def test_regular_solution_at_the_origin(
        self, l: int, value_at_origin: float
    ) -> None:
        """
        regular_solution returns j_l(kr), so j_0(0) = 1 and j_1(0) = 0.

        Note this is the spherical Bessel function itself, not the
        Riccati-Bessel function kr*j_l(kr) that vanishes for every l.
        """
        wave = SphericalContinuum(
            kinetic_energy=CONSTANTS.ev_to_hartree(1.0), l=l
        )
        values = wave.regular_solution(np.linspace(0.0, 20.0, 400))
        assert values[0] == pytest.approx(value_at_origin, abs=1e-12)

    @pytest.mark.parametrize("l", [0, 1])
    def test_regular_solution_decays_as_one_over_r(self, l: int) -> None:
        """
        j_l(kr) falls off as 1/r, so r*j_l(kr) stays bounded far out.

        This 1/r envelope is what keeps the OPW coupling integral finite.
        """
        wave = SphericalContinuum(kinetic_energy=CONSTANTS.ev_to_hartree(1.0), l=l)
        r = np.linspace(20.0, 200.0, 2000)
        envelope = np.abs(r * wave.regular_solution(r))
        assert envelope.max() < 2.0 / wave.k

    def test_p_wave_is_suppressed_at_short_range(self) -> None:
        """
        Near the origin the l=1 wave is smaller than the l=0 wave.

        The centrifugal barrier gives j_l(kr) ~ (kr)^l, so higher partial
        waves are pushed out of the molecular region — the reason l >= 2
        is neglected entirely.
        """
        energy = CONSTANTS.ev_to_hartree(0.1)
        r = np.array([0.5])
        s_wave = abs(SphericalContinuum(energy, l=0).regular_solution(r)[0])
        p_wave = abs(SphericalContinuum(energy, l=1).regular_solution(r)[0])
        assert p_wave < s_wave
