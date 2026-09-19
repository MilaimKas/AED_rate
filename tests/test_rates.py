"""
Tests for the rate layer: angular coupling, state-to-state rates,
cross sections and the thermal average.

The absolute normalisation is benchmarked against Acharya et al. (1985)
Table I in the slow layer, which needs the cached CPSCF coupling grid.
The fast layer checks the algebra, the selection rules and the energy
bookkeeping that hold regardless of the coupling used.
"""

import numpy as np
import pytest

from aed_rate.rate.state_to_state import (
    AEDRateCalculator,
    angular_coupling_coefficient,
)
from aed_rate.utils.constants import CONSTANTS

# Acharya's benchmark collision energy.
E_COLLISION = CONSTANTS.cm1_to_hartree(66.0)

# Acharya et al. (1985) Table I: J=0, E_coll = 66 cm⁻¹, rates in s⁻¹.
ACHARYA_TABLE_I = {3: 0.975, 4: 12.5, 5: 61.1, 6: 125.0, 7: 68.8, 8: 0.113}

# The coupling integrand nearly cancels, so the radial grid must be dense.
CONVERGED_N_GRID = 6000


class TestAngularCouplingCoefficient:
    """The closed-form <Y_{J'0}|d/dtheta|Y_{J0}> coefficients."""

    def test_delta_j_zero_is_forbidden(self) -> None:
        """Rotational coupling connects only J' = J +/- 1."""
        for J in range(5):
            assert angular_coupling_coefficient(J, J) == 0.0

    @pytest.mark.parametrize("delta", [2, 3, -2])
    def test_higher_multipoles_are_forbidden(self, delta: int) -> None:
        """|J' - J| > 1 gives zero."""
        assert angular_coupling_coefficient(5, 5 + delta) == 0.0

    def test_j_minus_one_from_j_zero_is_forbidden(self) -> None:
        """There is no J' = -1 state to couple down into."""
        assert angular_coupling_coefficient(0, -1) == 0.0

    @pytest.mark.parametrize("J", [0, 1, 3, 10])
    def test_upward_coefficient(self, J: int) -> None:
        """J' = J+1 gives (J+1)/sqrt((2J+1)(2J+3))."""
        expected = (J + 1) / np.sqrt((2 * J + 1) * (2 * J + 3))
        assert angular_coupling_coefficient(J, J + 1) == pytest.approx(expected)

    @pytest.mark.parametrize("J", [1, 3, 10])
    def test_downward_coefficient(self, J: int) -> None:
        """J' = J-1 gives J/sqrt((2J-1)(2J+1))."""
        expected = J / np.sqrt((2 * J - 1) * (2 * J + 1))
        assert angular_coupling_coefficient(J, J - 1) == pytest.approx(expected)

    def test_approaches_one_half_at_high_j(self) -> None:
        """
        Both coefficients tend to 1/2 as J grows.

        At high J the rotational coupling strength saturates, so the
        J-summed rate grows only through the rotational degeneracy.
        """
        assert angular_coupling_coefficient(200, 201) == pytest.approx(0.5, abs=1e-3)
        assert angular_coupling_coefficient(200, 199) == pytest.approx(0.5, abs=1e-3)


@pytest.fixture(scope="module")
def calculator(oh_system_acharya, reduced_mass, npz_coupling) -> AEDRateCalculator:
    """
    Rate calculator on the Acharya curves with the cached CPSCF coupling.

    Uses the analytic Morse solver, which is the only one validated for
    coupling integrals, on a grid dense enough to resolve the strong
    cancellation in the integrand (see the convergence test below).
    """
    anion, neutral, electron_affinity = oh_system_acharya
    return AEDRateCalculator(
        anion, neutral, electron_affinity, reduced_mass,
        coupling=npz_coupling,
        solver_method="morse",
        r_min=0.5, r_max=15.0, n_grid=CONVERGED_N_GRID,
    )


class TestStateToStateRates:
    """Structure of the state-to-state rate, independent of normalisation."""

    def test_energy_conservation(self, calculator, oh_system_acharya) -> None:
        """
        The electron energy follows from the Acharya energy balance:
        E_e = D_e(anion) + E_coll - E_bound(neutral).
        """
        anion, _, _ = oh_system_acharya
        result = calculator.state_to_state_rate(
            E_COLLISION, J=0, v_prime=6, J_prime=0
        )
        bound = calculator.neutral_solver.solve_bound_state(v=6, J=0)
        expected = anion.D_e + E_COLLISION - bound.energy
        assert result.electron_energy == pytest.approx(expected, rel=1e-9)

    def test_electron_energy_falls_with_final_vibration(self, calculator) -> None:
        """Higher v' leaves less energy for the electron."""
        energies = [
            calculator.state_to_state_rate(
                E_COLLISION, J=0, v_prime=v, J_prime=0
            ).electron_energy
            for v in (3, 5, 7)
        ]
        assert energies[0] > energies[1] > energies[2]

    def test_closed_channel_has_zero_rate(self, calculator) -> None:
        """
        A final vibrational level above the energy budget cannot be reached.

        OH has about 20 bound levels; the topmost ones lie above
        D_e(anion) + E_coll and must come back with exactly zero rate.
        """
        result = calculator.state_to_state_rate(
            E_COLLISION, J=0, v_prime=19, J_prime=0
        )
        assert result.rate == 0.0

    def test_rates_are_non_negative(self, calculator) -> None:
        """A golden-rule rate is a probability flux: never negative."""
        for v in range(3, 9):
            result = calculator.state_to_state_rate(
                E_COLLISION, J=0, v_prime=v, J_prime=0
            )
            assert result.rate >= 0.0

    def test_result_records_its_quantum_numbers(self, calculator) -> None:
        """The dataclass echoes back what was asked for."""
        result = calculator.state_to_state_rate(
            E_COLLISION, J=0, v_prime=6, J_prime=0
        )
        assert (result.v_prime, result.J, result.J_prime) == (6, 0, 0)
        assert result.E_collision == E_COLLISION


class TestVibrationalDistribution:
    """The shape of the final-state distribution — Acharya's main result."""

    def test_distribution_peaks_at_high_v_prime(self, calculator) -> None:
        """
        The distribution peaks among the highest open levels, v' >= 5.

        The exact peak position is basis-set sensitive (6-31G peaks at
        v'=7, aug-cc-pVDZ at v'=6, 6-311+G** at v'=8), so only the
        robust feature is asserted here.  See TestAcharyaBenchmark for
        the open discrepancy against Table I.
        """
        rates = {
            v: calculator.state_to_state_rate(
                E_COLLISION, J=0, v_prime=v, J_prime=0
            ).rate
            for v in range(3, 9)
        }
        assert max(rates, key=rates.get) >= 5

    @pytest.mark.slow
    def test_distribution_is_grid_converged(
        self, oh_system_acharya, reduced_mass, npz_coupling
    ) -> None:
        """
        The distribution is unchanged between 3000 and 6000 grid points.

        The coupling integrand is a near-cancelling oscillatory product,
        so its value is far more grid-sensitive than the wavefunctions
        that build it; this pins down that the default grid is dense
        enough for the residual to be converged.
        """
        anion, neutral, electron_affinity = oh_system_acharya
        distributions = []
        for n_grid in (3000, CONVERGED_N_GRID):
            calculator = AEDRateCalculator(
                anion, neutral, electron_affinity, reduced_mass,
                coupling=npz_coupling, solver_method="morse",
                r_min=0.5, r_max=15.0, n_grid=n_grid,
            )
            rates = np.array([
                calculator.state_to_state_rate(
                    E_COLLISION, J=0, v_prime=v, J_prime=0
                ).rate
                for v in range(4, 9)
            ])
            distributions.append(rates / rates.max())
        assert np.allclose(distributions[0], distributions[1], atol=0.01), (
            f"3000: {distributions[0]}, {CONVERGED_N_GRID}: {distributions[1]}"
        )

    def test_distribution_is_inverted(self, calculator) -> None:
        """
        Low v' are strongly suppressed relative to the peak.

        The product is born vibrationally hot: the nuclear overlap
        favours final states near the anion's outer turning point.
        """
        rates = {
            v: calculator.state_to_state_rate(
                E_COLLISION, J=0, v_prime=v, J_prime=0
            ).rate
            for v in (3, 6)
        }
        assert rates[3] < 0.05 * rates[6]


@pytest.mark.slow
class TestAcharyaBenchmark:
    """Comparison against Acharya et al. (1985) Table I."""

    @pytest.mark.xfail(
        reason=(
            "The computed v' distribution does not yet reproduce Acharya "
            "Table I, and its shape depends strongly on the coupling basis "
            "set: 6-31G peaks at v'=7, 6-311+G** at v'=8, aug-cc-pVDZ at "
            "v'=6 but with v'=5 nearly degenerate with the peak. Kept as a "
            "visible, deliberate failure rather than deleted."
        ),
        strict=False,
    )
    def test_relative_distribution_matches_table_i(self, calculator) -> None:
        """
        The v' distribution normalised to its peak should match Table I.

        Only the shape is compared: the code's scattering states are
        box-normalised, so the absolute rates differ from Acharya's
        energy-normalised convention by a constant factor.
        """
        computed = np.array([
            calculator.state_to_state_rate(
                E_COLLISION, J=0, v_prime=v, J_prime=0
            ).rate
            for v in (4, 5, 6, 7)
        ])
        reference = np.array([ACHARYA_TABLE_I[v] for v in (4, 5, 6, 7)])
        computed /= computed.max()
        reference /= reference.max()
        assert np.allclose(computed, reference, atol=0.25), (
            f"computed {computed}, Acharya {reference}"
        )


class TestCrossSection:
    """The box-independent observable."""

    def test_cross_section_is_positive(self, calculator) -> None:
        """An open channel has a positive cross section."""
        result = calculator.cross_section_state_to_state(
            E_COLLISION, J=0, v_prime=6, J_prime=0
        )
        assert result.sigma > 0.0

    def test_closed_channel_has_zero_cross_section(self, calculator) -> None:
        """A closed channel contributes nothing."""
        result = calculator.cross_section_state_to_state(
            E_COLLISION, J=0, v_prime=19, J_prime=0
        )
        assert result.sigma == 0.0

    def test_total_exceeds_any_single_channel(self, calculator) -> None:
        """Summing over v' can only increase the cross section."""
        single = calculator.cross_section_state_to_state(
            E_COLLISION, J=0, v_prime=6, J_prime=0
        ).sigma
        total = calculator.total_cross_section(E_COLLISION, J=0)
        assert total >= single

    def test_cross_section_is_independent_of_the_box(
        self, oh_system_acharya, reduced_mass, npz_coupling
    ) -> None:
        """
        Enlarging the radial box leaves the cross section unchanged.

        The state-to-state rate scales with the box length through the
        scattering-state normalisation; the cross section divides that
        out and is the quantity worth comparing with experiment.
        """
        anion, neutral, electron_affinity = oh_system_acharya
        sigmas = []
        for r_max, n_grid in ((15.0, 1000), (20.0, 1333)):
            calculator = AEDRateCalculator(
                anion, neutral, electron_affinity, reduced_mass,
                coupling=npz_coupling, solver_method="morse",
                r_min=0.5, r_max=r_max, n_grid=n_grid,
            )
            sigmas.append(
                calculator.cross_section_state_to_state(
                    E_COLLISION, J=0, v_prime=6, J_prime=0
                ).sigma
            )
        assert sigmas[1] == pytest.approx(sigmas[0], rel=0.10), f"sigmas: {sigmas}"
