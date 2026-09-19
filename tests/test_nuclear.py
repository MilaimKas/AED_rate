"""
Tests for the nuclear radial wavefunction solvers (DVR, Numerov, Morse).

The physical content checked here is solver-independent: bound states must
be normalised, carry v nodes, and reproduce the Morse spectrum; scattering
states must oscillate in the asymptotic region with a finite phase shift.
"""

import numpy as np
import pytest
from scipy.integrate import simpson

from aed_rate.nuclear.nuclear_wavefunction import create_wavefunction_solver
from aed_rate.utils.constants import CONSTANTS

# Grid shared by every solver in this module: wide enough for the asymptotic
# region, coarse enough to keep the DVR diagonalisation fast.
GRID = dict(r_min=0.5, r_max=15.0, n_grid=500)


def count_nodes(wavefunction: np.ndarray, threshold: float = 1e-6) -> int:
    """
    Count sign changes of a wavefunction, ignoring its numerical noise floor.

    Points below `threshold` times the peak amplitude are dropped first:
    in the classically forbidden tails the amplitude underflows and its
    sign flips randomly, which would otherwise swamp the true node count.
    """
    significant = wavefunction[np.abs(wavefunction) > threshold * np.max(np.abs(wavefunction))]
    return int(np.sum(np.diff(np.sign(significant)) != 0))


@pytest.fixture(scope="module")
def neutral_solver(oh_system, reduced_mass):
    """DVR solver on the neutral OH curve."""
    _, neutral, _ = oh_system
    return create_wavefunction_solver(neutral, reduced_mass, method="dvr", **GRID)


@pytest.fixture(scope="module")
def anion_solver(oh_system, reduced_mass):
    """DVR solver on the OH⁻ curve, used for the scattering states."""
    anion, _, _ = oh_system
    return create_wavefunction_solver(anion, reduced_mass, method="dvr", **GRID)


@pytest.fixture(scope="module")
def scattering_state(anion_solver):
    """A 500 cm⁻¹ collision-energy s-wave scattering state of OH⁻."""
    return anion_solver.solve_scattering_state(CONSTANTS.cm1_to_hartree(500.0), J=0)


@pytest.fixture(scope="module")
def neutral_bound_states(neutral_solver):
    """All J=0 bound vibrational states of neutral OH."""
    return neutral_solver.solve_all_bound_states(J=0)


class TestSolverFactory:
    """Dispatch behaviour of create_wavefunction_solver."""

    @pytest.mark.parametrize("method", ["dvr", "numerov", "morse"])
    def test_known_methods_construct(self, oh_system, reduced_mass, method) -> None:
        """Each documented method name returns a solver."""
        _, neutral, _ = oh_system
        solver = create_wavefunction_solver(
            neutral, reduced_mass, method=method, **GRID
        )
        assert solver.r_grid.shape == (GRID["n_grid"],)

    def test_unknown_method_raises(self, oh_system, reduced_mass) -> None:
        """An unrecognised method name is rejected, not silently defaulted."""
        _, neutral, _ = oh_system
        with pytest.raises(ValueError, match="Unknown method"):
            create_wavefunction_solver(neutral, reduced_mass, method="shooting")


class TestBoundStates:
    """Bound vibrational states from the DVR solver."""

    def test_bound_state_count(self, neutral_bound_states) -> None:
        """Neutral OH supports roughly 20 bound levels."""
        assert 15 < len(neutral_bound_states) < 30

    def test_fundamental_transition(self, neutral_bound_states) -> None:
        """The v=0 -> v=1 spacing is near 3570 cm⁻¹."""
        spacing = CONSTANTS.hartree_to_cm1 * (
            neutral_bound_states[1].energy - neutral_bound_states[0].energy
        )
        assert spacing == pytest.approx(3570, rel=0.10)

    def test_energies_increase_with_v(self, neutral_bound_states) -> None:
        """Levels come back sorted, with v matching the list index."""
        energies = [state.energy for state in neutral_bound_states]
        assert energies == sorted(energies)
        assert [state.v for state in neutral_bound_states] == list(
            range(len(neutral_bound_states))
        )

    @pytest.mark.parametrize("v", [0, 3])
    def test_normalisation(self, neutral_bound_states, v: int) -> None:
        """Bound states integrate to unity on the radial grid."""
        state = neutral_bound_states[v]
        norm = simpson(state.wavefunction**2, x=state.r_grid)
        assert norm == pytest.approx(1.0, abs=0.01)

    @pytest.mark.parametrize("v", [0, 1, 2, 5])
    def test_node_count(self, neutral_bound_states, v: int) -> None:
        """The v-th vibrational state has exactly v interior nodes."""
        assert count_nodes(neutral_bound_states[v].wavefunction) == v

    def test_ground_state_peaks_near_equilibrium(
        self, neutral_bound_states, oh_system
    ) -> None:
        """v=0 has its maximum amplitude close to r_e."""
        _, neutral, _ = oh_system
        state = neutral_bound_states[0]
        r_peak = state.r_grid[np.argmax(np.abs(state.wavefunction))]
        assert r_peak == pytest.approx(neutral.r_eq, abs=0.1)


class TestGridSolverAgainstAnalyticMorse:
    """
    The grid spectrum must converge to the analytic Morse eigenvalues.

    The two routines share no code, so this is the strongest available
    check on the grid solver's kinetic-energy matrix.
    """

    @staticmethod
    def energy_errors(neutral, reduced_mass, n_grid: int, n_levels: int = 4):
        """Grid-minus-analytic level energies, in cm⁻¹, for v = 0..n_levels-1."""
        grid_solver = create_wavefunction_solver(
            neutral, reduced_mass, method="dvr",
            r_min=GRID["r_min"], r_max=GRID["r_max"], n_grid=n_grid,
        )
        morse = create_wavefunction_solver(
            neutral, reduced_mass, method="morse", **GRID
        )
        return np.array([
            CONSTANTS.hartree_to_cm1 * (
                grid_solver.solve_bound_state(v=v, J=0).energy
                - morse.solve_bound_state(v=v, J=0).energy
            )
            for v in range(n_levels)
        ])

    def test_low_lying_levels_agree_on_a_fine_grid(
        self, oh_system, reduced_mass
    ) -> None:
        """At 2000 points the low-lying levels agree to a few cm⁻¹."""
        _, neutral, _ = oh_system
        errors = self.energy_errors(neutral, reduced_mass, n_grid=2000)
        assert np.all(np.abs(errors) < 5.0), f"errors (cm⁻¹): {errors}"

    @pytest.mark.slow
    def test_convergence_is_second_order(self, oh_system, reduced_mass) -> None:
        """
        Halving the grid spacing cuts the error by a factor of ~4.

        The solver is named "DVR" but its kinetic operator is the 3-point
        finite-difference Laplacian, so the error is O(dr^2) rather than
        the exponential convergence of a true sinc-DVR.  This test pins
        that behaviour down: if the kinetic matrix is ever upgraded, it
        will fail and should be updated deliberately.
        """
        _, neutral, _ = oh_system
        coarse = self.energy_errors(neutral, reduced_mass, n_grid=500)
        fine = self.energy_errors(neutral, reduced_mass, n_grid=1000)
        ratios = np.abs(coarse) / np.abs(fine)
        assert np.all(np.abs(ratios - 4.0) < 0.5), f"error ratios: {ratios}"


class TestScatteringStates:
    """Continuum states of the anion curve."""

    def test_normalisation(self, scattering_state) -> None:
        """The DVR scattering state is box-normalised to unity."""
        norm = simpson(scattering_state.wavefunction**2, x=scattering_state.r_grid)
        assert norm == pytest.approx(1.0, abs=0.05)

    def test_phase_shift_is_finite(self, scattering_state) -> None:
        """A real, finite phase shift is extracted from the asymptotic tail."""
        assert np.isfinite(scattering_state.phase_shift)

    def test_oscillates_in_asymptotic_region(self, scattering_state) -> None:
        """
        The tail oscillates: beyond the well the state is free, so it must
        cross zero repeatedly rather than decay.
        """
        tail = scattering_state.wavefunction[-200:-50]
        half_periods = int(np.sum(np.diff(np.sign(tail)) != 0))
        assert half_periods >= 3

    def test_wavelength_matches_free_particle(
        self, scattering_state, reduced_mass
    ) -> None:
        """
        The asymptotic node spacing matches the free-particle half
        wavelength pi/k, with k = sqrt(2*mu*E).

        Far outside the well the anion potential is flat, so the radial
        equation reduces to a free particle and the nodes must be evenly
        spaced by pi/k.
        """
        k_nuclear = np.sqrt(2.0 * reduced_mass * scattering_state.E)
        r_grid = scattering_state.r_grid
        tail_slice = slice(-200, -20)
        tail = scattering_state.wavefunction[tail_slice]
        crossings = np.where(np.diff(np.sign(tail)) != 0)[0]
        node_positions = r_grid[tail_slice][crossings]
        mean_spacing = float(np.mean(np.diff(node_positions)))
        assert mean_spacing == pytest.approx(np.pi / k_nuclear, rel=0.05)

    def test_higher_energy_oscillates_faster(self, anion_solver) -> None:
        """Raising the collision energy shortens the asymptotic wavelength."""
        node_counts = []
        for energy_cm1 in (200.0, 2000.0):
            state = anion_solver.solve_scattering_state(
                CONSTANTS.cm1_to_hartree(energy_cm1), J=0
            )
            node_counts.append(count_nodes(state.wavefunction[-200:]))
        assert node_counts[1] > node_counts[0]


class TestWavefunctionDerivative:
    """The dF/dR used by the coupling integrand."""

    def test_derivative_vanishes_at_the_peak(self, neutral_solver, neutral_bound_states) -> None:
        """dF/dR is small where the wavefunction is extremal."""
        state = neutral_bound_states[0]
        derivative = neutral_solver.wavefunction_derivative(state)
        peak = int(np.argmax(np.abs(state.wavefunction)))
        assert abs(derivative[peak]) < 0.5 * np.max(np.abs(derivative))

    def test_derivative_matches_numpy_gradient(
        self, neutral_solver, neutral_bound_states
    ) -> None:
        """The solver's 3-point derivative agrees with np.gradient in the bulk."""
        state = neutral_bound_states[0]
        derivative = neutral_solver.wavefunction_derivative(state)
        reference = np.gradient(state.wavefunction, state.r_grid)
        bulk = slice(5, -5)
        assert np.allclose(derivative[bulk], reference[bulk], atol=1e-6)
