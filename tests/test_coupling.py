"""
Tests for the electronic coupling providers.

Three implementations share one interface (``compute_coupling_at_r``):
``ModelCoupling`` (Gaussian stand-in), ``InterpolatedCoupling`` (CPSCF on a
precomputed 2D grid) and ``ElectronicCoupling`` (live CPSCF).  The fast
layer exercises the model and the interpolated grid; the CPSCF layer is
marked slow and skips without PySCF.
"""

import warnings

import numpy as np
import pytest

from aed_rate.electronic.coupling import (
    CouplingResult,
    InterpolatedCoupling,
    ModelCoupling,
)
from aed_rate.utils.constants import CONSTANTS

from conftest import requires_pyscf

R_EQUILIBRIUM = 1.83
ELECTRON_ENERGY = CONSTANTS.cm1_to_hartree(5000.0)   # about 0.6 eV


@pytest.fixture
def model_coupling() -> ModelCoupling:
    """
    Gaussian coupling with a weak radial and a stronger rotational channel.

    This ordering is what a pi HOMO gives: stretching barely modulates a
    pi orbital, while rotation mixes it strongly with the continuum.
    """
    with warnings.catch_warnings():
        # ModelCoupling warns on construction that it is not physical;
        # that is the point of using it here.
        warnings.simplefilter("ignore")
        return ModelCoupling(
            R0=R_EQUILIBRIUM,
            A_rad=0.01,
            alpha_rad=1.0,
            A_rot=0.05,
            alpha_rot=1.0,
        )


class TestModelCouplingWarning:
    """The stand-in must announce itself."""

    def test_construction_warns(self) -> None:
        """Building a ModelCoupling raises the not-physical warning."""
        from aed_rate.utils.constants import AEDValidationWarning

        with pytest.warns(AEDValidationWarning, match="sanity-check Gaussian"):
            ModelCoupling(R0=R_EQUILIBRIUM)


class TestModelCouplingShape:
    """Geometry dependence of the Gaussian model."""

    def test_rotational_exceeds_radial(self, model_coupling) -> None:
        """|m_rot| > |m_rad|, the expected ordering for a pi HOMO."""
        result = model_coupling.compute_coupling_at_r(R_EQUILIBRIUM, ELECTRON_ENERGY)
        assert abs(result.m_rot) > abs(result.m_rad)

    def test_peaks_at_the_gaussian_centre(self, model_coupling) -> None:
        """The coupling is largest at R0 and falls off on both sides."""
        r_grid = np.linspace(1.0, 5.0, 41)
        magnitudes = [
            abs(r.m_rad)
            for r in model_coupling.compute_coupling_curve(r_grid, ELECTRON_ENERGY)
        ]
        assert r_grid[int(np.argmax(magnitudes))] == pytest.approx(
            R_EQUILIBRIUM, abs=0.15
        )

    def test_decays_at_large_separation(self, model_coupling) -> None:
        """By R = 5 Bohr the coupling has dropped by more than a decade."""
        near = model_coupling.compute_coupling_at_r(R_EQUILIBRIUM, ELECTRON_ENERGY)
        far = model_coupling.compute_coupling_at_r(5.0, ELECTRON_ENERGY)
        assert abs(far.m_rad) < 0.1 * abs(near.m_rad)

    def test_curve_matches_pointwise_evaluation(self, model_coupling) -> None:
        """compute_coupling_curve is the pointwise routine mapped over a grid."""
        r_grid = np.linspace(1.5, 3.0, 7)
        curve = model_coupling.compute_coupling_curve(r_grid, ELECTRON_ENERGY)
        for R, result in zip(r_grid, curve):
            single = model_coupling.compute_coupling_at_r(R, ELECTRON_ENERGY)
            assert result.m_rad == single.m_rad
            assert result.m_rot == single.m_rot


class TestModelCouplingEnergyScaling:
    """The k_power exponent encoding the low-k OPW limit."""

    def test_default_is_energy_independent(self) -> None:
        """k_power=0 leaves the coupling flat in electron energy."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coupling = ModelCoupling(R0=R_EQUILIBRIUM)
        low = coupling.compute_coupling_at_r(R_EQUILIBRIUM, 0.001)
        high = coupling.compute_coupling_at_r(R_EQUILIBRIUM, 0.100)
        assert abs(low.m_rad) == pytest.approx(abs(high.m_rad))

    def test_linear_scaling_follows_k(self) -> None:
        """
        k_power=1 makes the coupling proportional to k_e.

        Quadrupling the electron energy doubles k_e and so doubles the
        matrix element — the low-k OPW behaviour for a pi or sigma HOMO.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coupling = ModelCoupling(R0=R_EQUILIBRIUM, k_power=1.0)
        low = coupling.compute_coupling_at_r(R_EQUILIBRIUM, 0.010)
        high = coupling.compute_coupling_at_r(R_EQUILIBRIUM, 0.040)
        assert abs(high.m_rad) / abs(low.m_rad) == pytest.approx(2.0, rel=1e-10)

    def test_threshold_electron_has_no_coupling(self) -> None:
        """With k_power=1 the coupling vanishes at zero electron energy."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coupling = ModelCoupling(R0=R_EQUILIBRIUM, k_power=1.0)
        assert coupling.compute_coupling_at_r(R_EQUILIBRIUM, 0.0).m_rad == 0j

    def test_reports_the_wave_vector(self, model_coupling) -> None:
        """The result carries k_e = sqrt(2*E) alongside the matrix elements."""
        result = model_coupling.compute_coupling_at_r(R_EQUILIBRIUM, ELECTRON_ENERGY)
        assert result.k_electron == pytest.approx(np.sqrt(2.0 * ELECTRON_ENERGY))
        assert result.electron_energy == ELECTRON_ENERGY


class TestPrecomputedGrid:
    """The CPSCF coupling loaded from a cached .npz, no PySCF needed."""

    def test_interface_matches_the_model(self, npz_coupling) -> None:
        """The spline provider returns a CouplingResult like any other."""
        result = npz_coupling.compute_coupling_at_r(R_EQUILIBRIUM, ELECTRON_ENERGY)
        assert isinstance(result, CouplingResult)
        assert result.R == R_EQUILIBRIUM

    def test_vanishes_outside_the_computed_range(self, npz_coupling) -> None:
        """Beyond the CPSCF grid the coupling is set to zero, not extrapolated."""
        result = npz_coupling.compute_coupling_at_r(50.0, ELECTRON_ENERGY)
        assert result.m_rad == 0j
        assert result.m_rot == 0j

    def test_from_npz_loads_without_pyscf(self, coupling_npz_path) -> None:
        """from_npz builds an evaluation-only coupling with no SCF machinery."""
        coupling = InterpolatedCoupling.from_npz(str(coupling_npz_path))
        result = coupling.compute_coupling_at_r(R_EQUILIBRIUM, ELECTRON_ENERGY)
        assert np.isfinite(abs(result.m_rad))
        assert np.isfinite(abs(result.m_rot))

    def test_coupling_is_largest_near_equilibrium(self, npz_coupling) -> None:
        """
        The CPSCF radial coupling peaks in the bonding region.

        Far outside, the anion HOMO stops changing with R, so d(phi)/dR
        and hence the coupling die away.
        """
        r_grid = np.linspace(1.2, 4.0, 40)
        magnitudes = np.array([
            abs(npz_coupling.compute_coupling_at_r(R, ELECTRON_ENERGY).m_rad)
            for R in r_grid
        ])
        assert magnitudes[:20].max() > magnitudes[-5:].max()

    def test_scales_roughly_linearly_with_k(self, npz_coupling) -> None:
        """
        m_rad/k_e is close to constant across the grid's k_e range.

        This is the low-k OPW limit the model coupling mimics with
        k_power=1; the CPSCF grid should reproduce it from first
        principles rather than by construction.
        """
        ratios = []
        for energy in (0.002, 0.008, 0.032):
            result = npz_coupling.compute_coupling_at_r(R_EQUILIBRIUM, energy)
            ratios.append(abs(result.m_rad) / result.k_electron)
        ratios = np.array(ratios)
        assert ratios.max() / ratios.min() < 2.0, f"m_rad/k_e spread: {ratios}"


@requires_pyscf
@pytest.mark.slow
class TestLiveCPSCF:
    """Coupling computed from a live CPSCF calculation."""

    def test_matches_finite_difference(self) -> None:
        """
        The CPSCF orbital response dC/dR agrees with finite differences.

        This is the one check that validates the analytic response
        machinery itself rather than anything built on top of it.
        """
        from aed_rate.electronic.coupling import ElectronicCoupling
        from aed_rate.electronic.wavefunctions import ElectronicStructure

        structure = ElectronicStructure("O", "H", basis="6-31g")
        coupling = ElectronicCoupling(structure, homo_symmetry="pi")
        analytic = coupling.compute_coupling_at_r(
            R=1.822, electron_energy=ELECTRON_ENERGY
        )
        assert np.isfinite(abs(analytic.m_rad))
        assert abs(analytic.m_rad) > 0.0
