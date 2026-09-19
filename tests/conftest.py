"""
Shared fixtures for the aed_rate test suite.

The suite is split into a fast default layer (numpy/scipy only) and an
opt-in slow layer that needs PySCF or a precomputed coupling file.  Tests
in the slow layer skip themselves rather than fail when their prerequisite
is missing, so a bare ``pytest`` run is always green on a clean checkout.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pytest

from aed_rate.electronic.potential import (
    MorsePotential,
    create_oh_system,
    create_oh_system_acharya,
)
from aed_rate.utils.constants import get_reduced_mass
from aed_rate.utils.paths import data_file

OHSystem = Tuple[MorsePotential, MorsePotential, float]


def pyscf_available() -> bool:
    """Report whether PySCF can be imported in the current environment."""
    try:
        import pyscf  # noqa: F401
    except ImportError:
        return False
    return True


requires_pyscf = pytest.mark.skipif(
    not pyscf_available(), reason="PySCF is not installed"
)


@pytest.fixture(scope="session")
def reduced_mass() -> float:
    """Reduced mass of OH in atomic units (electron masses)."""
    return get_reduced_mass("O", "H")


@pytest.fixture(scope="session")
def oh_system() -> OHSystem:
    """Spectroscopic OH⁻/OH Morse pair and the electron affinity, in Hartree."""
    return create_oh_system()


@pytest.fixture(scope="session")
def oh_system_acharya() -> OHSystem:
    """OH⁻/OH Morse pair with the exact parameters of Acharya et al. (1985)."""
    return create_oh_system_acharya()


@pytest.fixture(scope="session")
def coupling_npz_path() -> Path:
    """
    Path to the 6-31G CPSCF coupling grid, skipping the test if absent.

    The .npz grids are generated artefacts (gitignored) living under
    ``data/``; regenerate them with ``scripts/precompute_coupling.py``.
    """
    path = data_file("oh_minus_coupling_6-31g.npz")
    if not path.exists():
        pytest.skip(
            f"{path.name} not found — run scripts/precompute_coupling.py first"
        )
    return path


@pytest.fixture(scope="session")
def npz_coupling(coupling_npz_path: Path):
    """
    Coupling provider backed by the precomputed 2D spline grid, no PySCF.

    Mirrors the ``InterpolatedCoupling`` interface used by
    ``AEDRateCalculator``: only ``compute_coupling_at_r`` is required.
    """
    from scipy.interpolate import RectBivariateSpline

    from aed_rate.electronic.coupling import CouplingResult

    grid = np.load(coupling_npz_path)
    k_e_grid = grid["k_e_grid"]
    R_min = float(grid["R_min"][0])
    R_cutoff = float(grid["R_cutoff"][0])
    spline_rad = RectBivariateSpline(
        grid["R_grid"], k_e_grid, grid["m_rad_2d"], kx=3, ky=3
    )
    spline_rot = RectBivariateSpline(
        grid["R_grid"], k_e_grid, grid["m_rot_2d"], kx=3, ky=3
    )

    class SplineCoupling:
        """Minimal coupling provider interpolating a precomputed m(R, k_e)."""

        def compute_coupling_at_r(
            self, R: float, electron_energy: float, **_
        ) -> CouplingResult:
            """Interpolate m_rad and m_rot; return zero outside the R grid."""
            k_e = float(np.sqrt(max(2.0 * electron_energy, 0.0)))
            if R < R_min or R > R_cutoff:
                return CouplingResult(
                    R=R,
                    m_rad=0j,
                    m_rot=0j,
                    electron_energy=electron_energy,
                    k_electron=k_e,
                )
            # The CPSCF grid spans a finite k_e range; clamp rather than
            # extrapolate, since the spline diverges quickly outside it.
            k_e_clamped = float(np.clip(k_e, k_e_grid[0], k_e_grid[-1]))
            return CouplingResult(
                R=R,
                m_rad=complex(float(spline_rad(R, k_e_clamped, grid=False))),
                m_rot=complex(float(spline_rot(R, k_e_clamped, grid=False))),
                electron_energy=electron_energy,
                k_electron=k_e,
            )

    return SplineCoupling()
