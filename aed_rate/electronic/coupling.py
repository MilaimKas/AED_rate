"""
Electronic non-Born-Oppenheimer coupling for AED.

Computes the electronic coupling matrix elements m_rad(R) and m_rot(R)
that enter the Fermi Golden Rule rate expression:

    m_rad(R) = <phi_k | d(phi_HOMO)/dR >       (radial / vibrational)
    m_rot(R) = <phi_k | (1/R) d(phi_HOMO)/dtheta >  (rotational)

where phi_HOMO is the anion's detaching orbital and phi_k is the
continuum electron described by an orthogonalized plane wave (OPW).

The MO derivative d(phi_HOMO)/dR is computed analytically via PySCF's
coupled-perturbed SCF (CPSCF) infrastructure, avoiding finite differences.

References
----------
[1] Acharya, Kendall, Simons, JACS 106, 3402 (1984)
[2] Acharya, Das, Simons, JCP 83, 3888 (1985)
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass

try:
    from pyscf import dft
    PYSCF_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False

from .wavefunctions import ElectronicStructure
from .continuum import ContinuumOrbital


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class CouplingResult:
    """Result of electronic coupling calculation at one geometry."""

    R: float                    # bond length (Bohr)
    m_rad: complex              # radial coupling matrix element
    m_rot: complex              # rotational coupling matrix element
    electron_energy: float      # continuum electron kinetic energy (Hartree)
    k_electron: float           # electron wave vector magnitude (a.u.)


# ---------------------------------------------------------------------------
# Ab initio coupling via CPSCF
# ---------------------------------------------------------------------------

class ElectronicCoupling:
    """
    Ab initio non-BO electronic coupling using CPSCF.

    Uses PySCF's analytical coupled-perturbed SCF to compute
    MO coefficient derivatives, then overlaps with OPW continuum
    orbitals on a Becke integration grid.

    Parameters
    ----------
    electronic_structure : ElectronicStructure
        Electronic structure calculator (provides SCF objects)
    homo_symmetry : str
        Symmetry of the anion HOMO: 'pi' or 'sigma'.
        Controls the angular form of the low-k OPW.
    grid_level : int
        PySCF Becke grid quality (0-9). Default 3 (medium).
    k_switchover : float
        When k_e * r_e > k_switchover, use full numerical OPW
        instead of analytical low-k form. Default 0.5.
    """

    def __init__(
        self,
        electronic_structure: ElectronicStructure,
        homo_symmetry: str = "pi",
        grid_level: int = 3,
        k_switchover: float = 0.5,
    ):
        if not PYSCF_AVAILABLE:
            raise ImportError("PySCF is required for ab initio coupling.")

        self.es = electronic_structure
        self.homo_symmetry = homo_symmetry
        self.grid_level = grid_level
        self.k_switchover = k_switchover

        # Cache: R -> (mol, mf, mo1, homo_idx)
        self._cpscf_cache: dict = {}

    # ------------------------------------------------------------------
    # CPSCF: solve once per geometry, extract all derivatives
    # ------------------------------------------------------------------

    def _solve_cpscf(
        self, R: float, charge: int = -1, spin: int = 0,
    ) -> Tuple:
        """
        Run SCF + CPSCF at bond length R and cache the result.

        Returns (mol, mf, mo1, homo_idx) where mo1 has shape
        (natm, 3, nao, nocc) — derivatives of occupied MO coefficients
        with respect to each atom's Cartesian displacement.
        """
        if R in self._cpscf_cache:
            return self._cpscf_cache[R]

        mol, mf = self.es._run_scf(R, charge=charge, spin=spin)
        homo_idx = self.es._get_homo_index(mf)

        # Solve coupled-perturbed SCF equations
        hess = mf.Hessian()
        h1ao = hess.make_h1(mf.mo_coeff, mf.mo_occ)
        mo1, _mo_e1 = hess.solve_mo1(
            mf.mo_energy, mf.mo_coeff, mf.mo_occ, h1ao,
        )
        mo1 = np.array(mo1)  # (natm, 3, nao, nocc)

        result = (mol, mf, mo1, homo_idx)
        self._cpscf_cache[R] = result
        return result

    # ------------------------------------------------------------------
    # Evaluate HOMO derivative on a real-space grid
    # ------------------------------------------------------------------

    def _evaluate_homo_derivatives(
        self, R: float, charge: int = -1, spin: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate d(phi_HOMO)/dR and d(phi_HOMO)/dtheta on a Becke grid.

        For a diatomic A-B along z (A at origin, B at z=R):
        - Radial:      dC/dR     = mo1[1, 2, :, :]   (atom B, z-direction)
        - Rotational:  dC/dtheta = R * mo1[1, 0, :, :] (atom B, x-direction * R)

        The rotational identity comes from: for a diatomic along z,
        an infinitesimal rotation dtheta displaces atom B by dx_B = R*dtheta,
        so d/dtheta = R * d/dx_B.

        Returns
        -------
        coords : np.ndarray
            Grid coordinates (N, 3)
        weights : np.ndarray
            Integration weights (N,)
        dphi_dR : np.ndarray
            Radial derivative of HOMO on grid (N,)
        dphi_dtheta : np.ndarray
            Angular derivative of HOMO on grid (N,)
        """
        mol, mf, mo1, homo_idx = self._solve_cpscf(R, charge, spin)

        # Build Becke integration grid
        grids = dft.gen_grid.Grids(mol)
        grids.level = self.grid_level
        grids.build()

        # Evaluate all AOs on the grid: (N_grid, N_ao)
        ao_values = mol.eval_gto("GTOval_sph", grids.coords)

        # Extract MO coefficient derivatives for HOMO
        # mo1 shape: (natm, 3, nao, nocc)
        # Radial: atom B (idx 1), z-direction (idx 2)
        dC_dR = mo1[1, 2, :, homo_idx]           # (nao,)
        # Rotational: atom B (idx 1), x-direction (idx 0), scaled by R
        dC_dtheta = R * mo1[1, 0, :, homo_idx]   # (nao,)

        # Contract AO values with coefficient derivatives
        dphi_dR = ao_values @ dC_dR               # (N_grid,)
        dphi_dtheta = ao_values @ dC_dtheta        # (N_grid,)

        return grids.coords, grids.weights, dphi_dR, dphi_dtheta

    # ------------------------------------------------------------------
    # OPW continuum orbital evaluation
    # ------------------------------------------------------------------

    def _evaluate_opw_on_grid(
        self,
        R: float,
        coords: np.ndarray,
        electron_energy: float,
        symmetry: str = "same",
        charge: int = -1,
        spin: int = 0,
    ) -> np.ndarray:
        """
        Evaluate the OPW continuum orbital on a Becke grid.

        Uses analytical low-k forms (Acharya 1983 Eqs. 5-6) when
        k*R < k_switchover, otherwise falls back to full numerical OPW.

        Parameters
        ----------
        R : float
            Bond length (Bohr), used for switchover criterion
        coords : np.ndarray
            Grid coordinates (N, 3)
        electron_energy : float
            Continuum electron kinetic energy (Hartree)
        symmetry : str
            OPW symmetry to use: 'pi', 'sigma', or 'same' (use HOMO symmetry).
            Important: the radial derivative preserves HOMO symmetry, but the
            rotational derivative swaps it (pi <-> sigma). The caller must
            select the correct symmetry for each coupling term.

        Returns
        -------
        np.ndarray
            OPW values on grid (N,), complex
        """
        k_e = np.sqrt(2.0 * electron_energy)

        if symmetry == "same":
            symmetry = self.homo_symmetry

        if k_e * R < self.k_switchover:
            return self._evaluate_opw_low_k(coords, k_e, symmetry)
        else:
            return self._evaluate_opw_full(
                R, coords, electron_energy, charge, spin,
            )

    @staticmethod
    def _evaluate_opw_low_k(
        coords: np.ndarray, k_e: float, symmetry: str,
    ) -> np.ndarray:
        """
        Analytical low-k OPW from Acharya 1983.

        In the long-wavelength limit (k*R_mol << 1), only the leading
        partial wave survives. The L^{-3/2} normalization from the
        box-normalized plane wave cancels with L^3 in the density of
        states, so we omit it here.

        For pi symmetry:   phi_k ~ k_x * x   [1, Eq. 6]
        For sigma symmetry: phi_k ~ k_z * z   [1, Eq. 5]

        Parameters
        ----------
        coords : np.ndarray
            Grid coordinates (N, 3)
        k_e : float
            Electron wave vector magnitude
        symmetry : str
            'pi' or 'sigma'

        Returns
        -------
        np.ndarray
            Low-k OPW values (N,), real-valued
        """
        if symmetry == "pi":
            # pi: l=1, m=+/-1 → proportional to x (or y)
            return k_e * coords[:, 0]
        elif symmetry == "sigma":
            # sigma: l=1, m=0 → proportional to z
            return k_e * coords[:, 2]
        else:
            raise ValueError(
                f"Unknown OPW symmetry '{symmetry}'. Use 'pi' or 'sigma'."
            )

    def _evaluate_opw_full(
        self,
        R: float,
        coords: np.ndarray,
        electron_energy: float,
        charge: int = -1,
        spin: int = 0,
    ) -> np.ndarray:
        """
        Full numerical OPW on a Becke grid for high-k cases.

        Constructs a ContinuumOrbital with the neutral molecule's
        occupied MOs for orthogonalization, then evaluates on the grid.
        The neutral SCF uses the same basis as the anion to ensure
        consistent AO dimensions.
        """
        # Build neutral molecule with same basis/geometry
        # Use the anion mol's basis to guarantee AO dimension match
        mol_neutral = self.es._build_molecule(R, charge=0, spin=1)
        from pyscf import scf
        mf_neutral = scf.UHF(mol_neutral)
        mf_neutral.kernel()

        # Use alpha MOs for orthogonalization
        # UHF stores mo_coeff as ndarray (2, nao, nmo) or tuple of arrays
        mo_coeff = mf_neutral.mo_coeff
        mo_occ = mf_neutral.mo_occ
        if isinstance(mo_coeff, np.ndarray) and mo_coeff.ndim == 3:
            mo_coeff = mo_coeff[0]
            mo_occ = mo_occ[0]
        elif isinstance(mo_coeff, (tuple, list)):
            mo_coeff = mo_coeff[0]
            mo_occ = mo_occ[0]

        opw = ContinuumOrbital(
            kinetic_energy=electron_energy,
            mo_coeff=mo_coeff,
            mo_occ=mo_occ,
            overlap_matrix=mol_neutral.intor("int1e_ovlp"),
        )

        ao_values = mol_neutral.eval_gto("GTOval_sph", coords)

        # Default k-direction along z (bond axis)
        return opw.evaluate_opw(coords, ao_values)

    # ------------------------------------------------------------------
    # Coupling matrix elements
    # ------------------------------------------------------------------

    def compute_coupling_at_r(
        self,
        R: float,
        electron_energy: float,
        charge: int = -1,
        spin: int = 0,
    ) -> CouplingResult:
        """
        Compute m_rad(R) and m_rot(R) at a single geometry.

        m_rad = integral[ phi_k^*(r) * d(phi_HOMO)/dR * w(r) ]
        m_rot = integral[ phi_k^*(r) * (1/R) d(phi_HOMO)/dtheta * w(r) ]

        where w(r) are Becke integration weights.

        Parameters
        ----------
        R : float
            Bond length in Bohr
        electron_energy : float
            Kinetic energy of ejected electron (Hartree)
        charge : int
            Anion charge (default -1)
        spin : int
            Anion spin (default 0 for singlet)

        Returns
        -------
        CouplingResult
            Contains m_rad, m_rot, and metadata
        """
        k_e = np.sqrt(2.0 * electron_energy)

        # Get HOMO derivatives on Becke grid
        coords, weights, dphi_dR, dphi_dtheta = (
            self._evaluate_homo_derivatives(R, charge, spin)
        )

        # Radial coupling: ∂φ_HOMO/∂R preserves HOMO symmetry
        # → use same symmetry OPW as the HOMO
        phi_k_rad = self._evaluate_opw_on_grid(
            R, coords, electron_energy,
            symmetry=self.homo_symmetry, charge=charge, spin=spin,
        )

        # Rotational coupling: ∂φ_HOMO/∂θ swaps symmetry (pi <-> sigma)
        # Rotation mixes π_x into σ and vice versa, so the derivative
        # has the complementary symmetry
        rot_symmetry = "sigma" if self.homo_symmetry == "pi" else "pi"
        phi_k_rot = self._evaluate_opw_on_grid(
            R, coords, electron_energy,
            symmetry=rot_symmetry, charge=charge, spin=spin,
        )

        # Numerical integration: m = sum_i w_i * phi_k*(r_i) * dphi(r_i)
        m_rad = np.sum(weights * np.conj(phi_k_rad) * dphi_dR)
        m_rot = np.sum(weights * np.conj(phi_k_rot) * dphi_dtheta / R)

        return CouplingResult(
            R=R,
            m_rad=complex(m_rad),
            m_rot=complex(m_rot),
            electron_energy=electron_energy,
            k_electron=k_e,
        )

    def compute_coupling_curve(
        self,
        R_grid: np.ndarray,
        electron_energy: float,
        charge: int = -1,
        spin: int = 0,
    ) -> List[CouplingResult]:
        """
        Compute coupling matrix elements along a grid of R values.

        Parameters
        ----------
        R_grid : np.ndarray
            Bond lengths in Bohr
        electron_energy : float
            Continuum electron kinetic energy (Hartree)

        Returns
        -------
        List[CouplingResult]
            Coupling results at each R
        """
        results: List[CouplingResult] = []
        for R in R_grid:
            result = self.compute_coupling_at_r(
                R, electron_energy, charge, spin,
            )
            results.append(result)
        return results

    def clear_cache(self) -> None:
        """Clear the CPSCF cache to free memory."""
        self._cpscf_cache.clear()


# ---------------------------------------------------------------------------
# Model (Gaussian) coupling for testing / calibration
# ---------------------------------------------------------------------------

class ModelCoupling:
    """
    Gaussian model for electronic coupling matrix elements.

    Geometric part (R-dependent):
        g_rad(R) = A_rad * exp(-alpha_rad * (R - R0)^2)
        g_rot(R) = A_rot * exp(-alpha_rot * (R - R0)^2)

    Optional electron-energy scaling via ``k_power``:
        m(R, E_e) = g(R) * k_e^k_power,   k_e = sqrt(2 * E_e)

    The ``k_power`` parameter encodes the low-k OPW behaviour.  For a
    π-HOMO in the low-k limit (Acharya 1983), φ_k^π ∝ k_x · x, so the
    electronic matrix element scales as k_e^1.  For a σ-HOMO, φ_k^σ ∝
    k_z · z gives the same k_e^1 scaling.  The default ``k_power=0``
    leaves the coupling energy-independent (geometry-only model).

    Parameters
    ----------
    R0 : float
        Centre of Gaussian (equilibrium bond length, Bohr)
    A_rad : float
        Amplitude of radial coupling
    alpha_rad : float
        Width parameter for radial coupling (Bohr^-2)
    A_rot : float
        Amplitude of rotational coupling
    alpha_rot : float
        Width parameter for rotational coupling (Bohr^-2)
    k_power : float
        Exponent of k_e in the energy scaling factor.
        0 → energy-independent (default);
        1 → low-k OPW limit for π or σ HOMO (Acharya 1983, Eq. 5–6).
    """

    def __init__(
        self,
        R0: float,
        A_rad: float = 0.01,
        alpha_rad: float = 1.0,
        A_rot: float = 0.05,
        alpha_rot: float = 1.0,
        k_power: float = 0.0,
    ):
        self.R0 = R0
        self.A_rad = A_rad
        self.alpha_rad = alpha_rad
        self.A_rot = A_rot
        self.alpha_rot = alpha_rot
        self.k_power = k_power

    def compute_coupling_at_r(
        self, R: float, electron_energy: float, **kwargs,
    ) -> CouplingResult:
        """Evaluate model coupling at one geometry."""
        k_e = np.sqrt(max(2.0 * electron_energy, 0.0))
        k_factor = k_e ** self.k_power if k_e > 0 else 0.0

        gauss_rad = self.A_rad * np.exp(-self.alpha_rad * (R - self.R0) ** 2)
        gauss_rot = self.A_rot * np.exp(-self.alpha_rot * (R - self.R0) ** 2)

        return CouplingResult(
            R=R,
            m_rad=complex(gauss_rad * k_factor),
            m_rot=complex(gauss_rot * k_factor),
            electron_energy=electron_energy,
            k_electron=k_e,
        )

    def compute_coupling_curve(
        self, R_grid: np.ndarray, electron_energy: float, **kwargs,
    ) -> List[CouplingResult]:
        """Evaluate model coupling along R grid."""
        return [
            self.compute_coupling_at_r(R, electron_energy)
            for R in R_grid
        ]
