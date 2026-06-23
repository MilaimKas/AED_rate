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

import warnings

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
from .potential import PotentialEnergyCurve
from ..utils.constants import AEDValidationWarning


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class CouplingResult:
    """Result of electronic coupling calculation at one geometry."""

    R: float                    # bond length (Bohr)
    m_rad: complex              # radial coupling matrix element   (l=1 p-wave)
    m_rot: complex              # rotational coupling matrix element (l=1 p-wave)
    electron_energy: float      # continuum electron kinetic energy (Hartree)
    k_electron: float           # electron wave vector magnitude (a.u.)
    # l=0 s-wave coupling — non-zero only in the A1/σ-symmetry channel
    # (radial for a σ HOMO, rotational for a π HOMO).  Distinct final electron
    # state from the l=1 term, so it adds incoherently in the cross section.
    m_swave: complex = 0j
    swave_channel: Optional[str] = None   # 'rad', 'rot', or None (no s-wave)


@dataclass
class CouplingIntermediates:
    """
    Real-space intermediates of the electronic coupling at one geometry.

    These are the per-grid-point quantities that the coupling matrix elements
    are built from (m_rad = Σ_i w_i φ_k*(r_i) ∂φ_HOMO/∂R(r_i)), exposed for
    inspection/visualization.  Returned by
    ElectronicCoupling.compute_coupling_intermediates (PySCF only).
    """

    R: float                    # bond length (Bohr)
    electron_energy: float      # continuum electron kinetic energy (Hartree)
    coords: np.ndarray          # (N, 3) Becke grid points (Bohr)
    weights: np.ndarray         # (N,)   Becke integration weights
    dphi_dR: np.ndarray         # (N,)   ∂φ_HOMO/∂R on the grid
    dphi_dtheta: np.ndarray     # (N,)   ∂φ_HOMO/∂θ on the grid
    phi_k_rad: np.ndarray       # (N,)   OPW (HOMO symmetry) for the radial channel
    phi_k_rot: np.ndarray       # (N,)   OPW (complementary symmetry) for rotational
    m_rad: complex              # Σ w φ_k_rad* ∂φ/∂R
    m_rot: complex              # Σ w φ_k_rot* ∂φ/∂θ / R


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
        # UHF (open-shell anion) returns (mo1_α, mo1_β) — a 2-tuple of arrays
        # with different shapes (nocc differs), so np.array(mo1) fails.  The
        # detaching electron is the α SOMO (homo_idx is its α index), so keep α.
        # RHF returns a single array.
        is_uhf = np.ndim(mf.mo_occ) == 2
        mo1 = np.asarray(mo1[0] if is_uhf else mo1)  # (natm, 3, nao, nocc[_α])

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
        # Build the neutral (N−1 electrons) with the same basis/geometry.
        # Detaching one electron: closed-shell anion → doublet neutral; open-shell
        # anion → multiplicity drops by one (e.g. LiH⁻ ²Σ⁺ → LiH ¹Σ⁺ singlet).
        neutral_spin = 1 if spin == 0 else spin - 1
        mol_neutral = self.es._build_molecule(R, charge=charge + 1, spin=neutral_spin)
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
        inter = self.compute_coupling_intermediates(
            R, electron_energy, charge=charge, spin=spin,
        )
        return CouplingResult(
            R=R,
            m_rad=inter.m_rad,
            m_rot=inter.m_rot,
            electron_energy=electron_energy,
            k_electron=float(np.sqrt(2.0 * electron_energy)),
        )

    def compute_coupling_intermediates(
        self,
        R: float,
        electron_energy: float,
        charge: int = -1,
        spin: int = 0,
    ) -> CouplingIntermediates:
        """
        Compute the real-space coupling intermediates at one geometry.

        Returns the Becke grid, the HOMO derivatives ∂φ/∂R and ∂φ/∂θ, the two
        OPW continuum functions φ_k, and the resulting m_rad, m_rot.  Use this
        to inspect/visualize the electronic ingredients (∂φ_HOMO/∂R, the OPW)
        that compute_coupling_at_r sums over.  Requires PySCF.

        Parameters
        ----------
        R : float
            Bond length in Bohr.
        electron_energy : float
            Ejected-electron kinetic energy (Hartree).
        charge, spin : int
            Anion charge and spin (default −1, 0).

        Returns
        -------
        CouplingIntermediates
        """
        # HOMO derivatives on the Becke grid (CPSCF response)
        coords, weights, dphi_dR, dphi_dtheta = (
            self._evaluate_homo_derivatives(R, charge, spin)
        )

        # Radial coupling: ∂φ_HOMO/∂R preserves HOMO symmetry
        # → use the same-symmetry OPW as the HOMO.
        phi_k_rad = self._evaluate_opw_on_grid(
            R, coords, electron_energy,
            symmetry=self.homo_symmetry, charge=charge, spin=spin,
        )

        # Rotational coupling: ∂φ_HOMO/∂θ swaps symmetry (π ↔ σ), so the
        # continuum partner has the complementary symmetry.
        rot_symmetry = "sigma" if self.homo_symmetry == "pi" else "pi"
        phi_k_rot = self._evaluate_opw_on_grid(
            R, coords, electron_energy,
            symmetry=rot_symmetry, charge=charge, spin=spin,
        )

        # Numerical integration: m = Σ_i w_i φ_k*(r_i) ∂φ(r_i)
        m_rad = np.sum(weights * np.conj(phi_k_rad) * dphi_dR)
        m_rot = np.sum(weights * np.conj(phi_k_rot) * dphi_dtheta / R)

        return CouplingIntermediates(
            R=R, electron_energy=electron_energy,
            coords=coords, weights=weights,
            dphi_dR=dphi_dR, dphi_dtheta=dphi_dtheta,
            phi_k_rad=phi_k_rad, phi_k_rot=phi_k_rot,
            m_rad=complex(m_rad), m_rot=complex(m_rot),
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
        warnings.warn(
            "ModelCoupling is a sanity-check Gaussian, not a physical coupling. "
            "Its parameters were fitted to reproduce Acharya's final rates and may "
            "absorb unrelated errors. For physical results use ElectronicCoupling "
            "(CPSCF) or InterpolatedCoupling on a diffuse-augmented basis.",
            AEDValidationWarning,
            stacklevel=2,
        )
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


# ---------------------------------------------------------------------------
# Precomputed ab initio coupling with spline interpolation
# ---------------------------------------------------------------------------


class InterpolatedCoupling:
    """
    Ab initio CPSCF coupling precomputed on a 2D (R, k_e) grid and
    interpolated to any (R, k_e) on demand.

    OPW strategy
    ------------
    The continuum electron is described by the symmetry-projected l=1
    partial wave of the plane wave (Rayleigh expansion):

        φ_k^{Γ}(r) = 3 × j₁(k_e r) × (coord/r)

    where coord is x (E1x/π_x), y (E1y/π_y), or z (A1/σ) depending on
    the symmetry Γ matching ∂φ_HOMO/∂Q.  The factor 3 ensures:

        3 j₁(k_e r)/r → k_e   as k_e r → 0

    recovering the linear (low-k) OPW in the limit while keeping the
    integral bounded at large r (j₁(x) ~ cos x / x² for large x).

    HOMO symmetry is read automatically from PySCF's symmetry labels at
    the first geometry, so the OPW direction is determined without the
    user specifying it.

    Grid strategy
    -------------
    - CPSCF is run once per R geometry (expensive).
    - The OPW integral is re-evaluated at each k_e in ``k_e_grid``
      (cheap — just a weighted sum on the existing Becke grid).
    - The result is a 2D array m(R, k_e) stored for spline interpolation.
    - ``save()``/``load()`` persist both dimensions to disk.

    To use a precomputed grid without PySCF, construct via the
    :meth:`from_npz` classmethod instead of ``__init__``.

    Parameters
    ----------
    electronic_structure : ElectronicStructure
        PySCF wrapper (provides the atom symbols and basis for the SCF/CPSCF).
    anion_potential : PotentialEnergyCurve
        Anion Morse potential, used to determine R_cutoff automatically.
    R_min : float
        Inner boundary of coupling region (Bohr).
    R_cutoff : float, optional
        Outer boundary; defaults to R_e + ln(1/(1-√0.9))/β ≈ R_e + 3/β
        (where V_anion ≈ 0.9 D_e).
    n_points : int
        Number of CPSCF evaluations on the coarse R grid.
    k_e_grid : np.ndarray, optional
        Electron wave-vector values (a.u.) at which to evaluate the OPW
        integral.  Defaults to six log-spaced values (0.01 – 0.4 a.u.).
    homo_symmetry : str
        HOMO symmetry ('pi' or 'sigma').  Orbital tracking is tuned for a π
        HOMO (E1x); a σ HOMO falls back to a generic maximum-overlap criterion.
    grid_level : int
        PySCF Becke integration grid quality (0–9).
    charge : int
        Total charge of the anion (default −1).
    spin : int
        Spin (2S) of the anion (default 0, closed-shell singlet).
    """

    # Default k_e grid: log-spaced, covers near-threshold to ~3 eV electrons
    _DEFAULT_K_E_GRID: np.ndarray = np.array(
        [0.01, 0.03, 0.07, 0.15, 0.25, 0.40]
    )

    def __init__(
        self,
        electronic_structure: ElectronicStructure,
        anion_potential: PotentialEnergyCurve,
        R_min: float = 0.8,
        R_cutoff: Optional[float] = None,
        n_points: int = 30,
        k_e_grid: Optional[np.ndarray] = None,
        homo_symmetry: str = "pi",
        grid_level: int = 3,
        charge: int = -1,
        spin: int = 0,
    ) -> None:
        if not PYSCF_AVAILABLE:
            raise ImportError(
                "PySCF is required to precompute InterpolatedCoupling. "
                "To load a precomputed .npz without PySCF, use "
                "InterpolatedCoupling.from_npz(path)."
            )

        self.es = electronic_structure
        self.anion_potential = anion_potential
        self.homo_symmetry = homo_symmetry
        self.grid_level = grid_level
        self.charge = charge
        self.spin = spin
        self.R_min = R_min
        self.k_e_grid: np.ndarray = (
            k_e_grid if k_e_grid is not None
            else self._DEFAULT_K_E_GRID.copy()
        )

        # Outer cutoff: where V_anion ≈ 0.9 * D_e (90 % of way to dissociation).
        # V(R) = D_e*(1-exp(-β*(R-R_e)))^2 = 0.9*D_e
        # => R = R_e + ln(1/(1-sqrt(0.9)))/β ≈ R_e + 3.0/β
        # For OH⁻ this gives ~4.4 Bohr, well past the v'=8 outer turning point.
        if R_cutoff is None:
            pot = anion_potential
            R_cutoff = pot.r_e + np.log(1.0 / (1.0 - 0.9 ** 0.5)) / pot.beta
        self.R_cutoff = R_cutoff

        self.R_grid = np.linspace(R_min, R_cutoff, n_points)

        # Populated by precompute() — 2D arrays (n_R, n_ke)
        self._m_rad_2d: Optional[np.ndarray] = None
        self._m_rot_2d: Optional[np.ndarray] = None
        self._m_swave_2d: Optional[np.ndarray] = None
        self._spl_m_rad = None   # RectBivariateSpline(R, k_e)
        self._spl_m_rot = None
        self._spl_m_swave = None
        self.swave_channel: Optional[str] = None   # 'rad', 'rot', or None
        self._is_precomputed: bool = False

    # ------------------------------------------------------------------
    # Precomputation
    # ------------------------------------------------------------------

    def precompute(self, verbose: bool = True) -> None:
        """
        Run CPSCF at every point in R_grid and build cubic splines.

        This is the expensive step (~60 s for 30 points with aug-cc-pVTZ).
        Call once, then persist with ``save()``.

        Orbital tracking strategy
        -------------------------
        OH⁻ has a doubly degenerate π HOMO (E1x / E1y in C∞v).
        Without symmetry constraints the SCF returns a random linear
        combination at each geometry, causing the coupling to oscillate
        wildly.  We enforce ``mol.symmetry = True`` so PySCF keeps E1x
        (π_x, B1 in C2v) and E1y (π_y, B2) in separate irreps.

        At the first geometry the E1x HOMO is seeded by symmetry label.
        At each subsequent geometry we apply the **Maximum Overlap
        Criterion (MOC)**: choose the occupied E1x orbital whose
        S-weighted overlap ⟨prev_HOMO|S_AO|j⟩ is largest.  This
        correctly handles orbital reordering and energy crossings that
        occur at large R, where a different orbital may acquire the E1x
        label.  The sign of the overlap is used to phase-align the
        coefficient vector, giving a smooth, single-valued coupling curve.
        """
        from pyscf import gto, scf, dft, symm
        from scipy.interpolate import RectBivariateSpline
        from scipy.special import spherical_jn

        # Closed-shell anion (spin 0) → RHF/CPSCF; open-shell → UHF/CPSCF with
        # the detaching electron in the α channel (SOMO).
        is_uhf = self.spin != 0

        n = len(self.R_grid)
        n_ke = len(self.k_e_grid)
        # 2D arrays: rows = R geometry, columns = k_e value
        m_rad_2d = np.zeros((n, n_ke))      # l=1 radial   (p-wave)
        m_rot_2d = np.zeros((n, n_ke))      # l=1 rotational (p-wave)
        m_swave_2d = np.zeros((n, n_ke))    # l=0 s-wave (A1/σ channel only)
        swave_channel = "rot"               # set below from the HOMO symmetry

        # Phase reference: HOMO MO coefficient vector at previous geometry
        prev_homo_coeff: Optional[np.ndarray] = None
        # HOMO symmetry label detected at first geometry (used for OPW direction)
        homo_irrep_detected: Optional[str] = None

        if verbose:
            print(
                f"InterpolatedCoupling: precomputing CPSCF on {n} R points "
                f"[{self.R_grid[0]:.3f}, {self.R_grid[-1]:.3f}] Bohr, "
                f"{n_ke} k_e values [{self.k_e_grid[0]:.3f}, {self.k_e_grid[-1]:.3f}] a.u."
            )

        for i, R in enumerate(self.R_grid):
            if verbose:
                print(f"  [{i+1:2d}/{n}] R = {R:.3f} Bohr ...", end=" ", flush=True)

            # Build molecule with symmetry enforced (atoms from the
            # ElectronicStructure, so the precompute is system-agnostic).
            mol = gto.Mole()
            mol.atom = f"{self.es.atom1} 0 0 0; {self.es.atom2} 0 0 {R}"
            mol.basis = self.es.basis
            mol.charge = self.charge
            mol.spin = self.spin
            mol.unit = "Bohr"
            mol.symmetry = True   # keeps E1x and E1y in separate irreps
            mol.verbose = 0
            mol.build()

            mf = scf.UHF(mol) if is_uhf else scf.RHF(mol)
            mf.verbose = 0
            mf.kernel()

            # Work in the α channel: for an open-shell anion the detaching
            # electron is the α SOMO; for a closed shell this is the only channel.
            c_mo, occ_arr, _ = self._alpha_channel(mf)

            # Maximum Overlap Criterion (MOC): track the same physical orbital
            # across geometries by maximising the S-weighted overlap with the
            # reference HOMO from the previous geometry.  This correctly handles
            # orbital reordering / energy crossings that occur at large R, where
            # a different orbital may acquire the E1x symmetry label.
            S_AO = mol.intor("int1e_ovlp")  # AO overlap matrix, (nao, nao)
            occ_idx = np.where(np.asarray(occ_arr) > 0.5)[0]

            if prev_homo_coeff is None:
                # Seed at the first geometry using the HOMO symmetry label.
                homo_idx = self._find_homo_by_symmetry(mol, mf, self.homo_symmetry)
                sign = 1.0
            else:
                # Restrict candidates to the HOMO irrep (E1x for π, A1 for σ).
                try:
                    orbsym = symm.label_orb_symm(
                        mol, mol.irrep_name, mol.symm_orb, c_mo
                    )
                    homo_labels = self._homo_irrep_labels(self.homo_symmetry)
                    cand_idx = [j for j in occ_idx if orbsym[j] in homo_labels]
                    if not cand_idx:
                        cand_idx = list(occ_idx)
                except Exception:
                    cand_idx = list(occ_idx)

                # S-weighted overlap ⟨prev | S | j⟩ accounts for AO
                # non-orthogonality; a bare dot product can silently pick
                # the wrong orbital when basis functions overlap strongly.
                overlaps = {
                    j: float(prev_homo_coeff @ S_AO @ c_mo[:, j])
                    for j in cand_idx
                }
                homo_idx = max(overlaps, key=lambda j: abs(overlaps[j]))
                sign = float(np.sign(overlaps[homo_idx]))

            coeff = sign * c_mo[:, homo_idx]
            prev_homo_coeff = coeff  # update reference for next R

            # Solve CPSCF
            hess = mf.Hessian()
            h1ao = hess.make_h1(mf.mo_coeff, mf.mo_occ)
            mo1, _ = hess.solve_mo1(
                mf.mo_energy, mf.mo_coeff, mf.mo_occ, h1ao
            )
            # UHF returns (mo1_α, mo1_β) with different shapes (nocc differs); the
            # detaching electron is α.  RHF returns a single array.  Either way,
            # mo1 is then (natm, 3, nao, nocc) for the channel we track.
            mo1 = np.asarray(mo1[0] if is_uhf else mo1)

            # Becke integration grid
            grids = dft.gen_grid.Grids(mol)
            grids.level = self.grid_level
            grids.build()
            ao_values = mol.eval_gto("GTOval_sph", grids.coords)

            # HOMO derivatives on grid (apply phase sign)
            # Radial: atom B (idx 1), z-direction (bond axis, idx 2)
            dC_dR = sign * mo1[1, 2, :, homo_idx]
            # Rotational: atom B (idx 1), x-direction (idx 0), scaled by R
            dC_dtheta = sign * R * mo1[1, 0, :, homo_idx]

            dphi_dR = ao_values @ dC_dR         # (N_grid,)
            dphi_dtheta = ao_values @ dC_dtheta  # (N_grid,)

            # Determine OPW symmetry from HOMO irrep (auto-detected once).
            # Radial: ∂φ_HOMO/∂R has same symmetry as HOMO.
            # Rotational: ∂φ_HOMO/∂θ swaps π↔σ (A1 perturbation cross E1).
            #
            # l=1 partial-wave OPW:
            #   φ_k^{E1x}(r) = 3 j₁(k_e r) × (x/r)   for π_x (radial)
            #   φ_k^{A1}(r)  = 3 j₁(k_e r) × (z/r)   for σ  (rotational)
            # Swapped for σ HOMO (A1 radial, E1x rotational).
            #
            # The factor 3 ensures: 3 j₁(kr)/r → k_e as k_e r → 0,
            # recovering the linear OPW limit and keeping m(R,k_e)/k_e
            # comparable to the old m_geom at small k_e.
            if homo_irrep_detected is None:
                try:
                    orbsym_now = symm.label_orb_symm(
                        mol, mol.irrep_name, mol.symm_orb, c_mo
                    )
                    homo_irrep_detected = orbsym_now[homo_idx]
                except Exception:
                    homo_irrep_detected = "unknown"

            # Decide π vs σ from the detected HOMO irrep (fall back to the
            # homo_symmetry setting if the label could not be read).
            if homo_irrep_detected not in (None, "unknown"):
                is_pi = homo_irrep_detected in self._homo_irrep_labels("pi")
            else:
                is_pi = self.homo_symmetry == "pi"

            if is_pi:
                # π HOMO: radial OPW along x (E1x); rotational along z (A1/σ)
                coord_rad = grids.coords[:, 0]   # x
                coord_rot = grids.coords[:, 2]   # z
            else:
                # σ HOMO: radial OPW along z (A1); rotational along x (E1x/π)
                coord_rad = grids.coords[:, 2]   # z
                coord_rot = grids.coords[:, 0]   # x

            # |r| for the j₁ envelope; avoid division by zero at origin
            r_mag = np.linalg.norm(grids.coords, axis=1)
            r_safe = np.maximum(r_mag, 1e-10)

            # ----------------------------------------------------------
            # Neutral SCF at the same geometry for OPW orthogonalization.
            #
            # Acharya (1984) eq. 4: the continuum electron is described
            # by a plane wave *orthogonalized* to the occupied orbitals
            # of the neutral molecule:
            #
            #   φ_OPW = φ_k - Σ_i |φ_i⟩⟨φ_i|φ_k⟩
            #
            # Without this, the bare plane wave overlaps with the
            # occupied space, inflating the coupling matrix element.
            # ----------------------------------------------------------
            # Detaching one electron changes the spin: a closed-shell anion
            # (spin 0) → doublet neutral; an open-shell anion → multiplicity
            # drops by one (e.g. LiH⁻ ²Σ⁺ → LiH ¹Σ⁺).
            neutral_spin = 1 if self.spin == 0 else self.spin - 1
            mol_neutral = gto.Mole()
            mol_neutral.atom = f"{self.es.atom1} 0 0 0; {self.es.atom2} 0 0 {R}"
            mol_neutral.basis = self.es.basis
            mol_neutral.charge = self.charge + 1   # neutral = anion + 1 charge
            mol_neutral.spin = neutral_spin
            mol_neutral.unit = "Bohr"
            mol_neutral.symmetry = True
            mol_neutral.verbose = 0
            mol_neutral.build()

            # RHF for a singlet neutral, ROHF otherwise.  Only the occupied
            # orbitals are needed (for orthogonalization) — no Hessian here,
            # so ROHF's limited Hessian support is irrelevant.
            mf_neutral = (scf.RHF(mol_neutral) if neutral_spin == 0
                          else scf.ROHF(mol_neutral))
            mf_neutral.verbose = 0
            mf_neutral.kernel()

            # Occupied neutral MOs evaluated on the Becke grid
            # (use the anion grid coords — same atom positions)
            ao_neutral = mol_neutral.eval_gto("GTOval_sph", grids.coords)
            neutral_occ_idx = np.where(np.asarray(mf_neutral.mo_occ) > 0.5)[0]
            # MO values on grid: (N_grid, n_occ)
            neutral_mo_grid = ao_neutral @ mf_neutral.mo_coeff[:, neutral_occ_idx]

            # The A1/σ-symmetry derivative — the one that admits the l=0 s-wave —
            # is the rotational channel for a π HOMO, the radial channel for a σ
            # HOMO (the other channel is E1/π and gets no s-wave by symmetry).
            swave_channel = "rot" if is_pi else "rad"
            w = grids.weights

            # Evaluate OPW integrals at each k_e: m(R, k_e) = ∫ OPW × dphi d³r
            for j, k_e in enumerate(self.k_e_grid):
                # l=1: 3 j₁(k_e r)/r → k_e as k_e r → 0 (linear limit).
                j1_over_r = spherical_jn(1, k_e * r_safe) / r_safe
                # l=0: j₀(k_e r) → 1 as k_e r → 0 (Rayleigh coeff 2l+1 = 1).
                j0 = spherical_jn(0, k_e * r_safe)

                # Bare partial-wave components
                pw_rad = 3.0 * j1_over_r * coord_rad   # l=1, radial-channel direction
                pw_rot = 3.0 * j1_over_r * coord_rot   # l=1, rotational-channel direction
                pw_sw = j0                             # l=0, spherically symmetric

                # Orthogonalize each to the occupied neutral MOs:
                # φ_OPW = pw - Σ_i φ_i ⟨φ_i|pw⟩
                opw_rad, opw_rot, opw_sw = pw_rad.copy(), pw_rot.copy(), pw_sw.copy()
                for k_occ in range(neutral_mo_grid.shape[1]):
                    phi_i = neutral_mo_grid[:, k_occ]
                    opw_rad -= phi_i * float(np.sum(w * phi_i * pw_rad))
                    opw_rot -= phi_i * float(np.sum(w * phi_i * pw_rot))
                    opw_sw  -= phi_i * float(np.sum(w * phi_i * pw_sw))

                m_rad_2d[i, j] = float(np.sum(w * opw_rad * dphi_dR))
                # 1/R factor: rotational coupling uses plain F_E (not F_E/R)
                m_rot_2d[i, j] = float(np.sum(w * opw_rot * dphi_dtheta / R))
                # s-wave overlaps the A1/σ derivative: ∂φ/∂R (σ HOMO, radial) or
                # (1/R)∂φ/∂θ (π HOMO, rotational).
                if swave_channel == "rad":
                    m_swave_2d[i, j] = float(np.sum(w * opw_sw * dphi_dR))
                else:
                    m_swave_2d[i, j] = float(np.sum(w * opw_sw * dphi_dtheta / R))

            if verbose:
                # Print values at the representative middle k_e
                mid = n_ke // 2
                print(
                    f"m_rad(k_e={self.k_e_grid[mid]:.3f}) = {m_rad_2d[i, mid]:+.4e}  "
                    f"m_rot(k_e={self.k_e_grid[mid]:.3f}) = {m_rot_2d[i, mid]:+.4e}"
                )

        # 2D bicubic splines on (R_grid, k_e_grid)
        self._m_rad_2d = m_rad_2d
        self._m_rot_2d = m_rot_2d
        self._m_swave_2d = m_swave_2d
        self.swave_channel = swave_channel
        self._spl_m_rad = RectBivariateSpline(
            self.R_grid, self.k_e_grid, m_rad_2d, kx=3, ky=3
        )
        self._spl_m_rot = RectBivariateSpline(
            self.R_grid, self.k_e_grid, m_rot_2d, kx=3, ky=3
        )
        self._spl_m_swave = RectBivariateSpline(
            self.R_grid, self.k_e_grid, m_swave_2d, kx=3, ky=3
        )
        self._is_precomputed = True

        if verbose:
            print(f"Precomputation complete.  HOMO irrep detected: "
                  f"{homo_irrep_detected}; s-wave channel: {swave_channel}")

    # ------------------------------------------------------------------
    # Symmetry-based HOMO identification
    # ------------------------------------------------------------------

    @staticmethod
    def _homo_irrep_labels(symmetry: str) -> set:
        """
        PySCF irrep labels (C∞v/Coov and C2v) matching the HOMO symmetry.

        - ``'pi'``   → E1x / B1 (π_x); the σ partner of the degenerate Π pair
          is excluded so the random E1x/E1y mixing is avoided.
        - ``'sigma'``→ A1 (Σ⁺); non-degenerate, so this just restricts the
          maximum-overlap search to the right irrep.
        """
        if symmetry == "sigma":
            return {"A1", "a1", "A1g", "a1g"}
        # default: pi (π_x)
        return {"E1x", "e1x", "B1", "b1"}

    @staticmethod
    def _alpha_channel(mf):
        """
        Return (mo_coeff, mo_occ, mo_energy) for the α spin channel.

        For UHF these are spin-resolved (mo_occ is 2-D); the detaching electron
        of an open-shell anion is the α SOMO, so we work in the α channel.  For
        RHF (mo_occ is 1-D) the single channel is returned unchanged.
        """
        occ = mf.mo_occ
        if np.ndim(occ) == 2:                      # UHF: (2, nmo)
            return mf.mo_coeff[0], occ[0], mf.mo_energy[0]
        return mf.mo_coeff, occ, mf.mo_energy

    @staticmethod
    def _find_homo_by_symmetry(mol, mf, symmetry: str = "pi") -> int:
        """
        Index of the highest occupied orbital matching the HOMO symmetry.

        With ``mol.symmetry = True``, PySCF assigns irrep labels.  Selecting the
        detaching orbital by irrep (π_x = E1x, σ = A1) seeds the tracking and —
        for the degenerate π case — avoids the random E1x/E1y mixing that would
        otherwise make the coupling curve oscillate.  For an open-shell anion the
        α channel is used, so the highest α-occupied orbital of the irrep is the
        SOMO.  Falls back to the highest occupied orbital if no matching irrep.

        Parameters
        ----------
        mol, mf : PySCF Mole / SCF objects (with mol.symmetry = True).
        symmetry : str
            'pi' (default) or 'sigma'.
        """
        from pyscf import symm

        mo_coeff, mo_occ, _ = InterpolatedCoupling._alpha_channel(mf)
        occupied_idx = np.where(np.asarray(mo_occ) > 0.5)[0]

        labels = InterpolatedCoupling._homo_irrep_labels(symmetry)
        try:
            orbsym = symm.label_orb_symm(
                mol, mol.irrep_name, mol.symm_orb, mo_coeff
            )
            for idx in reversed(occupied_idx):
                if orbsym[idx] in labels:
                    return int(idx)
        except Exception:
            pass

        # Fallback: highest occupied
        return int(occupied_idx[-1])

    @staticmethod
    def _find_e1x_homo(mol, mf) -> int:
        """Backward-compatible π_x HOMO finder; see _find_homo_by_symmetry."""
        return InterpolatedCoupling._find_homo_by_symmetry(mol, mf, "pi")

    # ------------------------------------------------------------------
    # Query interface (same as ModelCoupling)
    # ------------------------------------------------------------------

    def compute_coupling_at_r(
        self, R: float, electron_energy: float, **kwargs
    ) -> CouplingResult:
        """
        Return m_rad and m_rot at (R, k_e) by 2D spline interpolation.

        The coupling is exactly zero outside [R_min, R_cutoff] and for
        electron energies below the k_e grid minimum.
        """
        if not self._is_precomputed:
            raise RuntimeError(
                "Call precompute() (or load()) before querying coupling."
            )

        k_e = float(np.sqrt(max(2.0 * electron_energy, 0.0)))

        # Hard zero outside the interaction region
        if R < self.R_min or R > self.R_cutoff:
            return CouplingResult(
                R=R, m_rad=0j, m_rot=0j,
                electron_energy=electron_energy, k_electron=k_e,
            )

        # Clamp k_e to the precomputed grid range (extrapolation not reliable)
        k_e_clamped = float(np.clip(k_e, self.k_e_grid[0], self.k_e_grid[-1]))

        # 2D spline: evaluate at (R, k_e) — grid=False for scalar query
        m_rad = complex(float(self._spl_m_rad(R, k_e_clamped, grid=False)))
        m_rot = complex(float(self._spl_m_rot(R, k_e_clamped, grid=False)))
        m_swave = (complex(float(self._spl_m_swave(R, k_e_clamped, grid=False)))
                   if self._spl_m_swave is not None else 0j)

        return CouplingResult(
            R=R, m_rad=m_rad, m_rot=m_rot,
            electron_energy=electron_energy, k_electron=k_e,
            m_swave=m_swave, swave_channel=self.swave_channel,
        )

    def compute_coupling_curve(
        self, R_grid: np.ndarray, electron_energy: float, **kwargs
    ) -> List[CouplingResult]:
        """Evaluate coupling along a grid (uses interpolation)."""
        return [
            self.compute_coupling_at_r(R, electron_energy)
            for R in R_grid
        ]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Save precomputed 2D coupling arrays to an .npz file.

        Parameters
        ----------
        path : str
            File path (e.g. 'oh_minus_coupling.npz')
        """
        if not self._is_precomputed:
            raise RuntimeError("Nothing to save — run precompute() first.")
        np.savez(
            path,
            R_grid=self.R_grid,
            k_e_grid=self.k_e_grid,
            m_rad_2d=self._m_rad_2d,
            m_rot_2d=self._m_rot_2d,
            m_swave_2d=self._m_swave_2d,
            swave_channel=np.array([self.swave_channel or "none"]),
            R_min=np.array([self.R_min]),
            R_cutoff=np.array([self.R_cutoff]),
        )

    def load(self, path: str) -> None:
        """
        Load precomputed 2D coupling arrays and rebuild splines.

        Parameters
        ----------
        path : str
            File path written by ``save()``
        """
        from scipy.interpolate import RectBivariateSpline

        data = np.load(path)
        self.R_grid = data["R_grid"]
        self.k_e_grid = data["k_e_grid"]
        self._m_rad_2d = data["m_rad_2d"]
        self._m_rot_2d = data["m_rot_2d"]
        self.R_min = float(data["R_min"][0])
        self.R_cutoff = float(data["R_cutoff"][0])

        self._spl_m_rad = RectBivariateSpline(
            self.R_grid, self.k_e_grid, self._m_rad_2d, kx=3, ky=3
        )
        self._spl_m_rot = RectBivariateSpline(
            self.R_grid, self.k_e_grid, self._m_rot_2d, kx=3, ky=3
        )

        # s-wave channel — absent in pre-multichannel .npz files (→ no s-wave).
        if "m_swave_2d" in data:
            self._m_swave_2d = data["m_swave_2d"]
            ch = str(data["swave_channel"][0]) if "swave_channel" in data else "none"
            self.swave_channel = None if ch == "none" else ch
            self._spl_m_swave = RectBivariateSpline(
                self.R_grid, self.k_e_grid, self._m_swave_2d, kx=3, ky=3
            )
        else:
            self._m_swave_2d = None
            self.swave_channel = None
            self._spl_m_swave = None

        self._is_precomputed = True

    @classmethod
    def from_npz(cls, path: str) -> "InterpolatedCoupling":
        """
        Build an evaluation-only coupling from a precomputed .npz file.

        Unlike the constructor, this does **not** require PySCF — it only
        rebuilds the interpolating splines from the saved grid.  The returned
        object supports ``compute_coupling_at_r`` / ``compute_coupling_curve``
        but not ``precompute`` (its ``es``/``anion_potential`` are unset).

        Parameters
        ----------
        path : str
            File path written by :meth:`save`.

        Returns
        -------
        InterpolatedCoupling
            Ready to query; PySCF not needed.
        """
        obj = cls.__new__(cls)          # bypass __init__ (which requires PySCF)
        # PySCF-only attributes — unused for evaluation:
        obj.es = None
        obj.anion_potential = None
        obj.homo_symmetry = None
        obj.grid_level = None
        obj.charge = None
        obj.spin = None
        obj.load(path)                  # sets grids, splines, _is_precomputed
        return obj


# ---------------------------------------------------------------------------
# Standalone HOMO-tracking diagnostic (SCF-only, no CPSCF)
# ---------------------------------------------------------------------------

def scan_homo_tracking(
    basis: str,
    R_grid: np.ndarray,
    n_virt: int = 3,
    verbose: bool = False,
) -> dict:
    """
    Run SCF-only HOMO tracking on an R grid and return diagnostic data.

    Applies the same Maximum Overlap Criterion (MOC) as
    ``InterpolatedCoupling.precompute()``: the highest E1x orbital at the
    first geometry seeds the reference; subsequent geometries pick the
    occupied orbital maximising |⟨prev_HOMO | S_AO | j⟩|.

    Because no CPSCF response equations are solved, this is ~10–20× faster
    than a full precomputation and can be run on a dense R grid (100+ points)
    in seconds.  Use it to produce orbital correlation diagrams and inspect
    where level crossings cause the tracked orbital index to jump.

    Parameters
    ----------
    basis : str
        PySCF basis string (e.g. '6-31g', 'aug-cc-pvdz')
    R_grid : np.ndarray
        Bond lengths in Bohr at which to run SCF
    n_virt : int
        Number of lowest virtual (unoccupied) orbital energies to record
        for the correlation diagram.  Default 3.
    verbose : bool
        Print one progress line per geometry.

    Returns
    -------
    dict with keys
        R             : (n_R,)           bond lengths (Bohr)
        homo_idx      : (n_R,) int       MO index of the tracked HOMO
        homo_energy   : (n_R,)           HOMO orbital energy (Hartree)
        homo_irrep    : list[str]        symmetry label of tracked HOMO
        moc_overlap   : (n_R,)           |⟨prev|S|chosen⟩|  (NaN at R[0])
        moc_overlap_2nd : (n_R,)         second-best candidate overlap
                                         (NaN at R[0] or if only 1 candidate)
        occ_energies  : (n_R, n_occ)     all occupied orbital energies
        occ_irreps    : list[list[str]]  symmetry labels (n_R × n_occ)
        virt_energies : (n_R, n_virt)    lowest virtual orbital energies
    """
    if not PYSCF_AVAILABLE:
        raise ImportError("PySCF is required for HOMO tracking scan.")

    from pyscf import gto, scf, symm as pyscf_symm

    n = len(R_grid)
    homo_idx_arr     = np.zeros(n, dtype=int)
    homo_energy_arr  = np.zeros(n)
    homo_irrep_list  = []
    moc_overlap_arr  = np.full(n, np.nan)
    moc_overlap_2nd  = np.full(n, np.nan)
    occ_energies_lst = []   # list of 1-D arrays (varying length guard)
    occ_irreps_lst   = []   # list of list[str]
    virt_energies_lst = []

    prev_homo_coeff: Optional[np.ndarray] = None

    for i, R in enumerate(R_grid):
        if verbose:
            print(f"  [{i+1:3d}/{n}]  R = {R:.4f} Bohr ... ", end="", flush=True)

        mol = gto.Mole()
        mol.atom    = f"O 0 0 0; H 0 0 {R}"
        mol.basis   = basis
        mol.charge  = -1
        mol.spin    = 0
        mol.unit    = "Bohr"
        mol.symmetry = True
        mol.verbose  = 0
        mol.build()

        mf = scf.RHF(mol)
        mf.verbose = 0
        mf.kernel()

        S_AO    = mol.intor("int1e_ovlp")
        occ_idx = np.where(np.asarray(mf.mo_occ) > 0.5)[0]

        # Symmetry labels for all MOs
        try:
            orbsym = list(pyscf_symm.label_orb_symm(
                mol, mol.irrep_name, mol.symm_orb, mf.mo_coeff
            ))
        except Exception:
            orbsym = ["?"] * mf.mo_coeff.shape[1]

        # --- HOMO selection ---
        if prev_homo_coeff is None:
            # Seed: use symmetry-based E1x selection
            homo_idx = InterpolatedCoupling._find_e1x_homo(mol, mf)
            sign = 1.0
        else:
            # MOC: restrict candidates to E1x (π_x) when available
            pi_x_labels = {"E1x", "e1x", "B1", "b1"}
            cand_idx = [j for j in occ_idx if orbsym[j] in pi_x_labels]
            if not cand_idx:
                cand_idx = list(occ_idx)

            # S-weighted overlap with the previous-step HOMO
            overlaps = {
                j: abs(float(prev_homo_coeff @ S_AO @ mf.mo_coeff[:, j]))
                for j in cand_idx
            }
            sorted_cands = sorted(overlaps, key=lambda j: overlaps[j], reverse=True)
            homo_idx = sorted_cands[0]
            sign = float(np.sign(
                float(prev_homo_coeff @ S_AO @ mf.mo_coeff[:, homo_idx])
            ))

            moc_overlap_arr[i] = overlaps[homo_idx]
            if len(sorted_cands) > 1:
                moc_overlap_2nd[i] = overlaps[sorted_cands[1]]

        prev_homo_coeff = sign * mf.mo_coeff[:, homo_idx]

        # --- Record ---
        homo_idx_arr[i]    = homo_idx
        homo_energy_arr[i] = float(mf.mo_energy[homo_idx])
        homo_irrep_list.append(str(orbsym[homo_idx]))
        occ_energies_lst.append(mf.mo_energy[occ_idx].copy())
        occ_irreps_lst.append([str(orbsym[j]) for j in occ_idx])

        virt_idx = np.where(np.asarray(mf.mo_occ) < 0.5)[0][:n_virt]
        ve = mf.mo_energy[virt_idx] if len(virt_idx) else np.array([])
        if len(ve) < n_virt:
            ve = np.pad(ve, (0, n_virt - len(ve)), constant_values=np.nan)
        virt_energies_lst.append(ve[:n_virt])

        if verbose:
            ovlp_str = (f"overlap={moc_overlap_arr[i]:.3f}"
                        if not np.isnan(moc_overlap_arr[i]) else "seed")
            print(f"HOMO={homo_idx:2d} ({orbsym[homo_idx]:4s})  "
                  f"ε={mf.mo_energy[homo_idx]:+.4f} Ha  {ovlp_str}")

    # --- Pack into uniform 2-D arrays ---
    n_occ_max = max(len(e) for e in occ_energies_lst)
    occ_energies_2d = np.full((n, n_occ_max), np.nan)
    for i, e in enumerate(occ_energies_lst):
        occ_energies_2d[i, :len(e)] = e

    return {
        "R":               R_grid,
        "homo_idx":        homo_idx_arr,
        "homo_energy":     homo_energy_arr,
        "homo_irrep":      homo_irrep_list,
        "moc_overlap":     moc_overlap_arr,
        "moc_overlap_2nd": moc_overlap_2nd,
        "occ_energies":    occ_energies_2d,
        "occ_irreps":      occ_irreps_lst,
        "virt_energies":   np.array(virt_energies_lst),
    }
