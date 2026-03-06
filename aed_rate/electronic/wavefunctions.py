"""
Electronic wavefunction calculations using PySCF.

Computes electronic wavefunctions for anion and neutral species
at various bond lengths for use in AED rate calculations.
"""

import numpy as np
from typing import Tuple, List, Dict
from dataclasses import dataclass, field
try:
    from pyscf import gto, scf

    PYSCF_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False


@dataclass
class MolecularOrbitals:
    """Container for molecular orbital information at a geometry."""

    r: float  # Bond length in Bohr
    mo_coeff: np.ndarray  # MO coefficients (AO x MO)
    mo_energy: np.ndarray  # MO energies
    mo_occ: np.ndarray  # MO occupations
    overlap: np.ndarray  # AO overlap matrix
    total_energy: float  # Total electronic energy


@dataclass
class ElectronicState:
    """Container for electronic state information across geometries."""

    atom1: str
    atom2: str
    charge: int
    spin: int
    basis: str
    r_points: np.ndarray
    mo_data: List[MolecularOrbitals] = field(default_factory=list)


class ElectronicStructure:
    """
    Electronic structure calculator using PySCF.

    Computes Hartree-Fock wavefunctions for anion and neutral
    diatomic molecules at multiple geometries.

    Parameters
    ----------
    atom1 : str
        Symbol of first atom
    atom2 : str
        Symbol of second atom
    basis : str, optional
        Basis set name. Default is 'aug-cc-pVTZ'.
    """

    def __init__(self, atom1: str, atom2: str, basis: str = "aug-cc-pVTZ"):
        if not PYSCF_AVAILABLE:
            raise ImportError(
                "PySCF is required for electronic structure calculations. "
                "Install with: pip install pyscf"
            )

        self.atom1 = atom1
        self.atom2 = atom2
        self.basis = basis

        # Cache for computed wavefunctions
        self._anion_cache: Dict[float, MolecularOrbitals] = {}
        self._neutral_cache: Dict[float, MolecularOrbitals] = {}

    def _build_molecule(
        self, r: float, charge: int = 0, spin: int = 0
    ) -> "gto.Mole":
        """
        Build PySCF molecule object.

        Parameters
        ----------
        r : float
            Bond length in Bohr
        charge : int
            Total charge
        spin : int
            Spin multiplicity (2S)

        Returns
        -------
        gto.Mole
            PySCF molecule object
        """
        mol = gto.Mole()
        mol.atom = f"{self.atom1} 0 0 0; {self.atom2} 0 0 {r}"
        mol.basis = self.basis
        mol.charge = charge
        mol.spin = spin
        mol.unit = "Bohr"
        mol.build()
        return mol

    def compute_anion(
        self, r: float, spin: int = 0, use_cache: bool = True
    ) -> MolecularOrbitals:
        """
        Compute anion electronic wavefunction at given geometry.

        Parameters
        ----------
        r : float
            Bond length in Bohr
        spin : int
            Spin multiplicity (2S). Default is 0 (singlet).
        use_cache : bool
            Whether to use cached results

        Returns
        -------
        MolecularOrbitals
            Molecular orbital information
        """
        if use_cache and r in self._anion_cache:
            return self._anion_cache[r]

        mol = self._build_molecule(r, charge=-1, spin=spin)

        if spin == 0:
            mf = scf.RHF(mol)
        else:
            mf = scf.UHF(mol)

        mf.kernel()

        mo_data = MolecularOrbitals(
            r=r,
            mo_coeff=mf.mo_coeff,
            mo_energy=mf.mo_energy,
            mo_occ=mf.mo_occ,
            overlap=mol.intor("int1e_ovlp"),
            total_energy=mf.e_tot,
        )

        if use_cache:
            self._anion_cache[r] = mo_data

        return mo_data

    def compute_neutral(
        self, r: float, spin: int = 1, use_cache: bool = True
    ) -> MolecularOrbitals:
        """
        Compute neutral electronic wavefunction at given geometry.

        Parameters
        ----------
        r : float
            Bond length in Bohr
        spin : int
            Spin multiplicity (2S). Default is 1 (doublet).
        use_cache : bool
            Whether to use cached results

        Returns
        -------
        MolecularOrbitals
            Molecular orbital information
        """
        if use_cache and r in self._neutral_cache:
            return self._neutral_cache[r]

        mol = self._build_molecule(r, charge=0, spin=spin)

        if spin == 0:
            mf = scf.RHF(mol)
        else:
            mf = scf.UHF(mol)

        mf.kernel()

        # Handle UHF case - use alpha orbitals
        if isinstance(mf.mo_coeff, tuple):
            mo_coeff = mf.mo_coeff[0]
            mo_energy = mf.mo_energy[0]
            mo_occ = mf.mo_occ[0]
        else:
            mo_coeff = mf.mo_coeff
            mo_energy = mf.mo_energy
            mo_occ = mf.mo_occ

        mo_data = MolecularOrbitals(
            r=r,
            mo_coeff=mo_coeff,
            mo_energy=mo_energy,
            mo_occ=mo_occ,
            overlap=mol.intor("int1e_ovlp"),
            total_energy=mf.e_tot,
        )

        if use_cache:
            self._neutral_cache[r] = mo_data

        return mo_data

    def compute_along_curve(
        self,
        r_points: np.ndarray,
        charge: int = -1,
        spin: int = 0,
    ) -> ElectronicState:
        """
        Compute electronic structure at multiple geometries.

        Parameters
        ----------
        r_points : np.ndarray
            Bond lengths in Bohr
        charge : int
            Total charge (-1 for anion, 0 for neutral)
        spin : int
            Spin multiplicity (2S)

        Returns
        -------
        ElectronicState
            Electronic state information at all geometries
        """
        state = ElectronicState(
            atom1=self.atom1,
            atom2=self.atom2,
            charge=charge,
            spin=spin,
            basis=self.basis,
            r_points=r_points,
        )

        for r in r_points:
            if charge == -1:
                mo_data = self.compute_anion(r, spin=spin)
            else:
                mo_data = self.compute_neutral(r, spin=spin)
            state.mo_data.append(mo_data)

        return state

    def _align_mo_phases(
        self, coeff1: np.ndarray, coeff_ref: np.ndarray
    ) -> np.ndarray:
        """
        Align MO phases between two sets of coefficients.

        Ensures consistent phase for numerical derivatives.
        """
        aligned = coeff1.copy()
        for i in range(coeff1.shape[1]):
            overlap = np.dot(coeff1[:, i], coeff_ref[:, i])
            if overlap < 0:
                aligned[:, i] *= -1
        return aligned

    def _run_scf(
        self, r: float, charge: int, spin: int
    ) -> Tuple["gto.Mole", "scf.hf.SCF"]:
        """
        Run SCF calculation and return (mol, mf) pair.

        Parameters
        ----------
        r : float
            Bond length in Bohr
        charge : int
            Total charge
        spin : int
            Spin multiplicity (2S)

        Returns
        -------
        Tuple[gto.Mole, scf.hf.SCF]
            PySCF molecule and converged SCF objects
        """
        mol = self._build_molecule(r, charge=charge, spin=spin)
        if spin == 0:
            mf = scf.RHF(mol)
        else:
            mf = scf.UHF(mol)
        mf.kernel()
        return mol, mf

    def _get_homo_index(self, mf: "scf.hf.SCF") -> int:
        """Return index of highest occupied MO."""
        occ = mf.mo_occ
        if isinstance(occ, tuple):
            occ = occ[0]
        return int(np.where(occ > 0.5)[0][-1])

