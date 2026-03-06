"""
Physical constants and atomic data for AED calculations.

All constants are in atomic units unless otherwise specified.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class PhysicalConstants:
    """Physical constants in atomic units."""

    # Fundamental constants
    hbar: float = 1.0  # Reduced Planck constant (a.u.)
    me: float = 1.0  # Electron mass (a.u.)
    e: float = 1.0  # Elementary charge (a.u.)
    a0: float = 1.0  # Bohr radius (a.u.)

    # Conversion factors
    hartree_to_ev: float = 27.211386245988  # eV per Hartree
    hartree_to_cm1: float = 219474.63136320  # cm^-1 per Hartree
    bohr_to_angstrom: float = 0.529177210903  # Angstrom per Bohr
    amu_to_me: float = 1822.888486209  # electron masses per amu

    # Time conversion
    au_time_to_s: float = 2.4188843265857e-17  # seconds per a.u. time

    # Speed of light in atomic units
    c_au: float = 137.035999084  # speed of light in a.u.

    # Boltzmann constant in atomic units
    k_B: float = 3.1668115634556e-6  # Hartree per Kelvin

    # Volume conversion for bimolecular rate constants
    bohr_to_cm: float = 5.29177210903e-9  # cm per Bohr
    bohr3_to_cm3: float = 1.48185e-25  # cm³ per Bohr³ (bohr_to_cm ** 3)

    def cm1_to_hartree(self, energy_cm1: float) -> float:
        """Convert energy from cm^-1 to Hartree."""
        return energy_cm1 / self.hartree_to_cm1

    def ev_to_hartree(self, energy_ev: float) -> float:
        """Convert energy from eV to Hartree."""
        return energy_ev / self.hartree_to_ev

    def angstrom_to_bohr(self, distance_ang: float) -> float:
        """Convert distance from Angstrom to Bohr."""
        return distance_ang / self.bohr_to_angstrom

    def rate_au_to_s1(self, rate_au: float) -> float:
        """Convert rate from a.u. to s^-1."""
        return rate_au / self.au_time_to_s

    def rate_au_to_cm3s(self, rate_au: float) -> float:
        """Convert bimolecular rate constant from a.u. (bohr³/a.u.time) to cm³/s."""
        return rate_au * self.bohr3_to_cm3 / self.au_time_to_s


# Global constants instance
CONSTANTS = PhysicalConstants()


# Atomic masses in amu (most abundant isotope)
ATOMIC_MASSES: Dict[str, float] = {
    "H": 1.00782503207,
    "D": 2.01410177785,  # Deuterium
    "He": 4.00260325415,
    "Li": 7.01600455,
    "Be": 9.0121822,
    "B": 11.0093054,
    "C": 12.0000000,
    "N": 14.0030740048,
    "O": 15.99491461956,
    "F": 18.99840322,
    "Ne": 19.9924401754,
    "Na": 22.9897692809,
    "Mg": 23.9850417,
    "Al": 26.98153863,
    "Si": 27.9769265325,
    "P": 30.97376163,
    "S": 31.97207100,
    "Cl": 34.96885268,
    "Ar": 39.9623831225,
    "K": 38.96370668,
    "Ca": 39.96259098,
    "Br": 78.9183371,
    "I": 126.904473,
}


# Electron affinities in eV (experimental values)
ELECTRON_AFFINITIES: Dict[str, float] = {
    "H": 0.754195,
    "Li": 0.618049,
    "B": 0.279723,
    "C": 1.262119,
    "N": -0.07,  # Negative (unstable)
    "O": 1.461112,
    "F": 3.401190,
    "Na": 0.547926,
    "Si": 1.389521,
    "P": 0.746607,
    "S": 2.077104,
    "Cl": 3.612724,
    "Br": 3.363588,
    "I": 3.059038,
    "OH": 1.8276,  # Hydroxyl radical
    "SH": 2.314,
    "NH": 0.370,
    "CH": 1.238,
    "NO": 0.026,
    "CN": 3.862,
}


def get_reduced_mass(atom1: str, atom2: str) -> float:
    """
    Calculate reduced mass for a diatomic system in atomic units (electron masses).

    Parameters
    ----------
    atom1 : str
        Symbol of first atom
    atom2 : str
        Symbol of second atom

    Returns
    -------
    float
        Reduced mass in atomic units (electron masses)
    """
    m1 = ATOMIC_MASSES[atom1] * CONSTANTS.amu_to_me
    m2 = ATOMIC_MASSES[atom2] * CONSTANTS.amu_to_me
    return (m1 * m2) / (m1 + m2)


def get_total_mass(atom1: str, atom2: str) -> float:
    """
    Calculate total mass for a diatomic system in atomic units.

    Parameters
    ----------
    atom1 : str
        Symbol of first atom
    atom2 : str
        Symbol of second atom

    Returns
    -------
    float
        Total mass in atomic units (electron masses)
    """
    m1 = ATOMIC_MASSES[atom1] * CONSTANTS.amu_to_me
    m2 = ATOMIC_MASSES[atom2] * CONSTANTS.amu_to_me
    return m1 + m2
