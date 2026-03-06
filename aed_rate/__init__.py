"""
AED Rate: Associative Electron Detachment rate calculations.

Computes rates for reactions like A⁻ + B → AB + e⁻ using
non-Born-Oppenheimer coupling theory (Acharya & Simons 1983, 1985).

Quick start
-----------
>>> from aed_rate import AEDSystem
>>> sys = AEDSystem.oh_system()
>>> print(sys.summary())
>>> k300 = sys.thermal_rate(300.0)   # cm³/s
"""

__version__ = "0.1.0"

from .aed_calculator import AEDSystem, VibDistribution
