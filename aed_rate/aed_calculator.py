"""
High-level interface for Associative Electron Detachment (AED) calculations.

Ties together potentials, electronic coupling, nuclear wavefunctions, and
rate calculators into a single AEDSystem object.

Typical usage
-------------
>>> sys = AEDSystem.oh_system()
>>> k = sys.thermal_rate(300.0)          # cm³/s
>>> dist = sys.vibrational_distribution(E_coll=0.001, J=0)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

from .electronic.potential import create_oh_system_acharya
from .electronic.coupling import ModelCoupling
from .rate.state_to_state import AEDRateCalculator
from .rate.thermal import ThermalRateCalculator
from .utils.constants import CONSTANTS, get_reduced_mass


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class VibDistribution:
    """Vibrational population distribution from an AED process."""

    E_collision: float            # Collision energy (Hartree)
    J: int                        # Initial angular momentum
    rates: Dict[int, float]       # {v': rate (a.u.)}
    total_rate: float             # Sum over all v' (a.u.)

    def normalized(self) -> Dict[int, float]:
        """Return fractional population per v'."""
        if self.total_rate == 0.0:
            return {}
        return {v: r / self.total_rate for v, r in self.rates.items()}

    def peak_v(self) -> int:
        """v' with highest rate."""
        return max(self.rates, key=self.rates.get)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class AEDSystem:
    """
    High-level interface for AED rate calculations.

    Bundles potential energy curves, electronic coupling, nuclear wavefunction
    solvers, and Fermi Golden Rule rate calculators into a single object.
    Once constructed, call :meth:`thermal_rate`, :meth:`vibrational_distribution`,
    or :meth:`state_to_state_rate` to compute observables.

    Parameters
    ----------
    anion_potential : PotentialEnergyCurve
        Anion PEC (e.g. OH⁻ Morse).
    neutral_potential : PotentialEnergyCurve
        Neutral PEC (e.g. OH Morse).
    EA : float
        Electron affinity in Hartree.
    reduced_mass : float
        Nuclear reduced mass in a.u. (electron masses).
    coupling : ModelCoupling or ElectronicCoupling
        Provides compute_coupling_at_r(R, E_e) → CouplingResult.
    solver_method : str
        Wavefunction solver: ``'dvr'`` (default) or ``'numerov'``.
    r_min, r_max : float
        Radial grid bounds in Bohr.
    n_grid : int
        Number of radial grid points.
    """

    def __init__(
        self,
        anion_potential,
        neutral_potential,
        EA: float,
        reduced_mass: float,
        coupling,
        solver_method: str = "dvr",
        r_min: float = 0.5,
        r_max: float = 15.0,
        n_grid: int = 500,
    ) -> None:
        self.anion_potential = anion_potential
        self.neutral_potential = neutral_potential
        self.EA = EA
        self.mu = reduced_mass
        self.coupling = coupling

        self._rate_calc = AEDRateCalculator(
            anion_potential=anion_potential,
            neutral_potential=neutral_potential,
            EA=EA,
            reduced_mass=reduced_mass,
            coupling=coupling,
            solver_method=solver_method,
            r_min=r_min,
            r_max=r_max,
            n_grid=n_grid,
        )
        self._thermal_calc = ThermalRateCalculator(
            rate_calculator=self._rate_calc,
            anion_potential=anion_potential,
            reduced_mass=reduced_mass,
        )

    # ------------------------------------------------------------------
    # Factory constructors
    # ------------------------------------------------------------------

    @classmethod
    def oh_system(
        cls,
        coupling=None,
        A_rad: float = 0.145,
        alpha_rad: float = 3.0,
        A_rot: float = 3.12,
        alpha_rot: float = 3.0,
        k_power: float = 1.0,
        solver_method: str = "morse",
        r_min: float = 0.5,
        r_max: float = 15.0,
        n_grid: int = 500,
    ) -> "AEDSystem":
        """
        Factory for the O⁻ + H → OH(v',J') + e⁻ system.

        Uses the exact Morse parameters of Acharya, Kendall, Simons (1984/1985).
        Pass ``coupling`` to supply a physical (ab initio / precomputed) coupling;
        if omitted, a Gaussian :class:`~aed_rate.electronic.coupling.ModelCoupling`
        stand-in is used.

        Parameters
        ----------
        coupling : ElectronicCoupling | InterpolatedCoupling | ModelCoupling, optional
            Electronic coupling provider.  Recommended: an ab initio coupling,
            e.g. ``InterpolatedCoupling.from_npz('oh.npz')``.  If None, a Gaussian
            ModelCoupling is built from the ``A_rad … k_power`` parameters below
            (sanity-check only).
        A_rad, alpha_rad, A_rot, alpha_rot, k_power : float
            Gaussian ModelCoupling parameters, used only when ``coupling`` is None.
            (amplitudes a.u.; widths Bohr⁻²; k_power=1 for the low-k OPW limit.)
        solver_method : str
            Nuclear wavefunction solver: ``'morse'`` (default; required for cross
            sections), ``'dvr'``, or ``'numerov'``.
        r_min, r_max : float
            Radial grid bounds in Bohr.
        n_grid : int
            Number of radial grid points (use ≳ 6000 for converged cross sections).

        Returns
        -------
        AEDSystem
            Pre-configured for the OH⁻/OH system.
        """
        anion_pot, neutral_pot, EA = create_oh_system_acharya()
        mu = get_reduced_mass("O", "H")

        if coupling is None:
            # Gaussian stand-in, centred at the anion equilibrium (sanity-check).
            coupling = ModelCoupling(
                R0=anion_pot.r_e,
                A_rad=A_rad,
                alpha_rad=alpha_rad,
                A_rot=A_rot,
                alpha_rot=alpha_rot,
                k_power=k_power,
            )

        return cls(
            anion_potential=anion_pot,
            neutral_potential=neutral_pot,
            EA=EA,
            reduced_mass=mu,
            coupling=coupling,
            solver_method=solver_method,
            r_min=r_min,
            r_max=r_max,
            n_grid=n_grid,
        )

    # ------------------------------------------------------------------
    # State-resolved rates
    # ------------------------------------------------------------------

    def state_to_state_rate(
        self,
        E_collision: float,
        J: int,
        v_prime: int,
        J_prime: Optional[int] = None,
    ) -> float:
        """
        Rate for a single (E, J) → v' transition.

        If J_prime is given, returns the rate for that specific final
        angular momentum channel. Otherwise returns the sum over
        J' = {J-1, J, J+1} (the physically observable quantity).

        Parameters
        ----------
        E_collision : float
            Collision energy above the anion dissociation limit (Hartree).
        J : int
            Initial angular momentum.
        v_prime : int
            Final vibrational quantum number.
        J_prime : int, optional
            Final angular momentum. If None, sums over all allowed J'.

        Returns
        -------
        float
            Rate in atomic units (Hartree / ħ).
        """
        if J_prime is not None:
            result = self._rate_calc.state_to_state_rate(
                E_collision, J, v_prime, J_prime
            )
            return result.rate
        return self._rate_calc.summed_rate(E_collision, J, v_prime)

    def vibrational_distribution(
        self,
        E_collision: float,
        J: int = 0,
        v_max: Optional[int] = None,
    ) -> VibDistribution:
        """
        Rate into each v' at fixed (E, J), summed over J'.

        Parameters
        ----------
        E_collision : float
            Collision energy in Hartree.
        J : int
            Initial angular momentum.
        v_max : int, optional
            Maximum v' to include. Defaults to all neutral bound states.

        Returns
        -------
        VibDistribution
            Contains dict {v': rate}, total rate, and helper methods.
        """
        raw = self._rate_calc.vibrational_distribution(E_collision, J, v_max)
        total = sum(raw.values())
        return VibDistribution(
            E_collision=E_collision,
            J=J,
            rates=raw,
            total_rate=total,
        )

    # ------------------------------------------------------------------
    # Cross sections (absolute, box-independent — the recommended observable)
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_sigma(sigma_a0: float, unit: str) -> float:
        """Convert a cross section from a₀² to the requested unit."""
        if unit in ("a0^2", "au", "bohr^2"):
            return sigma_a0
        a = CONSTANTS.bohr_to_angstrom
        if unit in ("Angstrom^2", "ang^2", "A^2"):
            return sigma_a0 * a ** 2
        if unit == "cm^2":
            return sigma_a0 * (a * 1.0e-8) ** 2
        raise ValueError(
            f"Unknown unit {unit!r}. Use 'a0^2', 'Angstrom^2', or 'cm^2'."
        )

    def cross_section(
        self,
        E_collision: float,
        J: int,
        v_prime: int,
        J_prime: Optional[int] = None,
        unit: str = "a0^2",
    ) -> float:
        """
        State-to-state AED cross section σ_{v',J→J'}(E).

        Uses the Čížek (2001) convention (unit-amplitude scattering state) — an
        absolute, box-independent observable, unlike :meth:`state_to_state_rate`.

        Parameters
        ----------
        E_collision : float
            Collision energy in Hartree.
        J, v_prime : int
            Initial angular momentum and final vibrational quantum number.
        J_prime : int, optional
            Final angular momentum. If None, sums over J' ∈ {J−1, J, J+1}.
        unit : str
            'a0^2' (default), 'Angstrom^2', or 'cm^2'.

        Returns
        -------
        float
            Cross section in the requested unit.
        """
        if J_prime is not None:
            sigma = self._rate_calc.cross_section_state_to_state(
                E_collision, J, v_prime, J_prime,
            ).sigma
        else:
            sigma = 0.0
            for Jp in (J - 1, J, J + 1):
                if Jp < 0:
                    continue
                sigma += self._rate_calc.cross_section_state_to_state(
                    E_collision, J, v_prime, Jp,
                ).sigma
        return self._convert_sigma(sigma, unit)

    def total_cross_section(
        self,
        E_collision: float,
        J: int,
        unit: str = "a0^2",
    ) -> float:
        """
        Total AD cross section for initial partial wave J (sum over v', J').

        Parameters
        ----------
        E_collision : float
            Collision energy in Hartree.
        J : int
            Initial angular momentum.
        unit : str
            'a0^2' (default), 'Angstrom^2', or 'cm^2'.

        Returns
        -------
        float
            σ(E; J) in the requested unit.
        """
        sigma = self._rate_calc.total_cross_section(E_collision, J)
        return self._convert_sigma(sigma, unit)

    def sigma_AD(
        self,
        E_collision: float,
        unit: str = "a0^2",
    ) -> float:
        """
        Total AD cross section σ_AD(E), summed over all initial partial waves J.

        Parameters
        ----------
        E_collision : float
            Collision energy in Hartree.
        unit : str
            'a0^2' (default), 'Angstrom^2', or 'cm^2'.

        Returns
        -------
        float
            σ_AD(E) in the requested unit.
        """
        sigma = self._rate_calc.total_cross_section_all_J(E_collision)
        return self._convert_sigma(sigma, unit)

    # ------------------------------------------------------------------
    # Thermal rate constant
    # ------------------------------------------------------------------

    def thermal_rate(
        self,
        T: float,
        n_energy: int = 32,
        v_prime_max: Optional[int] = None,
        J_step: int = 1,
    ) -> float:
        """
        Thermal rate constant k(T) in cm³/s.

        Uses Gauss-Laguerre quadrature for the Boltzmann energy integral
        and a partial-wave sum over J up to J_max(E).

        Parameters
        ----------
        T : float
            Temperature in Kelvin.
        n_energy : int
            Number of Gauss-Laguerre quadrature points (default 32).
        v_prime_max : int, optional
            Maximum v' to include. Defaults to all neutral bound states.
        J_step : int
            Step size for J summation (use > 1 to speed up).

        Returns
        -------
        float
            k(T) in cm³/s.
        """
        return self._thermal_calc.thermal_rate_constant(
            T, n_energy=n_energy, v_prime_max=v_prime_max, J_step=J_step
        )

    def thermal_rate_vs_temperature(
        self,
        T_array: np.ndarray,
        n_energy: int = 32,
        v_prime_max: Optional[int] = None,
        J_step: int = 1,
    ) -> np.ndarray:
        """
        k(T) for an array of temperatures.

        Parameters
        ----------
        T_array : np.ndarray
            Temperatures in Kelvin.

        Returns
        -------
        np.ndarray
            k(T) in cm³/s, same shape as T_array.
        """
        return self._thermal_calc.thermal_rate_vs_temperature(
            T_array, n_energy=n_energy, v_prime_max=v_prime_max, J_step=J_step
        )

    # ------------------------------------------------------------------
    # Bound state info
    # ------------------------------------------------------------------

    def neutral_bound_states(self, J: int = 0) -> list:
        """
        Return all neutral bound state energies at angular momentum J.

        Returns
        -------
        list of BoundState
            Energies on the absolute PEC scale (anion min = 0).
        """
        return self._rate_calc.neutral_solver.solve_all_bound_states(J=J)

    def anion_bound_states(self, J: int = 0) -> list:
        """
        Return all anion bound state energies at angular momentum J.

        Returns
        -------
        list of BoundState
            Energies on the absolute PEC scale (anion min = 0).
        """
        return self._rate_calc.anion_solver.solve_all_bound_states(J=J)

    # ------------------------------------------------------------------
    # Diagnostics / visualization
    # ------------------------------------------------------------------

    def diagnostic(
        self,
        E_collision: float,
        J: int = 0,
        v_prime: Optional[int] = None,
        n_R_coupling: int = 60,
        save_dir: Optional[str] = None,
        slab: float = 0.4,
    ) -> Dict[str, object]:
        """
        Render every step of the cross-section calculation as a figure.

        Walks the pipeline and calls the matching :mod:`aed_rate.utils.plotting`
        function for each ingredient, for a representative transition
        (E_collision, J → v', J'=J):

          1. ``potentials``    — anion/neutral PECs + EA gap
          2. ``bound_states``  — neutral bound levels on the PEC
          3. ``scattering``    — anion scattering state F_E(R) + V_eff
          4. ``scatt_deriv``   — dF_E/dR (the fast oscillator)
          5. ``coupling``      — m_rad(R), m_rot(R) over a coarse R grid
          6. ``integrand``     — χ_{v'}·m_rad·dF_E/dR and its running integral
          7. ``electronic``    — ∂φ/∂R and the OPW φ_k *(only if the coupling
             exposes ``compute_coupling_intermediates``, e.g. ElectronicCoupling)*

        The coupling is sampled on a coarse R grid (``n_R_coupling`` points) and
        spline-interpolated onto the solver grid for the integrand, so the method
        stays fast for any coupling type (it never evaluates an ab initio coupling
        on the full fine grid).

        Parameters
        ----------
        E_collision : float
            Collision energy in Hartree.
        J : int
            Initial angular momentum.
        v_prime : int, optional
            Final vibrational level for the scattering/integrand panels.
            Default: the highest energetically accessible v' at this (E, J).
        n_R_coupling : int
            Number of R points for the coupling-curve panel.
        save_dir : str, optional
            If given, write ``<step>.png`` into this directory.
        slab : float
            Molecular-plane slab half-thickness (Bohr) for the electronic panel.

        Returns
        -------
        dict
            ``{step_name: matplotlib.figure.Figure}`` for each rendered step.
        """
        from .utils import plotting
        from scipy.interpolate import interp1d

        rc = self._rate_calc
        anion, neutral = self.anion_potential, self.neutral_potential
        R_grid = rc.anion_solver.r_grid

        # --- pick a representative v' from energetics alone (no coupling eval) ---
        neutral_states = rc.neutral_solver.solve_all_bound_states(J=J)
        accessible = [
            s.v for s in neutral_states
            if (E_collision + anion.D_e - s.energy) > 0.0   # E_e > 0
        ]
        if not accessible:
            raise ValueError(
                f"No vibrational channel is open at E_coll={E_collision:.3e} Ha, J={J}."
            )
        if v_prime is None:
            v_prime = accessible[-1]
        elif v_prime not in accessible:
            raise ValueError(f"v'={v_prime} is not accessible; open channels: {accessible}")

        bound = rc.neutral_solver.solve_bound_state(v_prime, J)
        E_e = E_collision + anion.D_e - bound.energy        # ejected-electron KE

        # --- scattering state (unit amplitude if the solver supports it) ---
        try:
            scatt = rc.anion_solver.solve_scattering_state(
                E_collision, J, normalization="unit_amplitude",
            )
        except TypeError:
            # Solver lacks the normalization kwarg (e.g. DVR/Numerov).
            scatt = rc.anion_solver.solve_scattering_state(E_collision, J)
        except ValueError as exc:
            # The anion scattering channel is closed at this J (Pekeris barrier).
            raise ValueError(
                f"The anion scattering channel is closed at J={J} for "
                f"E_coll={E_collision:.3e} Ha (centrifugal barrier exceeds E). "
                "Use a lower J for the diagnostic."
            ) from exc
        dF_dR = rc.anion_solver.wavefunction_derivative(scatt)

        # --- coupling on a coarse R grid, then spline m_rad onto the solver grid ---
        R_coarse = np.linspace(R_grid[0], R_grid[-1], n_R_coupling)
        coup = self.coupling.compute_coupling_curve(R_coarse, E_e)
        R_c = np.array([c.R for c in coup])
        m_rad_c = np.array([c.m_rad.real for c in coup])
        m_rad_grid = interp1d(
            R_c, m_rad_c, kind="cubic", bounds_error=False, fill_value=0.0,
        )(R_grid)

        figs: Dict[str, object] = {}
        figs["potentials"] = plotting.plot_potential_curves(anion, neutral, self.EA)[0]
        figs["bound_states"] = plotting.plot_bound_states(neutral_states, neutral)[0]
        figs["scattering"] = plotting.plot_scattering_state(scatt, anion, self.mu, J=J)[0]
        figs["scatt_deriv"] = plotting.plot_scattering_derivative(scatt, dF_dR)[0]
        figs["coupling"] = plotting.plot_coupling_curve(coup)[0]
        figs["integrand"] = plotting.plot_coupling_integrand(
            R_grid, bound.wavefunction, m_rad_grid, dF_dR,
            label=f"v'={v_prime}, k_e={np.sqrt(2*E_e):.3f}",
        )[0]

        # Step 7 only if the coupling exposes the electronic intermediates.
        if hasattr(self.coupling, "compute_coupling_intermediates"):
            inter = self.coupling.compute_coupling_intermediates(anion.r_e, E_e)
            figs["electronic"] = plotting.plot_electronic_intermediates(inter, slab=slab)[0]

        if save_dir is not None:
            import os
            os.makedirs(save_dir, exist_ok=True)
            for name, fig in figs.items():
                fig.savefig(os.path.join(save_dir, f"{name}.png"), dpi=130)

        return figs

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """
        Return a human-readable summary of the system parameters.

        Returns
        -------
        str
            Multi-line system description.
        """
        anion = self.anion_potential
        neutral = self.neutral_potential

        # Count bound states
        anion_states = self.anion_bound_states(J=0)
        neutral_states = self.neutral_bound_states(J=0)

        ea_ev = self.EA * CONSTANTS.hartree_to_ev

        lines = [
            "=" * 60,
            "  AED System Summary",
            "=" * 60,
            "",
            "  Anion PEC:",
            f"    D_e   = {anion.D_e:.4f} Ha  ({anion.D_e * CONSTANTS.hartree_to_ev:.3f} eV)",
            f"    R_e   = {anion.r_e:.4f} Bohr",
            f"    beta  = {anion.beta:.4f} Bohr⁻¹",
            f"    Bound states (J=0): {len(anion_states)}",
            "",
            "  Neutral PEC:",
            f"    D_e   = {neutral.D_e:.4f} Ha  ({neutral.D_e * CONSTANTS.hartree_to_ev:.3f} eV)",
            f"    R_e   = {neutral.r_e:.4f} Bohr",
            f"    beta  = {neutral.beta:.4f} Bohr⁻¹",
            f"    V_0   = {neutral.V_0:.4f} Ha  (= EA)",
            f"    Bound states (J=0): {len(neutral_states)}",
            "",
            f"  Electron affinity: {ea_ev:.4f} eV",
            f"  Reduced mass:      {self.mu:.1f} a.u.",
            "",
            "  Coupling:",
            f"    Type   = {type(self.coupling).__name__}",
        ]

        # Add ModelCoupling-specific info
        if hasattr(self.coupling, "A_rad"):
            lines += [
                f"    R0     = {self.coupling.R0:.4f} Bohr",
                f"    A_rad  = {self.coupling.A_rad:.4f}  alpha_rad = {self.coupling.alpha_rad:.2f} Bohr⁻²",
                f"    A_rot  = {self.coupling.A_rot:.4f}  alpha_rot = {self.coupling.alpha_rot:.2f} Bohr⁻²",
                f"    k_power= {self.coupling.k_power:.1f}",
            ]

        lines.append("=" * 60)
        return "\n".join(lines)

    def __repr__(self) -> str:
        n_anion = len(self.anion_bound_states())
        n_neutral = len(self.neutral_bound_states())
        return (
            f"AEDSystem(EA={self.EA * CONSTANTS.hartree_to_ev:.3f} eV, "
            f"mu={self.mu:.0f} a.u., "
            f"coupling={type(self.coupling).__name__}, "
            f"anion_v={n_anion}, neutral_v={n_neutral})"
        )
