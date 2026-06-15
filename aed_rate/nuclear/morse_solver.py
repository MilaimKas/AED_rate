"""
Analytical/hybrid Morse wavefunction solver with Pekeris approximation for J > 0.

Provides exact or near-exact wavefunctions for the effective Morse potential:

  V_eff(R, J) = D_e(1 - e^{-β(R-R_e)})² + V_0 + J(J+1)/(2μR²)

For J = 0 the centrifugal term vanishes and the solutions are exact.
For J > 0 the Pekeris (1934) approximation replaces 1/R² with a
three-term Morse-basis expansion that restores the equation to Morse
form with modified parameters.

Bound states      — exact generalised-Laguerre polynomials via scipy
Scattering states — Numerov outward shooting on the Pekeris-corrected V_eff
Derivatives       — cumulative TISE integration (F'' = 2μ(V-E)F, exact)

The hybrid approach resolves two DVR pathologies that cause wrong coupling
integrals:

  1. Phase error  — DVR box-quantises continuum states (F(R_max)=0 forced),
     selecting an arbitrary standing-wave phase.  Numerov outward shooting
     from the inner wall (F(R_min)=0) gives the physical regular solution.
  2. Derivative amplitude error — the 3-point central difference
     underestimates dF/dR by sin(kΔ)/(kΔ) ≈ 0.89–0.99 in the inner well
     (k_local ≈ 25 Bohr⁻¹ for OH near R_e).  The TISE-based cumulative
     integral of F'' avoids this systematic damping entirely.

Why NOT the Tricomi-U analytical formula for scattering states
--------------------------------------------------------------
The exact formula Φ = z^s e^{-z/2} U(a,b,z) with a ≈ −λ + ik/β (λ ≈ 22)
requires computing U for small z (large R).  At small z the two linearly
independent terms each have magnitude ~Γ(−22±ik/β)^{-1} ≈ 10^{+21}, and
their difference must reproduce an O(1) oscillatory function — 21 digits of
catastrophic cancellation.  Even mpmath at 200 decimal places is unreliable
for arbitrary λ ∈ [10, 50], and the loop over 500 grid points is slow.
The Numerov/TISE approach is faster, unconditionally stable, and equally
accurate for coupling integrals.

References
----------
Morse (1929) Phys. Rev. 34, 57.
Pekeris (1934) Phys. Rev. 45, 98.
Dahl & Springborg (1988) Mol. Phys. 64, 629.
"""

import numpy as np
from scipy.special import genlaguerre
from typing import Optional, List

from ..electronic.potential import MorsePotential
from ._base_solver import BoundState, ScatteringState, WavefunctionSolver


# ---------------------------------------------------------------------------
# Pekeris parameter computation
# ---------------------------------------------------------------------------

def _pekeris_params(D_e: float, r_e: float, beta: float, V_0: float,
                    mu: float, J: int) -> dict:
    """
    Compute effective Morse parameters for angular momentum J via the
    Pekeris (1934) approximation.

    The centrifugal term J(J+1)/(2μR²) is expanded around R_e in a
    Morse-type basis:

        1/R² ≈ (1/R_e²) [c₀ + c₁ ξ + c₂ ξ²],  ξ = exp(-β(R-R_e))

    The coefficients c₀, c₁, c₂ match the function and its first two
    derivatives at R_e (three-point Hermite conditions).

    The resulting V_eff is a Morse potential with the same β but shifted
    D_e, R_e, V_0.  For J=0 the exact parameters are returned unchanged.

    Parameters
    ----------
    D_e, r_e, beta, V_0 : float
        Original Morse parameters.
    mu : float
        Reduced mass (a.u.).
    J : int
        Angular momentum quantum number.

    Returns
    -------
    dict with keys:
        D_e_eff, R_e_eff, V_0_eff — effective Morse parameters
        lam_eff                   — λ_eff = √(2μ D_e_eff) / β
        B_J                       — centrifugal constant at R_e (Hartree)
        c0                        — Pekeris c₀ coefficient (needed for E_coll_eff)
    """
    if J == 0:
        lam = np.sqrt(2.0 * mu * D_e) / beta
        return dict(D_e_eff=D_e, R_e_eff=r_e, V_0_eff=V_0,
                    lam_eff=lam, B_J=0.0, c0=1.0)

    # Dimensionless expansion parameter  p = 1/(β R_e)
    p = 1.0 / (beta * r_e)

    # Pekeris coefficients (matched to 1/R² and first two derivatives at R_e)
    c0 = 1.0 - 3.0 * p + 3.0 * p ** 2
    c1 = 4.0 * p - 6.0 * p ** 2
    c2 = 3.0 * p ** 2 - p           # = p(3p - 1)

    B_J = J * (J + 1) / (2.0 * mu * r_e ** 2)   # centrifugal constant (Ha)

    # Effective potential coefficients in V_eff = a₀ - a₁ξ + a₂ξ²  (+ V_0)
    a0 = D_e + B_J * c0
    a1 = 2.0 * D_e - B_J * c1
    a2 = D_e + B_J * c2

    if a1 <= 0.0 or a2 <= 0.0:
        raise ValueError(
            f"Pekeris approximation breaks down at J={J}: "
            f"a1={a1:.4f}, a2={a2:.4f}. "
            "The centrifugal term is too large for this β·R_e product."
        )

    # Map to Morse form  A(1 - B_coeff·ξ)² + C  ≡  A - 2AB_coeff·ξ + AB_coeff²·ξ²
    #   → AB_coeff² = a2,  2AB_coeff = a1  →  B_coeff = 2a2/a1
    B_coeff  = 2.0 * a2 / a1
    D_e_eff  = a1 ** 2 / (4.0 * a2)    # = A  (effective well depth)
    # New equilibrium: 1 - B_coeff·ξ = 0  →  ξ = 1/B_coeff  →  R_e_eff = R_e + ln(B_coeff)/β
    R_e_eff  = r_e + np.log(B_coeff) / beta
    # Constant offset:  C + V_0 = V_0 + a0 - A
    V_0_eff  = V_0 + a0 - D_e_eff

    lam_eff  = np.sqrt(2.0 * mu * D_e_eff) / beta

    return dict(D_e_eff=D_e_eff, R_e_eff=R_e_eff, V_0_eff=V_0_eff,
                lam_eff=lam_eff, B_J=B_J, c0=c0)


# ---------------------------------------------------------------------------
# Analytical wavefunction kernels
# ---------------------------------------------------------------------------

def _bound_wavefunction_kernel(
    R_grid: np.ndarray,
    v: int,
    lam_eff: float,
    R_e_eff: float,
    beta: float,
) -> np.ndarray:
    """
    Un-normalised Morse bound-state wavefunction on R_grid.

        ψ_v(R) = z^(λ-v-½) exp(-z/2) L_v^{2λ-2v-1}(z)

    where  z = 2λ_eff exp(-β(R - R_e_eff)).
    """
    z = 2.0 * lam_eff * np.exp(-beta * (R_grid - R_e_eff))

    # Laguerre order  α = 2λ-2v-1  (must be > -1 for a bound state)
    alpha = 2.0 * lam_eff - 2.0 * v - 1.0

    # Prefactor computed in log-space to prevent underflow deep in classically
    # forbidden region (where z is large and the exponential dominates)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_z = np.where(z > 0.0, np.log(z), -1e300)
    prefactor = np.exp((lam_eff - v - 0.5) * log_z - 0.5 * z)

    L_poly = genlaguerre(v, alpha)
    wf = prefactor * L_poly(z)
    return np.nan_to_num(wf, nan=0.0, posinf=0.0, neginf=0.0)


def _numerov_scattering_wf_and_deriv(
    R_grid: np.ndarray,
    V_eff_grid: np.ndarray,
    E_total: float,
    mu: float,
) -> tuple:
    """
    Outward Numerov shooting for the scattering wavefunction F(R) and its
    radial derivative dF/dR, both on R_grid.

    Boundary conditions (physical, inner-wall):
        F(R_min) = 0,   F'(R_min) = 1  (overall normalisation fixed later)

    The derivative is obtained by cumulative trapezoid integration of the
    TISE second derivative:

        F''(R) = 2μ[V_eff(R) - E_total] F(R)          (exact from TISE)

        dF/dR[n+1] = dF/dR[n] + dr/2 × (F''[n] + F''[n+1])

    This avoids the sin(kΔ)/(kΔ) amplitude damping of central-difference
    formulas.  For k_local ≈ 25 Bohr⁻¹ (inner Morse well) and dr ≈ 0.029
    Bohr, the 3-point central difference underestimates dF/dR by ~9%;
    the TISE integral introduces only O(h²) quadrature error, whose
    oscillatory nature prevents linear accumulation across the grid.

    Parameters
    ----------
    R_grid : np.ndarray
        Uniformly spaced radial grid (Bohr).
    V_eff_grid : np.ndarray
        Potential energy on the same grid (Hartree).  For J > 0 this
        should be the Pekeris-corrected effective Morse potential.
    E_total : float
        Total energy (Hartree), same reference frame as V_eff_grid.
    mu : float
        Reduced mass (a.u.).

    Returns
    -------
    (F, dF_dR) : tuple of np.ndarray
        Un-normalised wavefunction and its TISE-based radial derivative.
    """
    N  = len(R_grid)
    dr = R_grid[1] - R_grid[0]

    # g[n] = 2μ(E - V[n]) > 0 in classically allowed region
    g = 2.0 * mu * (E_total - V_eff_grid)

    # ------------------------------------------------------------------
    # Numerov outward propagation
    # ------------------------------------------------------------------
    F = np.zeros(N)
    F[0] = 0.0
    F[1] = dr   # → F'(R_min) ≈ 1.0 (normalisation applied after)

    c12 = dr ** 2 / 12.0
    for n in range(1, N - 1):
        num = (2.0 * F[n] * (1.0 - 5.0 * c12 * g[n])
               - F[n - 1] * (1.0 + c12 * g[n - 1]))
        den = 1.0 + c12 * g[n + 1]
        F[n + 1] = num / den

    # ------------------------------------------------------------------
    # Derivative via TISE:  F'' = -g F  (= 2μ(V-E)F)
    # Cumulative trapezoid integration avoids sin(kΔ)/(kΔ) damping.
    # ------------------------------------------------------------------
    F_pp = -g * F   # second derivative at each grid point (exact from TISE)

    dF = np.zeros(N)
    dF[0] = 1.0     # consistent with F[1] = dr, F[0] = 0
    for n in range(N - 1):
        dF[n + 1] = dF[n] + 0.5 * dr * (F_pp[n] + F_pp[n + 1])

    return F, dF


# ---------------------------------------------------------------------------
# Solver class
# ---------------------------------------------------------------------------

class MorseAnalyticSolver(WavefunctionSolver):
    """
    Nuclear wavefunction solver using analytical Morse wavefunctions.

    Bound states use the exact generalised-Laguerre formula; scattering
    states use Numerov outward shooting on the Pekeris-corrected effective
    Morse potential.  The scattering derivative is computed via cumulative
    TISE integration (F'' = 2μ(V-E)F), eliminating the sin(kΔ)/(kΔ)
    amplitude error of central-difference approaches.

    Parameters
    ----------
    potential : MorsePotential
        Must have attributes D_e, r_e, beta, V_0.
    reduced_mass : float
        Nuclear reduced mass in atomic units.
    r_min, r_max : float
        Radial grid bounds in Bohr.
    n_grid : int
        Number of radial grid points.
    """

    def __init__(
        self,
        potential: MorsePotential,
        reduced_mass: float,
        r_min: float = 0.5,
        r_max: float = 15.0,
        n_grid: int = 500,
    ) -> None:
        if not isinstance(potential, MorsePotential):
            raise TypeError(
                "MorseAnalyticSolver requires a MorsePotential instance. "
                "Use DVRWavefunctionSolver for general (tabulated) potentials."
            )
        super().__init__(potential, reduced_mass, r_min, r_max, n_grid)

        # Store Morse parameters directly for convenience
        self.D_e  = potential.D_e
        self.r_e  = potential.r_e
        self.beta = potential.beta
        self.V_0  = potential.V_0

        # J=0 λ
        self.lam = np.sqrt(2.0 * reduced_mass * self.D_e) / self.beta

        # Caches
        self._bound_cache:      dict = {}   # (v, J)      → BoundState
        self._scattering_cache: dict = {}   # (E_key, J)  → ScatteringState
        self._pekeris_cache:    dict = {}   # J           → params dict

    # ------------------------------------------------------------------
    # Pekeris parameters (cached per J)
    # ------------------------------------------------------------------

    def _pekeris(self, J: int) -> dict:
        """Return (cached) Pekeris effective Morse parameters for angular momentum J."""
        if J not in self._pekeris_cache:
            self._pekeris_cache[J] = _pekeris_params(
                self.D_e, self.r_e, self.beta, self.V_0, self.mu, J
            )
        return self._pekeris_cache[J]

    # ------------------------------------------------------------------
    # Bound states
    # ------------------------------------------------------------------

    def _n_max_bound(self, J: int) -> int:
        """Highest bound vibrational quantum number for given J."""
        p = self._pekeris(J)
        return int(p["lam_eff"] - 0.5)

    def _bound_energy_abs(self, v: int, J: int) -> float:
        """Absolute energy of bound state (v, J) using Pekeris parameters."""
        p = self._pekeris(J)
        lam_eff  = p["lam_eff"]
        D_e_eff  = p["D_e_eff"]
        V_0_eff  = p["V_0_eff"]
        omega_eff    = self.beta * np.sqrt(2.0 * D_e_eff / self.mu)
        # ω_e × x_e  =  ω_e² / (4 D_e)  (NOT just x_e = ω_e/(4D_e))
        omega_xe_eff = omega_eff ** 2 / (4.0 * D_e_eff)
        vp = v + 0.5
        return V_0_eff + omega_eff * vp - omega_xe_eff * vp ** 2

    def solve_bound_state(
        self,
        v: int,
        J: int = 0,
        energy_guess: Optional[float] = None,   # API compatibility only
    ) -> BoundState:
        """
        Bound-state wavefunction for quantum number (v, J).

        Uses the exact Morse Laguerre-polynomial formula with Pekeris-
        corrected effective Morse parameters for J > 0.
        """
        cache_key = (v, J)
        if cache_key in self._bound_cache:
            return self._bound_cache[cache_key]

        v_max = self._n_max_bound(J)
        if v > v_max:
            raise ValueError(
                f"State v={v} not bound at J={J} "
                f"(λ_eff={self._pekeris(J)['lam_eff']:.3f}, v_max={v_max})."
            )

        p = self._pekeris(J)
        energy = self._bound_energy_abs(v, J)

        wf = _bound_wavefunction_kernel(
            self.r_grid, v, p["lam_eff"], p["R_e_eff"], self.beta
        )
        wf, norm = self._box_normalize(wf)

        state = BoundState(
            v=v, J=J, energy=energy,
            r_grid=self.r_grid.copy(),
            wavefunction=wf,
            normalization=norm,
        )
        self._bound_cache[cache_key] = state
        return state

    def solve_all_bound_states(self, J: int = 0) -> List[BoundState]:
        """All bound vibrational states for given J."""
        v_max = self._n_max_bound(J)
        return [self.solve_bound_state(v, J) for v in range(v_max + 1)]

    # ------------------------------------------------------------------
    # Scattering states
    # ------------------------------------------------------------------

    def solve_scattering_state(
        self, E_collision: float, J: int = 0, normalization: str = "box"
    ) -> ScatteringState:
        """
        Numerov scattering wavefunction on the Pekeris-corrected Morse potential.

        Outward Numerov shooting from the inner wall (F(R_min)=0) gives the
        physical regular solution with the correct standing-wave phase, unlike
        DVR box eigenstates which enforce F(R_max)=0 as well.

        For J > 0, the Pekeris-corrected effective Morse potential is used and
        the effective collision energy E_coll_eff = E_coll - B_J × c₀ is
        checked to be positive (otherwise raises ValueError).

        Parameters
        ----------
        E_collision : float
            Collision energy above the anion dissociation limit (Hartree).
        J : int
            Collisional angular momentum.
        normalization : {"box", "unit_amplitude"}
            How to scale the wavefunction (and its derivative):
              - "box": ∫|F|² dR = 1 over [r_min, r_max]. L-dependent; used by
                the legacy box-normalized rate.
              - "unit_amplitude": asymptotically F → sin(kR − Jπ/2 + δ) with
                unit amplitude (the spherical-Bessel / amplitude-1 convention
                of Čížek 2001, Eq. 2.7). L-independent; required for absolute
                AED cross sections.

        The returned ScatteringState carries state._analytical_derivative,
        which wavefunction_derivative() returns directly (TISE-based, no
        sin(kΔ)/(kΔ) amplitude damping).  It also carries
        state._asymptotic_amplitude (the asymptotic amplitude of the returned,
        normalized F — ≈ 1.0 for "unit_amplitude") and state._normalization.
        """
        if normalization not in ("box", "unit_amplitude"):
            raise ValueError(
                f"normalization must be 'box' or 'unit_amplitude', "
                f"got {normalization!r}"
            )
        E_key = round(E_collision, 14)
        cache_key = (E_key, J, normalization)
        if cache_key in self._scattering_cache:
            return self._scattering_cache[cache_key]

        p = self._pekeris(J)
        B_J = p["B_J"]
        c0  = p["c0"]

        # Check accessibility: effective collision energy must be positive
        E_coll_eff = E_collision - B_J * c0
        if E_coll_eff <= 0.0:
            raise ValueError(
                f"Effective collision energy E_coll_eff = {E_coll_eff:.3e} Ha ≤ 0 "
                f"at J={J} (B_J·c₀ = {B_J*c0:.3e} Ha). "
                "The Pekeris barrier exceeds the collision energy."
            )

        # Asymptotic wavenumber (used for phase-shift extraction)
        k_eff = np.sqrt(2.0 * self.mu * E_coll_eff)

        # Pekeris-corrected effective Morse potential on the grid
        V_eff_grid = (
            p["D_e_eff"] * (1.0 - np.exp(-self.beta * (self.r_grid - p["R_e_eff"]))) ** 2
            + p["V_0_eff"]
        )
        # Total energy: dissociation limit of effective potential + E_coll_eff
        # = (D_e_eff + V_0_eff) + E_coll_eff = D_e + V_0 + E_collision (conserved)
        E_total = p["D_e_eff"] + p["V_0_eff"] + E_coll_eff

        wf, dwf = _numerov_scattering_wf_and_deriv(
            self.r_grid, V_eff_grid, E_total, self.mu
        )

        wf  = np.nan_to_num(wf,  nan=0.0, posinf=0.0, neginf=0.0)
        dwf = np.nan_to_num(dwf, nan=0.0, posinf=0.0, neginf=0.0)

        # Rescale the raw Numerov solution (arbitrary shooting amplitude).
        if normalization == "box":
            # ∫|F(R)|² dR = 1 over [r_min, r_max].  L-dependent (legacy path).
            wf, norm = self._box_normalize(wf)
            dwf /= norm
        else:  # "unit_amplitude"
            # Scale so the asymptotic envelope F → sin(kR − Jπ/2 + δ) has
            # unit amplitude (Čížek/Bessel convention; box length drops out).
            scale = self._asymptotic_amplitude(wf, dwf, k_eff)
            wf  = wf / scale
            dwf = dwf / scale

        # Amplitude of the returned (normalized) F — a built-in self-check:
        # ≈ 1.0 for "unit_amplitude", ≈ √(2/L) for "box".
        amp_out = self._asymptotic_amplitude(wf, dwf, k_eff)

        phase_shift = self._extract_phase_shift(
            self.r_grid[-100:], wf[-100:], k_eff
        )

        state = ScatteringState(
            E=E_collision, J=J,
            r_grid=self.r_grid.copy(),
            wavefunction=wf,
            phase_shift=phase_shift,
        )
        # Attach the exact analytical derivative for wavefunction_derivative()
        state._analytical_derivative = dwf
        state._asymptotic_amplitude  = amp_out
        state._normalization         = normalization

        self._scattering_cache[cache_key] = state
        return state

    def _extract_phase_shift(
        self, r: np.ndarray, f: np.ndarray, k_eff: float
    ) -> float:
        """Asymptotic phase shift via root-finding on two points."""
        from scipy.optimize import brentq
        r1, r2 = r[-50], r[-10]
        f1, f2 = f[-50], f[-10]
        def eq(delta: float) -> float:
            return f1 * np.sin(k_eff * r2 + delta) - f2 * np.sin(k_eff * r1 + delta)
        try:
            return brentq(eq, -np.pi, np.pi)
        except ValueError:
            return 0.0

    def _asymptotic_amplitude(
        self, f: np.ndarray, dwf: np.ndarray, k_eff: float
    ) -> float:
        """
        Asymptotic amplitude A of a scattering wavefunction F → A sin(kR + φ).

        Uses the envelope identity: for F = A sin(kR + φ) in the flat asymptotic
        region (local wavenumber → k_eff), F² + (F'/k)² = A² is constant, since
        sin² + cos² = 1.  This needs no phase fit and is robust to the standing-
        wave phase.  The median over the outer tail (excluding grid edges)
        suppresses the small residual wobble from the not-perfectly-flat
        potential and any edge effects in the derivative.

        Parameters
        ----------
        f : np.ndarray
            Wavefunction on r_grid.
        dwf : np.ndarray
            Its derivative dF/dR on r_grid (the analytical TISE derivative).
        k_eff : float
            Asymptotic wavenumber √(2μ E_coll_eff).

        Returns
        -------
        float
            Asymptotic amplitude A (falls back to an L²-based estimate if k_eff
            is non-positive or the envelope is degenerate).
        """
        n = len(f)
        lo = max(0, n - 200)
        hi = max(lo + 1, n - 5)          # drop the last few points (edge effects)
        if k_eff <= 0.0:
            # Standing wave ⟨sin²⟩ = ½ ⇒ A ≈ √2 · rms(f)
            return float(np.sqrt(2.0 * np.mean(f[lo:hi] ** 2))) or 1.0
        env = np.sqrt(f[lo:hi] ** 2 + (dwf[lo:hi] / k_eff) ** 2)
        A = float(np.median(env))
        return A if A > 0.0 else 1.0

    # ------------------------------------------------------------------
    # Derivative
    # ------------------------------------------------------------------

    def wavefunction_derivative(self, state) -> np.ndarray:
        """
        Radial derivative dF/dR of wavefunction.

        For scattering states produced by this solver (any J): returns the
        exact analytical derivative, avoiding finite-difference amplitude
        errors.

        For bound states or states from other solvers: falls back to the
        standard 3-point central-difference formula.
        """
        if hasattr(state, "_analytical_derivative"):
            return state._analytical_derivative

        # Fallback: central differences
        f = state.wavefunction
        deriv = np.zeros_like(f)
        deriv[1:-1] = (f[2:] - f[:-2]) / (2.0 * self.dr)
        deriv[0]    = (f[1] - f[0]) / self.dr
        deriv[-1]   = (f[-1] - f[-2]) / self.dr
        return deriv

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_interpolated_wavefunction(self, state: BoundState, kind: str = "cubic"):
        """Return an interpolating function for the wavefunction."""
        from scipy.interpolate import interp1d
        return interp1d(
            state.r_grid, state.wavefunction,
            kind=kind, bounds_error=False, fill_value=0.0,
        )
