# pipeline/uvlf_conversion.py
"""
Stellar mass ↔ UV magnitude conversion + UVLF Jacobian
=======================================================

Converts stellar mass to observed UV absolute magnitude and provides
the Jacobian dM★/dM_UV needed for the UV luminosity function prediction.

PIPELINE POSITION
-----------------
    hmf.py  →  THIS FILE (SHMR + M★→M_UV)  →  uvlf_theory.py  →  uvlf_likelihood.py

CALIBRATION SOURCES
-------------------
    SHMR:           Stefanon et al. 2021 (ApJ 922, 29), Eq. 5, page 22.
                    Redshift-independent double power law fitted to z=6–10 merged data.

    M★–M_UV:        Stefanon et al. 2021, Table 3.
                    log₁₀(M★/M☉) = C(z) + D(z) × M_UV
                    We fit D(z) and the normalization as linear functions of z from the
                    five calibration redshifts z = 6, 7, 8, 9, 10.  This yields a
                    continuous relation valid at any z.

    UV slope β:     Cullen et al. 2023 (MNRAS 520, 14).
                    β_UV = −5.40 + (−0.17) × M_UV^df
                    Empirical correlation measured from JWST + ground-based data at z ≈ 8–16.

    Dust A_1600:    Meurer, Heckman & Calzetti 1999 (ApJ 521, 64).
                    A_1600 = max(4.43 + 1.99 × β_UV, 0)
                    IRX–β relation calibrated on local starbursts.
                    The max(0, ...) floor prevents unphysical negative attenuation for
                    galaxies bluer than β = −2.226 (typical at z > 10).

SPS ASSUMPTIONS (baked into the Stefanon calibration, not free parameters)
--------------------------------------------------------------------------
    SPS models:     Bruzual & Charlot (2003)
    IMF:            Salpeter (1955)  — matches Jiang et al. 2024 explicitly
    Metallicity:    0.2 Z☉
    SFH:            Constant, age ∈ [10⁶ yr, age of universe at z]
    Dust in SED:    Calzetti et al. (2000), A_V = 0–3 mag
    Nebular:        Lines + continuum via Cloudy v17.02
    Cosmology:      Ωm=0.3, ΩΛ=0.7, H₀=70 (reference cosmology for mass estimates)

    These assumptions are FIXED by the calibration.  The M★→M_UV conversion
    does not change with MCMC cosmology — only the HMF and comoving volumes do.

UNIT CONVENTIONS
----------------
    M_h, M_star :   M_sun (physical, linear — NOT log, NOT h-scaled)
    M_UV        :   AB absolute magnitude
    A_1600      :   magnitudes (≥ 0)
    dM★/dM_UV   :   M_sun per mag (negative; take |...| in UVLF calculation)
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
#  SHMR PARAMETERS
#  Source: Stefanon et al. 2021 (ApJ 922, 29), Eq. 5 and Section 6.3.2.
#          Fitted to abundance-matched data at z = 6–10, ALL redshifts merged.
#          Jiang et al. 2024 Eq. 5 uses the same form and cites the same source.
# ═══════════════════════════════════════════════════════════════════════════════

SHMR_N      = 0.0297    # Peak star-formation efficiency M★/M_h at M_h = M_c
                        # (the factor of 2 in 2N cancels the denominator of 2
                        # at M_h = M_c, so actual peak is N, not 2N)
SHMR_LOG_MC = 11.5      # log₁₀(M_c / M☉) — characteristic halo mass where
                        # efficiency peaks.  Stefanon: 11.5 ± 0.2
SHMR_BETA   = 1.35      # Low-mass slope (supernova feedback suppresses SF
                        # in low-mass halos).  Stefanon: 1.35 ± 0.26
SHMR_GAMMA  = 0.01      # High-mass slope (AGN feedback).  Set to 0.01 following
                        # Jiang et al. 2024 — AGN feedback negligible at z ≥ 8.
                        # NOTE: Stefanon's own fit gives γ = 0.4, but that was
                        # constrained by z ~ 0 data where AGN feedback matters.

# ═══════════════════════════════════════════════════════════════════════════════
#  M★–M_UV CALIBRATION PARAMETERS
#  Source: Stefanon et al. 2021 (ApJ 922, 29), Table 3.
#
#  The Stefanon relation at each z:
#      log₁₀(M★/M☉) = C(z) + D(z) × M_UV
#
#  Table 3 gives two quantities at z = 6, 7, 8, 9, 10:
#      D       — the slope (how fast log M★ changes per mag of M_UV)
#      log M★  — the stellar mass at the reference magnitude M_UV = −20.5
#
#  We fit both as linear functions of z (OLS, 5 points each):
#
#      D(z) = _D0 + _D1 × z
#      log M★|_{−20.5}(z) = _N0 + _N1 × z
#
#  Then C(z) = log M★|_{−20.5} + 20.5 × D(z)
#            = (_N0 + 20.5 _D0) + (_N1 + 20.5 _D1) z
#            = −5.862 + 0.5875 z
#
#  Verification (residuals at all 5 calibration redshifts):
#      z=6:  Δ(log M★) = +0.02     z=9:  Δ(log M★) = +0.03
#      z=7:  Δ(log M★) = −0.01     z=10: Δ(log M★) = +0.00
#      z=8:  Δ(log M★) = −0.04
#  All well within the ±0.1 dex measurement uncertainties.
# ═══════════════════════════════════════════════════════════════════════════════

_D0 = -0.764    # D(z) intercept: D(z) = _D0 + _D1 * z
_D1 =  0.035    # D(z) slope per unit redshift

_N0 =  9.80     # log M★|_{−20.5} intercept
_N1 = -0.130    # log M★|_{−20.5} slope per unit redshift


# ═══════════════════════════════════════════════════════════════════════════════
#  DUST PARAMETERS
#
#  Step 1: UV spectral slope β from intrinsic magnitude
#      β_UV = _P0 + _P1 × M_UV^df
#      Source: Cullen et al. 2023 (MNRAS 520, 14)
#      Physics: brighter galaxies are more massive → more dust → redder UV slope
#
#  Step 2: Dust attenuation from β
#      A_1600 = max(_Q0 + _Q1 × β_UV, 0)
#      Source: Meurer, Heckman & Calzetti 1999 (ApJ 521, 64)
#      Physics: dust simultaneously reddens UV (changes β) and dims it (A_1600)
#      The max(0,...) floor: Meurer was calibrated on local starbursts with
#      β > −2.  High-z galaxies can have β < −2.23 where the formula gives
#      A < 0 (unphysical).  We clamp to zero = no dust.
#
#  Combined: A_1600 = max(Q0 + Q1(P0 + P1 × M_UV^df), 0)
#  Zero-crossing: β_crit = −Q0/Q1 = −2.226, at M_UV^df = −18.67
#  Galaxies fainter than M_UV^df ≈ −18.7 get no dust correction.
# ═══════════════════════════════════════════════════════════════════════════════

_P0 = -5.40     # β–M_UV intercept      (Cullen et al. 2023)
_P1 = -0.17     # β–M_UV slope           (Cullen et al. 2023)

_Q0 =  4.43     # Meurer A_1600 intercept (Meurer et al. 1999)
_Q1 =  1.99     # Meurer A_1600 slope     (Meurer et al. 1999)

# Dust Jacobian factor: η = dM_UV/dM_UV^df when dust is active.
#
# Since M_UV = M_UV^df + A_1600(M_UV^df):
#     dM_UV/dM_UV^df = 1 + dA_1600/dM_UV^df = 1 + Q1 × P1
#

SIGMA_BETA = 0.34   # Bouwens et al. 2014 (ApJ 793, 115), scatter in beta at fixed M_UV

# ═══════════════════════════════════════════════════════════════════════════════
#  SHMR: Stellar-to-Halo Mass Relation
# ═══════════════════════════════════════════════════════════════════════════════

def shmr_mstar(M_h, N=SHMR_N, log_Mc=SHMR_LOG_MC, beta=SHMR_BETA, gamma=SHMR_GAMMA):
    """
    Stellar mass from halo mass via the Stefanon 2021 double power law.

        M★ / M_h = 2N / [(M_h/M_c)^{-β} + (M_h/M_c)^{γ}]

    At the pivot mass M_h = M_c: both power-law terms equal 1,
    denominator = 2, so M★(M_c) = N × M_c.  Peak efficiency is N.

    Parameters
    ----------
    M_h : array_like
        Halo mass(es) in M_sun (physical, linear).
    N : float
        Peak efficiency (dimensionless).
    Mc : float
        Characteristic halo mass in M_sun (linear, not log).
    beta : float
        Low-mass slope (> 0).
    gamma : float
        High-mass slope (> 0).

    Returns
    -------
    M_star : np.ndarray
        Stellar mass(es) in M_sun (physical), same shape as M_h.
    """
    M_h = np.atleast_1d(np.asarray(M_h, dtype=np.float64))
    
    Mc = 10**log_Mc
    
    ratio = M_h / Mc
    return M_h * (2.0 * N) / (ratio ** (-beta) + ratio ** gamma)


def shmr_mstar_and_jacobian(M_h, N=SHMR_N, log_Mc=SHMR_LOG_MC, beta=SHMR_BETA, gamma=SHMR_GAMMA):
    """
    Stellar mass AND analytical d ln M★ / d ln M_h in a single call.

    Avoids the numerical np.gradient which caused the sampling surfaces to be rough. The analytical
    derivative of the Stefanon double power law is:

        d ln M★        (1 + β) x^{-β}  +  (1 - γ) x^{γ}
        ────────  =   ──────────────────────────────────────
        d ln M_h              x^{-β}  +  x^{γ}

    where x = M_h / M_c.  Always positive for β > 0, γ < 1.

    Returns
    -------
    M_star        : np.ndarray — stellar mass [M_sun]
    dlnMs_dlnMh   : np.ndarray — dimensionless log-log derivative (> 0)
    """
    M_h = np.atleast_1d(np.asarray(M_h, dtype=np.float64))
    Mc    = 10**log_Mc
    x     = M_h / Mc
    xnb   = x ** (-beta)
    xg    = x ** gamma
    denom = xnb + xg

    M_star        = M_h * (2.0 * N) / denom
    dlnMs_dlnMh   = ((1.0 + beta) * xnb + (1.0 - gamma) * xg) / denom

    return M_star, dlnMs_dlnMh


# ═══════════════════════════════════════════════════════════════════════════════
#  STEFANON PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

def stefanon_params(z):
    """
    Redshift-dependent intercept C(z) and slope D(z) of the M★–M_UV relation.

        log₁₀(M★/M☉) = C(z) + D(z) × M_UV

    Fitted as linear functions of z from Stefanon et al. 2021 Table 3
    (five calibration redshifts z = 6, 7, 8, 9, 10).

    Parameters
    ----------
    z : float
        Redshift.

    Returns
    -------
    C : float
        Intercept of the log M★–M_UV relation.
    D : float
        Slope of the log M★–M_UV relation (negative).
    """
    D = _D0 + _D1 * z
    C = (_N0 + _N1 * z) + 20.5 * D
    return C, D


# ═══════════════════════════════════════════════════════════════════════════════
#  DUST-FREE UV MAGNITUDE
# ═══════════════════════════════════════════════════════════════════════════════

def dust_free_M_UV(M_star, z):
    """
    Intrinsic (dust-free) UV absolute magnitude from stellar mass.

        M_UV^df = [log₁₀(M★/M☉) − C(z)] / D(z)

    Since M_star is in solar mass units, log₁₀(M_star) already gives
    log₁₀(M★/M☉) — no additional division needed.

    Parameters
    ----------
    M_star : array_like
        Stellar mass in M_sun (linear, not log).
    z : float
        Redshift.

    Returns
    -------
    M_UV_df : np.ndarray
        Dust-free UV absolute magnitude (AB mag).
    """
    C, D = stefanon_params(z)
    return (np.log10(M_star) - C) / D


# ═══════════════════════════════════════════════════════════════════════════════
#  DUST ATTENUATION
# ═══════════════════════════════════════════════════════════════════════════════

def dust_attenuation(M_UV_df):
    """
    Dust attenuation at 1600 Å from the Meurer (1999) IRX–β relation.

        β_UV   = −5.40 + (−0.17) × M_UV^df        [Cullen et al. 2023]
        A_1600 = max(4.43 + 1.99 × β_UV, 0)        [Meurer et al. 1999]

    The max(0, ...) floor prevents unphysical negative attenuation for
    galaxies bluer than β = −2.226 (M_UV^df fainter than ~−18.7).
    Uses np.maximum for ELEMENT-WISE comparison (not np.max which
    returns the single largest value in the array).

    Parameters
    ----------
    M_UV_df : array_like
        Dust-free UV absolute magnitude.

    Returns
    -------
    A_1600 : np.ndarray
        Dust attenuation in magnitudes (≥ 0), same shape as input.
    """
    M_UV_df = np.asarray(M_UV_df, dtype=np.float64)
    beta_uv = _P0 + _P1 * M_UV_df
    return np.maximum(_Q0 + _Q1 * beta_uv, 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
#  OBSERVED UV MAGNITUDE
# ═══════════════════════════════════════════════════════════════════════════════

def observed_M_UV(M_star, z):
    """
    Observed UV absolute magnitude (including dust) from stellar mass.

        M_UV = M_UV^df + A_1600

    Combines the Stefanon M★→M_UV^df conversion with Meurer dust.

    Parameters
    ----------
    M_star : array_like
        Stellar mass in M_sun (linear, not log).
    z : float
        Redshift.

    Returns
    -------
    M_UV : np.ndarray
        Observed UV absolute magnitude (AB mag), same shape as M_star.
    """
    M_UV_df = dust_free_M_UV(M_star, z)
    A_1600  = dust_attenuation(M_UV_df)
    return M_UV_df + A_1600


# ═══════════════════════════════════════════════════════════════════════════════
#  COMBINED CONVERSION + JACOBIAN
# ═══════════════════════════════════════════════════════════════════════════════

def convert_and_jacobian(M_star, z):
    """
    Observed M_UV and Jacobian dM★/dM_UV in a single call.

    Returns both the observed UV magnitude and the derivative dM★/dM_UV
    needed for the UVLF:

        dn/dM_UV = (dn/dlnM_h) × (1/M_h) × |dM_h/dM★| × |dM★/dM_UV|

    Derivation of dM★/dM_UV
    -----------------------
    Full chain: M★ → M_UV^df → M_UV = M_UV^df + A_1600(M_UV^df)

    By chain rule:
        dM_UV/dM★ = (dM_UV/dM_UV^df) × (dM_UV^df/dM★)

    dM_UV^df/dM★ = 1 / (D(z) × M★ × ln10)
        from M_UV^df = [log₁₀(M★) − C] / D
        and d(log₁₀ x)/dx = 1/(x ln10)

    dM_UV/dM_UV^df = 1 + dA_1600/dM_UV^df
        = 1 + Q1 × P1 = 0.6617     when A_1600 > 0  (dusty)
        = 1                          when A_1600 = 0  (dust-free)
    This factor (η) captures the negative feedback: dust increases
    with intrinsic brightness, partially offsetting the brightening.

    Inverting:
        dM★/dM_UV = D(z) × M★ × ln(10) / η

    Parameters
    ----------
    M_star : array_like
        Stellar mass in M_sun (linear, not log).
    z : float
        Redshift.

    Returns
    -------
    M_UV : np.ndarray
        Observed UV absolute magnitude (AB mag).
    dMstar_dMUV : np.ndarray
        Jacobian dM★/dM_UV in M_sun/mag.  NEGATIVE (increasing M★ →
        decreasing M_UV).  The calling code (uvlf_theory.py) should
        take the absolute value when computing the UVLF.
    """
    M_star = np.atleast_1d(np.asarray(M_star, dtype=np.float64))

    # --- Stefanon parameters at this redshift ---
    C, D = stefanon_params(z)

    # --- Dust-free magnitude ---
    M_UV_df = (np.log10(M_star) - C) / D

    # --- Dust attenuation (element-wise max with zero) ---
    from scipy.special import ndtr
    
    beta = _P0 + _P1 * M_UV_df
    A_raw = _Q0 + _Q1 * beta                     # can be negative
    s = _Q1 * SIGMA_BETA                         # = 1.99 * 0.34 = 0.677
    t = A_raw / s
    Phi_t = ndtr(t)                              # Φ(t), vectorised
    phi_t = np.exp(-0.5 * t**2) / np.sqrt(2.0 * np.pi)  # φ(t)

    A_UV = A_raw * Phi_t + s * phi_t             # soft rectifier
    M_UV = M_UV_df + A_UV

    # Jacobian — smooth transition from eta=1 to eta=0.6617
    eta = 1.0 + _Q1 * _P1 * Phi_t
    dMstar_dMUV = (D * M_star * np.log(10.0)) / eta

    return M_UV, dMstar_dMUV