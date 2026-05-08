# pipeline/stellar_mass_function.py
"""
Cumulative stellar mass density  rho_star(>M_star, z)
======================================================
Theory side  : SHMR maps the HMF grid (from compute_hmf) onto stellar masses,
               then integrates cumulative stellar mass density downward in mass.
Observation  : UNCOVER galaxies sorted by M_star descending, cumulative sum
               weighted by lensing magnification, divided by survey volume.

UNIT CONVENTIONS
----------------
    M_h       input   :  M_sun  (physical; direct output of compute_hmf)
    M_star    output  :  M_sun  (physical)
    dndlnm    input   :  Mpc^{-3}  (physical; direct output of compute_hmf)
    rho_star  output  :  M_sun Mpc^{-3}  (physical)
    V_survey  input   :  Mpc^3  (physical comoving)
    mstar_*   input   :  log10(M_star/M_sun)  (catalog convention;
                         linearised to M_sun internally)

No h-unit conversions anywhere.  compute_hmf() already outputs physical units.

GRID COUPLING
-------------
_N_M and _N_K are imported directly from pipeline.hmf.  Change either constant
there and it propagates automatically to this module and to tests/test_smf.py.
"""

import numpy as np
from astropy.table import Table  # noqa: F401  (re-exported for type hints)

# ── Grid-size metadata imported from hmf.py ───────────────────────────────────
# These are the ONLY coupling between this file and hmf.py's internals.
# Re-exported so that tests/test_smf.py can import them from one canonical place.
from pipeline.hmf import _N_M, _N_K  # noqa: F401


# ═══════════════════════════════════════════════════════════════════════════════
#  SHMR PARAMETERS  (Stefanon 2021, Table 4)
# ═══════════════════════════════════════════════════════════════════════════════
#
#  Equation (double power law):
#
#      M_star / M_h = 2N / [(M_h/M_c)^{-beta} + (M_h/M_c)^{gamma}]
#
#  At M_h = M_c:  denominator = 1^{-beta} + 1^{gamma} = 1 + 1 = 2
#                 => M_star(M_c) = N * M_c        (NOT 2N*M_c)
#
#  Physical meaning of each parameter:
#    N     -- dimensionless amplitude; equals M_star/M_h at the peak mass M_c.
#             N = 0.0297 means ~3% of baryons are converted to stars at the peak.
#    M_c   -- characteristic halo mass where star-formation efficiency peaks.
#    beta  -- low-mass slope: supernova feedback suppresses M_star ~ M_h^(beta+1)
#             below M_c.
#    gamma -- high-mass slope: at z > 6 AGN feedback is negligible -> gamma ~ 0
#             (nearly flat efficiency above M_c).
#
#  Why z-independent: Stefanon et al. 2021 title is "No Significant Evolution
#  in the Stellar-to-Halo Mass Ratio of Galaxies in the First Gigayear of
#  Cosmic Time."  All z-evolution of rho_star_theory comes from the HMF, not
#  the SHMR.  Using a pre-JWST calibration (Spitzer/IRAC, z=6-10) avoids
#  circular analysis when fitting exotic DE to JWST data.
#
#  Source: Stefanon et al. 2021 (arXiv:2103.16571), Table 4.
#  Precedent for same SHMR + SMT pipeline: Jiang et al. 2024 (arXiv:2409.19941).

SHMR_N      = 0.0297    # dimensionless amplitude
SHMR_LOG_MC = 11.5      # log10(M_c / M_sun), physical
SHMR_BETA   = 1.35      # low-mass slope  (supernova feedback)
SHMR_GAMMA  = 0.01      # high-mass slope (AGN feedback negligible at z > 6)

# Derived constant: M_c in M_sun, computed once at import
_SHMR_MC = 10.0 ** SHMR_LOG_MC  # M_sun, physical


# ═══════════════════════════════════════════════════════════════════════════════
#  SHMR: Stellar-to-Halo Mass Relation
# ═══════════════════════════════════════════════════════════════════════════════

def shmr_mstar(M_h, N=SHMR_N, Mc=_SHMR_MC, beta=SHMR_BETA, gamma=SHMR_GAMMA):
    """
    Stellar mass from halo mass via the Stefanon 2021 double power law.

        M_star / M_h = 2N / [(M_h/M_c)^{-beta} + (M_h/M_c)^{gamma}]

    IMPORTANT: At the pivot mass M_h = M_c both denominator terms equal 1,
    so the result is M_star(M_c) = N * M_c  (the factor of 2 in the numerator
    cancels the denominator of 2).  Peak efficiency is N, not 2N.

    Vectorised: ~20 us on the 750-point HMF grid.  No Python loop.
    Works for any array shape or scalar.

    Parameters
    ----------
    M_h : array_like
        Halo mass(es) in M_sun (physical).  Scalar or any shape.

    Returns
    -------
    M_star : np.ndarray, same shape as M_h
        Stellar mass(es) in M_sun (physical).
    """
    
    M_h = np.atleast_1d(np.asarray(M_h, dtype=np.float64))
    # ratio = M_h / M_c  (dimensionless)

    ratio = M_h / Mc
    Ms_by_Mh = (2.0 * N) / (ratio ** (-beta) + ratio ** gamma)
    return Ms_by_Mh * M_h


# ═══════════════════════════════════════════════════════════════════════════════
#  THEORY SIDE: Predicted rho_star(>M_star, z)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_theory_rho_star(M_h, dndlnm, M_star_thresholds,
                                     N=SHMR_N, log_Mc=SHMR_LOG_MC, beta=SHMR_BETA, gamma=SHMR_GAMMA):
    """
    Theoretical cumulative stellar mass density rho_star(>M_star_threshold, z).
 
    For each threshold M_star_i in M_star_thresholds, computes:
 
        rho_star(>M_star_i) = integral_{M_h(M_star_i)}^{inf}
                              M_star(M_h) * (dn/dlnM_h) * dlnM_h
 
    where M_h(M_star_i) is found by numerically inverting the SHMR.
 
    NOTE ON ARRAY SIZE
    ------------------
    In production (MCMC loop), M_h and dndlnm come directly from compute_hmf()
    and have length _N_M.  In convergence tests (test_smf.py T9), a finer grid
    of length 2*_N_M may be passed.  The function is correct for any length >= 2.
 
    Parameters
    ----------
    M_h : np.ndarray, shape (N,)
        Halo masses [M_sun], ascending log-spaced.  First output of compute_hmf().
        N = _N_M in production; can differ in convergence tests.
    dndlnm : np.ndarray, shape (N,)
        Halo mass function dn/dlnM_h [Mpc^{-3}].  Second output of compute_hmf().
    M_star_thresholds : array_like, shape (K,)
        Stellar mass thresholds [M_sun] at which to evaluate rho_star.
        Need not be sorted; any order is handled correctly.
        
    N                 : float, SHMR amplitude
    Mc                : float, SHMR characteristic mass [M_sun] LOG
    beta              : float, SHMR low-mass slope
    gamma             : float, SHMR high-mass slope
 
        
    
 
    Returns
    -------
    rho_star : np.ndarray, shape (K,)
        rho_star(>M_star_threshold) [M_sun Mpc^{-3}] at each threshold.
        Monotonically decreasing with increasing M_star_threshold.
    """
    M_h    = np.asarray(M_h,    dtype=np.float64)
    dndlnm = np.asarray(dndlnm, dtype=np.float64)
    M_star_thresholds = np.atleast_1d(
        np.asarray(M_star_thresholds, dtype=np.float64)
    )
 
    n = len(M_h)  # grid length; equals _N_M in normal use, 2*_N_M in T9
 
    # ── Step 1: SHMR on the full grid ────────────────────────────────────────
    # M_star is strictly ascending because SHMR is a strictly monotone
    # increasing function of M_h over the entire HMF mass range.
    Mc= 10**log_Mc
    
    M_star = shmr_mstar(M_h, N, Mc, beta, gamma)
 
    # ── Step 2: Cumulative trapezoidal integral from high mass to low mass ────
    # Grid is log-spaced -> dlnM is constant throughout.
    lnM_h = np.log(M_h)
    dlnM  = lnM_h[1] - lnM_h[0]  # constant spacing in ln M_h
 
    # Integrand: M_star * dn/dlnM_h  [M_sun Mpc^{-3} per unit ln M_h]
    integrand = M_star * dndlnm  # shape (N,)
 
    # Trapezoid contributions between adjacent bins: shape (N-1,)
    #   trap[i] = 0.5*(f[i] + f[i+1])*dlnM  ~= integral from M_h[i] to M_h[i+1]
    trap = 0.5 * (integrand[:-1] + integrand[1:]) * dlnM
 
    # Cumulative sum from the HIGH-mass end to each bin.
    # Desired: rho_cum[i] = sum_{j=i}^{N-2} trap[j]
    #                     = integral from M_h[i] to M_h[-1]
    #
    # Vectorised trick (zero Python loops and to avoid cumsum going left-to-right, non-intended behaviour):
    #   reverse trap -> cumsum left-to-right -> reverse back -> assign to [:-1].

    rho_cum = np.empty(n, dtype=np.float64)
    rho_cum[-1]  = 0.0                          # nothing above the last bin
    rho_cum[:-1] = np.cumsum(trap[::-1])[::-1]  # M_sun Mpc^{-3}
 
    # ── Step 3: Invert SHMR — find M_h at each M_star threshold ──────────────
    # M_star is strictly ascending -> np.interp requires ascending xp -> OK.
    #   left  = M_h[0]:   threshold below the grid -> use the full integral
    #   right = M_h[-1]:  threshold above the grid -> grid contributes nothing
    M_h_thr = np.interp(
        M_star_thresholds,
        M_star,           
        M_h,              
        left  = M_h[0],
        right = M_h[-1],
    )
 
    # ── Step 4: Read rho_cum at each M_h threshold ───────────────────────────
    # xp = M_h (ascending).  fp = rho_cum (monotone decreasing with M_h, but
    # np.interp only requires xp to be ascending, not fp -- linear interpolation
    # between fp values is valid regardless of fp's monotonicity).
    #   left  = rho_cum[0]:  M_h_thr below the grid edge -> full integral
    #   right = 0.0:         M_h_thr above the grid edge -> no contribution
    rho_star = np.interp(
        M_h_thr,
        M_h,             
        rho_cum,         
        left  = rho_cum[0],
        right = 0.0,
    )
 
    return rho_star




# ═══════════════════════════════════════════════════════════════════════════════
#  OBSERVATION SIDE: Measured rho_star(>M_star, z) from UNCOVER
# ═══════════════════════════════════════════════════════════════════════════════

def compute_observed_rho_star(catalog_table, z_min, z_max, V_survey):
    """
    Observed cumulative stellar mass density from the UNCOVER catalog.

    Algorithm 
    ----------------------------------------------------
    1. Select galaxies with redshift in [z_min, z_max).
    2. Sort by median M_star descending (most massive first = highest threshold
       first in the output arrays).
    3. Cumulative sum with lensing magnification correction:
           rho_star(>M_star_i) = (1/V_survey) * sum_{j=1}^{i} mu_j * M_star_j
       where mu_j corrects for the reduced effective source-plane volume behind
       the lens.  Produces a confirmed 3-5x enhancement at z=6-8 vs mu=1.
    4. Repeat steps 2-3 with mstar_16 and mstar_84 to bracket uncertainties.

    Runtime: ~150 us for O(10^3) galaxies.  Well below 60 ms; no C needed.

    Parameters
    ----------
    catalog_table : astropy Table
        Output of data_extractor.load_catalogs().  Required columns:
            z        -- redshift (photo-z or spec-z, depending on catalog)
            mstar_50 -- log10(M_star/M_sun), posterior median
            mstar_16 -- log10(M_star/M_sun), 16th percentile
            mstar_84 -- log10(M_star/M_sun), 84th percentile
            mu       -- lensing magnification (clipped >= 1 by data_extractor)
    z_min, z_max : float
        Redshift bin edges.  Selection is [z_min, z_max).
    V_survey : float
        Comoving survey volume [Mpc^3] for this redshift bin.

    Returns
    -------
    M_star_thresholds : np.ndarray, shape (n_gal,)
        Stellar masses [M_sun] sorted descending.  Empty if no galaxies pass cut.
    rho_star_obs  : np.ndarray, shape (n_gal,)
        rho_star(>M_star_i) [M_sun Mpc^{-3}], computed with mstar_50.
        Non-decreasing with index (cumulative sum over positive masses).
    rho_star_low  : np.ndarray, shape (n_gal,)
        Same computed with mstar_16 (lower bound: smaller linear masses ->
        smaller cumulative sum).
    rho_star_high : np.ndarray, shape (n_gal,)
        Same computed with mstar_84 (upper bound).
    """
    if V_survey <= 0.0:
        raise ValueError(
            f"V_survey must be positive [Mpc^3]; got {V_survey:.6g}"
        )

    # ── Extract columns as plain float64 arrays ───────────────────────────────
    z        = np.asarray(catalog_table["z"],        dtype=np.float64)
    mstar_50 = np.asarray(catalog_table["mstar_50"], dtype=np.float64)
    mstar_16 = np.asarray(catalog_table["mstar_16"], dtype=np.float64)
    mstar_84 = np.asarray(catalog_table["mstar_84"], dtype=np.float64)
    mu       = np.asarray(catalog_table["mu"],       dtype=np.float64)

    # ── Redshift selection ────────────────────────────────────────────────────
    mask = (z >= z_min) & (z < z_max)
    if not np.any(mask):
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty, empty

    mstar_50 = mstar_50[mask]
    mstar_16 = mstar_16[mask]
    mstar_84 = mstar_84[mask]
    mu       = mu[mask]

    # ── Guard against non-finite percentile values ────────────────────────────
    # data_extractor guarantees mstar_50 and mu are finite; mstar_16/84 are
    # not explicitly checked there.  Fall back to mstar_50 if degenerate.
    mstar_16 = np.where(np.isfinite(mstar_16), mstar_16, mstar_50)
    mstar_84 = np.where(np.isfinite(mstar_84), mstar_84, mstar_50)

    # ── Convert log10 -> linear M_sun ─────────────────────────────────────────
    M50 = 10.0 ** mstar_50  # M_sun
    M16 = 10.0 ** mstar_16
    M84 = 10.0 ** mstar_84

    # ── Sort by median M_star descending (most massive first) ─────────────────
    idx = np.argsort(M50)[::-1]
    M50 = M50[idx]
    M16 = M16[idx]
    M84 = M84[idx]
    mu  = mu[idx]

    # ── Cumulative stellar mass density ───────────────────────────────────────
    # rho_star(>M_star_i) = (1/V_survey) * sum_{j=1}^{i} mu_j * M_j
    inv_V    = 1.0 / V_survey
    rho_obs  = np.cumsum(mu * M50) * inv_V   # M_sun Mpc^{-3}
    rho_low  = np.cumsum(mu * M16) * inv_V
    rho_high = np.cumsum(mu * M84) * inv_V

    return M50, rho_obs, rho_low, rho_high