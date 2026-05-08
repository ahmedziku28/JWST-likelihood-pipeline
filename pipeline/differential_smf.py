# pipeline/differential_smf.py
"""
Differential (binned) stellar mass function — theory and observation
=====================================================================
Companion to pipeline/stellar_mass_function.py, which provides the CUMULATIVE
SMF.  The differential version bins galaxies into disjoint stellar mass bins,
producing approximately independent data points and a valid diagonal likelihood.

PIPELINE POSITION
-----------------
    CLASS -> compute_hmf (hmf.py) -> [this file] -> jwst_likelihood.py

UNIT CONVENTIONS  (identical to stellar_mass_function.py)
---------------------------------------------------------
    M_h       :  M_sun  (physical)
    M_star    :  M_sun  (physical)
    dndlnm    :  Mpc^{-3}
    phi       :  Mpc^{-3} dex^{-1}       (differential number density)
    rho_bin   :  M_sun Mpc^{-3} dex^{-1} (differential mass density)
    V_survey  :  Mpc^3  (physical comoving)
    mstar_*   :  log10(M_star/M_sun)  (catalog; linearised internally)

BIN EDGE CONVENTION
-------------------
Bin edges are fixed at module level for comparability.
    Bin width   0.5 dex      — Standard in the high-z SMF literature.  Wide
                               enough for nonzero counts even in the z=15-20
                               bin; narrow enough to resolve the SHMR slope.
Bins with zero observed galaxies are skipped at likelihood evaluation time
via the n_gal > 0 mask — no adaptive logic is needed or used.

TWO OBSERVABLES
---------------
Both the number density phi = dn/dlog10(M_star) [Mpc^{-3} dex^{-1}] and the
mass density rho = drho_star/dlog10(M_star) [M_sun Mpc^{-3} dex^{-1}] are
provided.  The cumulative observable in stellar_mass_function.py is a mass
density, so compute_*_differential_rho keeps the analysis more directly
comparable.  compute_*_differential_smf (number density) is more sensitive to
the low-mass slope of the SHMR.  Both are available; the likelihood chooses.
"""

import numpy as np

from pipeline.stellar_mass_function import (
    shmr_mstar,
    SHMR_N,
    SHMR_LOG_MC,
    SHMR_BETA,
    SHMR_GAMMA,
)
from pipeline.hmf import _N_M, _N_K


# ═══════════════════════════════════════════════════════════════════════════════
#  FIXED BIN EDGES
# ═══════════════════════════════════════════════════════════════════════════════

# Default stellar mass bin edges fixed for comparability between redshift bins.
DEFAULT_LOG10_MSTAR_BINS = np.arange(5.0, 15.5, 0.5)


# ═══════════════════════════════════════════════════════════════════════════════
#  INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _invert_shmr(log10_edges, M_star_grid, M_h_grid):
    """
    Convert stellar-mass bin edges to halo-mass edges by inverting the SHMR.

    The SHMR is strictly monotone increasing over the HMF grid, so np.interp
    with ascending xp is exact.  Left/right clamping to grid endpoints is safe:
    the lowest observed stellar mass (~10^5.6 M_sun) maps to a halo mass well
    above the grid floor, so the left clamp is never activated by real data.

    Parameters
    ----------
    log10_edges  : np.ndarray (N_edges,)  — bin edges in log10(M_star/M_sun)
    M_star_grid  : np.ndarray (N_grid,)   — M_star on HMF grid [M_sun], ascending
    M_h_grid     : np.ndarray (N_grid,)   — M_h on HMF grid [M_sun], ascending

    Returns
    -------
    M_h_edges : np.ndarray (N_edges,)  — halo mass edges [M_sun]
    """
    return np.interp(10.0 ** log10_edges, M_star_grid, M_h_grid,
                     left=M_h_grid[0], right=M_h_grid[-1])


def _integrate_bin(lnM_h, integrand, M_h, M_h_lo, M_h_hi):
    """
    Trapezoidal integral of integrand over HMF grid points in [M_h_lo, M_h_hi].

    searchsorted with left/right sides gives a half-open interval so abutting
    bins share no grid points and there is no double-counting at boundaries.

    Returns 0.0 if fewer than 2 grid points fall in range.
    """
    idx_lo = np.searchsorted(M_h, M_h_lo, side='left')
    idx_hi = np.searchsorted(M_h, M_h_hi, side='right')
    if idx_hi - idx_lo < 2:
        return 0.0
    return np.trapz(integrand[idx_lo:idx_hi], lnM_h[idx_lo:idx_hi])


def _bin_galaxies(mstar_assign, mu, edges):
    """
    Bin galaxies by mstar_assign, accumulate mu-weighted sums for poisson uncertainty.

    Parameters
    ----------
    mstar_assign : np.ndarray (N_gal,)    — log10(M_star) for bin assignment
    mu           : np.ndarray (N_gal,)    — lensing magnification weights
    edges        : np.ndarray (N_bins+1,) — bin edges in log10(M_star/M_sun)

    Returns
    -------
    sum_mu  : np.ndarray (N_bins,) float64  — sum of mu_j per bin
    sum_mu2 : np.ndarray (N_bins,) float64  — sum of mu_j^2 per bin
    n_gal   : np.ndarray (N_bins,) int      — raw galaxy count per bin
    """
    n_bins  = len(edges) - 1
    bin_idx = np.digitize(mstar_assign, edges) - 1   # 0-based
    valid   = (bin_idx >= 0) & (bin_idx < n_bins)
    idx_v   = bin_idx[valid]
    mu_v    = mu[valid]

    sum_mu  = np.bincount(idx_v, weights=mu_v,    minlength=n_bins).astype(np.float64)
    sum_mu2 = np.bincount(idx_v, weights=mu_v**2, minlength=n_bins).astype(np.float64)
    n_gal   = np.bincount(idx_v,                  minlength=n_bins).astype(int)

    return sum_mu, sum_mu2, n_gal


def _bin_galaxies_mass_weighted(mstar_assign, mstar_mass, mu, edges):
    """
    Bin galaxies by mstar_assign, accumulate (mu * M_star)-weighted sums.
    No Python loop.

    Both bin assignment and mass value can be shifted to mstar_16/84 together
    to capture the joint effect of mass uncertainty on the density estimate.

    Parameters
    ----------
    mstar_assign : np.ndarray (N_gal,)    — log10(M_star) for bin assignment
    mstar_mass   : np.ndarray (N_gal,)    — log10(M_star) for mass value
    mu           : np.ndarray (N_gal,)    — lensing magnification
    edges        : np.ndarray (N_bins+1,)

    Returns
    -------
    sum_muM  : np.ndarray (N_bins,) float64  — sum of mu_j * M_star_j [M_sun]
    sum_muM2 : np.ndarray (N_bins,) float64  — sum of (mu_j * M_star_j)^2
    n_gal    : np.ndarray (N_bins,) int
    """
    n_bins  = len(edges) - 1
    M_star  = 10.0 ** np.asarray(mstar_mass, dtype=np.float64)
    bin_idx = np.digitize(mstar_assign, edges) - 1
    valid   = (bin_idx >= 0) & (bin_idx < n_bins)
    idx_v   = bin_idx[valid]
    w       = mu[valid] * M_star[valid]

    sum_muM  = np.bincount(idx_v, weights=w,    minlength=n_bins).astype(np.float64)
    sum_muM2 = np.bincount(idx_v, weights=w**2, minlength=n_bins).astype(np.float64)
    n_gal    = np.bincount(idx_v,               minlength=n_bins).astype(int)

    return sum_muM, sum_muM2, n_gal


def _extract_catalog_columns(catalog_table, z_min, z_max):
    """
    Extract and redshift-select catalog columns, guarding non-finite percentiles.

    Returns
    -------
    mstar_50, mstar_16, mstar_84, mu : np.ndarray (N_sel,)
        Empty arrays if no galaxies pass the redshift cut.
    """
    z        = np.asarray(catalog_table["z"],        dtype=np.float64)
    mstar_50 = np.asarray(catalog_table["mstar_50"], dtype=np.float64)
    mstar_16 = np.asarray(catalog_table["mstar_16"], dtype=np.float64)
    mstar_84 = np.asarray(catalog_table["mstar_84"], dtype=np.float64)
    mu       = np.asarray(catalog_table["mu"],       dtype=np.float64)

    mask     = (z >= z_min) & (z < z_max)
    mstar_50 = mstar_50[mask]
    mu       = mu[mask]
    mstar_16 = np.where(np.isfinite(mstar_16[mask]), mstar_16[mask], mstar_50)
    mstar_84 = np.where(np.isfinite(mstar_84[mask]), mstar_84[mask], mstar_50)

    return mstar_50, mstar_16, mstar_84, mu


# ═══════════════════════════════════════════════════════════════════════════════
#  THEORY — DIFFERENTIAL NUMBER DENSITY  dn/dlog10(M_star)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_theory_differential_smf(M_h, dndlnm, log10_M_star_bin_edges=DEFAULT_LOG10_MSTAR_BINS,
                                     N=SHMR_N, log_Mc=SHMR_LOG_MC,
                                     beta=SHMR_BETA, gamma=SHMR_GAMMA):
    """
    Theoretical differential stellar mass function dn/dlog10(M_star).

    For each bin [log10(M_star,lo), log10(M_star,hi)]:

        phi = (1 / Delta_log10M) *
              integral_{M_h,lo}^{M_h,hi} (dn/dlnM_h) dlnM_h

    where M_h,lo and M_h,hi are found by inverting the SHMR at the bin edges,
    and the integral is evaluated with np.trapz on the HMF grid.

    UNIT CHAIN
    ----------
    dndlnm [Mpc^{-3}] * dlnM_h [dimless] / Delta_log10M [dex]
        -> phi [Mpc^{-3} dex^{-1}]

    Parameters
    ----------
    M_h : np.ndarray, shape (N_grid,)
        Halo masses [M_sun], ascending.  Output of compute_hmf().
    dndlnm : np.ndarray, shape (N_grid,)
        HMF dn/dlnM_h [Mpc^{-3}].  Output of compute_hmf().
    log10_M_star_bin_edges : np.ndarray, shape (N_bins+1,)
        Bin edges in log10(M_star/M_sun).  Default = DEFAULT_LOG10_MSTAR_BINS.
    N, log_Mc, beta, gamma : float
        SHMR parameters.  Defaults = Stefanon 2021.

    Returns
    -------
    bin_centers : np.ndarray, shape (N_bins,)
        Bin centres in log10(M_star/M_sun).
    phi : np.ndarray, shape (N_bins,)
        dn/dlog10(M_star) [Mpc^{-3} dex^{-1}].
        Zero for bins with fewer than 2 HMF grid points in range.
    """
    M_h    = np.asarray(M_h,    dtype=np.float64)
    dndlnm = np.asarray(dndlnm, dtype=np.float64)
    edges  = np.asarray(log10_M_star_bin_edges, dtype=np.float64)

    Mc       = 10.0 ** log_Mc
    M_star   = shmr_mstar(M_h, N, Mc, beta, gamma)
    lnM_h    = np.log(M_h)
    M_h_edges = _invert_shmr(edges, M_star, M_h)

    n_bins      = len(edges) - 1
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    delta_logM  = edges[1:] - edges[:-1]
    phi         = np.zeros(n_bins, dtype=np.float64)

    for i in range(n_bins):
        integral = _integrate_bin(lnM_h, dndlnm, M_h,
                                  M_h_edges[i], M_h_edges[i + 1])
        phi[i] = integral / delta_logM[i]

    return bin_centers, phi


# ═══════════════════════════════════════════════════════════════════════════════
#  THEORY — DIFFERENTIAL MASS DENSITY  drho_star/dlog10(M_star)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_theory_differential_rho(M_h, dndlnm, log10_M_star_bin_edges=DEFAULT_LOG10_MSTAR_BINS,
                                     N=SHMR_N, log_Mc=SHMR_LOG_MC,
                                     beta=SHMR_BETA, gamma=SHMR_GAMMA):
    """
    Theoretical differential stellar mass density drho_star/dlog10(M_star).

    For each bin [log10(M_star,lo), log10(M_star,hi)]:

        rho_bin = (1 / Delta_log10M) *
                  integral_{M_h,lo}^{M_h,hi} M_star(M_h) * (dn/dlnM_h) dlnM_h

    The integrand is M_star(M_h) * dn/dlnM_h, i.e. the same as for rho_star
    in the cumulative case but integrated only over the bin's halo mass range.

    Relation to number density:
        rho_bin ~= <M_star>_bin * phi    (exact in the limit of narrow bins)

    This observable is more directly comparable to the existing cumulative
    rho_star(>M_star) analysis because it weights by stellar mass.

    UNIT CHAIN
    ----------
    M_star [M_sun] * dndlnm [Mpc^{-3}] * dlnM_h [dimless] / Delta_log10M [dex]
        -> rho_bin [M_sun Mpc^{-3} dex^{-1}]

    Parameters
    ----------
    M_h : np.ndarray, shape (N_grid,)
        Halo masses [M_sun], ascending.  Output of compute_hmf().
    dndlnm : np.ndarray, shape (N_grid,)
        HMF dn/dlnM_h [Mpc^{-3}].  Output of compute_hmf().
    log10_M_star_bin_edges : np.ndarray, shape (N_bins+1,)
        Bin edges in log10(M_star/M_sun).  Default = DEFAULT_LOG10_MSTAR_BINS.
    N, log_Mc, beta, gamma : float
        SHMR parameters.  Defaults = Stefanon 2021.

    Returns
    -------
    bin_centers : np.ndarray, shape (N_bins,)
        Bin centres in log10(M_star/M_sun).
    rho_bin : np.ndarray, shape (N_bins,)
        drho_star/dlog10(M_star) [M_sun Mpc^{-3} dex^{-1}].
        Zero for bins with fewer than 2 HMF grid points in range.
    """
    M_h    = np.asarray(M_h,    dtype=np.float64)
    dndlnm = np.asarray(dndlnm, dtype=np.float64)
    edges  = np.asarray(log10_M_star_bin_edges, dtype=np.float64)

    Mc       = 10.0 ** log_Mc
    M_star   = shmr_mstar(M_h, N, Mc, beta, gamma)
    lnM_h    = np.log(M_h)
    M_h_edges = _invert_shmr(edges, M_star, M_h)

    integrand_rho = M_star * dndlnm   # [M_sun Mpc^{-3}]

    n_bins      = len(edges) - 1
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    delta_logM  = edges[1:] - edges[:-1]
    rho_bin     = np.zeros(n_bins, dtype=np.float64)

    for i in range(n_bins):
        integral   = _integrate_bin(lnM_h, integrand_rho, M_h,
                                    M_h_edges[i], M_h_edges[i + 1])
        rho_bin[i] = integral / delta_logM[i]

    return bin_centers, rho_bin


# ═══════════════════════════════════════════════════════════════════════════════
#  OBSERVATION — DIFFERENTIAL NUMBER DENSITY  dn/dlog10(M_star)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_observed_differential_smf(catalog_table, z_min, z_max, V_survey,
                                       log10_M_star_bin_edges=DEFAULT_LOG10_MSTAR_BINS):
    """
    Observed differential stellar mass function dn/dlog10(M_star).

    For each stellar mass bin:

        phi_obs = (1 / V_survey / Delta_log10M) * sum_{j in bin} mu_j

    Magnification correction: each galaxy j with magnification mu_j occupies
    an effective source-plane volume V_eff = V_image / mu_j.  Summing mu_j
    and dividing by V_image gives the lensing-unbiased number density.

    UNIT CHAIN
    ----------
    sum(mu_j) [dimless] / V_survey [Mpc^3] / Delta_log10M [dex]
        -> phi [Mpc^{-3} dex^{-1}]

    Parameters
    ----------
    catalog_table : astropy Table
        Columns: z, mstar_50, mstar_16, mstar_84, mu.
    z_min, z_max : float
        Redshift slice [z_min, z_max).
    V_survey : float
        Comoving survey volume [Mpc^3].
    log10_M_star_bin_edges : np.ndarray, shape (N_bins+1,)
        Default = DEFAULT_LOG10_MSTAR_BINS.

    Returns
    -------
    bin_centers : np.ndarray, shape (N_bins,)
    phi_obs : np.ndarray, shape (N_bins,)
        dn/dlog10(M_star) [Mpc^{-3} dex^{-1}].
    sigma_poisson : np.ndarray, shape (N_bins,)
        Poisson uncertainty = sqrt(sum mu_j^2) / V / Delta_log10M.
        np.inf for empty bins.
    sigma_mass : np.ndarray, shape (N_bins,)
        Mass measurement uncertainty = |phi(mstar_84) - phi(mstar_16)| / 2.
        Both bin assignment and count shift to mstar_16/84 together.
    n_gal : np.ndarray, shape (N_bins,), int
        Raw (unweighted) galaxy count per bin.
    """
    if V_survey <= 0.0:
        raise ValueError(f"V_survey must be positive [Mpc^3]; got {V_survey:.6g}")

    edges  = np.asarray(log10_M_star_bin_edges, dtype=np.float64)
    n_bins = len(edges) - 1

    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    delta_logM  = edges[1:] - edges[:-1]
    inv_VdM     = 1.0 / (V_survey * delta_logM)

    mstar_50, mstar_16, mstar_84, mu = _extract_catalog_columns(
        catalog_table, z_min, z_max
    )

    if len(mstar_50) == 0:
        return (bin_centers,
                np.zeros(n_bins),
                np.full(n_bins, np.inf),
                np.zeros(n_bins),
                np.zeros(n_bins, dtype=int))

    # ── Central estimate ──────────────────────────────────────────────────────
    sum_mu, sum_mu2, n_gal = _bin_galaxies(mstar_50, mu, edges)
    phi_obs       = sum_mu * inv_VdM
    sigma_poisson = np.where(n_gal > 0, np.sqrt(sum_mu2) * inv_VdM, np.inf)

    # ── Mass uncertainty ──────────────────────────────────────────────────────
    sum_lo, _, _ = _bin_galaxies(mstar_16, mu, edges)
    sum_hi, _, _ = _bin_galaxies(mstar_84, mu, edges)
    sigma_mass   = 0.5 * np.abs(sum_hi - sum_lo) * inv_VdM

    return bin_centers, phi_obs, sigma_poisson, sigma_mass, n_gal


# ═══════════════════════════════════════════════════════════════════════════════
#  OBSERVATION — DIFFERENTIAL MASS DENSITY  drho_star/dlog10(M_star)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_observed_differential_rho(catalog_table, z_min, z_max, V_survey,
                                       log10_M_star_bin_edges=DEFAULT_LOG10_MSTAR_BINS):
    """
    Observed differential stellar mass density drho_star/dlog10(M_star).

    For each stellar mass bin:

        rho_obs = (1 / V_survey / Delta_log10M) *
                  sum_{j in bin} mu_j * M_star_j

    where M_star_j = 10^{mstar_50_j} [M_sun].  Magnification correction is
    identical to the number density case but the count is mass-weighted.

    This observable is more directly comparable to the existing cumulative
    rho_star(>M_star) analysis because both weight by stellar mass.

    UNIT CHAIN
    ----------
    sum(mu_j * M_star_j) [M_sun] / V_survey [Mpc^3] / Delta_log10M [dex]
        -> rho_bin [M_sun Mpc^{-3} dex^{-1}]

    Parameters
    ----------
    catalog_table : astropy Table
        Columns: z, mstar_50, mstar_16, mstar_84, mu.
    z_min, z_max : float
        Redshift slice [z_min, z_max).
    V_survey : float
        Comoving survey volume [Mpc^3].
    log10_M_star_bin_edges : np.ndarray, shape (N_bins+1,)
        Default = DEFAULT_LOG10_MSTAR_BINS.

    Returns
    -------
    bin_centers : np.ndarray, shape (N_bins,)
    rho_obs : np.ndarray, shape (N_bins,)
        drho_star/dlog10(M_star) [M_sun Mpc^{-3} dex^{-1}].
    sigma_poisson : np.ndarray, shape (N_bins,)
        Poisson uncertainty = sqrt(sum (mu_j*M_star_j)^2) / V / Delta_log10M.
        np.inf for empty bins.
    sigma_mass : np.ndarray, shape (N_bins,)
        Mass measurement uncertainty = |rho(mstar_84) - rho(mstar_16)| / 2.
        Both bin assignment AND mass value shift to mstar_16/84 together.
    n_gal : np.ndarray, shape (N_bins,), int
        Raw (unweighted) galaxy count per bin.
    """
    if V_survey <= 0.0:
        raise ValueError(f"V_survey must be positive [Mpc^3]; got {V_survey:.6g}")

    edges  = np.asarray(log10_M_star_bin_edges, dtype=np.float64)
    n_bins = len(edges) - 1

    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    delta_logM  = edges[1:] - edges[:-1]
    inv_VdM     = 1.0 / (V_survey * delta_logM)

    mstar_50, mstar_16, mstar_84, mu = _extract_catalog_columns(
        catalog_table, z_min, z_max
    )

    if len(mstar_50) == 0:
        return (bin_centers,
                np.zeros(n_bins),
                np.full(n_bins, np.inf),
                np.zeros(n_bins),
                np.zeros(n_bins, dtype=int))

    # ── Central estimate ──────────────────────────────────────────────────────
    sum_muM, sum_muM2, n_gal = _bin_galaxies_mass_weighted(
        mstar_50, mstar_50, mu, edges
    )
    rho_obs       = sum_muM * inv_VdM
    sigma_poisson = np.where(n_gal > 0, np.sqrt(sum_muM2) * inv_VdM, np.inf)

    # ── Mass uncertainty: both assignment and mass value shift to 16/84 ───────
    sum_lo, _, _ = _bin_galaxies_mass_weighted(mstar_16, mstar_16, mu, edges)
    sum_hi, _, _ = _bin_galaxies_mass_weighted(mstar_84, mstar_84, mu, edges)
    sigma_mass   = 0.5 * np.abs(sum_hi - sum_lo) * inv_VdM

    return bin_centers, rho_obs, sigma_poisson, sigma_mass, n_gal