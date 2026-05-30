# pipeline/hmf_plugin.py
#
# Direct Sheth-Mo-Tormen HMF using CLASS P(k, z_target).
# No dependency on hmf.MassFunction in production code.
#
# implement SMT directly rather than
# wrapping hmf.MassFunction, avoiding silent internal rescaling of P(k).
#
# UNIT CHAIN (all four lines auditable here or in comments below):
#   P(k, z_target) from CLASS:  k [Mpc^-1],  P [Mpc^3]
#   rho_m0                    : [M_sun / Mpc^3]  (no h units)
#   R(M)                      : [Mpc]
#   sigma(M)                  : dimensionless
#   dn/dlnM                   : [Mpc^-3]
#   M_h output                : [M_sun]
#   All physical
#
# HALO MASS DEFINITION:
#   SOVirial (Bryan & Norman 1998). SMT was calibrated on FOF halos
#   whose mass is closest to M_vir rather than M_200m. This is also
#   the definition used by Bolshoi-Planck (Klypin et al. 2016), to
#   which the Stefanon 2021 SHMR was abundance-matched
#   (Rodriguez-Puebla et al. 2016, MNRAS 462, 893).
#   R(M) = (3M / 4pi rho_m0)^(1/3) is the Lagrangian radius, which
#   is self-consistent with the SMT sigma(M) calculation — no
#   overdensity conversion needed.
#
# PERFORMANCE DESIGN:
#   The hot path per MCMC step (Omega_m, h, a_exo, b_exo all float):
#       (1) R(M): one power operation on 600 elements — negligible
#       (2) P(k, z_target): single get_pk_array C call — ~2ms
#       (3) sigma^2(M) + dsigma^2/dR: C kernel with OpenMP — ~2ms on 8 cores
#       (4) SMT + final assembly: vectorized numpy on 600 elements — ~0.1ms
#   No caching of W_grid — Omega_m and h are sampled parameters that
#   change every MCMC step, making any cache useless.

import numpy as np
import ctypes
import os




# ---------------------------------------------------------------------------
# SMT parameters (Sheth, Mo & Tormen 2001, MNRAS 323, 1, Eq. 6)
# ---------------------------------------------------------------------------
DELTA_C = 1.68647     # linear collapse threshold (Einstein-de Sitter)
A_SMT   = 0.3222      # normalization (ensures integral of f(nu) = 1)
a_SMT   = 0.707       # sharpness of exponential cutoff
q_SMT   = 0.3         # low-mass power-law index

# ---------------------------------------------------------------------------
# Module-level k grid — computed once at import
# with margin. n_k = 2500 gives < 0.1% convergence error in sigma(M).
# ---------------------------------------------------------------------------
_K_MIN, _K_MAX, _N_K = 1e-4, 150.0, 1050
_K_GRID    = np.logspace(np.log10(_K_MIN), np.log10(_K_MAX), _N_K)  # Mpc^-1
_LNK_GRID  = np.log(_K_GRID)

# ---------------------------------------------------------------------------
# Module-level mass grid — fixed, physical M_sun
# ---------------------------------------------------------------------------
_LOG10M_MIN, _LOG10M_MAX, _N_M = 8.0,  15.0 , 600
_M_GRID    = np.logspace(_LOG10M_MIN, _LOG10M_MAX, _N_M)  # M_sun
_LNM_GRID  = np.log(_M_GRID)


_lib_path = os.path.join(os.path.dirname(__file__), 'hmf_sigma.so')
_lib = ctypes.CDLL(_lib_path)

_lib.compute_sigma_batch.restype = None
_lib.compute_sigma_batch.argtypes = [
    ctypes.c_void_p,   # k_grid
    ctypes.c_void_p,   # lnk_grid
    ctypes.c_void_p,   # Pk
    ctypes.c_void_p,   # R_grid
    ctypes.c_void_p,   # sigma2 (output)
    ctypes.c_void_p,   # dsigma2_dR (output)
    ctypes.c_int,      # n_k
    ctypes.c_int,      # n_M
]

def _get_pk_vec(cosmo_class, z_target):
    """
    Return P(k, z_target) on _K_GRID in Mpc^3.

    Uses cosmo_class.get_pk_array(). 
    """
    z_arr    = np.array([z_target])
    
    Pk_flat = cosmo_class.get_pk_array(
    _K_GRID, z_arr, _N_K, 1, nonlinear=False
    )
    
    return np.asarray(Pk_flat)             # (n_k,)


def _sigma_and_deriv(Pk, R_grid):
    """
    Computes sigma^2 and the derivative of sigma^2 in the C program and outputs them.
    """
    n_k = len(_K_GRID)
    n_M = len(R_grid)
    sigma2     = np.zeros(n_M)
    dsigma2_dR = np.zeros(n_M)

    _lib.compute_sigma_batch(
        _K_GRID.ctypes.data_as(ctypes.c_void_p),
        _LNK_GRID.ctypes.data_as(ctypes.c_void_p),
        Pk.ctypes.data_as(ctypes.c_void_p),
        R_grid.ctypes.data_as(ctypes.c_void_p),
        sigma2.ctypes.data_as(ctypes.c_void_p),
        dsigma2_dR.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(n_k),
        ctypes.c_int(n_M),
    )
    return sigma2, dsigma2_dR


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_hmf(cosmo_class, z_target):
    """
    Compute the SMT halo mass function at z_target using CLASS P(k,z).

    Returns
    -------
    M_h    : np.ndarray, log spaced [M_sun]
    dndlnm : np.ndarray [Mpc^-3]
    sigma  : np.ndarray dimensionless
    """
    h   = cosmo_class.h()
    Om0 = cosmo_class.Omega_m()

    # Mean matter density today [M_sun / Mpc^3]
    rho_m0 = Om0 * 2.775e11 * h**2

    # Lagrangian radius R(M) [Mpc]
    R = (3.0 * _M_GRID / (4.0 * np.pi * rho_m0))**(1.0/3.0)

    # P(k, z_target) from CLASS [Mpc^3]
    Pk = _get_pk_vec(cosmo_class, z_target)

    # sigma^2(M) and dsigma^2/dR via C kernel (parallel, fused)
    Pk     = np.ascontiguousarray(Pk,  dtype=np.float64)
    R_grid = np.ascontiguousarray(R,   dtype=np.float64)
    sigma2, dsigma2_dR = _sigma_and_deriv(Pk, R_grid)

    sigma = np.sqrt(np.maximum(sigma2, 0.0))

    # Analytical d ln sigma / d ln M
    # dsigma^2/dM = dsigma^2/dR * dR/dM,  dR/dM = R / (3M)
    # d ln sigma / d ln M = M/(2 sigma^2) * dsigma^2/dM
    #                     = R * dsigma2_dR / (6 * sigma^2)
    dlnsigma_dlnM = R * dsigma2_dR / (6.0 * np.maximum(sigma2, 1e-300))

    # SMT multiplicity
    nu   = DELTA_C / sigma
    anu2 = a_SMT * nu**2
    f_nu = (A_SMT * np.sqrt(2.0 * a_SMT / np.pi)
            * (1.0 + anu2**(-q_SMT))
            * nu * np.exp(-anu2 / 2.0))

    dndlnm = (rho_m0 / _M_GRID) * f_nu * np.abs(dlnsigma_dlnM)

    return _M_GRID.copy(), dndlnm, sigma, Pk


def compute_cosmic_variance(Pk, M_h, dndlnm, sigma, V_survey,
                             log10_M_star_bin_edges, shmr_func, shmr_kwargs=None):
    """
    Fractional cosmic variance sigma_CV per stellar mass bin.
 
    Uses the Moster et al. (2011, ApJ 731, 113) framework:
 
        sigma_CV(bin) = b_eff(bin) * sigma_DM
 
    where sigma_DM is the rms dark matter density fluctuation smoothed over the
    survey volume, and b_eff is the effective linear halo bias averaged over all
    halos that host galaxies in the given stellar mass bin.
 
    All inputs are accepted directly from the logp() hot path so that no
    redundant CLASS or HMF calls are made inside this function.
 
    SIGMA_DM
    --------
    The survey volume is treated as an effective sphere of radius:
 
        R_eff = (3 V_survey / 4 pi)^{1/3}
 
    sigma_DM^2 is computed by passing R_eff as a length-1 array to the existing
    C kernel _sigma_and_deriv(), which evaluates:
 
        sigma_DM^2 = (1 / 2pi^2) int k^3 P(k,z) W^2(k R_eff) d ln k
 
    where W is the spherical top-hat window in Fourier space.  This reuses the
    same OpenMP kernel already running for the 750-point HMF sigma(M) — no new
    C code.  The marginal cost is one extra kernel call for a single radius.
 
    EFFECTIVE BIAS
    --------------
    Sheth, Mo & Tormen (2001, MNRAS 323, 1) linear halo bias (Eq. 8):
 
        b(nu) = 1 + (a_SMT nu^2 - 1) / delta_c
                  + 2 q_SMT / (delta_c (1 + (a_SMT nu^2)^q_SMT))
 
    where nu = delta_c / sigma(M_h).  Uses module constants DELTA_C, a_SMT,
    q_SMT already defined in hmf.py.
 
    Mass-weighted average over halos in a stellar mass bin:
 
        b_eff = integral_{M_h,lo}^{M_h,hi} b(nu) (dn/dlnM_h) dlnM_h
                -------------------------------------------------------
                integral_{M_h,lo}^{M_h,hi} (dn/dlnM_h) dlnM_h
 
    UNIT CHAIN
    ----------
    sigma_DM  : dimensionless  (rms of delta_rho / rho_m)
    b_eff     : dimensionless  (linear bias)
    sigma_CV  : dimensionless  (fractional uncertainty on any density estimate)
 
    Usage in logp():
        sigma_tot^2 = sigma_poisson^2 + sigma_mass^2 + (sigma_CV * phi_obs)^2
 
    Parameters
    ----------
    Pk : np.ndarray, shape (_N_K,)
        P(k, z) [Mpc^3] on _K_GRID.  Direct output of _get_pk_vec().
    M_h : np.ndarray, shape (_N_M,)
        Halo masses [M_sun], ascending.  Direct output of compute_hmf().
    dndlnm : np.ndarray, shape (_N_M,)
        HMF dn/dlnM_h [Mpc^{-3}].  Direct output of compute_hmf().
    sigma : np.ndarray, shape (_N_M,)
        Dimensionless sigma(M_h).  Direct output of compute_hmf().
    V_survey : float
        Comoving survey volume [Mpc^3].  Direct output of V_survey().
    log10_M_star_bin_edges : np.ndarray, shape (N_bins+1,)
        Stellar mass bin edges in log10(M_star/M_sun).  Must match the edges
        passed to compute_theory_differential_smf / _rho and their observed
        counterparts so that sigma_CV[i] corresponds to the same bin.
    shmr_func : callable
        M_star = shmr_func(M_h, **shmr_kwargs).
    shmr_kwargs : dict or None
        Keyword arguments for shmr_func.  None uses shmr_func's own defaults.
 
    Returns
    -------
    sigma_cv : np.ndarray, shape (N_bins,)
        Fractional cosmic variance per stellar mass bin (dimensionless).
        Increases with stellar mass because more massive galaxies live in
        more biased halos.
 
    Notes
    -----
    sigma_CV > 1 is physically possible in very small volumes at very high
    stellar masses where the halo bias is large.  The logp() good mask
    (n_gal > 0) already discards empty bins before sigma_CV is applied.
    """
    if shmr_kwargs is None:
        shmr_kwargs = {}
 
    edges  = np.asarray(log10_M_star_bin_edges, dtype=np.float64)
    n_bins = len(edges) - 1
 
    # ── Step 1: sigma_DM via existing C kernel ────────────────────────────────
    # Pass R_eff as a length-1 contiguous array.
    R_eff = (3.0 * V_survey / (4.0 * np.pi)) ** (1.0 / 3.0)   # Mpc
    R_arr = np.ascontiguousarray([R_eff], dtype=np.float64)
    Pk_c  = np.ascontiguousarray(Pk,      dtype=np.float64)
 
    sigma2_dm, _ = _sigma_and_deriv(Pk_c, R_arr)
    sigma_DM     = np.sqrt(max(sigma2_dm[0], 0.0))
 
    # ── Step 2: SMT halo bias on the full M grid ──────────────────────────────
    nu   = DELTA_C / np.maximum(sigma, 1.0e-300)
    anu2 = a_SMT * nu**2
 
    b_SMT = (1.0
             + (anu2 - 1.0) / DELTA_C
             + 2.0 * q_SMT / (DELTA_C * (1.0 + anu2 ** q_SMT)))
 
    # ── Step 3: SHMR on M grid, invert at bin edges ───────────────────────────
    M_star_grid = shmr_func(M_h, **shmr_kwargs)          # ascending (_N_M,)
    M_h_edges   = np.interp(10.0 ** edges, M_star_grid, M_h,
                            left=M_h[0], right=M_h[-1])
    lnM_h       = np.log(M_h)
 
    # ── Step 4: b_eff per bin ─────────────────────────────────────────────────
    sigma_cv = np.zeros(n_bins, dtype=np.float64)

    for i in range(n_bins):
        M_h_lo = M_h_edges[i]
        M_h_hi = M_h_edges[i + 1]

        idx_lo = np.searchsorted(M_h, M_h_lo, side='left')
        idx_hi = np.searchsorted(M_h, M_h_hi, side='right')

        if idx_hi - idx_lo < 2:
            idx_mid     = min(max(idx_lo, 0), len(b_SMT) - 1)
            sigma_cv[i] = b_SMT[idx_mid] * sigma_DM
            continue

        # Interpolate dndlnm and b_SMT at the exact bin edges
        lnM_lo = np.log(M_h_lo)
        lnM_hi = np.log(M_h_hi)

        dndlnm_lo = np.interp(lnM_lo, lnM_h, dndlnm)
        dndlnm_hi = np.interp(lnM_hi, lnM_h, dndlnm)
        b_lo      = np.interp(lnM_lo, lnM_h, b_SMT)
        b_hi      = np.interp(lnM_hi, lnM_h, b_SMT)

        # Build arrays with exact endpoints prepended/appended
        lnM_sl    = np.concatenate([[lnM_lo],    lnM_h[idx_lo:idx_hi],    [lnM_hi]])
        dndlnm_sl = np.concatenate([[dndlnm_lo], dndlnm[idx_lo:idx_hi],   [dndlnm_hi]])
        b_sl      = np.concatenate([[b_lo],       b_SMT[idx_lo:idx_hi],    [b_hi]])

        norm = np.trapz(dndlnm_sl, lnM_sl)

        if norm <= 0.0:
            sigma_cv[i] = b_sl[-1] * sigma_DM
            continue

        b_eff       = np.trapz(b_sl * dndlnm_sl, lnM_sl) / norm
        sigma_cv[i] = b_eff * sigma_DM

    return sigma_cv

