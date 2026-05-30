# pipeline/uvlf_theory.py
"""
UV Luminosity Function (UVLF) — theory prediction and chi-squared likelihood.

Theory prediction chain at each redshift z:

    compute_hmf(cosmo, z)  →  M_h [M_sun],  dn/dlnM_h [Mpc^-3]
            ↓  shmr_mstar()
    M_star [M_sun]
            ↓  numerical Jacobian (log-log gradient)
    |dM_star/dM_h|  [dimensionless × M_star/M_h]
            ↓  convert_and_jacobian()
    M_UV [mag],  |dM_star/dM_UV|  [M_sun mag^-1]
            ↓  Jiang et al. 2024 Eq. 3
    phi(M_UV)  [mag^-1 Mpc^-3]
            ↓  interpolate onto data bin centres
    chi^2  vs  Donnan+2024  +  Finkelstein+2024

Redshift integration (optional):
    When integrate_bin=True, phi(M_UV) is computed as a comoving-volume-
    weighted average over the full redshift bin using Gauss-Legendre
    quadrature with n_gl nodes. This is more accurate than evaluating at
    the nominal bin centre but costs n_gl CLASS calls per bin instead of 1.
    Use for diagnostics; single-z mode is recommended for production MCMC.

Volume corrections (optional):
    Published phi_obs values assume a survey-specific reference cosmology.
    When use_volume_correction=True, phi_obs and both sigma arrays are
    rescaled by V_ref/V_MCMC before chi^2 is computed. The reference CLASS
    objects are built once before the loop and reused across all bins.

References
----------
    Jiang et al. 2024, arXiv:2409.19941, Eq. 3
    Donnan et al. 2024, MNRAS 533, 3222  (PRIMER)
    Finkelstein et al. 2024, ApJL 969, L2 (CEERS)
    Song et al. 2016 — M_star to M_UV conversion
    Meurer et al. 1999 — dust attenuation
    Stefanon et al. 2021 — stellar-to-halo mass relation
"""

from classy import Class
from .uvlf_conversion import shmr_mstar, shmr_mstar_and_jacobian, convert_and_jacobian
from .hmf import compute_hmf
import os
import numpy as np


# ── path setup ────────────────────────────────────────────────────────────────
_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
DATA_PATH     = os.path.join(_PROJECT_ROOT, "data")


# ── redshift bin edges ────────────────────────────────────────────────────────

# Donnan et al. 2024 (PRIMER, MNRAS 533, 3222) — 5 redshift bins
# TODO: verify z=12.5 and z=14.5 edges against Donnan+2024 Table 1;
#       z=9, 10, 11 edges are confirmed from the paper.
DONNAN_Z_EDGES = {
    9.0  : [8.5,  9.5],
    10.0 : [9.5,  10.5],
    11.0 : [10.5, 11.5],
    12.5 : [11.5, 13.5],
    14.5 : [13.5, 15.5],
}

# Finkelstein et al. 2024 (CEERS, ApJL 969, L2) — 3 redshift bins
# Edges confirmed from paper Table 3 caption.
FINKELSTEIN_Z_EDGES = {
    8.9  : [8.5,  9.7],
    10.9 : [9.7,  13.0],
    14.0 : [13.0, 15.0],
}


# ── reference cosmologies for volume correction ───────────────────────────────

# Donnan 2024 reference cosmology
_DONNAN_H0 = 70.0
_DONNAN_OM = 0.30
_DONNAN_OL = 0.70

DONNAN_REF_COSMO = {"h": _DONNAN_H0/100, "Omega_m": _DONNAN_OM, "Omega_Lambda": _DONNAN_OL}

# Finkelstein 2024 reference cosmology (Planck 2018)
_FINKELSTEIN_H0 = 67.36
_FINKELSTEIN_OM = 0.3153
_FINKELSTEIN_OL = 0.6847

FINKELSTEIN_REF_COSMO = {"h": _FINKELSTEIN_H0/100, "Omega_m": _FINKELSTEIN_OM, "Omega_Lambda": _FINKELSTEIN_OL}


# unit conversion factor: data files report phi in 10^-6 mag^-1 Mpc^-3
_PHI_UNIT = 1e-6


# ─────────────────────────────────────────────────────────────────────────────
def load_donnan():
    """
    Load Donnan et al. 2024 (PRIMER, MNRAS 533, 3222) Table 2 UVLF data.

    Reads data/donnan2024.txt and applies the 1e-6 unit conversion so all
    phi and sigma values are returned in mag^-1 Mpc^-3.

    Reference cosmology used by the survey: H0=70, Om=0.3, OL=0.7.
    Contains 29 data points across 5 redshift bins: z = 9, 10, 11, 12.5, 14.5.

    Returns
    -------
    np.ndarray, structured, with fields:
        z          : float64 — nominal bin redshift
        M_UV       : float64 — bin centre [mag]
        phi        : float32 — number density [mag^-1 Mpc^-3]
        sigma_up   : float32 — upper (Poisson+cosmic variance) error [mag^-1 Mpc^-3]
        sigma_down : float32 — lower error [mag^-1 Mpc^-3]
    """
    filepath      = os.path.join(DATA_PATH, "donnan2024.txt")
    names_n_types = [('z', 'f8'), ('M_UV', 'f8'),
                 ('phi', 'f4'), ('sigma_up', 'f4'), ('sigma_down', 'f4')]
    data = np.genfromtxt(filepath, delimiter=',',
                                  comments='#', dtype=names_n_types)

    # convert from 10^-6 mag^-1 Mpc^-3 → mag^-1 Mpc^-3
    data['phi']        *= _PHI_UNIT
    data['sigma_up']   *= _PHI_UNIT
    data['sigma_down'] *= _PHI_UNIT

    return data


# ─────────────────────────────────────────────────────────────────────────────
def load_finkelstein():
    """
    Load Finkelstein et al. 2024 (CEERS, ApJL 969, L2) Table 3 UVLF data.

    Reads data/finkelstein2024.txt and applies the 1e-6 unit conversion.
    The nominal redshifts (8.9, 10.9, 14.0) are completeness-weighted medians
    of the galaxy redshift distributions — use them exactly, do not round.

    Reference cosmology used by the survey: H0=67.36, Om=0.3153 (Planck 2018).
    Contains 11 data points across 3 redshift bins.

    Bin ranges (from paper):
        z ~ 8.9  : 8.5  <= z <= 9.7
        z ~ 10.9 : 9.7  <= z <= 13.0
        z ~ 14.0 : 13.0 <= z <= 15.0

    Returns
    -------
    np.ndarray, structured, with fields:
        z          : float64 — nominal bin redshift (completeness-weighted median)
        M_UV       : float64 — bin centre [mag]
        phi        : float32 — number density [mag^-1 Mpc^-3]
        sigma_up   : float32 — upper error [mag^-1 Mpc^-3]
        sigma_down : float32 — lower error [mag^-1 Mpc^-3]
        V_eff      : int32   — effective survey volume [Mpc^3]
    """
    filepath      = os.path.join(DATA_PATH, "finkelstein2024.txt")
    names_n_types = [('z', 'f8'), ('M_UV', 'f8'),
                 ('phi', 'f4'), ('sigma_up', 'f4'), ('sigma_down', 'f4'),
                 ('V_eff', 'i4')]
    data = np.genfromtxt(filepath, delimiter=',',
                                  comments='#', dtype=names_n_types)

    # convert from 10^-6 mag^-1 Mpc^-3 → mag^-1 Mpc^-3
    data['phi']        *= _PHI_UNIT
    data['sigma_up']   *= _PHI_UNIT
    data['sigma_down'] *= _PHI_UNIT

    return data


# ─────────────────────────────────────────────────────────────────────────────
def _phi_single_z(M_h, dndlnm, z, M_UV_bins, **shmr_kwargs):
    """
    Compute the theoretical UVLF at a single redshift, interpolated onto the
    requested M_UV bin centres.

    Implements the change-of-variables identity (Jiang et al. 2024, Eq. 3):

        phi(M_UV) = (dn/dlnM_h) / M_h / |dM*/dM_h| * |dM*/dM_UV|

    where:
        dn/dlnM_h  comes from compute_hmf()           [Mpc^-3]
        |dM*/dM_h| is the SHMR Jacobian, computed     [dimensionless]
                   via log-log numerical gradient
        |dM*/dM_UV| is returned by convert_and_jacobian() [M_sun mag^-1]

    Parameters
    ----------
    M_h      : np.ndarray (N,) — halo masses [M_sun], log-spaced, ascending
    dndlnm   : np.ndarray (N,) — HMF dn/dlnM_h [Mpc^-3]
    z        : float           — evaluation redshift
    M_UV_bins: np.ndarray (B,) — data bin centres to interpolate onto [mag]

    Returns
    -------
    np.ndarray (B,) — phi [mag^-1 Mpc^-3] at each requested bin centre.
    Bins outside the theory M_UV range return 0.0 (see np.interp left/right).
    """
    # Step 1+2 — SHMR + analytical Jacobian (exact, no numerical gradient noise)
    M_star, dlnMs_dlnMh = shmr_mstar_and_jacobian(M_h, **shmr_kwargs)
    dMs_dMh = dlnMs_dlnMh * (M_star / M_h)                # always positive

    # Step 3 — M* → M_UV conversion + Jacobian |dM*/dM_UV|.
    # M_UV is descending (brighter = more negative mag as M_star increases).
    # dMstar_dMUV is negative by convention; take absolute value.
    M_UV, dMstar_dMUV = convert_and_jacobian(M_star, z)
    dMstar_dMUV       = np.abs(dMstar_dMUV)               # [M_sun mag^-1]

    # Step 4 — assemble phi on the irregular M_UV grid (one value per M_h node)
    phi = (dndlnm / (M_h * dMs_dMh)) * dMstar_dMUV       # [mag^-1 Mpc^-3]

    # Step 5 — M_UV is always descending (brighter = more massive, monotonic
    # SHMR confirmed by U04). Flip to ascending for np.interp.
    M_UV_sorted = M_UV[::-1]
    phi_sorted  = phi[::-1]

    # Step 6 — interpolate onto observational bin centres;
    # return 0 outside the theory range rather than silently extrapolating
    phi_interpolated = np.interp(M_UV_bins, M_UV_sorted, phi_sorted,
                                 left=0.0, right=0.0)

    return phi_interpolated


# ─────────────────────────────────────────────────────────────────────────────
def compute_uvlf_theory(M_h, dndlnm, z, M_UV_bins,
                         integrate_bin=False,
                         z_lo=None, z_hi=None,
                         cosmo=None, n_gl=2, **shmr_kwargs):
    """
    Predict phi(M_UV) at the specified bin centres, with optional redshift
    integration via Gauss-Legendre quadrature.

    Single-z mode (integrate_bin=False, default):
        Evaluates phi at the nominal redshift z using the pre-computed M_h
        and dndlnm arrays. Costs zero additional CLASS calls. Recommended
        for production MCMC.

    Integrated mode (integrate_bin=True):
        Computes a comoving-volume-weighted average of phi(z) across the
        redshift bin [z_lo, z_hi] using n_gl Gauss-Legendre nodes:

            phi_theory = sum_i(omega_i * w_i * phi_i) / sum_i(omega_i * w_i)

        where omega_i are GL weights, w_i = d_C(z_i)^2 / H(z_i) is the
        comoving volume element (the Delta_z/2 Jacobian cancels in the ratio),
        and phi_i is evaluated by calling compute_hmf + _phi_single_z at each
        node redshift z_i. Costs n_gl CLASS calls per bin. Use for diagnostics
        to quantify the single-z approximation error.

    Gauss-Legendre nodes for n_gl=2 are placed at z_mid +/- 0.289*(Delta_z/2),
    giving exact integration of cubics — equivalent in practice to ~5-point
    trapezoid for the smooth phi(z) encountered here.
    For n_gl=3, the middle node falls exactly at z_mid = (z_lo+z_hi)/2, which
    coincides with z_nominal for all symmetric Donnan bins, allowing the
    central compute_hmf call to be reused (see optimisation note in source).

    Parameters
    ----------
    M_h          : np.ndarray (N,) — halo masses [M_sun], log-spaced, ascending
    dndlnm       : np.ndarray (N,) — HMF dn/dlnM_h [Mpc^-3]
    z            : float           — nominal bin redshift
    M_UV_bins    : np.ndarray (B,) — data bin centres [mag]
    integrate_bin: bool            — enable redshift bin integration
    z_lo, z_hi   : float           — bin edges; required if integrate_bin=True
    cosmo        : classy.Class    — CLASS object; required if integrate_bin=True
    n_gl         : int             — number of GL quadrature nodes (default 2)

    Returns
    -------
    np.ndarray (B,) — phi_theory [mag^-1 Mpc^-3] at each M_UV bin centre
    """
    # ── single-z path ─────────────────────────────────────────────────────────
    if not integrate_bin:
        phi = _phi_single_z(M_h, dndlnm, z, M_UV_bins, **shmr_kwargs)
        return phi

    # ── GL-integrated path ────────────────────────────────────────────────────

    # GL nodes xi on [-1, 1] and their quadrature weights omega
    xi, omega = np.polynomial.legendre.leggauss(n_gl)

    # remap nodes from [-1,1] to [z_lo, z_hi]:
    # z_i = midpoint + half_width * xi_i
    z_arr = ((z_hi + z_lo) / 2) + (((z_hi - z_lo) / 2) * xi)

    # pre-allocate output arrays
    phi   = np.zeros((n_gl, len(M_UV_bins)))  # phi at each node × bin
    w_arr = np.zeros(n_gl)                    # comoving volume weight per node

    for i, z_i in enumerate(z_arr):

        # HMF at this GL node (one CLASS call per node)
        M_h_i, dndlnm_i, _, _ = compute_hmf(cosmo, z_i)

        # UVLF at this node, interpolated onto the data bin centres
        phi[i] = _phi_single_z(M_h_i, dndlnm_i, z_i, M_UV_bins, **shmr_kwargs)

        d_A_i    = cosmo.angular_distance(float(z_i))
        w_arr[i] = d_A_i**2 / cosmo.Hubble(float(z_i))

    # combined weight per node: GL quadrature weight × physical volume weight.
    # The Delta_z/2 Jacobian from the [-1,1]→[z_lo,z_hi] remapping appears
    # in both numerator and denominator and cancels exactly — never computed.
    combined_weights = omega * w_arr                         # shape (n_gl,)

    # volume-weighted average across GL nodes
    phi_theory = (np.sum(combined_weights[:, None] * phi, axis=0)
                  / np.sum(combined_weights))                # shape (B,)

    return phi_theory


# ─────────────────────────────────────────────────────────────────────────────
def chi_squared(phi_theory, phi_obs, sigma_up, sigma_down):
    """
    Compute the split-Gaussian chi^2 between theory and observation.

    Uses asymmetric published error bars following the standard convention
    in the high-z UVLF literature (e.g. Jiang et al. 2024):

        sigma_i = sigma_up_i   if phi_theory_i > phi_obs_i  (overprediction)
        sigma_i = sigma_down_i if phi_theory_i <= phi_obs_i (underprediction)

        chi^2 = sum_i  (phi_theory_i - phi_obs_i)^2 / sigma_i^2

    Parameters
    ----------
    phi_theory : np.ndarray (B,) — theoretical UVLF [mag^-1 Mpc^-3]
    phi_obs    : np.ndarray (B,) — observed UVLF    [mag^-1 Mpc^-3]
    sigma_up   : np.ndarray (B,) — upper error bars [mag^-1 Mpc^-3]
    sigma_down : np.ndarray (B,) — lower error bars [mag^-1 Mpc^-3]

    Returns
    -------
    float — chi^2 contribution from these bins
    """
    # pick the appropriate error bar per bin based on over/under prediction
    sigmas = np.where(phi_theory > phi_obs, sigma_up, sigma_down)  

    chi2 = np.sum((phi_theory - phi_obs)**2 / sigmas**2)           

    return chi2


# ─────────────────────────────────────────────────────────────────────────────
def _build_ref_cosmo(dataset):
    """
    Build and return a background-only CLASS cosmology object for the
    reference cosmology of the requested survey dataset.

    Called once per dataset before the MCMC loop. The returned object is
    reused across all redshift bins via volume_ratio(). Never call this
    inside the MCMC hot path.

    Only background quantities (H(z), angular_distance(z)) are needed, so
    output='' is set to skip all power spectrum computation — making this
    CLASS call very fast (~0.05s).

    The Omega_b / Omega_cdm split is derived from a standard physical baryon
    density omega_b = Omega_b * h^2. The exact split does not affect H(z)
    or d_A(z) — only the total matter density Omega_m matters for the
    background.

    Parameters
    ----------
    dataset : str — 'Donnan' or 'Finkelstein'

    Returns
    -------
    classy.Class — initialised and computed CLASS object
    """
    if dataset == "Donnan":

        params = {

            'h': _DONNAN_H0/100,

            'Omega_Lambda': _DONNAN_OL,


            'Omega_m': _DONNAN_OM,


            'output': ''

        }

    elif dataset == "Finkelstein" :

        params = {

            'h': _FINKELSTEIN_H0/100,

            'Omega_Lambda': _FINKELSTEIN_OL,

            'Omega_m': _FINKELSTEIN_OM,

            'output': ''

        }
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. "   
                         f"Expected 'Donnan' or 'Finkelstein'.")

    # derive Omega_b and Omega_cdm from the known total Omega_m

    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()

    return cosmo


# ─────────────────────────────────────────────────────────────────────────────
def volume_ratio(z, cosmo_mcmc, cosmo_ref):
    """
    Compute the comoving volume ratio V_ref / V_MCMC at redshift z.

    Published phi_obs values assume a reference cosmology for the survey
    volume. When the MCMC samples a different cosmology, the observed number
    density must be rescaled by this ratio before comparing to theory:

        phi_obs_corrected = phi_obs_published * (V_ref / V_MCMC)
        sigma_corrected   = sigma_published   * (V_ref / V_MCMC)

    Derivation: the survey area Omega and the redshift bin width Delta_z are
    fixed (they cancel). The ratio reduces to:

        V_ref / V_MCMC = d_A_ref^2(z) * H_MCMC(z)
                       / d_A_MCMC^2(z) * H_ref(z)

    The (1+z)^2 factors from d_C = d_A*(1+z) cancel identically in the
    ratio, so angular diameter distances are used directly.

    Parameters
    ----------
    z          : float        — redshift at which to evaluate the ratio
    cosmo_mcmc : classy.Class — current MCMC step cosmology
    cosmo_ref  : classy.Class — survey reference cosmology (pre-built, fixed)

    Returns
    -------
    float — V_ref / V_MCMC  (multiply onto phi_obs and sigmas)
    """
    z = float(z)
    d_A_mcmc = cosmo_mcmc.angular_distance(z)
    d_A_ref  = cosmo_ref.angular_distance(z)
    H_mcmc   = cosmo_mcmc.Hubble(z)
    H_ref    = cosmo_ref.Hubble(z)
    return (d_A_ref**2 * H_mcmc) / (d_A_mcmc**2 * H_ref)


# ─────────────────────────────────────────────────────────────────────────────
def compute_total_chi2(cosmo, integrate_bin=False, n_gl=2,
                        use_volume_correction=False, **shmr_kwargs):
    """
    Compute the total chi^2 across all UVLF data from Donnan+2024 and
    Finkelstein+2024 against the theoretical prediction from the given
    CLASS cosmology.

    Loops over 5 Donnan redshift bins and 3 Finkelstein redshift bins (8 bins
    total, 40 data points). At each bin:
        1. Calls compute_hmf(cosmo, z_nom) to get the HMF
        2. Calls compute_uvlf_theory() to get the theoretical phi
        3. Optionally applies the volume correction to phi_obs and sigmas
        4. Accumulates chi^2 via chi_squared()

    The two datasets cover independent sky fields — no cross-covariance.
    Their chi^2 contributions are simply summed.

    Volume corrections are applied to the DATA (phi_obs, sigma_up,
    sigma_down), not to phi_theory. The reference CLASS objects are built
    once before the bin loop and never rebuilt inside it.

    Parameters
    ----------
    cosmo                : classy.Class — current evaluation cosmology
    integrate_bin        : bool — pass through to compute_uvlf_theory
    n_gl                 : int  — GL nodes per bin; used if integrate_bin=True
    use_volume_correction: bool — rescale phi_obs for cosmology-dependent volume

    Returns
    -------
    float — total chi^2 summed over all 40 data points
    """
    donnan_data      = load_donnan()
    finkelstein_data = load_finkelstein()

    # build reference CLASS objects once — never inside the bin loop
    if use_volume_correction:
        cosmo_ref_don  = _build_ref_cosmo('Donnan')
        cosmo_ref_fink = _build_ref_cosmo('Finkelstein')
        
    elif not use_volume_correction:
        cosmo_ref_don  = None
        cosmo_ref_fink = None

    chi2_total = 0.0

    # ── Donnan loop — 5 redshift bins ─────────────────────────────────────────
    for z_nom in np.unique(donnan_data['z']):

        # slice data for this redshift bin
        mask       = donnan_data['z'] == z_nom
        M_UV_bins  = donnan_data['M_UV'][mask]
        phi_obs    = donnan_data['phi'][mask].copy()       # copy: may be rescaled below
        sigma_up   = donnan_data['sigma_up'][mask].copy()
        sigma_down = donnan_data['sigma_down'][mask].copy()

        # HMF at the nominal bin redshift
        M_h, dndlnm, _, _ = compute_hmf(cosmo, z_nom)

        # theoretical UVLF (single-z or GL-integrated)
        z_lo, z_hi  = DONNAN_Z_EDGES[z_nom]
        phi_theory  = compute_uvlf_theory(M_h, dndlnm, z_nom, M_UV_bins,
                                           integrate_bin, z_lo, z_hi,
                                           cosmo, n_gl, **shmr_kwargs)

        # volume correction: rescale observed phi and errors onto MCMC cosmology
        if use_volume_correction:
            r          = volume_ratio(z_nom, cosmo, cosmo_ref_don)
            phi_obs   *= r
            sigma_up  *= r
            sigma_down *= r

        chi2_total += chi_squared(phi_theory, phi_obs, sigma_up, sigma_down)

    # ── Finkelstein loop — 3 redshift bins ────────────────────────────────────
    for z_nom in np.unique(finkelstein_data['z']):

        mask       = finkelstein_data['z'] == z_nom
        M_UV_bins  = finkelstein_data['M_UV'][mask]
        phi_obs    = finkelstein_data['phi'][mask].copy()
        sigma_up   = finkelstein_data['sigma_up'][mask].copy()
        sigma_down = finkelstein_data['sigma_down'][mask].copy()

        M_h, dndlnm, _, _ = compute_hmf(cosmo, z_nom)

        z_lo, z_hi = FINKELSTEIN_Z_EDGES[z_nom]
        phi_theory = compute_uvlf_theory(M_h, dndlnm, z_nom, M_UV_bins,
                                          integrate_bin, z_lo, z_hi,
                                          cosmo, n_gl=3, **shmr_kwargs)

        if use_volume_correction:
            r          = volume_ratio(z_nom, cosmo, cosmo_ref_fink)
            phi_obs   *= r
            sigma_up  *= r
            sigma_down *= r

        chi2_total += chi_squared(phi_theory, phi_obs, sigma_up, sigma_down)

    return chi2_total


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    """
    Sanity check: compute the LCDM UVLF at z=9 and compare to Donnan+2024.

    Prints a table of phi_single_z vs phi_integrated vs phi_obs at all
    z=9 bin centres, plus the chi^2 for each mode on the z=9 bin alone.

    Uses Donnan reference cosmology (H0=70, Om=0.3) for the LCDM run so
    the comparison is maximally clean (no volume correction needed).

    Note: requires the full CLASS setup with mPk output and sufficient
    k and z range for compute_hmf. Check against test_likelihood_differential
    for the exact parameter dict used in this project.
    """
    h       = _DONNAN_H0 / 100
    omega_b = 0.022
    Om      = _DONNAN_OM

    # full CLASS params — mPk needed for compute_hmf (P(k,z) integration)
    params_lcdm = {
        'h'              : h,
        'Omega_b'        : omega_b / h**2,
        'Omega_cdm'      : Om - omega_b / h**2,
        'Omega_Lambda'   : _DONNAN_OL,
        'n_s'            : 0.965,
        'A_s'            : 2.1e-9,
        'tau_reio'       : 0.054,
        'output'         : 'mPk',
        'P_k_max_1/Mpc'  : 150.0,    # must match _K_MAX in hmf.py
        'z_max_pk'       : 15.0,     # above the highest redshift bin
    }

    cosmo = Class()
    cosmo.set(params_lcdm)
    cosmo.compute()
    
    for Z_TEST in [9.0, 10.0, 11.0, 12.5, 14.5]:

        # HMF at z=9
        M_h, dndlnm, _, _ = compute_hmf(cosmo, Z_TEST)

        # Donnan z=9 data
        donnan      = load_donnan()
        mask        = donnan['z'] == Z_TEST
        M_UV_bins   = donnan['M_UV'][mask]
        phi_obs     = donnan['phi'][mask]
        sigma_up    = donnan['sigma_up'][mask]
        sigma_down  = donnan['sigma_down'][mask]


        # theory predictions — single-z and GL-integrated
        phi_single = compute_uvlf_theory(M_h, dndlnm, Z_TEST, M_UV_bins)
        phi_integ  = compute_uvlf_theory(M_h, dndlnm, Z_TEST, M_UV_bins,
                                          integrate_bin=True,
                                          z_lo=np.min(DONNAN_Z_EDGES[Z_TEST]), z_hi=np.max(DONNAN_Z_EDGES[Z_TEST]), cosmo=cosmo, n_gl=2)

        # chi^2 for z=9 bin only
        chi2_single = chi_squared(phi_single, phi_obs, sigma_up, sigma_down)
        chi2_integ  = chi_squared(phi_integ,  phi_obs, sigma_up, sigma_down)

        print(f"\n redshift for donnan's test. z = {Z_TEST}")

        # print comparison table
        print(f"\n{'M_UV':>8}  {'phi_single':>12}  {'phi_integ':>12}  "
              f"{'phi_obs':>12}  {'ratio_s':>8}  {'ratio_i':>8}")
        print('-' * 72)

        for i in range(len(M_UV_bins)):
            print(f"{M_UV_bins[i]:>8.2f}  "
                  f"{phi_single[i]:>12.3e}  "
                  f"{phi_integ[i]:>12.3e}  "
                  f"{phi_obs[i]:>12.3e}  "
                  f"{phi_single[i]/phi_obs[i]:>8.3f}  "
                  f"{phi_integ[i]/phi_obs[i]:>8.3f}")

        print('-' * 72)
        print(f"\nchi2 single-z   : {chi2_single:.4f}")
        print(f"chi2 integrated : {chi2_integ:.4f}")
        print(f"Delta chi2      : {chi2_integ - chi2_single:.4f}  "
              f"({'negligible' if abs(chi2_integ - chi2_single) < 1 else 'SIGNIFICANT'})")

    
    
    finkelstein = load_finkelstein()

    for Z_TEST in [8.9, 10.9, 14.0]:

        M_h, dndlnm, _, _ = compute_hmf(cosmo, Z_TEST)

        mask       = finkelstein['z'] == Z_TEST
        M_UV_bins  = finkelstein['M_UV'][mask]
        phi_obs    = finkelstein['phi'][mask]
        sigma_up   = finkelstein['sigma_up'][mask]
        sigma_down = finkelstein['sigma_down'][mask]

        phi_single = compute_uvlf_theory(M_h, dndlnm, Z_TEST, M_UV_bins)
        phi_integ  = compute_uvlf_theory(M_h, dndlnm, Z_TEST, M_UV_bins,
                                          integrate_bin=True,
                                          z_lo=np.min(FINKELSTEIN_Z_EDGES[Z_TEST]),
                                          z_hi=np.max(FINKELSTEIN_Z_EDGES[Z_TEST]),
                                          cosmo=cosmo, n_gl=3)

        chi2_single = chi_squared(phi_single, phi_obs, sigma_up, sigma_down)
        chi2_integ  = chi_squared(phi_integ,  phi_obs, sigma_up, sigma_down)

        print(f"\n Finkelstein z = {Z_TEST}")
        print(f"\n{'M_UV':>8}  {'phi_single':>12}  {'phi_integ':>12}  "
              f"{'phi_obs':>12}  {'ratio_s':>8}  {'ratio_i':>8}")
        print('-' * 72)

        for i in range(len(M_UV_bins)):
            print(f"{M_UV_bins[i]:>8.2f}  "
                  f"{phi_single[i]:>12.3e}  "
                  f"{phi_integ[i]:>12.3e}  "
                  f"{phi_obs[i]:>12.3e}  "
                  f"{phi_single[i]/phi_obs[i]:>8.3f}  "
                  f"{phi_integ[i]/phi_obs[i]:>8.3f}")

        print('-' * 72)
        delta = chi2_integ - chi2_single
        print(f"\nchi2 single-z   : {chi2_single:.4f}")
        print(f"chi2 integrated : {chi2_integ:.4f}")
        print(f"Delta chi2      : {delta:.4f}  "
              f"({'negligible' if abs(delta) < 1 else 'SIGNIFICANT'})")
    
    
    cosmo.struct_cleanup()
    cosmo.empty()