# likelihood/jwst_likelihood_uvlf.py
#
# Cobaya external likelihood — UVLF (UV luminosity function)
# Constrains exotic transient dark energy via UVLF comparison to
# Donnan et al. 2024 (PRIMER, MNRAS 533) + Finkelstein et al. 2024
# (CEERS, ApJL 969) JWST data.
#
# ══════════════════════════════════════════════════════════════════════════════
# DESIGN NOTES
# ══════════════════════════════════════════════════════════════════════════════
#
# PARAMETER FLOW :
#   Cobaya samples (a_samp, s).  The YAML declares derived parameters:
#       a_exo = a_samp
#       b_exo = s - a_samp
#   These derived params are passed to CLASS (class_omx), which computes H(z)
#   with the exotic component already encoded.  This likelihood receives the
#   CLASS cosmo object AFTER CLASS has run — it is completely blind to the
#   reparametrisation.  It never reads a_exo, b_exo, a_samp, or s in logp().
#
# PHYSICALITY POLYGON (H²(z) >= 0 for all z):
#   Enforced in prior() — called by Cobaya BEFORE CLASS runs.
#   Prevents CLASS ODE crashes from unphysical (a_samp, s) proposals.
#   Binding constraint at z* ≈ 13.9:
#       s >= _POLY_SLOPE * a_samp + _POLY_INTERCEPT
#       s >= -0.07202 * a_samp - 1381.5969
#
# DATA:
#   Donnan et al. 2024 (PRIMER) — 29 data points, 5 redshift bins
#       z = 9.0, 10.0, 11.0, 12.5, 14.5
#       Reference cosmology: H0=70, Om=0.3, OL=0.7
#   Finkelstein et al. 2024 (CEERS) — 11 data points, 3 redshift bins
#       z = 8.9, 10.9, 14.0  (completeness-weighted medians — use exactly)
#       Reference cosmology: H0=67.36, Om=0.3153 (Planck 2018)
#   Both datasets are hardcoded in pipeline/uvlf.py via load_donnan()
#   and load_finkelstein(). No external file paths needed.
#   The two surveys cover independent sky fields — chi2 is simply summed.
#
# LIKELIHOOD:
#   Split-Gaussian chi-squared (Jiang et al. 2024, Eq. 3 convention):
#       sigma_i = sigma_up_i   if phi_theory > phi_obs  (overprediction)
#       sigma_i = sigma_down_i if phi_theory <= phi_obs  (underprediction)
#       chi2 = sum_i (phi_theory_i - phi_obs_i)^2 / sigma_i^2
#   logp = -0.5 * chi2_total
#
# VOLUME CORRECTIONS:
#   The published phi_obs values assume the survey reference cosmology.
#   When Cobaya varies H0 and Om (joint runs with BAO/CMB), the data must
#   be rescaled by V_ref/V_MCMC at each step (use_volume_correction=True).
#   For UVLF-only runs (fixed cosmology diagnostics): set False (default).
#   The reference CLASS objects (one per dataset) are built once in
#   initialize() and reused. Never rebuilt in logp().
#
# REDSHIFT EVALUATION:
#   Single-z mode (integrate_bin=False, default):
#       phi evaluated at the nominal bin redshift. Validated against the
#       GL-integrated mode — differences are < 1% for Donnan bins (Dz~1)
#       and < 5% for Finkelstein z~10.9 (Dz=3.3). Use for production MCMC.
#   GL-integrated mode (integrate_bin=True):
#       Volume-weighted average over the bin via n_gl-point Gauss-Legendre
#       quadrature. Costs n_gl extra CLASS calls per bin per step.
#       Use for diagnostics only.
#
# RESTRICTED LIKELIHOOD:
#   z9_donnan and z89_finkelstein bins show LCDM overprediction driven by
#   the Stefanon 2021 SHMR calibrated on pre-JWST data. The z_min_donnan
#   and z_min_finkelstein attributes allow excluding low-z bins to assess
#   sensitivity of the exotic DE signal to this systematic (robustness test).
#   Primary result always uses the full likelihood (defaults).
#
# ══════════════════════════════════════════════════════════════════════════════
# COBAYA YAML REFERENCE (minimal UVLF-only fixed-SHMR run)
# ══════════════════════════════════════════════════════════════════════════════
#
#   likelihood:
#     jwst_likelihood_uvlf.UVLFLikelihood:
#       python_path: /path/to/exo_de_project
#       use_volume_correction: false     # true for joint BAO/CMB runs
#       integrate_bin: false             # true for diagnostic GL integration
#       z_min_donnan: 9.0               # set to 10.0 for restricted likelihood
#       z_min_finkelstein: 8.9          # set to 10.9 for restricted likelihood
#
#   params:
#     a_samp:
#       prior: {min: -1838.0, max: -1e-5}
#       latex: a_\mathrm{samp}
#     s:
#       prior: {min: -1381.597, max: 0.0}
#       latex: s
#     a_exo:
#       derived: "lambda a_samp: a_samp"
#       latex: a_\mathrm{exo}
#     b_exo:
#       derived: "lambda a_samp, s: s - a_samp"
#       latex: b_\mathrm{exo}
#
#   theory:
#     classy:
#       extra_args:
#         z_c_exo: 16.0
#         sigma_z_exo: 3.25
#
# ══════════════════════════════════════════════════════════════════════════════

import os
import sys
import numpy as np
from cobaya.likelihood import Likelihood

# ── Pipeline imports ──────────────────────────────────────────────────────────
# __file__ = .../likelihood/jwst_likelihood_uvlf.py
# project root = one level up from likelihood/
_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pipeline.hmf import compute_hmf
from pipeline.uvlf import (
    load_donnan,
    load_finkelstein,
    compute_uvlf_theory,
    chi_squared,
    DONNAN_Z_EDGES,
    FINKELSTEIN_Z_EDGES,
)


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE-LEVEL CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# Physicality polygon — H²(z) >= 0 for all z.
# Bottom edge of the viable (a_samp, s) region, binding at z* ≈ 13.9.
# Enforced in prior() to reject proposals before CLASS runs.
_POLY_SLOPE     = -0.07202
_POLY_INTERCEPT = -1381.5969

# Reference cosmologies for volume correction.
# Donnan 2024 (PRIMER): H0=70, Om=0.3, OL=0.7
# Finkelstein 2024 (CEERS): H0=67.36, Om=0.3153 (Planck 2018)
# Built once in initialize() via _build_ref_cosmo(); never rebuilt in logp().
_DONNAN_REF_H0  = 70.0
_DONNAN_REF_OM  = 0.30
_DONNAN_REF_OL  = 0.70

_FINK_REF_H0    = 67.36
_FINK_REF_OM    = 0.3153
_FINK_REF_OL    = 0.6847

# ── Fixed SHMR (Stefanon 2021, vary_SHMR = False) ─────────────────────
_SHMR_GAMMA_FIXED = 0.01   # AGN feedback negligible at z >= 8

# ── Floating SHMR priors (vary_SHMR = True) ───────────────────────────
# N, log_Mc, beta: Gaussian from Stefanon et al. 2021 (arXiv:2103.16571) uncertainties.

_SHMR_N_MU,     _SHMR_N_SIG     = 0.0297, 0.0065
_SHMR_LOGMC_MU, _SHMR_LOGMC_SIG = 11.5,   0.2
_SHMR_BETA_MU,  _SHMR_BETA_SIG  = 1.35,   0.26


# ══════════════════════════════════════════════════════════════════════════════
#  LIKELIHOOD CLASS
# ══════════════════════════════════════════════════════════════════════════════

class UVLFLikelihood(Likelihood):
    """
    Cobaya likelihood for the JWST UV luminosity function.

    Constrains exotic transient dark energy (a_exo, b_exo) by comparing
    the predicted UVLF phi(M_UV, z) from the HMF + SHMR + M_UV conversion
    pipeline to published Donnan 2024 (PRIMER) and Finkelstein 2024 (CEERS)
    JWST data across z = 8.9 – 14.5.

    SHMR uses fixed Stefanon 2021 parameters throughout (gamma=0.01, N=0.0297,
    Mc=10^11.5, beta=1.35). No SHMR floating — see DESIGN NOTES above.

    All class attributes below are settable from the Cobaya YAML file.
    """

    # ── YAML-settable attributes ───────────────────────────────────────────
    use_volume_correction: bool  = True
    # True  -> rescale phi_obs by V_ref/V_MCMC each step (joint cosmo runs)
    # False -> phi_obs taken at face value (UVLF-only fixed-cosmo runs)

    integrate_bin: bool = True
    # True  -> GL-integrated phi over each redshift bin (diagnostic only)
    # False -> single-z evaluation at nominal bin redshift (production MCMC)

    n_gl: int = 2
    # Number of Gauss-Legendre nodes per bin when integrate_bin=True.
    # n_gl=2 gives cubic accuracy; adequate for all Donnan bins (Dz~1).
    # Use n_gl=3 for the wide Finkelstein z~10.9 bin (Dz=3.3).

    z_min_donnan: float = 9.0
    # Exclude Donnan bins with z_nominal < z_min_donnan.
    # Default 9.0 = full likelihood. Set to 10.0 for restricted likelihood
    # that excludes the z=9 bin affected by the pre-JWST SHMR systematic.

    z_min_finkelstein: float = 8.9
    # Exclude Finkelstein bins with z_nominal < z_min_finkelstein.
    # Default 8.9 = full likelihood. Set to 10.9 for restricted likelihood.
    
    vary_SHMR: bool = False
    
    vary_beta: bool = False
        
    use_donnan_bins: bool = True
        
    use_finkelstein_bins: bool = True
        
    

    # ══════════════════════════════════════════════════════════════════════
    #  initialize()
    # ══════════════════════════════════════════════════════════════════════

    def initialize(self):
        """
        Called once by Cobaya before any MCMC step.

        Loads both UVLF datasets and pre-filters to the active redshift bins.
        Builds reference CLASS objects for volume correction if enabled.
        Data arrays are cosmology-independent and stored as instance attributes.
        The V_ref/V_MCMC correction is applied inside logp() where the current
        CLASS cosmology is available.
        """
        
        if not self.use_donnan_bins and not self.use_finkelstein_bins:
            raise ValueError(
        "Both use_donnan_bins and use_finkelstein_bins are False. "
        "The UVLF likelihood requires at least one dataset."
            )
        # ── Load data ──────────────────────────────────────────────────────
        
        if self.use_donnan_bins:
            donnan_raw      = load_donnan()
            
            self._donnan_bins = [
            z for z in np.unique(donnan_raw['z'])
            if z >= self.z_min_donnan
            ]
            
            self._donnan_data = {}
            for z_nom in self._donnan_bins:
                mask = donnan_raw['z'] == z_nom
                self._donnan_data[z_nom] = (
                    donnan_raw['M_UV'][mask],
                    donnan_raw['phi'][mask].copy(),
                    donnan_raw['sigma_up'][mask].copy(),
                    donnan_raw['sigma_down'][mask].copy(),
                )
        else:
                self._donnan_bins = []
                self._donnan_data = {}
    
        if self.use_finkelstein_bins:
            finkelstein_raw = load_finkelstein()


            self._finkelstein_bins = [
                z for z in np.unique(finkelstein_raw['z'])
                if z >= self.z_min_finkelstein
            ]


            self._finkelstein_data = {}
            for z_nom in self._finkelstein_bins:
                mask = finkelstein_raw['z'] == z_nom
                self._finkelstein_data[z_nom] = (
                    finkelstein_raw['M_UV'][mask],
                    finkelstein_raw['phi'][mask].copy(),
                    finkelstein_raw['sigma_up'][mask].copy(),
                    finkelstein_raw['sigma_down'][mask].copy(),
                )
        else:
            self._finkelstein_bins = []
            self._finkelstein_data = {}
        
        if self.vary_SHMR and self.vary_beta:
            print("vary_SHMR and vary_beta cannot be both true simultaneously, please only make one of them is True")
            raise ValueError
            
        

        # ── Reference cosmologies for volume correction ────────────────────
        if self.use_volume_correction:
            self._cosmo_ref_don  = _build_ref_cosmo(
                _DONNAN_REF_H0, _DONNAN_REF_OM, _DONNAN_REF_OL,
            )
            self._cosmo_ref_fink = _build_ref_cosmo(
                _FINK_REF_H0, _FINK_REF_OM, _FINK_REF_OL,
            )
        else:
            self._cosmo_ref_don  = None
            self._cosmo_ref_fink = None

        # ── Collect all z-midpoints for Pk_interpolator request ───────────
        self._all_z_mids = list(self._donnan_bins) + list(self._finkelstein_bins)

        n_don  = sum(len(v[0]) for v in self._donnan_data.values())
        n_fink = sum(len(v[0]) for v in self._finkelstein_data.values())

        self.log.info(
            f"UVLFLikelihood initialized | "
            f"Donnan bins={len(self._donnan_bins)} ({n_don} pts) | "
            f"Finkelstein bins={len(self._finkelstein_bins)} ({n_fink} pts) | "
            f"volume_correction={self.use_volume_correction} | "
            f"integrate_bin={self.integrate_bin} | "
            f"z_min_donnan={self.z_min_donnan} | "
            f"z_min_finkelstein={self.z_min_finkelstein} | "
            f"varying SHMR parameters={self.vary_SHMR} | "
            f"varying beta only={self.vary_beta}"
        )
        
        # Claim SHMR params so CLASS (agnostic) does not receive them
        _shmr_active = []
        if self.vary_SHMR:
            _shmr_active = ['shmr_N', 'shmr_log_Mc', 'shmr_beta']
        elif self.vary_beta:
            _shmr_active = ['shmr_beta']
        if _shmr_active:
            self.input_params = list(getattr(self, 'input_params', ())) + _shmr_active
            
        if self.integrate_bin:
            all_edges = {**DONNAN_Z_EDGES, **FINKELSTEIN_Z_EDGES}
            gl_nodes = []
            for z_nom in self._donnan_bins + self._finkelstein_bins:
                z_lo, z_hi = all_edges[z_nom]
                n_gl_bin = 3 if (z_hi - z_lo) > 2.0 else self.n_gl
                xi, _ = np.polynomial.legendre.leggauss(n_gl_bin)
                z_nodes = (z_lo + z_hi) / 2.0 + (z_hi - z_lo) / 2.0 * xi
                gl_nodes.extend(z_nodes.tolist())
            self._all_z_mids = sorted(set(self._all_z_mids + gl_nodes))

    # ══════════════════════════════════════════════════════════════════════
    #  prior()  — evaluated BEFORE CLASS runs
    # ══════════════════════════════════════════════════════════════════════

    def prior(self, **params_values):
        """
        Physicality polygon check: H²(z) >= 0 for all z.

        Identical to jwst_likelihood_differential.py — see DESIGN NOTES.
        Returning -np.inf rejects the proposal before CLASS ever runs,
        preventing ODE crashes from unphysical (a_samp, s) combinations.

        Binding constraint (z* ≈ 13.9):
            s >= -0.07202 * a_samp - 1381.5969
        """
        a_samp = params_values.get('a_samp', 0.0)
        s      = params_values.get('s',      0.0)

        if s < (_POLY_SLOPE * a_samp + _POLY_INTERCEPT):
            return -np.inf

        return 0.0

    # ══════════════════════════════════════════════════════════════════════
    #  get_requirements()
    # ══════════════════════════════════════════════════════════════════════

    def get_requirements(self):
        """
        Tell Cobaya what CLASS must compute before logp() is called.

        Requests Pk_interpolator at all active bin redshifts. compute_hmf()
        calls cosmo.get_pk_array() on the raw CLASS object returned by
        provider.get_classy_cosmo() — the adapter built in logp() handles
        the translation.

        Also requests angular_distance and Hubble at all bin redshifts
        for the volume correction ratio V_ref/V_MCMC.
        """
        z_list = sorted(set(self._all_z_mids))

        # Pk_interpolator request tells Cobaya to run CLASS with mPk output.
        # We never use the interpolator itself — logp() calls CLASS directly
        # via _get_raw_classy().  This request just triggers the computation.
        return {
            'Pk_interpolator': {
                'z':          z_list,
                'k_max':      151.0,
                'nonlinear':  False,
                'vars_pairs': [['delta_tot', 'delta_tot']],
            },
        }

    # ══════════════════════════════════════════════════════════════════════
    #  logp()
    # ══════════════════════════════════════════════════════════════════════

    def logp(self, **params_values):
        """
        Log-likelihood for one MCMC step.

        The exotic DE component is already encoded in the CLASS cosmo object —
        Cobaya passed a_exo, b_exo to CLASS before calling this. This function
        is blind to the exotic DE reparametrisation.

        Loops over active Donnan and Finkelstein redshift bins. At each bin:
            1. Calls compute_hmf(cosmo, z_nom) for the matter power spectrum
            2. Calls compute_uvlf_theory() to predict phi(M_UV)
            3. Optionally applies the volume correction to phi_obs and sigmas
            4. Accumulates the split-Gaussian chi2

        Returns
        -------
        float
            Total log-likelihood = -0.5 * chi2_total.
            Returns -np.inf on any numerical failure without crashing.
        """          
        try:
            
            cosmo = _get_raw_classy(self.provider)

            chi2_total = 0.0
            
            if self.vary_SHMR:
                shmr_N      = params_values['shmr_N']
                shmr_log_Mc = params_values['shmr_log_Mc']
                shmr_beta   = params_values['shmr_beta']
                shmr_gamma = _SHMR_GAMMA_FIXED
                # Hard physicality bounds — these define the prior support.
                # N > 0:                   negative amplitude is unphysical
                # beta > 0:                negative low-mass slope is unphysical
                
                if (shmr_N     <= 0.0
                        or shmr_beta  <= 0.0):
                    return -np.inf

                # Gaussian prior on N, log_Mc, beta
                # Normalization constants omitted — cancel in MCMC ratio
                log_like = _shmr_prior_logp(shmr_N, shmr_log_Mc, shmr_beta)
                if not np.isfinite(log_like):
                    return -np.inf
                
            elif self.vary_beta:
                shmr_N      = _SHMR_N_MU
                shmr_log_Mc = _SHMR_LOGMC_MU
                shmr_beta   = params_values['shmr_beta']
                shmr_gamma = _SHMR_GAMMA_FIXED
                # Hard physicality bounds — these define the prior support.
                # N > 0:                   negative amplitude is unphysical
                # beta > 0:                negative low-mass slope is unphysical
                if (shmr_beta  <= 0.0):
                    return -np.inf

                # Gaussian prior on N, log_Mc, beta
                # Normalization constants omitted — cancel in MCMC ratio
                log_like = _beta_prior_logp(shmr_beta)
                if not np.isfinite(log_like):
                    return -np.inf
                
            else:
                # Fixed Stefanon 2021 SHMR — no prior contribution
                log_like   = 0.0
                
                
            if self.vary_SHMR:
                shmr_kwargs = dict(
                    N=shmr_N,
                    log_Mc=shmr_log_Mc,
                    beta=shmr_beta,
                    gamma=shmr_gamma
                )

            elif self.vary_beta:
                shmr_kwargs = dict(
                    beta=shmr_beta
                )
            else:
                shmr_kwargs = {}            

            # ── Donnan 2024 — active z-bins ────────────────────────────────
            for z_nom in self._donnan_bins:
                

                M_UV_bins, phi_obs, sigma_up, sigma_down = \
                    self._donnan_data[z_nom]

                M_h, dndlnm, _, _ = compute_hmf(cosmo, z_nom)
                

                    
                z_lo, z_hi = DONNAN_Z_EDGES[z_nom]
                phi_theory = compute_uvlf_theory(
                    M_h, dndlnm, z_nom, M_UV_bins,
                    integrate_bin = self.integrate_bin,
                    z_lo          = z_lo,
                    z_hi          = z_hi,
                    cosmo         = cosmo,
                    n_gl          = self.n_gl,
                    **shmr_kwargs
                )

                # Volume correction: rescale observed phi and errors
                # phi_obs_corrected = phi_obs * V_ref/V_MCMC
                if self.use_volume_correction:
                    r          = _volume_ratio(z_nom, cosmo, self._cosmo_ref_don)
                    phi_obs    = phi_obs    * r
                    sigma_up   = sigma_up   * r
                    sigma_down = sigma_down * r


                chi2_total += chi_squared(phi_theory, phi_obs, sigma_up, sigma_down)

            # ── Finkelstein 2024 — active z-bins ──────────────────────────
            for z_nom in self._finkelstein_bins:

                M_UV_bins, phi_obs, sigma_up, sigma_down = \
                    self._finkelstein_data[z_nom]

                M_h, dndlnm, _, _ = compute_hmf(cosmo, z_nom)
                
                    
                z_lo, z_hi = FINKELSTEIN_Z_EDGES[z_nom]
                n_gl_bin = (self.n_gl + 1) if (z_hi - z_lo) > 2.0 else self.n_gl
                phi_theory = compute_uvlf_theory(
                    M_h, dndlnm, z_nom, M_UV_bins,
                    integrate_bin = self.integrate_bin,
                    z_lo          = z_lo,
                    z_hi          = z_hi,
                    cosmo         = cosmo,
                    n_gl          = n_gl_bin,
                    **shmr_kwargs
                )

                if self.use_volume_correction:
                    r          = _volume_ratio(z_nom, cosmo, self._cosmo_ref_fink)
                    phi_obs    = phi_obs    * r
                    sigma_up   = sigma_up   * r
                    sigma_down = sigma_down * r
                    


                chi2_total += chi_squared(phi_theory, phi_obs, sigma_up, sigma_down)

            return float(-0.5 * chi2_total + log_like)

        except Exception as exc:
            import traceback as _tb
            self.log.warning(
                f"logp() raised {type(exc).__name__}: {exc}\n"
                f"{_tb.format_exc()}"
            )
            return -np.inf


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE-LEVEL HELPERS
#  Outside the class — no attribute lookup overhead in the MCMC hot path.
# ══════════════════════════════════════════════════════════════════════════════

def _get_raw_classy(provider):
    """
    Retrieve the live classy.Class object from Cobaya's theory layer.

    Bypasses Cobaya's Pk_interpolator / H(z) / d_A(z) interpolation
    products entirely.  All P(k,z), H(z), d_A(z) calls go directly
    to CLASS's C code — no spline, no log/exp roundtrip, no numerical
    noise from intermediate interpolation layers.

    Parameters
    ----------
    provider : cobaya.provider.Provider (or _MockProvider in tests)

    Returns
    -------
    classy.Class — the live, already-computed CLASS instance
    """
    for theory in provider.model.theory.values():
        if hasattr(theory, 'classy'):
            return theory.classy
    raise RuntimeError(
        "No CLASS theory code found in Cobaya model. "
        "Check that 'classy' is listed under 'theory' in the YAML."
    )
    

def _build_ref_cosmo(H0, Om, OL):
    """
    Build a background-only CLASS instance for the survey reference cosmology.

    Used exclusively for the volume correction ratio V_ref/V_MCMC.
    Only H(z) and angular_distance(z) are needed, so output='' skips
    all power spectrum computation — this CLASS call takes < 0.1s.

    The Omega_b / Omega_cdm split uses a standard physical baryon density.
    The exact split does not affect H(z) or d_A(z); only total Om matters.

    Called once in initialize(). Never called in logp().

    Parameters
    ----------
    H0 : float — Hubble constant [km/s/Mpc]
    Om : float — total matter density parameter
    OL : float — cosmological constant density parameter

    Returns
    -------
    classy.Class — initialised and computed CLASS instance
    """
    from classy import Class

    h         = H0 / 100.0
    omega_b   = 0.022          # standard physical baryon density
    Omega_b   = omega_b / h**2
    Omega_cdm = Om - Omega_b

    params = {
        'h'           : h,
        'Omega_b'     : Omega_b,
        'Omega_cdm'   : Omega_cdm,
        'Omega_Lambda': OL,
        'output'      : '',    # background only — no spectra needed
    }

    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()
    return cosmo


def _volume_ratio(z, cosmo_mcmc, cosmo_ref):
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


def _shmr_prior_logp(N, log_Mc, beta):
    """
    Log-probability of Gaussian priors on N, log_Mc, beta.

    Source: Stefanon et al. 2021.
    bounds are enforced in logp() and it is not included here.
    Normalization constants omitted — they cancel in the M-H ratio.
    """
    lp  = -0.5 * ((N      - _SHMR_N_MU)     / _SHMR_N_SIG)    **2
    lp += -0.5 * ((log_Mc - _SHMR_LOGMC_MU) / _SHMR_LOGMC_SIG)**2
    lp += -0.5 * ((beta   - _SHMR_BETA_MU)  / _SHMR_BETA_SIG) **2
    return lp

def _beta_prior_logp(beta):
    """
    Log-probability of Gaussian priors on beta.

    bounds are enforced in logp() and it is not included here.
    Normalization constants omitted — they cancel in the M-H ratio.
    """
    lp = -0.5 * ((beta   - _SHMR_BETA_MU)  / _SHMR_BETA_SIG) **2
    return lp