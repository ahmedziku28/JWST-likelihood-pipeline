# likelihood/jwst_likelihood.py
#
# Cobaya external likelihood — UNCOVER stellar mass density
# Constrains exotic transient dark energy via cumulative stellar mass density.
#
# ══════════════════════════════════════════════════════════════════════════════
# DESIGN NOTES
# ══════════════════════════════════════════════════════════════════════════════
#
# PARAMETER FLOW (critical to understand before modifying):
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
#   This prevents CLASS ODE crashes from unphysical (a_samp, s) proposals.
#   The polygon bottom edge (binding constraint at z* ≈ 13.9) is:
#       s >= -0.07202 * a_samp - 1381.5969
#   The YAML only needs flat box priors on a_samp and s — no lambda block.
#
# OBSERVED DATA (cosmology-independent parts computed once at initialize()):
#   Galaxies are selected per z-bin, sorted by M_star descending, and their
#   lensing-weighted cumulative mass sums are stored.  The normalisation
#   1/V_survey(cosmo) is applied at every logp() call, because V_survey
#   depends on chi(z) and H(z) from the current CLASS cosmology.
#   UNCOVER-only runs (fixed cosmo in YAML) -> V is effectively constant.
#   Runs with floating cosmo (BAO/CMB) -> V changes each step.
#
# SHMR MODES:
#   vary_SHMR_params = False (default):
#       Uses Stefanon 2021 fixed values from pipeline.stellar_mass_function.
#       gamma = 0.01 fixed.  compute_theory_rho_star() called directly.
#   vary_SHMR_params = True:
#       shmr_N, shmr_log_Mc, shmr_beta sampled by Cobaya.
#       Priors (all enforced here, not in YAML):
#           N      ~ N(0.0297, 0.0065),  N > 0      [Jiang 2024 / Stefanon 2021]
#           log_Mc ~ N(11.5, 0.2)                   [Jiang 2024 / Stefanon 2021]
#           beta   ~ N(1.35, 0.26),      beta > 0   [Jiang 2024 / Stefanon 2021]
#           gamma  = 0.01            fixed since UNCOVER cannot constrain high redshift feedback + highly degenerate behaviour with a
#
# LIKELIHOOD:
#   Split-Gaussian chi-squared summed over all z-bins and all M_star
#   thresholds within each bin (one threshold per galaxy in the catalog).
#
# ══════════════════════════════════════════════════════════════════════════════
# COBAYA YAML REFERENCE (minimal, UNCOVER-only fixed-SHMR run)
# ══════════════════════════════════════════════════════════════════════════════
#
#   likelihood:
#     jwst_likelihood.JWSTLikelihood:
#       python_path: /path/to/exo_de_project
#       mode: spectroscopic             # or photometric
#       vary_SHMR_params: false
#       phot_path:  /path/to/uncover_phot.fits
#       zspec_path: /path/to/uncover_zspec.fits
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
# __file__ = .../likelihood/jwst_likelihood.py
# project root = one level up from likelihood/
_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pipeline.data_extractor import load_catalogs, UNCOVER_AREA_ARCMIN2, UNCOVER_AREA_SR, UNCOVER_SKY_FRACTION      # edit these globals in-file if you'd like to edit the Survey's area

from pipeline.stellar_mass_function import compute_theory_rho_star, compute_observed_rho_star
from pipeline.hmf import compute_hmf

from pipeline.stellar_mass_function import shmr_mstar
 
from pipeline.differential_smf import (
    DEFAULT_LOG10_MSTAR_BINS,
    compute_theory_differential_smf,
    compute_theory_differential_rho,
    compute_observed_differential_smf,
    compute_observed_differential_rho,
)
from pipeline.hmf import compute_cosmic_variance



# ══════════════════════════════════════════════════════════════════════════════
#  MODULE-LEVEL CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# Redshift bin definitions: (z_min, z_max, z_mid)
# z_mid is the bin midpoint where theory is evaluated (Hashim's decision).
_ZBINS_PHOTOZ = [
    (6.0,  8.0,   7.0),
    (8.0,  10.0,  9.0),
    (10.0, 15.0, 12.5),
    (15.0, 20.0, 17.5),
]
_ZBINS_SPECZ = [
    (6.0,  8.0,   7.0),
    (8.0,  10.0,  9.0),
    (10.0, 15.0, 12.5),
    # 15-20 bin excluded: no spectroscopic galaxies past z = 15
]

# ── Physicality polygon — H²(z) >= 0 for all z ───────────────────────────────
# approximated Bottom edge of the viable (a_samp, s) polygon.
# Binding redshift z* ≈ 13.9.  Derived analytically via
# scipy.spatial.HalfspaceIntersection. through ab_grid_fast.py
# Physical condition: s >= _POLY_SLOPE * a_samp + _POLY_INTERCEPT
_POLY_SLOPE     = -0.07202
_POLY_INTERCEPT = -1381.5969

# ── Fixed SHMR (Stefanon 2021, vary_SHMR_params = False) ─────────────────────
_SHMR_GAMMA_FIXED = 0.01   # AGN feedback negligible at z >= 8

# ── Floating SHMR priors (vary_SHMR_params = True) ───────────────────────────
# N, log_Mc, beta: Gaussian from Stefanon et al. 2021 (arXiv:2103.16571) uncertainties.

_SHMR_N_MU,     _SHMR_N_SIG     = 0.0297, 0.0065
_SHMR_LOGMC_MU, _SHMR_LOGMC_SIG = 11.5,   0.2
_SHMR_BETA_MU,  _SHMR_BETA_SIG  = 1.35,   0.26

class _CosmoAdapter:
    """
    Duck-types the CLASS API subset used by compute_hmf() and V_survey().
    Built each logp() call from Cobaya's standard provider products.
    No version-dependent hacks — uses only Pk_interpolator, Hubble,
    comoving_radial_distance, and Omega_m (all BoltzmannBase staples).
    """
    __slots__ = ('_h', '_Om', '_pk_interp', '_dist')

    def __init__(self, h, Om, pk_interp, dist_func):
        self._h         = h
        self._Om        = Om
        self._pk_interp = pk_interp
        self._dist      = dist_func

    def h(self):
        return self._h

    def Omega_m(self):
        return self._Om

    def get_pk_array(self, k_arr, z_arr, n_k, n_z, nonlinear):
        """Matches CLASS get_pk_array signature used by _get_pk_vec."""
        return self._pk_interp.P(z_arr[0], k_arr)

    def comoving_distance(self, z):
        return self._dist(z)

# ══════════════════════════════════════════════════════════════════════════════
#  LIKELIHOOD CLASS
# ══════════════════════════════════════════════════════════════════════════════

class JWSTLikelihood(Likelihood):
    """
    Cobaya likelihood for UNCOVER cumulative stellar mass density.

    Constrains exotic transient dark energy (a_exo, b_exo) by comparing
    the predicted and observed cumulative stellar mass density rho_star(>M_star)
    in redshift bins z = 6-8, 8-10, 10-15 [and 15-20 for photoz mode].

    All class attributes below are settable from the Cobaya YAML file.
    """

    # ── YAML-settable class attributes ────────────────────────────────────
    mode:                str   = 'spectroscopic'   # 'spectroscopic' or 'photometric'
    vary_SHMR_params:    bool  = False     # True -> N, log_Mc, beta float
    phot_path:           str   = ''        # path to photometric FITS catalog
    zspec_path:          str   = ''        # path to spectroscopic FITS catalog
        
        

    use_differential:      bool  = False
    # True  -> binned differential SMF/rho likelihood
    # False -> cumulative rho_star likelihood
 
    differential_observable: str = 'smf'
    # 'smf' -> use dn/dlog10(M_star)   [Mpc^{-3} dex^{-1}]
    # 'rho' -> use drho/dlog10(M_star) [M_sun Mpc^{-3} dex^{-1}]


    # ══════════════════════════════════════════════════════════════════════
    #  initialize()
    # ══════════════════════════════════════════════════════════════════════
    
    def initialize(self):
        """
        Called once by Cobaya before any MCMC step.

        Loads catalogs, selects galaxies per z-bin, and pre-computes the
        cosmology-independent lensing-weighted cumulative mass sums.
        The 1/V_survey normalization is applied in logp() because V_survey
        depends on chi(z) and H(z) from the current CLASS cosmology.
        """
        
        if self.mode not in ('spectroscopic', 'photometric'):
            raise ValueError(
                f"JWSTLikelihood: mode must be 'spectroscopic' or 'photometric', "
                f"got '{self.mode}'"
            )
        if not self.phot_path or not self.zspec_path:
            raise ValueError(
                "JWSTLikelihood: phot_path and zspec_path must be set in the YAML."
            )

        self._zbins  = _ZBINS_SPECZ if self.mode == 'spectroscopic' else _ZBINS_PHOTOZ
        self._z_mids = [zb[2] for zb in self._zbins]

        # Load catalogs once
        phot_table, spec_table = load_catalogs(self.phot_path, self.zspec_path)
        catalog = spec_table if self.mode == 'spectroscopic' else phot_table

        self.log.info(
            f"JWSTLikelihood initialized | mode={self.mode} | "
            f"vary_SHMR_params={self.vary_SHMR_params} | "
            f"bins={len(self._zbins)}"
        )
        self._catalog = catalog

    # ══════════════════════════════════════════════════════════════════════
    #  prior()  — evaluated BEFORE CLASS runs
    # ══════════════════════════════════════════════════════════════════════

    def prior(self, **params_values):
        """
        Physicality polygon check: H²(z) >= 0 for all z.

        Called by Cobaya as part of prior evaluation, BEFORE the theory
        code (CLASS) runs.  Returning -np.inf here rejects the proposal
        immediately without ever calling CLASS, preventing ODE crashes
        from unphysical (a_samp, s) proposals.

        This completely replaces the h2_positivity lambda block that would
        otherwise live in the YAML prior: section.  The YAML only needs
        flat box priors on a_samp and s.

        Polygon bottom edge (binding constraint at z* ≈ 13.9):
            s >= _POLY_SLOPE * a_samp + _POLY_INTERCEPT
            s >= -0.07202 * a_samp - 1381.5969
        See PROJECT_STATUS.md Section 5a for derivation.
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

        Requesting Pk_interpolator at the bin midpoints ensures CLASS runs
        the full matter power spectrum solver at the target redshifts.
        compute_hmf() then calls cosmo.get_pk_array() on the raw CLASS
        object returned by provider.get_classy_cosmo().
        """
        z_edges = set()
        for z_min, z_max, _ in self._zbins:
            z_edges.update([z_min, z_max])

        return {
            'Pk_interpolator': {
                'z':          self._z_mids,
                'k_max':      510.0,
                'nonlinear':  False,
                'vars_pairs': [['delta_tot', 'delta_tot']],
            },
            'h': None,
            'comoving_radial_distance':   {'z': sorted(z_edges)},
            'Omega_m':                    None,
        }
    # ══════════════════════════════════════════════════════════════════════
    #  logp()
    # ══════════════════════════════════════════════════════════════════════

    def logp(self, **params_values):
        """
        Log-likelihood for one MCMC step.

        The exotic component is already encoded in the CLASS cosmo object
        (Cobaya passed a_exo, b_exo to CLASS before calling this method).
        This function is completely blind to the exotic DE reparametrisation.

        When vary_SHMR_params = True, reads shmr_N, shmr_log_Mc, shmr_beta
        from params_values and applies their priors internally.

        Returns
        -------
        float
            Total log-likelihood (including SHMR Gaussian priors when active).
            Returns -np.inf on any numerical failure without crashing.
        """
        try:
            # ── Build adapter from Cobaya provider ────────────────────
            h = self.provider.get_param('h')
            Om = self.provider.get_param('Omega_m')

            pk_interp = self.provider.get_Pk_interpolator(
                var_pair=('delta_tot', 'delta_tot'),
                nonlinear=False,
            )

            cosmo = _CosmoAdapter(
                h, Om, pk_interp,
                self.provider.get_comoving_radial_distance,
            )

            # ── SHMR parameters ────────────────────────────────────────────
            if self.vary_SHMR_params:
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

            else:
                # Fixed Stefanon 2021 SHMR — no prior contribution
                shmr_gamma = _SHMR_GAMMA_FIXED
                log_like   = 0.0
                
            if self.use_differential and self.differential_observable not in ('smf', 'rho'):
                raise ValueError(
                    f"differential_observable must be 'smf' or 'rho', "
                    f"got '{self.differential_observable}'"
                )
                

            for i, (z_min, z_max, z_mid) in enumerate(self._zbins):

                V = V_survey(cosmo, z_min, z_max)
                if not np.isfinite(V) or V <= 0.0:
                    return -np.inf

                M_h, dndlnm, hmf_sigma, Pk = compute_hmf(cosmo, z_mid)

                if self.vary_SHMR_params:
                    shmr_kwargs = dict(
                        N=shmr_N,
                        Mc=10.0 ** shmr_log_Mc,
                        beta=shmr_beta,
                        gamma=shmr_gamma,
                    )
                else:
                    shmr_kwargs = {}
                    
                

                # ══════════════════════════════════════════════════════
                #  DIFFERENTIAL MODE
                # ══════════════════════════════════════════════════════
                if self.use_differential:
                    
                    edges = DEFAULT_LOG10_MSTAR_BINS

                    if self.differential_observable == 'smf':
                        if self.vary_SHMR_params:
                            _, pred = compute_theory_differential_smf(
                                M_h, dndlnm, edges,
                                N=shmr_N, log_Mc=shmr_log_Mc,
                                beta=shmr_beta, gamma=shmr_gamma,
                            )
                        else:
                            _, pred = compute_theory_differential_smf(
                                M_h, dndlnm, edges,
                            )
                    else:
                        if self.vary_SHMR_params:
                            _, pred = compute_theory_differential_rho(
                                M_h, dndlnm, edges,
                                N=shmr_N, log_Mc=shmr_log_Mc,
                                beta=shmr_beta, gamma=shmr_gamma,
                            )
                        else:
                            _, pred = compute_theory_differential_rho(
                                M_h, dndlnm, edges,
                            )

                    if self.differential_observable == 'smf':
                        _, obs, sigma_poisson, sigma_mass, n_gal = \
                            compute_observed_differential_smf(
                                self._catalog, z_min, z_max, V, edges,
                            )
                    else:
                        _, obs, sigma_poisson, sigma_mass, n_gal = \
                            compute_observed_differential_rho(
                                self._catalog, z_min, z_max, V, edges,
                            )

                    sigma_cv = compute_cosmic_variance(
                        Pk, M_h, dndlnm, hmf_sigma, V,
                        edges, shmr_mstar, shmr_kwargs,
                    )

                    sigma_tot = np.sqrt(
                        sigma_poisson**2
                        + sigma_mass**2
                        + (sigma_cv * obs)**2
                    )

                    good = (n_gal > 0) & np.isfinite(sigma_tot) & (sigma_tot > 0.0)
                    if not np.any(good):
                        continue

                    log_like -= 0.5 * np.sum(((pred[good] - obs[good]) / sigma_tot[good])**2)

                # ══════════════════════════════════════════════════════
                #  CUMULATIVE MODE
                # ══════════════════════════════════════════════════════
                else:

                    M_star_thr, rho_obs, rho_low, rho_high = \
                        compute_observed_rho_star(self._catalog, z_min, z_max, V)

                    if self.vary_SHMR_params:
                        rho_theory = compute_theory_rho_star(
                            M_h, dndlnm, M_star_thr,
                            shmr_N, shmr_log_Mc, shmr_beta, shmr_gamma,
                        )
                    else:
                        rho_theory = compute_theory_rho_star(M_h, dndlnm, M_star_thr)

                    sigma = np.where(
                        rho_theory > rho_obs,
                        rho_high - rho_obs,
                        rho_obs  - rho_low,
                    )

                    bad = ~np.isfinite(sigma) | (sigma <= 0.0)
                    if np.any(bad):
                        self.log.debug(
                            f"z-bin ({z_min},{z_max}): {bad.sum()} degenerate "
                            f"sigma values — skipping those data points"
                        )
                        good = ~bad
                        if not np.any(good):
                            continue
                        rho_theory = rho_theory[good]
                        rho_obs    = rho_obs[good]
                        sigma      = sigma[good]

                    log_like -= 0.5 * np.sum(((rho_theory - rho_obs) / sigma)**2)

            return float(log_like)

        except Exception as exc:
            import traceback
            self.log.debug(f"logp failed: {exc}\n{traceback.format_exc()}")
            return -np.inf


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE-LEVEL HELPERS
#  Defined outside the class: no repeated attribute lookup, marginally faster
#  in the MCMC hot path where these are called thousands of times.
# ══════════════════════════════════════════════════════════════════════════════

def V_survey(cosmo, z_min, z_max):
    """Comoving survey volume [Mpc^3] for UNCOVER in redshift bin [z_lo, z_hi)."""
    
    chi_min = cosmo.comoving_distance(z_min) # Mpc
    chi_max = cosmo.comoving_distance(z_max) # Mpc
    
    vol =  (4.0/3.0) * np.pi * (chi_max**3 - chi_min**3)   # Volume of shell in a flat universe

    return vol * UNCOVER_SKY_FRACTION




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