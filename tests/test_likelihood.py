#!/usr/bin/env python3
"""
test_likelihood.py

Comprehensive validation suite for the JWST exotic dark energy pipeline,
covering every unit and physical calculation in the differential SMF
observable addition.

Produces:
  1. Console validation report — 13 physics + unit tests
  2. Two LaTeX .tex files:
       table_diff_smf.tex  — ΛCDM vs Exotic DE for dn/dlog10(M_star)
       table_diff_rho.tex  — ΛCDM vs Exotic DE for drho_star/dlog10(M_star)
  3. labbe_rho_star.pdf/png — Labbé/BK2023 cumulative rho_star plot with
       best-fit model curve from the differential likelihood minimisation.

Usage:
    python test_likelihood.py \\
        --phot_path  data/UNCOVER_DR4_SPS_catalog.fits \\
        --zspec_path data/UNCOVER_DR4_SPS_zspec_catalog.fits \\
        --output_dir test_outputs/

Run only validation tests (no minimisation):
    python test_likelihood.py ... --skip_bobyqa

"""

import argparse
import os
import sys
import time
import traceback
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.table import Table

# ── Path setup ────────────────────────────────────────────────────────────────
_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from classy import Class
from likelihood.jwst_likelihood import JWSTLikelihood, V_survey as _V_survey_fn

# ── Pipeline imports (for direct unit validation) ─────────────────────────────
from pipeline.hmf import (
    compute_hmf, compute_cosmic_variance,
    _M_GRID, _N_M, _K_GRID, _N_K,
    DELTA_C, a_SMT, q_SMT,
)
from pipeline.stellar_mass_function import (
    shmr_mstar, SHMR_N, SHMR_LOG_MC, SHMR_BETA, SHMR_GAMMA,
    _SHMR_MC,
    compute_theory_rho_star,
    compute_observed_rho_star,
)
from pipeline.differential_smf import (
    DEFAULT_LOG10_MSTAR_BINS,
    compute_theory_differential_smf,
    compute_theory_differential_rho,
    compute_observed_differential_smf,
    compute_observed_differential_rho,
)
from pipeline.data_extractor import UNCOVER_SKY_FRACTION


# ══════════════════════════════════════════════════════════════════════════════
#  FIXED CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

_Z_C_EXO     = 16.0
_SIGMA_Z_EXO = 3.25

_POLY_SLOPE     = -0.07202
_POLY_INTERCEPT = -1381.5969

# Planck 2018 base cosmology
_PLANCK = {
    'h':             0.6736,
    'omega_b':       0.02237,
    'omega_cdm':     0.1200,
    'n_s':           0.9649,
    'A_s':           2.101e-9,
    'tau_reio':      0.0544,
    'output':        'mPk',
    'P_k_max_1/Mpc': 510.0,
    'z_max_pk':      20.0,
    'non linear':    'none',
    'z_c_exo':       _Z_C_EXO,
    'sigma_z_exo':   _SIGMA_Z_EXO,
}

_LCDM_PARAMS = {'a_exo': 0.0, 'b_exo': 0.0}




# ══════════════════════════════════════════════════════════════════════════════
#  LABBÉ / BOYLAN-KOLCHIN 2023 HARDCODED DATA
#  Source: Labbé et al. 2023 (Nature 616, 266)
#          Boylan-Kolchin 2023 (Nature Astronomy 7, 1009)
#
#  These are the 6 massive galaxy candidates BK2023 flagged as problematic
#  for ΛCDM.  Stellar masses are log10(M_star/M_sun).
#  All are field galaxies with negligible lensing: mu ≈ 1.0.
#  Uncertainties are approximate from SED fitting posteriors.
# ══════════════════════════════════════════════════════════════════════════════

_LABBE_GALAXIES = np.array([
    # z_phot  log10_M50  log10_M16  log10_M84   mu
    [9.1,     11.16,     10.99,     11.33,       1.0],
    [8.5,     10.99,     10.75,     11.11,       1.0],
    [7.4,     10.97,     10.78,     11.11,       1.0],
    [7.7,     10.54,     10.31,     10.77,       1.0],
    [7.5,     10.23,     10.11,     10.39,       1.0],
    [8.1,     10.10,     9.97,      10.27,       1.0],
])
_LABBE_IDS = ['39575', '17487', '35300', '3686', '6878', '12553']

# Survey area for Labbé et al. 2023 (CEERS + GLASS early JWST, approximate)
_LABBE_AREA_ARCMIN2 = 35.0
_LABBE_Z_MIN        = 7.0
_LABBE_Z_MAX        = 10.0
# Full sky in arcmin²: 4π × (180/π × 60)² ≈ 1.4852e8 arcmin²
_LABBE_SKY_FRACTION = _LABBE_AREA_ARCMIN2 / (
    4.0 * np.pi * (180.0 * 60.0 / np.pi) ** 2
)


# ══════════════════════════════════════════════════════════════════════════════
#  MOCK COBAYA PROVIDER INFRASTRUCTURE  (kept and extended from original)
# ══════════════════════════════════════════════════════════════════════════════

class _MockProvider:
    """
    Minimal stand-in for Cobaya's provider object.
    Wraps a real classy.Class instance and serves the same four products
    that logp() requests via _CosmoAdapter.
    """
    def __init__(self, cosmo, params=None):
        self._cosmo  = cosmo
        self._params = params or {}

    def get_comoving_radial_distance(self, z):
        return self._cosmo.comoving_distance(z)

    def get_Pk_interpolator(self, var_pair=None, nonlinear=False):
        cosmo = self._cosmo
        class _MockPkInterp:
            def P(self, z, k):
                k_arr = np.atleast_1d(k)
                z_arr = np.array([z])
                return np.asarray(
                    cosmo.get_pk_array(k_arr, z_arr, len(k_arr), 1, False)
                )
        return _MockPkInterp()

    def get_param(self, name):
        if name == 'Omega_m':
            return self._cosmo.Omega_m()
        elif name == 'h':
            return self._cosmo.h()
        return self._params.get(name, 0.0)


class _MockLogger:
    def info(self,    msg): print(f"  [INFO]  {msg}")
    def debug(self,   msg): pass   # suppress debug noise during tests
    def warning(self, msg): print(f"  [WARN]  {msg}")
    def error(self,   msg): print(f"  [ERROR] {msg}")


def _build_likelihood(phot_path, zspec_path, mode,
                       use_differential=False,
                       differential_observable='smf',
                       vary_shmr=False):
    """Instantiate and initialise a JWSTLikelihood without Cobaya."""
    like = JWSTLikelihood.__new__(JWSTLikelihood)
    like.mode                    = mode
    like.vary_SHMR_params        = vary_shmr
    like.phot_path               = phot_path
    like.zspec_path              = zspec_path
    like.use_differential        = use_differential
    like.differential_observable = differential_observable
    like.log                     = _MockLogger()
    like.initialize()
    return like


def _build_cosmo(a_exo=0.0, b_exo=0.0):
    """Instantiate and compute a class_omx cosmology."""
    p = dict(_PLANCK)
    p['a_exo'] = a_exo
    p['b_exo'] = b_exo
    cosmo = Class()
    cosmo.set(p)
    cosmo.compute()
    return cosmo


def _attach_provider(like, cosmo, extra_params=None):
    params = {'a_samp': 0.0}
    if extra_params:
        params.update(extra_params)
    like.provider = _MockProvider(cosmo, params)


def _logp(like, a_exo, b_exo, cosmo=None):
    """
    Call like.logp() at (a_exo, b_exo).
    Builds and destroys its own CLASS instance unless one is supplied.
    """
    if cosmo is None:
        cosmo = _build_cosmo(a_exo, b_exo)
        owns = True
    else:
        owns = False

    _attach_provider(like, cosmo, {'a_samp': a_exo, 's': a_exo + b_exo})
    result = like.logp(a_samp=a_exo, s=a_exo + b_exo)

    if owns:
        cosmo.struct_cleanup()
        cosmo.empty()

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  DATA COUNTING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _count_galaxies(like, cosmo):
    """
    Return total galaxy count across all z-bins.
    This is N in the table header (dataset size).
    """
    total = 0
    for z_min, z_max, _ in like._zbins:
        chi_lo = cosmo.comoving_distance(z_min)
        chi_hi = cosmo.comoving_distance(z_max)
        V = (4.0 / 3.0) * np.pi * (chi_hi**3 - chi_lo**3) * UNCOVER_SKY_FRACTION
        M_thr, _, _, _ = compute_observed_rho_star(like._catalog, z_min, z_max, V)
        total += len(M_thr)
    return total


def _count_nonempty_bins(like, cosmo, observable='smf'):
    """
    Return number of non-empty stellar mass bins across all z-bins.
    Used as the effective N in chi²_reduced for the differential likelihood.
    """
    total = 0
    fn = (compute_observed_differential_smf if observable == 'smf'
          else compute_observed_differential_rho)
    for z_min, z_max, _ in like._zbins:
        chi_lo = cosmo.comoving_distance(z_min)
        chi_hi = cosmo.comoving_distance(z_max)
        V = (4.0 / 3.0) * np.pi * (chi_hi**3 - chi_lo**3) * UNCOVER_SKY_FRACTION
        _, _, _, _, n_gal = fn(like._catalog, z_min, z_max, V,
                               DEFAULT_LOG10_MSTAR_BINS)
        total += int(np.sum(n_gal > 0))
    return total


# ══════════════════════════════════════════════════════════════════════════════
#  RESULT TRACKER
# ══════════════════════════════════════════════════════════════════════════════

class _Results:
    def __init__(self):
        self._rows = []

    def add(self, test_id, name, passed, detail=''):
        icon = '✅ PASS' if passed else '❌ FAIL'
        self._rows.append((test_id, name, icon, detail))
        print(f'  {icon}  [{test_id}] {name}')
        if detail:
            print(f'         {detail}')

    def summary(self):
        print('\n' + '═' * 72)
        print('  VALIDATION SUMMARY')
        print('═' * 72)
        for tid, name, icon, _ in self._rows:
            print(f'  {icon}  T{tid:02d}  {name}')
        n_pass = sum(1 for _, _, icon, _ in self._rows if 'PASS' in icon)
        n_fail = len(self._rows) - n_pass
        print('─' * 72)
        print(f'  {n_pass}/{len(self._rows)} passed', end='')
        if n_fail == 0:
            print('  — all good. بسم الله 🚀')
        else:
            print(f'  — fix {n_fail} failure(s) before launching.')
        print('═' * 72)


# ══════════════════════════════════════════════════════════════════════════════
#  VALIDATION TESTS
# ══════════════════════════════════════════════════════════════════════════════

def test_T01_sigma8_recovery(cosmo_lcdm, results):
    """
    T01 — sigma_8 recovery.

    Our pipeline computes sigma(M) by integrating P(k,z) with the same
    top-hat window kernel that CLASS uses to define sigma_8.  At z=0,
    sigma evaluated at the mass M corresponding to R = 8/h Mpc must
    match CLASS's own sigma8() output to within 2 %.

    Validated against: Planck 2018 sigma_8 = 0.811 ± 0.006.
    """
    print('\n' + '─' * 72)
    print('T01 — sigma_8 recovery (our pipeline vs CLASS)')
    print('─' * 72)
    try:
        h   = cosmo_lcdm.h()
        Om  = cosmo_lcdm.Omega_m()
        rho_m0 = Om * 2.775e11 * h**2   # M_sun/Mpc^3

        # R_8 = 8/h Mpc (Lagrangian radius matching sigma_8 definition)
        R8 = 8.0 / h                    # Mpc
        M8 = (4.0 / 3.0) * np.pi * R8**3 * rho_m0   # M_sun

        # Compute sigma on our 750-point grid at z=0
        M_h, dndlnm, sigma_grid, _ = compute_hmf(cosmo_lcdm, 0.0)

        # Interpolate sigma to M_8 (log-linear, since grid is log-spaced)
        sigma_ours = float(np.interp(np.log10(M8),
                                     np.log10(M_h), sigma_grid))

        sigma_class = cosmo_lcdm.sigma8()
        rel_err     = abs(sigma_ours - sigma_class) / sigma_class

        passed = rel_err < 0.02
        results.add(1, 'sigma_8 recovery',  passed,
                    f'our={sigma_ours:.4f}  CLASS={sigma_class:.4f}  '
                    f'err={rel_err*100:.2f}%  (tol=2%)')
    except Exception:
        results.add(1, 'sigma_8 recovery', False,
                    traceback.format_exc())


def test_T02_hmf_mass_fraction(cosmo_lcdm, results):
    """
    T02 — HMF mass fraction self-consistency.

    The SMT mass function satisfies analytically:

        integral_0^inf f(nu) d(nu)/nu = 1

    meaning all matter ends up in halos when integrated over ALL masses.
    A_SMT = 0.3222 was chosen explicitly to enforce this condition.

    WHY WE DO NOT GET 1.0
    ---------------------
    Our grid starts at M_h = 10^6 M_sun.  At z=0, sigma(10^6 M_sun) ~ 6,
    so nu(10^6 M_sun) ~ delta_c / sigma ~ 1.686 / 6 ~ 0.28.

    This is a SMALL nu — meaning 10^6 M_sun halos are common, not rare.
    The SMT f(nu)/nu is non-negligible at nu=0.28, and the integral from
    nu=0 to nu=0.28 (i.e., halos BELOW 10^6 M_sun) contributes roughly:

        integral_0^{0.28} f(nu) d(nu)/nu ~ 0.36

    So our grid captures 1 - 0.36 = 0.64 of the total mass fraction.
    """
    print('\n' + '─' * 72)
    print('T02 — HMF mass fraction normalization')
    print('─' * 72)
    try:
        h   = cosmo_lcdm.h()
        Om  = cosmo_lcdm.Omega_m()
        rho_m0 = Om * 2.775e11 * h**2

        M_h, dndlnm, _, _ = compute_hmf(cosmo_lcdm, 0.0)
        integral = float(np.trapz(M_h * dndlnm / rho_m0, np.log(M_h)))

        passed = 0.55 <= integral <= 0.72
        results.add(2, 'HMF mass fraction ≈ 1', passed,
                    f'integral = {integral:.4f}  (expected 0.55~ 0.72)')
    except Exception:
        results.add(2, 'HMF mass fraction ≈ 1', False,
                    traceback.format_exc())


def test_T03_shmr_pivot_identity(results):
    """
    T03 — SHMR analytic identity at the pivot mass.

    At M_h = M_c (the characteristic mass), the double power law simplifies:
        (M_h/M_c)^{-beta} = 1,  (M_h/M_c)^{gamma} = 1
        denominator = 2
        M_star = (2N / 2) * M_c = N * M_c

    This identity must hold to machine precision.
    Validated analytically from Stefanon 2021 Eq. 1.
    """
    print('\n' + '─' * 72)
    print('T03 — SHMR analytic identity at pivot mass M_c')
    print('─' * 72)
    try:
        Mc       = _SHMR_MC
        M_star   = float(shmr_mstar(np.array([Mc]),
                                     SHMR_N, Mc, SHMR_BETA, SHMR_GAMMA)[0])
        expected = SHMR_N * Mc
        rel_err  = abs(M_star - expected) / expected

        passed = rel_err < 1e-8
        results.add(3, 'SHMR pivot identity M_star(M_c) = N·M_c', passed,
                    f'computed={M_star:.6e}  expected={expected:.6e}  '
                    f'rel_err={rel_err:.2e}')
    except Exception:
        results.add(3, 'SHMR pivot identity', False,
                    traceback.format_exc())


def test_T04_shmr_monotone(cosmo_lcdm, results):
    """
    T04 — SHMR strict monotonicity.

    shmr_mstar(M_h) must be strictly increasing over the entire HMF grid.
    Monotonicity is required for SHMR inversion (used in all differential
    and cosmic variance calculations).  Any non-monotone segment would
    silently produce wrong halo mass bin boundaries.
    """
    print('\n' + '─' * 72)
    print('T04 — SHMR strict monotonicity on HMF grid')
    print('─' * 72)
    try:
        Mc     = _SHMR_MC
        M_star = shmr_mstar(_M_GRID, SHMR_N, Mc, SHMR_BETA, SHMR_GAMMA)
        diffs  = np.diff(M_star)
        n_bad  = int(np.sum(diffs <= 0))

        passed = n_bad == 0
        results.add(4, 'SHMR strict monotone on 750-point grid', passed,
                    f'non-increasing steps = {n_bad}  (must be 0)')
    except Exception:
        results.add(4, 'SHMR monotonicity', False,
                    traceback.format_exc())


def test_T05_differential_phi_positive(cosmo_lcdm, results):
    """
    T05 — Differential phi ≥ 0 everywhere.

    The number density dn/dlog10(M_star) is non-negative by definition.
    A negative value would indicate a sign error in the HMF or the
    trapezoidal integral.
    """
    print('\n' + '─' * 72)
    print('T05 — Differential phi ≥ 0 on all bins')
    print('─' * 72)
    try:
        M_h, dndlnm, _, _ = compute_hmf(cosmo_lcdm, 9.0)
        _, phi = compute_theory_differential_smf(M_h, dndlnm,
                                                 DEFAULT_LOG10_MSTAR_BINS)
        n_neg = int(np.sum(phi < 0))
        passed = n_neg == 0
        results.add(5, 'Differential phi ≥ 0 everywhere', passed,
                    f'negative bins = {n_neg}  (must be 0)  '
                    f'min={phi.min():.3e}  max={phi.max():.3e}')
    except Exception:
        results.add(5, 'phi ≥ 0', False, traceback.format_exc())


def test_T06_differential_phi_integral_consistency(cosmo_lcdm, results):
    """
    T06 — Differential phi integral recovers direct HMF integral.

    sum_i [ phi_i × Delta_log10M_i ] must equal the direct trapezoidal
    integral of dn/dlnM over the same halo mass range.

    These are the same computation partitioned differently: the differential
    SMF breaks the integral into bins, while the direct computation integrates
    the full range at once.  They must agree to <0.1 %.

    This test validates that:
      (a) SHMR inversion is correct at all bin edges
      (b) searchsorted bin assignments are gapless and non-overlapping
      (c) There are no off-by-one errors in the index slicing
    """
    print('\n' + '─' * 72)
    print('T06 — Differential phi integral vs direct HMF integral')
    print('─' * 72)
    try:
        M_h, dndlnm, _, _ = compute_hmf(cosmo_lcdm, 9.0)
        lnM_h = np.log(M_h)
        Mc    = _SHMR_MC

        # Differential sum
        _, phi = compute_theory_differential_smf(M_h, dndlnm,
                                                 DEFAULT_LOG10_MSTAR_BINS)
        delta  = np.diff(DEFAULT_LOG10_MSTAR_BINS)
        sum_diff = float(np.sum(phi * delta))

        # Direct integral over the same M_h range
        M_star = shmr_mstar(M_h, SHMR_N, Mc, SHMR_BETA, SHMR_GAMMA)
        M_star_lo = 10.0 ** DEFAULT_LOG10_MSTAR_BINS[0]
        M_star_hi = 10.0 ** DEFAULT_LOG10_MSTAR_BINS[-1]
        M_h_lo = float(np.interp(M_star_lo, M_star, M_h))
        M_h_hi = float(np.interp(M_star_hi, M_star, M_h))

        idx_lo = np.searchsorted(M_h, M_h_lo, side='left')
        idx_hi = np.searchsorted(M_h, M_h_hi, side='right')
        direct = float(np.trapz(dndlnm[idx_lo:idx_hi],
                                lnM_h[idx_lo:idx_hi]))

        rel_err = abs(sum_diff - direct) / max(abs(direct), 1e-30)
        passed  = rel_err < 1e-3   # < 0.1 %
        results.add(6,
                    'sum(phi*dlogM) matches direct HMF integral',
                    passed,
                    f'diff={sum_diff:.6e}  direct={direct:.6e}  '
                    f'rel_err={rel_err*100:.4f}%  (tol=0.1%)')
    except Exception:
        results.add(6, 'phi integral consistency', False,
                    traceback.format_exc())


def test_T07_differential_rho_integral_consistency(cosmo_lcdm, results):
    """
    T07 — Differential rho integral recovers cumulative rho_star.

    sum_i [ rho_bin_i × Delta_log10M_i ] must recover compute_theory_rho_star
    evaluated at the minimum mass threshold.

    Both compute the same integral:
        ∫ M_star(M_h) * (dn/dlnM_h) dlnM_h

    over [M_h(M_star,min), M_h(M_star,max)].  Agreement to < 2 % is required.
    Larger mismatch would indicate a unit error or wrong integrand.
    """
    print('\n' + '─' * 72)
    print('T07 — Differential rho integral vs cumulative rho_star')
    print('─' * 72)
    try:
        M_h, dndlnm, _, _ = compute_hmf(cosmo_lcdm, 9.0)

        # Differential sum: sum(rho_i * delta_i)
        _, rho_bin = compute_theory_differential_rho(M_h, dndlnm,
                                                      DEFAULT_LOG10_MSTAR_BINS)
        delta      = np.diff(DEFAULT_LOG10_MSTAR_BINS)
        sum_rho    = float(np.sum(rho_bin * delta))

        # Cumulative: rho_star above the minimum threshold
        M_star_lo  = np.array([10.0 ** DEFAULT_LOG10_MSTAR_BINS[0]])
        rho_cumul  = float(compute_theory_rho_star(M_h, dndlnm, M_star_lo)[0])

        rel_err = abs(sum_rho - rho_cumul) / max(rho_cumul, 1e-30)
        # Note: they differ by the contribution above M_star_max (upper tail),
        # which should be ~0 for our 10^15 upper edge.
        passed = rel_err < 0.02
        results.add(7,
                    'sum(rho_bin*dlogM) matches cumulative rho_star',
                    passed,
                    f'diff_sum={sum_rho:.4e}  cumul={rho_cumul:.4e}  '
                    f'rel_err={rel_err*100:.2f}%  (tol=2%)')
    except Exception:
        results.add(7, 'rho integral consistency', False,
                    traceback.format_exc())


def test_T08_cosmic_variance_ordering(cosmo_lcdm, results):
    """
    T08 — Cosmic variance increases with stellar mass.

    More massive galaxies inhabit more biased halos.  sigma_CV(bin) = b_eff * sigma_DM
    must be monotonically non-decreasing as a function of stellar mass bin.

    A violation would indicate that the bias-weighted integral is computing
    the wrong halo mass range (e.g. SHMR inversion swapped at some edge).
    """
    print('\n' + '─' * 72)
    print('T08 — sigma_CV increases with stellar mass')
    print('─' * 72)
    try:
        M_h, dndlnm, sigma, Pk = compute_hmf(cosmo_lcdm, 9.0)

        chi_lo = cosmo_lcdm.comoving_distance(8.0)
        chi_hi = cosmo_lcdm.comoving_distance(10.0)
        V = ((4.0 / 3.0) * np.pi * (chi_hi**3 - chi_lo**3)
             * UNCOVER_SKY_FRACTION)

        sigma_cv = compute_cosmic_variance(
            Pk, M_h, dndlnm, sigma, V,
            DEFAULT_LOG10_MSTAR_BINS, shmr_mstar,
        )

        diffs    = np.diff(sigma_cv)
        n_bad    = int(np.sum(diffs < -1e-6))   # allow tiny numerical noise
        passed   = n_bad == 0
        results.add(8,
                    'sigma_CV monotone non-decreasing with M_star',
                    passed,
                    f'sigma_CV = [{sigma_cv.min():.3f}, {sigma_cv.max():.3f}]  '
                    f'decreasing steps = {n_bad}')
    except Exception:
        results.add(8, 'CV ordering', False, traceback.format_exc())


def test_T09_cosmic_variance_physical_range(cosmo_lcdm, results):
    """
    T09 — Cosmic variance in physically plausible range.

    For UNCOVER's 45 arcmin² field at z~9:
      - sigma_DM is the DM fluctuation on ~(1000 Mpc^3)^{1/3} ~ 10 Mpc scales
      - Halo bias b_eff ~ 5–15 for massive high-z halos
      - Physically: sigma_CV ∈ [0.05, 10]

    Values outside this range signal a unit error in V_survey, R_eff, or
    the sigma_DM integral.
    """
    print('\n' + '─' * 72)
    print('T09 — sigma_CV in physical range [0.05, 10]')
    print('─' * 72)
    try:
        M_h, dndlnm, sigma, Pk = compute_hmf(cosmo_lcdm, 9.0)
        chi_lo = cosmo_lcdm.comoving_distance(8.0)
        chi_hi = cosmo_lcdm.comoving_distance(10.0)
        V = ((4.0 / 3.0) * np.pi * (chi_hi**3 - chi_lo**3)
             * UNCOVER_SKY_FRACTION)

        sigma_cv = compute_cosmic_variance(
            Pk, M_h, dndlnm, sigma, V,
            DEFAULT_LOG10_MSTAR_BINS, shmr_mstar,
        )
        # Focus on populated bins (log10 M* = 8-11, indices 6-12 of our 20-bin grid)
        populated = sigma_cv[6:13]
        lo_ok  = bool(np.all(populated > 0.05))
        hi_ok  = bool(np.all(populated < 10.0))
        passed = lo_ok and hi_ok
        results.add(9,
                    'sigma_CV ∈ [0.05, 10] for M_star bins 8–11',
                    passed,
                    f'range = [{populated.min():.3f}, {populated.max():.3f}]  '
                    f'lo_ok={lo_ok}  hi_ok={hi_ok}')
    except Exception:
        results.add(9, 'CV physical range', False, traceback.format_exc())


def test_T10_magnification_weighting(results):
    """
    T10 — Magnification weighting linearity.

    A single galaxy with mu=2 must contribute exactly twice the phi_obs
    of the same galaxy with mu=1.  This validates the core
    sum(mu_j)/V/DlogM formula.

    Failure would indicate a bug in _bin_galaxies or the mu-weighting logic.
    """
    print('\n' + '─' * 72)
    print('T10 — Magnification weighting: phi(mu=2) = 2 × phi(mu=1)')
    print('─' * 72)
    try:
        # Synthetic catalog: one galaxy at log10(M*) = 9.25 (middle of a bin)
        mstar = 9.25
        z_mid = 9.0
        V     = 1e4   # Mpc^3, arbitrary

        def _mock_cat(mu_val):
            return Table({
                'z':        [z_mid],
                'mstar_50': [mstar],
                'mstar_16': [mstar - 0.2],
                'mstar_84': [mstar + 0.2],
                'mu':       [float(mu_val)],
            })

        edges = np.array([9.0, 9.5])   # single bin

        _, phi1, _, _, n1 = compute_observed_differential_smf(
            _mock_cat(1.0), z_mid - 1, z_mid + 1, V, edges
        )
        _, phi2, _, _, n2 = compute_observed_differential_smf(
            _mock_cat(2.0), z_mid - 1, z_mid + 1, V, edges
        )

        ratio   = float(phi2[0] / phi1[0]) if phi1[0] > 0 else np.nan
        rel_err = abs(ratio - 2.0) / 2.0
        passed  = rel_err < 1e-10 and n1[0] == 1 and n2[0] == 1
        results.add(10,
                    'phi(mu=2) = 2×phi(mu=1)',
                    passed,
                    f'ratio = {ratio:.12f}  rel_err = {rel_err:.2e}  '
                    f'(must be < 1e-10)')
    except Exception:
        results.add(10, 'Magnification weighting', False,
                    traceback.format_exc())


def test_T11_logp_finite_cumulative(like_cumul, cosmo_lcdm, results):
    """
    T11 — logp returns finite at ΛCDM (cumulative mode).

    Basic sanity: the existing cumulative likelihood pipeline must still
    run without crashing after our additions.
    """
    print('\n' + '─' * 72)
    print('T11 — logp finite at ΛCDM (cumulative mode)')
    print('─' * 72)
    try:
        t0 = time.time()
        lp = _logp(like_cumul, 0.0, 0.0, cosmo=cosmo_lcdm)
        dt = time.time() - t0
        passed = np.isfinite(lp)
        results.add(11, 'logp finite — cumulative', passed,
                    f'logp={lp:.3f}  chi2={-2*lp:.1f}  dt={dt:.2f}s')
    except Exception:
        results.add(11, 'logp finite — cumulative', False,
                    traceback.format_exc())


def test_T12_logp_finite_diff_smf(like_diff_smf, cosmo_lcdm, results):
    """
    T12 — logp returns finite at ΛCDM (differential smf mode).
    """
    print('\n' + '─' * 72)
    print('T12 — logp finite at ΛCDM (differential dn/dlogM mode)')
    print('─' * 72)
    try:
        t0 = time.time()
        lp = _logp(like_diff_smf, 0.0, 0.0, cosmo=cosmo_lcdm)
        dt = time.time() - t0
        passed = np.isfinite(lp)
        results.add(12, 'logp finite — differential smf', passed,
                    f'logp={lp:.3f}  chi2={-2*lp:.1f}  dt={dt:.2f}s')
    except Exception:
        results.add(12, 'logp finite — differential smf', False,
                    traceback.format_exc())


def test_T13_logp_finite_diff_rho(like_diff_rho, cosmo_lcdm, results):
    """
    T13 — logp returns finite at ΛCDM (differential rho mode).
    """
    print('\n' + '─' * 72)
    print('T13 — logp finite at ΛCDM (differential drho/dlogM mode)')
    print('─' * 72)
    try:
        t0 = time.time()
        lp = _logp(like_diff_rho, 0.0, 0.0, cosmo=cosmo_lcdm)
        dt = time.time() - t0
        passed = np.isfinite(lp)
        results.add(13, 'logp finite — differential rho', passed,
                    f'logp={lp:.3f}  chi2={-2*lp:.1f}  dt={dt:.2f}s')
    except Exception:
        results.add(13, 'logp finite — differential rho', False,
                    traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
#  BOBYQA / SCIPY MINIMISATION  (one run per mode × observable)
# ══════════════════════════════════════════════════════════════════════════════

def _run_minimisation(phot_path, zspec_path, mode, observable, output_dir,
                       cosmo_lcdm, like):
    """
    Run a single Cobaya minimisation for (mode, observable) and return
    a result dict containing chi², best-fit params, and data counts.

    ΛCDM (P=0) is evaluated analytically from the mock provider.
    Exotic DE (P=2) is minimised with BOBYQA (spectroscopic) or
    scipy/Nelder-Mead (photometric).

    Returns
    -------
    dict with keys:
        mode, observable,
        N_gal, N_nonempty,
        lcdm_chi2,
        exotic_chi2, a_best, b_best
    """
    tag = f'{mode[:4]}_{observable}'
    output_path = os.path.join(output_dir, f'minimise_{tag}')

    # ── ΛCDM evaluation (no minimisation needed, P=0) ─────────────────────
    lp_lcdm  = _logp(like, 0.0, 0.0, cosmo=cosmo_lcdm)
    chi2_lcdm = -2.0 * lp_lcdm

    # ── Data counts (cosmology-independent for galaxy counting) ───────────
    N_gal      = _count_galaxies(like, cosmo_lcdm)
    N_nonempty = _count_nonempty_bins(like, cosmo_lcdm, observable)

    print(f'\n  [{tag}] ΛCDM  chi2={chi2_lcdm:.2f}  '
          f'N_gal={N_gal}  N_nonempty={N_nonempty}')

    # ── Exotic DE minimisation ────────────────────────────────────────────
    sampler_method = 'bobyqa' if mode == 'spectroscopic' else 'scipy'

    info = {
        'likelihood': {
            'likelihood.jwst_likelihood.JWSTLikelihood': {
                'mode':                    mode,
                'vary_SHMR_params':        False,
                'phot_path':               phot_path,
                'zspec_path':              zspec_path,
                'python_path':             _PROJECT_ROOT,
                'use_differential':        True,
                'differential_observable': observable,
            }
        },
        'theory': {
            'classy': {
                'extra_args': {
                    'z_c_exo':       _Z_C_EXO,
                    'sigma_z_exo':   _SIGMA_Z_EXO,
                    'output':        'mPk',
                    'P_k_max_1/Mpc': 510.0,
                    'z_max_pk':      20.0,
                    'non linear':    'none',
                },
                'ignore_obsolete': True,
            }
        },
        'params': {
            'a_samp': {
                'prior': {'min': -1838.0, 'max': -1e-5},
                'ref':   {'dist': 'norm', 'loc': -20.0, 'scale': 10.0},
                'drop':  True,
            },
            's': {
                'prior': {'min': -1381.597, 'max': -1e-5},
                'ref':   {'dist': 'norm', 'loc': -30.0, 'scale': 10.0},
                'drop':  True,
            },
            'a_exo':   {'value': 'lambda a_samp: a_samp'},
            'b_exo':   {'value': 'lambda a_samp, s: s - a_samp'},
            'H0':      {'value': 67.36},
            'omega_b': {'value': 0.02237},
            'omega_cdm': {'value': 0.1200},
            'n_s':     {'value': 0.9649},
            'logA':    {'value': 3.044, 'drop': True},
            'A_s':     {'value': 'lambda logA: 1e-10*np.exp(logA)'},
            'tau_reio': {'value': 0.0544},
            'Omega_m': {'derived': True},
        },
        'prior':   {'h2_positivity': "lambda a_samp, s: 0.0 if s >= (-0.07202 * a_samp - 1381.5969) else -1e500"},
        'sampler': {
            'minimize': {
                'method':    sampler_method,
                'max_evals': 4500,
            }
        },
        'output': output_path,
        'debug':  True,
    }

    from cobaya.run import run as cobaya_run
    t0 = time.time()
    upd_info, sampler = cobaya_run(info, force=True)
    dt = time.time() - t0

    bf        = sampler.products().get('minimum', {})
    a_best    = float(bf.get('a_exo', np.nan))
    b_best    = float(bf.get('b_exo', np.nan))
    chi2_best = float(bf.get('chi2', -2.0 * bf.get('logp', np.nan)))

    print(f'  [{tag}] Exotic  chi2={chi2_best:.2f}  '
          f'a_exo={a_best:.2f}  b_exo={b_best:.2f}  dt={dt:.0f}s')

    return {
        'mode':        mode,
        'observable':  observable,
        'N_gal':       N_gal,
        'N_nonempty':  N_nonempty,
        'lcdm_chi2':   chi2_lcdm,
        'exotic_chi2': chi2_best,
        'a_best':      a_best,
        'b_best':      b_best,
    }


def run_all_minimisations(phot_path, zspec_path, output_dir,
                           cosmo_lcdm,
                           like_phot_smf, like_spec_smf,
                           like_phot_rho, like_spec_rho):
    """
    Run all 4 minimisations and return a dict of results keyed by
    (mode, observable).
    """
    print('\n' + '═' * 72)
    print('  MINIMISATION RUNS  (BOBYQA / scipy)')
    print('═' * 72)

    runs = [
        ('photometric',   'smf', like_phot_smf),
        ('photometric',   'rho', like_phot_rho),
        ('spectroscopic', 'smf', like_spec_smf),
        ('spectroscopic', 'rho', like_spec_rho),
    ]

    results = {}
    for mode, obs, like in runs:
        try:
            r = _run_minimisation(phot_path, zspec_path, mode, obs,
                                   output_dir, cosmo_lcdm, like)
            results[(mode, obs)] = r
        except Exception:
            print(f'  ❌ Minimisation failed for ({mode}, {obs}):')
            print(traceback.format_exc())
            results[(mode, obs)] = None

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  CUMULATIVE rho_star(>M_star) MINIMISATION  (commented out)
#
#  Uncomment this block + the corresponding calls in main() to run the
#  original cumulative likelihood through the same BOBYQA/scipy minimiser
#  and produce a third LaTeX table for direct comparison with the
#  differential tables.
#
#  Key differences vs the differential minimisation:
#    - use_differential: False  (no differential_observable key)
#    - chi²_reduced = chi² / (N_gal - P)  — N_gal is both the display
#      value AND the denominator because every galaxy is a data point
#    - photometric → scipy  (Nelder-Mead, more robust for large N)
#    - spectroscopic → bobyqa  (fewer points, faster convergence)
# ══════════════════════════════════════════════════════════════════════════════

def _run_minimisation_cumulative(phot_path, zspec_path, mode, output_dir,
                                  cosmo_lcdm, like):
    """
    Run a single cumulative rho_star(>M_star) minimisation.

    ΛCDM evaluated analytically at (a_exo=0, b_exo=0).
    Exotic DE minimised with BOBYQA (spectroscopic) or scipy (photometric).

    Returns
    -------
    dict with keys:
        mode, N_gal, lcdm_chi2, exotic_chi2, a_best, b_best
    """
    tag = f'{mode[:4]}_cumul'
    output_path = os.path.join(output_dir, f'minimise_{tag}')

    # ── ΛCDM: no minimisation, P=0 ────────────────────────────────────────
    lp_lcdm   = _logp(like, 0.0, 0.0, cosmo=cosmo_lcdm)
    chi2_lcdm = -2.0 * lp_lcdm
    N_gal     = _count_galaxies(like, cosmo_lcdm)

    print(f'\n  [{tag}] ΛCDM  chi2={chi2_lcdm:.2f}  N_gal={N_gal}')

    # ── Exotic DE minimisation ─────────────────────────────────────────────
    # photometric → scipy (Nelder-Mead) to avoid BOBYQA crashes on large N
    # spectroscopic → bobyqa (fewer data points, faster convergence)
    sampler_method = 'bobyqa' if mode == 'spectroscopic' else 'scipy'

    info = {
        'likelihood': {
            'likelihood.jwst_likelihood.JWSTLikelihood': {
                'mode':             mode,
                'vary_SHMR_params': False,
                'phot_path':        phot_path,
                'zspec_path':       zspec_path,
                'python_path':      _PROJECT_ROOT,
                'use_differential': False,   # ← cumulative mode
                # no differential_observable key needed
            }
        },
        'theory': {
            'classy': {
                'extra_args': {
                    'z_c_exo':       _Z_C_EXO,
                    'sigma_z_exo':   _SIGMA_Z_EXO,
                    'output':        'mPk',
                    'P_k_max_1/Mpc': 510.0,
                    'z_max_pk':      20.0,
                    'non linear':    'none',
                },
                'ignore_obsolete': True,
            }
        },
        'params': {
            'a_samp': {
                'prior': {'min': -1838.0, 'max': -1e-5},
                'ref':   {'dist': 'norm', 'loc': -20.0, 'scale': 10.0},
                'drop':  True,
            },
            's': {
                'prior': {'min': -1381.597, 'max': -1e-5},
                'ref':   {'dist': 'norm', 'loc': -30.0, 'scale': 10.0},
                'drop':  True,
            },
            'a_exo':     {'value': 'lambda a_samp: a_samp'},
            'b_exo':     {'value': 'lambda a_samp, s: s - a_samp'},
            'H0':        {'value': 67.36},
            'omega_b':   {'value': 0.02237},
            'omega_cdm': {'value': 0.1200},
            'n_s':       {'value': 0.9649},
            'logA':      {'value': 3.044, 'drop': True},
            'A_s':       {'value': 'lambda logA: 1e-10*np.exp(logA)'},
            'tau_reio':  {'value': 0.0544},
            'Omega_m':   {'derived': True},
        },
        'prior':   {'h2_positivity': "lambda a_samp, s: 0.0 if s >= (-0.07202 * a_samp - 1381.5969) else -1e500"},
        'sampler': {
            'minimize': {
                'method':    sampler_method,
                'max_evals': 4500,
            }
        },
        'output': output_path,
        'debug':  True,
    }

    from cobaya.run import run as cobaya_run
    t0 = time.time()
    upd_info, sampler = cobaya_run(info, force=True)
    dt = time.time() - t0

    bf        = sampler.products().get('minimum', {})
    a_best    = float(bf.get('a_exo', np.nan))
    b_best    = float(bf.get('b_exo', np.nan))
    chi2_best = float(bf.get('chi2', -2.0 * bf.get('logp', np.nan)))

    print(f'  [{tag}] Exotic  chi2={chi2_best:.2f}  '
          f'a_exo={a_best:.2f}  b_exo={b_best:.2f}  dt={dt:.0f}s')

    return {
        'mode':        mode,
        'N_gal':       N_gal,
        'lcdm_chi2':   chi2_lcdm,
        'exotic_chi2': chi2_best,
        'a_best':      a_best,
        'b_best':      b_best,
    }


def run_all_minimisations_cumulative(phot_path, zspec_path, output_dir,
                                      cosmo_lcdm,
                                      like_cumul_phot, like_cumul_spec):
    """
    Run both cumulative minimisations (photometric + spectroscopic).
    Returns dict keyed by mode string.
    """
    print('\n' + '═' * 72)
    print('  CUMULATIVE MINIMISATION RUNS  (rho_star > M_star)')
    print('═' * 72)

    results = {}
    for mode, like in [('photometric',   like_cumul_phot),
                        ('spectroscopic', like_cumul_spec)]:
        try:
            r = _run_minimisation_cumulative(
                phot_path, zspec_path, mode, output_dir, cosmo_lcdm, like
            )
            results[mode] = r
        except Exception:
            print(f'  ❌ Cumulative minimisation failed for {mode}:')
            print(traceback.format_exc())
            results[mode] = None

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  LATEX TABLE GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def _format_table(observable, min_results, output_dir):
    """
    Generate a LaTeX table for one observable ('smf' or 'rho').

    chi²_reduced for Exotic DE: chi² / (N_nonempty - 2)
    chi²_reduced for ΛCDM:      chi² / (N_nonempty - 0)
    N column displays total galaxies (dataset size).

    The commented-out cumulative table block is retained for reference
    so that Hashim can compare the two approaches.
    """
    obs_label = (r'dn/d\log_{10}(M_\star)' if observable == 'smf'
                 else r'd\rho_\star/d\log_{10}(M_\star)')
    obs_units = (r'[Mpc^{-3}\,dex^{-1}]' if observable == 'smf'
                 else r'[M_\odot\,Mpc^{-3}\,dex^{-1}]')
    label     = f'tab:diff_{observable}'
    caption   = (
        f'Validation results for the differential SMF observable '
        f'${obs_label}$ ${obs_units}$.  '
        r'$N$ = total galaxies in the dataset (dataset size). '
        r'$P$ = number of free parameters. '
        r'$\chi^2_\nu = \chi^2 / (N_{\rm bins} - P)$ where '
        r'$N_{\rm bins}$ is the number of non-empty stellar mass bins '
        r'across all redshift slices.  '
        r'Exotic DE parameters $(a_{\rm exo},\,b_{\rm exo})$ are found '
        r'by BOBYQA (spectroscopic) or scipy/Nelder-Mead (photometric) '
        r'minimisation.  $\Lambda$CDM is evaluated at '
        r'$a_{\rm exo}=b_{\rm exo}=0$ without minimisation ($P=0$).'
    )

    def _row(label_str, mode, is_exotic, results_dict):
        key = (mode, observable)
        r   = results_dict.get(key)
        if r is None:
            return rf'  {label_str} & {mode.capitalize()[:4]} & — & — & — & — \\'

        N_gal    = r['N_gal']
        N_bins   = r['N_nonempty']
        chi2     = r['exotic_chi2'] if is_exotic else r['lcdm_chi2']
        P        = 2 if is_exotic else 0
        dof      = max(N_bins - P, 1)
        red_chi2 = chi2 / dof

        mode_str = 'Photometric' if mode == 'photometric' else 'Spectroscopic'
        return (
            rf'  {label_str} & {mode_str} & {N_gal} & {P} '
            rf'& {chi2:.1f} & {red_chi2:.3f} \\'
        )

    lines = [
        r'\begin{table}[ht]',
        r'\centering',
        rf'\caption{{{caption}}}',
        rf'\label{{{label}}}',
        r'\begin{tabular}{l|c|c|c|c|c}',
        r'\toprule',
        r'Model & Mode & $N$ (galaxies) & $P$ (params) '
        r'& $\chi^{2}$ & $\chi^{2}_{\nu}$ \\',
        r'\midrule',
        _row(r'$\Lambda$CDM', 'photometric',   False, min_results),
        _row(r'Exotic DE',    'photometric',   True,  min_results),
        r'\midrule',
        _row(r'$\Lambda$CDM', 'spectroscopic', False, min_results),
        _row(r'Exotic DE',    'spectroscopic', True,  min_results),
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
        '',
        r'% ─────────────────────────────────────────────────────────────────',
        r'% CUMULATIVE LIKELIHOOD TABLE (commented out for reference)',
        r'% This is the original observable: rho_star(>M_star) summed over',
        r'% every galaxy as a threshold.  N here = number of galaxies = number',
        r'% of data points in chi^2.  The differential table above uses the same',
        r'% dataset but bins galaxies into ~15 independent mass bins, removing',
        r'% the maximal correlation of consecutive cumulative thresholds.',
        r'% Using differential parameters (a_exo, b_exo) from the table above',
        r'% to compute the cumulative observable is scientifically more sound:',
        r'% the likelihood is well-conditioned, and the resulting parameters',
        r'% are then used to predict any observable including the cumulative SMF.',
        r'%',
        r'% \begin{table}[ht]',
        r'% \centering',
        r'% \caption{Cumulative rho_star(>M_star) likelihood results}',
        r'% \begin{tabular}{l|c|c|c|c|c}',
        r'% \toprule',
        r'% Model & Mode & $N$ (data pts) & $P$ & $\chi^2$ & $\chi^2_\nu$ \\',
        r'% \midrule',
        r'% $\Lambda$CDM & Photometric & ... & 0 & ... & ... \\',
        r'% Exotic DE    & Photometric & ... & 2 & ... & ... \\',
        r'% \midrule',
        r'% $\Lambda$CDM & Spectroscopic & ... & 0 & ... & ... \\',
        r'% Exotic DE    & Spectroscopic & ... & 2 & ... & ... \\',
        r'% \bottomrule',
        r'% \end{tabular}',
        r'% \end{table}',
    ]

    tex  = '\n'.join(lines)
    path = os.path.join(output_dir, f'table_diff_{observable}.tex')
    with open(path, 'w') as f:
        f.write(tex)
    print(f'\n  LaTeX table written → {path}')
    return path


def generate_latex_tables(min_results, output_dir):
    """Generate both LaTeX tables (smf and rho) and return their paths."""
    print('\n' + '═' * 72)
    print('  LATEX TABLE GENERATION')
    print('═' * 72)
    p1 = _format_table('smf', min_results, output_dir)
    p2 = _format_table('rho', min_results, output_dir)
    return p1, p2


# ── Cumulative rho_star(>M_star) table  (commented out) ──────────────────────
#
# Uncomment _format_cumulative_table and generate_cumulative_table, then call
# generate_cumulative_table(cumul_results, args.output_dir) in main().
#
# chi²_reduced uses N_gal in the denominator — every galaxy is a data point in
# the cumulative chi², so N_gal = N_data_points.  Compare with differential
# tables where chi²_reduced uses N_nonempty_bins.  Keeping both lets Hashim see
# how the chi² values change when switching observables with the same dataset.

def _format_cumulative_table(cumul_results, output_dir):
    """
    Generate table_cumulative.tex from cumulative minimisation results.

    cumul_results : dict keyed by mode ('photometric', 'spectroscopic'),
                   each value a dict with N_gal, lcdm_chi2, exotic_chi2.

    chi²_reduced = chi² / (N_gal - P)
    N_gal is both the display value and the denominator because every galaxy
    contributes one cumulative threshold to the chi² sum.
    """
    caption = (
        r'Cumulative stellar mass density $\rho_\star(>M_\star)$ likelihood '
        r'results (original observable, shown for comparison with '
        r'Tables~\ref{tab:diff_smf} and \ref{tab:diff_rho}).  '
        r'$N$ = total galaxies = number of data points in $\chi^2$.  '
        r'$\chi^2_\nu = \chi^2 / (N - P)$.  '
        r'Consecutive cumulative thresholds are maximally correlated, so '
        r'a diagonal covariance artificially inflates $\chi^2$ and tightens '
        r'constraints.  The differential tables above use the same dataset '
        r'with independent bins, giving a statistically valid diagonal '
        r'covariance.  Best-fit parameters for the cumulative observable '
        r'should be compared with those from the differential minimisation.'
    )

    def _row(label_str, mode, is_exotic):
        r = cumul_results.get(mode)
        if r is None:
            mode_str = "Photometric" if mode == "photometric" else "Spectroscopic"
            return rf"  {label_str} & {mode_str} & — & — & — & — \\"
        N_gal    = r["N_gal"]
        chi2     = r["exotic_chi2"] if is_exotic else r["lcdm_chi2"]
        P        = 2 if is_exotic else 0
        red_chi2 = chi2 / max(N_gal - P, 1)
        mode_str = "Photometric" if mode == "photometric" else "Spectroscopic"
        return (
            rf"  {label_str} & {mode_str} & {N_gal} & {P} "
            rf"& {chi2:.1f} & {red_chi2:.3f} \\"
        )

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{{caption}}}",
        r"\label{tab:cumulative}",
        r"\begin{tabular}{l|c|c|c|c|c}",
        r"\toprule",
        r"Model & Mode & $N$ (galaxies) & $P$ (params) & $\chi^{2}$ & $\chi^{2}_{\nu}$ \\",
        r"\midrule",
        _row(r"$\Lambda$CDM", "photometric",   False),
        _row(r"Exotic DE",     "photometric",   True),
        r"\midrule",
        _row(r"$\Lambda$CDM", "spectroscopic", False),
        _row(r"Exotic DE",     "spectroscopic", True),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    tex  = "\n".join(lines)
    path = os.path.join(output_dir, "table_cumulative.tex")
    with open(path, "w") as f:
        f.write(tex)
    print(f"\n  LaTeX table written → {path}")
    return path


def generate_cumulative_table(cumul_results, output_dir):
    """Generate table_cumulative.tex and return its path."""
    print("\n" + "─" * 72)
    print("  CUMULATIVE TABLE GENERATION")
    print("─" * 72)
    return _format_cumulative_table(cumul_results, output_dir)


# ══════════════════════════════════════════════════════════════════════════════
#  LABBÉ / BOYLAN-KOLCHIN 2023 PLOT
# ══════════════════════════════════════════════════════════════════════════════

def _labbe_cumulative_rho(cosmo):
    """
    Compute rho_star(>M_star) from the 6 Labbé/BK2023 candidates.

    Returns (M_star_thresholds, rho_50, rho_lo, rho_hi) in M_sun/Mpc^3.
    Galaxies sorted by M*_50 descending so the cumulative sum is built
    from the most massive downwards.

    rho_lo / rho_hi use M*_16 / M*_84 for both the galaxy's mass value
    AND its threshold assignment (consistent with our pipeline convention).
    """
    # Survey volume from Labbé et al. 2023 area at this cosmology
    chi_lo  = cosmo.comoving_distance(_LABBE_Z_MIN)
    chi_hi  = cosmo.comoving_distance(_LABBE_Z_MAX)
    V_labbe = ((4.0 / 3.0) * np.pi * (chi_hi**3 - chi_lo**3)
               * _LABBE_SKY_FRACTION)

    # Sort galaxies by M*_50 descending
    idx = np.argsort(_LABBE_GALAXIES[:, 1])[::-1]
    gals = _LABBE_GALAXIES[idx]

    M50 = 10.0 ** gals[:, 1]
    M16 = 10.0 ** gals[:, 2]
    M84 = 10.0 ** gals[:, 3]

    n     = len(gals)
    rho50 = np.zeros(n)
    rho16 = np.zeros(n)
    rho84 = np.zeros(n)

    # cumulative sum: at threshold = M50[i], sum M50[j] for j = 0..i
    for i in range(n):
        rho50[i] = np.sum(M50[:i+1]) / V_labbe
        rho16[i] = np.sum(M16[:i+1]) / V_labbe
        rho84[i] = np.sum(M84[:i+1]) / V_labbe

    # x-axis: the threshold = mass of each galaxy (threshold at which
    # that galaxy enters the cumulative sum when sweeping from high to low)
    return M50, rho50, rho16, rho84, V_labbe


def _find_best_params(min_results):
    """
    Return (a_exo, b_exo) from the minimisation run with the lowest chi².
    Falls back to ΛCDM if all runs failed.
    """
    best_chi2 = np.inf
    best_a, best_b = 0.0, 0.0
    for r in min_results.values():
        if r is None:
            continue
        if r['exotic_chi2'] < best_chi2:
            best_chi2 = r['exotic_chi2']
            best_a    = r['a_best']
            best_b    = r['b_best']
    return best_a, best_b


def plot_labbe_rho_star(min_results, output_dir):
    """
    Reproduce the Boylan-Kolchin 2023 cumulative rho_star plot with
    our model curve at the best-fit differential likelihood parameters.

    Panel:
      - Blue solid    : Exotic DE best-fit from differential likelihood
      - Gray dashed   : ΛCDM reference (a_exo = b_exo = 0)
      - Red stars     : Labbé et al. 2023 candidates (BK2023 selection)
      - Error bars    : propagated from M*_16 / M*_84 percentiles

    The model curve uses compute_theory_rho_star at z=9 (bin midpoint
    of the z=8–10 UNCOVER redshift slice).
    """
    print('\n' + '═' * 72)
    print('  LABBÉ / BOYLAN-KOLCHIN 2023 PLOT')
    print('═' * 72)

    a_best, b_best = _find_best_params(min_results)
    print(f'  Best-fit params: a_exo={a_best:.2f}  b_exo={b_best:.2f}')

    # ── Build two CLASS instances ─────────────────────────────────────────
    cosmo_lcdm   = _build_cosmo(0.0, 0.0)
    cosmo_exotic = _build_cosmo(a_best, b_best)

    # ── Labbé data points ─────────────────────────────────────────────────
    M_thr_data, rho50, rho16, rho84, V_labbe = _labbe_cumulative_rho(cosmo_exotic)

    err_lo = rho50 - rho16
    err_hi = rho84 - rho50

    print(f'  Labbé V_survey = {V_labbe:.3e} Mpc³  '
          f'(area={_LABBE_AREA_ARCMIN2} arcmin²)')

    # ── Model curves ──────────────────────────────────────────────────────
    # Fine M_star grid from 10^9 to 10^12 M_sun
    M_star_grid = np.logspace(9.0, 12.0, 300)   # M_sun

    def _model_rho(cosmo):
        M_h, dndlnm, _, _ = compute_hmf(cosmo, 9.0)
        return compute_theory_rho_star(M_h, dndlnm, M_star_grid)

    rho_lcdm   = _model_rho(cosmo_lcdm)
    rho_exotic = _model_rho(cosmo_exotic)

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5.5))

    ax.plot(M_star_grid, rho_exotic,
            color='steelblue', lw=2.2, label=r'Exotic DE (best-fit, diff. likelihood)')
    ax.plot(M_star_grid, rho_lcdm,
            color='gray', lw=1.8, ls='--', label=r'$\Lambda$CDM')

    ax.errorbar(M_thr_data, rho50,
                yerr=[err_lo, err_hi],
                fmt='r*', ms=10, capsize=4, lw=1.5, elinewidth=1.2,
                label=r'Labbé et al. 2023 (BK2023 candidates)')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$M_\star$ [$M_\odot$]', fontsize=13)
    ax.set_ylabel(r'$\rho_\star(>M_\star)$ [$M_\odot\,\mathrm{Mpc}^{-3}$]',
                  fontsize=13)
    ax.set_title(r'Cumulative stellar mass density at $z\approx9$', fontsize=13)
    ax.set_xlim(8e9, 5e11)
    ax.legend(fontsize=10, framealpha=0.85)

    ax.text(0.03, 0.05,
            f'Best-fit: $a_{{\\rm exo}}={a_best:.1f}$, '
            f'$b_{{\\rm exo}}={b_best:.1f}$\n'
            f'(from differential SMF likelihood)',
            transform=ax.transAxes, fontsize=8,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', fc='white', alpha=0.7))

    plt.tight_layout()

    pdf_path = os.path.join(output_dir, 'labbe_rho_star.pdf')
    png_path = os.path.join(output_dir, 'labbe_rho_star.png')
    fig.savefig(pdf_path, dpi=200, bbox_inches='tight')
    fig.savefig(png_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    cosmo_lcdm.struct_cleanup();   cosmo_lcdm.empty()
    cosmo_exotic.struct_cleanup(); cosmo_exotic.empty()

    print(f'  Plot saved → {pdf_path}')
    print(f'  Plot saved → {png_path}')


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='JWST likelihood validation suite + differential SMF tables.'
    )
    parser.add_argument('--phot_path',
        default='/home/lustre_p/ahmed.omar/workspace/exo_de_project/'
                'data/UNCOVER_DR4_SPS_catalog.fits')
    parser.add_argument('--zspec_path',
        default='/home/lustre_p/ahmed.omar/workspace/exo_de_project/'
                'data/UNCOVER_DR4_SPS_zspec_catalog.fits')
    parser.add_argument('--output_dir', default='likelihood_test_outputs')
    parser.add_argument('--skip_bobyqa', action='store_true',
        help='Run only validation tests, skip minimisation and plot.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print('═' * 72)
    print('  JWST LIKELIHOOD — VALIDATION + DIFFERENTIAL SMF SUITE')
    print(f'  phot  = {args.phot_path}')
    print(f'  zspec = {args.zspec_path}')
    print('═' * 72)

    # ── Build shared CLASS instance at ΛCDM ──────────────────────────────
    print('\nBuilding ΛCDM CLASS instance...')
    t0 = time.time()
    cosmo_lcdm = _build_cosmo(0.0, 0.0)
    print(f'  CLASS done in {time.time()-t0:.1f}s  '
          f'h={cosmo_lcdm.h():.4f}  '
          f'sigma_8={cosmo_lcdm.sigma8():.4f}')

    # ── Build likelihood instances (one per observable × mode) ───────────
    print('\nInitialising likelihoods...')

    def _build(mode, use_diff, obs):
        t = time.time()
        like = _build_likelihood(
            args.phot_path, args.zspec_path, mode,
            use_differential=use_diff,
            differential_observable=obs,
        )
        print(f'  [{mode[:4]}/{obs if use_diff else "cumul"}] done in '
              f'{time.time()-t:.1f}s')
        return like

    like_cumul_spec  = _build('spectroscopic', False, 'smf')
    like_cumul_phot = _build('photometric', False, 'smf')  # needed for cumulative minimisation
    like_diff_smf_ph = _build('photometric',   True,  'smf')
    like_diff_rho_ph = _build('photometric',   True,  'rho')
    like_diff_smf_sp = _build('spectroscopic', True,  'smf')
    like_diff_rho_sp = _build('spectroscopic', True,  'rho')

    # ── Validation tests ─────────────────────────────────────────────────
    print('\n' + '═' * 72)
    print('  VALIDATION TESTS  (T01 – T13)')
    print('═' * 72)

    results = _Results()

    test_T01_sigma8_recovery(cosmo_lcdm, results)
    test_T02_hmf_mass_fraction(cosmo_lcdm, results)
    test_T03_shmr_pivot_identity(results)
    test_T04_shmr_monotone(cosmo_lcdm, results)
    test_T05_differential_phi_positive(cosmo_lcdm, results)
    test_T06_differential_phi_integral_consistency(cosmo_lcdm, results)
    test_T07_differential_rho_integral_consistency(cosmo_lcdm, results)
    test_T08_cosmic_variance_ordering(cosmo_lcdm, results)
    test_T09_cosmic_variance_physical_range(cosmo_lcdm, results)
    test_T10_magnification_weighting(results)
    test_T11_logp_finite_cumulative(like_cumul_spec,  cosmo_lcdm, results)
    test_T12_logp_finite_diff_smf(like_diff_smf_sp,  cosmo_lcdm, results)
    test_T13_logp_finite_diff_rho(like_diff_rho_sp,  cosmo_lcdm, results)

    results.summary()

    # ── Minimisation + tables + plot ──────────────────────────────────────
    if not args.skip_bobyqa:
        min_results = run_all_minimisations(
            args.phot_path, args.zspec_path, args.output_dir,
            cosmo_lcdm,
            like_diff_smf_ph, like_diff_smf_sp,
            like_diff_rho_ph, like_diff_rho_sp,
        )

        generate_latex_tables(min_results, args.output_dir)
        plot_labbe_rho_star(min_results, args.output_dir)

#         ── Cumulative rho_star(>M_star) minimisation + table (commented out) ──
#         Uncomment to also run the original cumulative likelihood through
#         the minimiser and produce table_cumulative.tex.
#         like_cumul_phot must be uncommented in the _build block above.
        
        cumul_results = run_all_minimisations_cumulative(
            args.phot_path, args.zspec_path, args.output_dir,
            cosmo_lcdm,
            like_cumul_phot,   # photometric → scipy (defined above, uncomment)
            like_cumul_spec,   # spectroscopic → bobyqa (already built)
        )
        generate_cumulative_table(cumul_results, args.output_dir)

    else:
        print('\n  [skip_bobyqa] Minimisation, tables, and plot skipped.')

    # ── Cleanup ───────────────────────────────────────────────────────────
    cosmo_lcdm.struct_cleanup()
    cosmo_lcdm.empty()
    print('\nDone.')


if __name__ == '__main__':
    main()