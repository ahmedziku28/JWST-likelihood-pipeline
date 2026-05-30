#!/usr/bin/env python3
"""
tests/test_uvlf.py

Validation suite and BOBYQA minimisation for the JWST UVLF exotic dark
energy pipeline (Donnan 2024 + Finkelstein 2024).

Validation tests  (U01 – U04)
──────────────────────────────
  U01  logp finite at ΛCDM — full likelihood (all 40 data points)
  U02  logp finite at ΛCDM — restricted likelihood (z≥10, 29 data points)
  U03  phi(M_UV) ≥ 0 everywhere on a dense M_UV grid at ΛCDM
  U04  SHMR strictly monotone on the full HMF mass grid

BOBYQA runs
───────────
  Run 1  Full likelihood, fixed SHMR          P=2   z_min_don=9.0   z_min_fink=8.9
  Run 2  Restricted likelihood, fixed SHMR    P=2   z_min_don=10.0  z_min_fink=10.9
  Run 3  Full likelihood, vary_beta           P=3   z_min_don=9.0   z_min_fink=8.9
  Run 4  Full likelihood, vary_SHMR           P=5   z_min_don=9.0   z_min_fink=8.9

  Each run:
    (a) Evaluates logp at ΛCDM analytically (no BOBYQA) → chi2_ΛCDM
    (b) Runs BOBYQA over (a_samp, s) [+ SHMR params where applicable]
        max_evals=2000
    → chi2_best, a_exo_best, b_exo_best, Ω_x0

Outputs
───────
  Console: per-test report + summary table
  table_uvlf_results.tex  — LaTeX table for Hashim's PDF
  table_uvlf_perbin.tex   — per-z-bin chi2 breakdown at best-fit (Run 1 & 2)

Usage
─────
  python tests/test_uvlf.py --output_dir test_outputs/uvlf/
  python tests/test_uvlf.py --output_dir test_outputs/uvlf/ --skip_bobyqa
"""

import argparse
import os
import sys
import time
import traceback
import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from classy import Class

from pipeline.hmf import compute_hmf
from pipeline.uvlf import (
    load_donnan,
    load_finkelstein,
    compute_uvlf_theory,
    chi_squared,
    DONNAN_Z_EDGES,
    FINKELSTEIN_Z_EDGES,
)
from pipeline.uvlf_conversion import (
    shmr_mstar,
    SHMR_N, SHMR_LOG_MC, SHMR_BETA, SHMR_GAMMA,
)
from likelihood.jwst_likelihood_uvlf import UVLFLikelihood


# ══════════════════════════════════════════════════════════════════════════════
#  FIXED CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

_Z_C_EXO     = 16.0
_SIGMA_Z_EXO = 3.25

_POLY_SLOPE     = -0.07202
_POLY_INTERCEPT = -1381.5969

# Ω_x(z=0) = a_exo × W(z=0) where W(z) is the Gaussian window.
# W(0) = exp(-z_c^2 / (2 σ_z^2)) = exp(-256 / 21.125) ≈ 5.43e-6
# a_exo is the peak Ω_x at z=z_c=16.  Today's value is essentially zero.
_OMEGA_X0_FACTOR = np.exp(-_Z_C_EXO**2 / (2.0 * _SIGMA_Z_EXO**2))

# Planck 2018 base — identical to all other pipeline test files
_PLANCK = {
    'h':             0.6736,
    'omega_b':       0.02237,
    'omega_cdm':     0.1200,
    'n_s':           0.9649,
    'A_s':           2.101e-9,
    'tau_reio':      0.0544,
    'output':        'mPk',
    'P_k_max_1/Mpc': 375.0,
    'z_max_pk':      20.0,
    'non linear':    'none',
    'z_c_exo':       _Z_C_EXO,
    'sigma_z_exo':   _SIGMA_Z_EXO,
}

# Stefanon 2021 SHMR defaults — used for ΛCDM evaluation in vary-SHMR runs
_SHMR_DEFAULTS = {
    'shmr_N':      SHMR_N,
    'shmr_log_Mc': SHMR_LOG_MC,
    'shmr_beta':   SHMR_BETA,
}


# ══════════════════════════════════════════════════════════════════════════════
#  MOCK COBAYA PROVIDER
# ══════════════════════════════════════════════════════════════════════════════

class _MockTheory:
    """Minimal stand-in so _get_raw_classy() can find the CLASS object."""
    def __init__(self, cosmo):
        self.classy = cosmo

class _MockModel:
    """Minimal stand-in for Cobaya's Model."""
    def __init__(self, cosmo):
        self.theory = {'classy': _MockTheory(cosmo)}

class _MockProvider:
    """
    Minimal stand-in for Cobaya's provider.

    The only thing logp() needs from the provider is the raw CLASS
    object, accessed via _get_raw_classy(provider) which traverses
    provider.model.theory['classy'].classy.  Everything else (h,
    Omega_m, P(k,z), H(z), d_A(z)) comes from the CLASS object
    directly.
    """
    def __init__(self, cosmo):
        self.model = _MockModel(cosmo)

class _MockLogger:
    def info(self,    msg): print(f"  [INFO]  {msg}")
    def debug(self,   msg): pass
    def warning(self, msg): print(f"  [WARN]  {msg}")
    def error(self,   msg): print(f"  [ERROR] {msg}")


# ══════════════════════════════════════════════════════════════════════════════
#  INFRASTRUCTURE
# ══════════════════════════════════════════════════════════════════════════════

def _build_uvlf_likelihood(z_min_don=9.0, z_min_fink=8.9,
                            vary_SHMR=False, vary_beta=False,
                            integrate_bin=False, use_volume_correction=False):
    like = UVLFLikelihood.__new__(UVLFLikelihood)
    like.use_volume_correction = use_volume_correction
    like.integrate_bin         = integrate_bin
    like.n_gl                  = 2
    like.z_min_donnan          = z_min_don
    like.z_min_finkelstein     = z_min_fink
    like.vary_SHMR             = vary_SHMR
    like.vary_beta             = vary_beta
    like.log                   = _MockLogger()
    like.initialize()
    return like


def _build_cosmo(a_exo=0.0, b_exo=0.0):
    """Build and compute a class_omx cosmology at (a_exo, b_exo)."""
    p = dict(_PLANCK)
    p['a_exo'] = a_exo
    p['b_exo'] = b_exo
    cosmo = Class()
    cosmo.set(p)
    cosmo.compute()
    return cosmo


def _logp_uvlf(like, a_exo=0.0, b_exo=0.0, shmr_overrides=None, cosmo=None):
    """
    Call like.logp() at (a_exo, b_exo) with optional SHMR overrides.

    Builds and destroys its own CLASS instance unless one is supplied.

    Parameters
    ----------
    like           : UVLFLikelihood instance (initialised)
    a_exo, b_exo   : exotic DE parameters
    shmr_overrides : dict — keys: shmr_N, shmr_log_Mc, shmr_beta.
                     If None and like.vary_SHMR or like.vary_beta, injects
                     Stefanon 2021 defaults so logp() doesn't KeyError.
    cosmo          : classy.Class — if supplied, reused (not destroyed).

    Returns
    -------
    float — logp value
    """
    owns_cosmo = cosmo is None
    if owns_cosmo:
        cosmo = _build_cosmo(a_exo, b_exo)

    like.provider = _MockProvider(cosmo)

    # Build params_values dict for logp().
    # SHMR params must be present when the likelihood is in vary mode.
    pv = {'a_samp': a_exo, 's': a_exo + b_exo}

    if like.vary_SHMR:
        if shmr_overrides is None:
            pv.update(_SHMR_DEFAULTS)
        else:
            pv.update({k: shmr_overrides.get(k, _SHMR_DEFAULTS[k])
                       for k in ('shmr_N', 'shmr_log_Mc', 'shmr_beta')})

    elif like.vary_beta:
        if shmr_overrides is None:
            pv['shmr_beta'] = _SHMR_DEFAULTS['shmr_beta']
        else:
            pv['shmr_beta'] = shmr_overrides.get('shmr_beta',
                                                   _SHMR_DEFAULTS['shmr_beta'])

    result = like.logp(**pv)
    if not np.isfinite(result):
        print(f"  [DEBUG] logp returned {result} — check WARNING lines above")

    if owns_cosmo:
        cosmo.struct_cleanup()
        cosmo.empty()

    return result


def _count_data_points(like):
    """
    Count total active data points in an initialised likelihood.

    Returns (N_don, N_fink, N_total).
    """
    n_don  = sum(len(v[0]) for v in like._donnan_data.values())
    n_fink = sum(len(v[0]) for v in like._finkelstein_data.values())
    return n_don, n_fink, n_don + n_fink


def _per_bin_chi2(like, cosmo):
    """
    Compute chi2 per redshift bin at the given cosmology.

    Returns list of (dataset, z_nom, chi2_bin, n_pts, pull_mean).
    pull_mean = mean of (phi_theory - phi_obs) / sigma over the bin.
    """
    rows = []

    for z_nom in like._donnan_bins:
        M_UV_bins, phi_obs, su, sd = like._donnan_data[z_nom]
        M_h, dndlnm, _, _ = compute_hmf(cosmo, z_nom)
        z_lo, z_hi = DONNAN_Z_EDGES[z_nom]
        phi_th = compute_uvlf_theory(M_h, dndlnm, z_nom, M_UV_bins,
                                     integrate_bin=False)
        chi2_bin = chi_squared(phi_th, phi_obs, su, sd)
        sigma    = np.where(phi_th > phi_obs, su, sd)
        pull     = float(np.mean((phi_th - phi_obs) / sigma))
        rows.append(('Donnan', z_nom, chi2_bin, len(M_UV_bins), pull))

    for z_nom in like._finkelstein_bins:
        M_UV_bins, phi_obs, su, sd = like._finkelstein_data[z_nom]
        M_h, dndlnm, _, _ = compute_hmf(cosmo, z_nom)
        z_lo, z_hi = FINKELSTEIN_Z_EDGES[z_nom]
        phi_th = compute_uvlf_theory(M_h, dndlnm, z_nom, M_UV_bins,
                                     integrate_bin=False)
        chi2_bin = chi_squared(phi_th, phi_obs, su, sd)
        sigma    = np.where(phi_th > phi_obs, su, sd)
        pull     = float(np.mean((phi_th - phi_obs) / sigma))
        rows.append(('Finkelstein', z_nom, chi2_bin, len(M_UV_bins), pull))

    return rows


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
            print(f'  {icon}  {tid}  {name}')
        n_pass = sum(1 for _, _, icon, _ in self._rows if 'PASS' in icon)
        n_fail = len(self._rows) - n_pass
        print('─' * 72)
        print(f'  {n_pass}/{len(self._rows)} passed', end='')
        if n_fail == 0:
            print('  — all good. بسم الله 🚀')
        else:
            print(f'  — fix {n_fail} failure(s) before launching BOBYQA.')
        print('═' * 72)
        return n_fail == 0


# ══════════════════════════════════════════════════════════════════════════════
#  VALIDATION TESTS  U01 – U04
# ══════════════════════════════════════════════════════════════════════════════

def test_U01_logp_finite_full(cosmo_lcdm, results):
    """
    U01 — logp finite at ΛCDM, full likelihood (all 40 data points).

    Smoke test: imports, data loading, _CosmoAdapter, chi2 computation.
    Expected chi2 ≈ 113.2 from the diagnostic grid scan at Planck cosmology.
    """
    print('\n' + '─' * 72)
    print('U01 — logp finite at ΛCDM [full likelihood]')
    print('─' * 72)
    try:
        like = _build_uvlf_likelihood(9.0, 8.9)
        t0   = time.time()
        lp   = _logp_uvlf(like, 0.0, 0.0, cosmo=cosmo_lcdm)
        dt   = time.time() - t0
        chi2 = -2.0 * lp
        passed = np.isfinite(lp) and chi2 > 0
        results.add('U01', 'logp finite — full likelihood', passed,
                    f'logp={lp:.3f}  chi2={chi2:.2f}  '
                    f'(expected ≈113)  dt={dt:.2f}s')
    except Exception:
        results.add('U01', 'logp finite — full likelihood', False,
                    traceback.format_exc())


def test_U02_logp_finite_restricted(cosmo_lcdm, results):
    """
    U02 — logp finite at ΛCDM, restricted likelihood (z≥10 only).

    Excludes Donnan z=9 and Finkelstein z=8.9 — the two bins dominated
    by the pre-JWST SHMR systematic. Validates the z_min threshold filtering.
    """
    print('\n' + '─' * 72)
    print('U02 — logp finite at ΛCDM [restricted: z_don≥10, z_fink≥10.9]')
    print('─' * 72)
    try:
        like = _build_uvlf_likelihood(10.0, 10.9)
        _, _, n_total = _count_data_points(like)
        t0   = time.time()
        lp   = _logp_uvlf(like, 0.0, 0.0, cosmo=cosmo_lcdm)
        dt   = time.time() - t0
        chi2 = -2.0 * lp
        passed = np.isfinite(lp) and chi2 > 0
        results.add('U02', 'logp finite — restricted likelihood', passed,
                    f'logp={lp:.3f}  chi2={chi2:.2f}  '
                    f'N_bins={n_total}  dt={dt:.2f}s')
    except Exception:
        results.add('U02', 'logp finite — restricted likelihood', False,
                    traceback.format_exc())


def test_U03_phi_non_negative(cosmo_lcdm, results):
    """
    U03 — phi(M_UV) ≥ 0 everywhere on a dense M_UV grid.

    Evaluates _phi_single_z on a 200-point M_UV grid from −23 to −14 at
    each of the 8 nominal bin redshifts. A negative value would indicate
    a sign error in the Jacobian chain or the interpolation.
    """
    print('\n' + '─' * 72)
    print('U03 — phi(M_UV) ≥ 0 at all M_UV and all active z-bins')
    print('─' * 72)
    try:
        M_UV_dense = np.linspace(-23.0, -14.0, 200)
        z_list = [9.0, 10.0, 11.0, 12.5, 14.5, 8.9, 10.9, 14.0]
        n_neg   = 0
        worst_z = None

        for z in z_list:
            M_h, dndlnm, _, _ = compute_hmf(cosmo_lcdm, z)
            phi = compute_uvlf_theory(M_h, dndlnm, z, M_UV_dense)
            bad = int(np.sum(phi < 0))
            if bad > 0:
                n_neg += bad
                worst_z = z

        passed = n_neg == 0
        detail = (f'negative bins = {n_neg}  (must be 0)'
                  + (f'  worst z={worst_z}' if worst_z else ''))
        results.add('U03', 'phi ≥ 0 on dense M_UV grid', passed, detail)
    except Exception:
        results.add('U03', 'phi ≥ 0', False, traceback.format_exc())


def test_U04_shmr_monotone(cosmo_lcdm, results):
    """
    U04 — SHMR strictly monotone on the full HMF mass grid.

    shmr_mstar(M_h) must be strictly increasing over the entire HMF grid
    at Stefanon default parameters. Non-monotone segments would corrupt
    the log-log Jacobian calculation in _phi_single_z.
    """
    print('\n' + '─' * 72)
    print('U04 — SHMR strict monotonicity on HMF grid')
    print('─' * 72)
    try:
        M_h, _, _, _ = compute_hmf(cosmo_lcdm, 9.0)
        M_star = shmr_mstar(M_h, SHMR_N, SHMR_LOG_MC, SHMR_BETA, SHMR_GAMMA)
        diffs  = np.diff(M_star)
        n_bad  = int(np.sum(diffs <= 0))
        passed = n_bad == 0
        results.add('U04', 'SHMR strictly monotone on HMF grid', passed,
                    f'non-increasing steps = {n_bad}  (must be 0)  '
                    f'M_star range = [{M_star.min():.2e}, {M_star.max():.2e}]')
    except Exception:
        results.add('U04', 'SHMR monotone', False, traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
#  BOBYQA MINIMISATION
# ══════════════════════════════════════════════════════════════════════════════

def _run_uvlf_minimisation(label, z_min_don, z_min_fink,
                            vary_SHMR, vary_beta,
                            output_dir, cosmo_lcdm,
                            max_evals=2000,
                            integrate_bin=False,
                            use_volume_correction=False,
                            resume=False):
    """
    Run one BOBYQA minimisation through Cobaya and return a result dict.

    ΛCDM chi2 is evaluated analytically from the mock provider before
    BOBYQA runs — no minimisation needed for the P=0 reference point.

    Parameters
    ----------
    label      : str  — short run label (e.g. 'run1_full')
    z_min_don  : float — minimum Donnan z-bin to include
    z_min_fink : float — minimum Finkelstein z-bin to include
    vary_SHMR  : bool — if True, float N + log_Mc + beta simultaneously
    vary_beta  : bool — if True, float beta only
    output_dir : str  — Cobaya output path prefix
    cosmo_lcdm : classy.Class — pre-built ΛCDM object (shared, not destroyed)
    max_evals  : int  — BOBYQA maximum function evaluations

    Returns
    -------
    dict with keys:
        label, N_don, N_fink, N_total, P,
        lcdm_chi2, exotic_chi2, delta_chi2,
        a_best, b_best, omega_x0,
        shmr_beta_best, shmr_N_best, shmr_log_Mc_best  (if applicable)
        chi2_red, elapsed_s
    """
    _print_run_header(label, z_min_don, z_min_fink, vary_SHMR, vary_beta,
                      integrate_bin, use_volume_correction, lcdm=False)

    # ── ΛCDM evaluation ───────────────────────────────────────────────────────
    like = _build_uvlf_likelihood(z_min_don, z_min_fink, vary_SHMR, vary_beta,
                               integrate_bin=integrate_bin,
                               use_volume_correction=use_volume_correction)
    N_don, N_fink, N_total = _count_data_points(like)
    P = 2 + (3 if vary_SHMR else 1 if vary_beta else 0)

    lp_lcdm   = _logp_uvlf(like, 0.0, 0.0, cosmo=cosmo_lcdm)
    chi2_lcdm = -2.0 * lp_lcdm
    
    print(f'  N_total={N_total}  P={P}  chi2_ΛCDM={chi2_lcdm:.2f}')
    


    # ── Cobaya BOBYQA info dict ────────────────────────────────────────────────
    output_path = os.path.join(output_dir, f'minimise_{label}')

    params = {
        # ── exotic DE reparametrisation ───────────────────────────────────────
        # Sample in (a_samp, s); CLASS receives a_exo = a_samp, b_exo = s-a_samp.
        # a_samp and s are dropped so they don't clobber CLASS directly.
        'a_samp': {
            'prior': {'min': -1838.0, 'max': -1e-10},
            'ref':   {'dist': 'norm', 'loc': -40.0, 'scale': 5.0},
            'drop':  True,
        },
        's': {
            'prior': {'min': -1381.597, 'max': -1e-10},
            'ref':   {'dist': 'norm', 'loc': -160.0, 'scale': 10.0},
            'drop':  True,
        },
        'a_exo':     {'value': 'lambda a_samp: a_samp'},
        'b_exo':     {'value': 'lambda a_samp, s: s - a_samp'},
        # ── fixed Planck cosmology ────────────────────────────────────────────
        'H0':        {'value': 67.36},
        'omega_b':   {'value': 0.02237},
        'omega_cdm': {'value': 0.1200},
        'n_s':       {'value': 0.9649},
        'logA':      {'value': 3.044, 'drop': True},
        'A_s':       {'value': 'lambda logA: 1e-10*np.exp(logA)'},
        'tau_reio':  {'value': 0.0544},
        'Omega_m':   {'derived': True},
    }

    # ── SHMR parameters (only added when floating) ────────────────────────────
    # These parameters go to the likelihood via params_values.
    # Cobaya-classy ignores them when setting CLASS parameters because
    # class_omx does not declare them as supported.
    if vary_SHMR:
        params['shmr_N'] = {
            'prior': {'min': 0.005, 'max': 0.12},
            'ref':   SHMR_N,
            'latex': r'N_{\rm SHMR}',
        }
        params['shmr_log_Mc'] = {
            'prior': {'min': 10.5, 'max': 12.5},
            'ref':   SHMR_LOG_MC,
            'latex': r'\log_{10}(M_c)',
        }
        params['shmr_beta'] = {
            'prior': {'min': 0.1, 'max': 5.0},
            'ref':   SHMR_BETA,
            'latex': r'\beta_{\rm SHMR}',
        }
    elif vary_beta:
        params['shmr_beta'] = {
            'prior': {'min': 0.1, 'max': 5.0},
            'ref':   SHMR_BETA,
            'latex': r'\beta_{\rm SHMR}',
        }
        
#     minimizer_method = 'scipy' if (use_volume_correction or integrate_bin) else 'bobyqa'
#     max_evals = 1300 * P if (use_volume_correction or integrate_bin) else max_evals
    
    minimizer_method = 'bobyqa'
    
    info = {
        'likelihood': {
            'likelihood.jwst_likelihood_uvlf.UVLFLikelihood': {
                'python_path':         _PROJECT_ROOT,
                'use_volume_correction': use_volume_correction,
                'integrate_bin':         integrate_bin,
                'n_gl':                  2,
                'z_min_donnan':          z_min_don,
                'z_min_finkelstein':     z_min_fink,
                'vary_SHMR':             vary_SHMR,
                'vary_beta':             vary_beta,
            }
        },
        'theory': {
            'classy': {
                'extra_args': {
                    'z_c_exo':       _Z_C_EXO,
                    'sigma_z_exo':   _SIGMA_Z_EXO,
                    'output':        'mPk',
                    'P_k_max_1/Mpc': 375.0,
                    'z_max_pk':      20.0,
                    'non linear':    'none',
                },
                'ignore_obsolete': True,
            }
        },
        'params': params,
        'prior': {
            'h2_positivity': (
                "lambda a_samp, s: "
                "0.0 if s >= (-0.07202 * a_samp - 1381.5969) else -1e500"
            ),
        },
        'sampler': {
            'minimize': {
                'method':    minimizer_method,
                'max_evals': max_evals,
                'best_of':   4,
            }
        },
        'output': output_path,
        'debug':  False,
    }

    # ── Run Cobaya ─────────────────────────────────────────────────────────────
    from cobaya.run import run as cobaya_run
    t0 = time.time()
    upd_info, sampler = cobaya_run(info, resume=resume, force=(not resume))
    elapsed = time.time() - t0

    best   = sampler.products()
    bf     = best['minimum']

    a_best = float(bf['a_samp'])
    b_best = float(bf['s']) - float(bf['a_samp'])
    chi2_best = float(bf['chi2'])
    omega_x0  = a_best * _OMEGA_X0_FACTOR

    beta_best  = float(bf['shmr_beta'])   if (vary_SHMR or vary_beta) else SHMR_BETA
    N_best     = float(bf['shmr_N'])      if vary_SHMR                else SHMR_N
    logMc_best = float(bf['shmr_log_Mc']) if vary_SHMR                else SHMR_LOG_MC

    delta_chi2 = chi2_best - chi2_lcdm
    chi2_red   = chi2_best / max(N_total - P, 1)
        
    result = {
        'label':            label,
        'N_don':            N_don,
        'N_fink':           N_fink,
        'N_total':          N_total,
        'P':                P,
        'lcdm_chi2':        chi2_lcdm,
        'exotic_chi2':      chi2_best,
        'delta_chi2':       delta_chi2,
        'a_best':   a_best,
        'b_best':   b_best,
        'omega_x0': omega_x0,
        'shmr_beta_best':   beta_best,
        'shmr_N_best':      N_best,
        'shmr_log_Mc_best': logMc_best,
        'chi2_red':         chi2_red,
        'elapsed_s':        elapsed,
    }
    _print_run_result(result)
    return result

def _run_lcdm_shmr_minimisation(label, z_min_don, z_min_fink,
                                  vary_SHMR, vary_beta,
                                  integrate_bin, use_volume_correction,
                                  output_dir, cosmo_lcdm,
                                  max_evals=3000,
                                  resume=False):
    """
    ΛCDM-only BOBYQA run — a_exo=b_exo=0 fixed throughout.

    Varies only SHMR parameters (shmr_beta, or shmr_N+shmr_log_Mc+shmr_beta)
    to find the best chi2 ΛCDM can achieve through astrophysical adjustment alone.
    If vary_SHMR=False and vary_beta=False, no BOBYQA is run — returns the
    pure ΛCDM analytical evaluation directly.
    """
    _print_run_header(label, z_min_don, z_min_fink, vary_SHMR, vary_beta,
                  integrate_bin, use_volume_correction, lcdm=True)

    like = _build_uvlf_likelihood(z_min_don, z_min_fink,
                                   vary_SHMR, vary_beta,
                                   integrate_bin, use_volume_correction)
    N_don, N_fink, N_total = _count_data_points(like)
    P = 3 if vary_SHMR else 1 if vary_beta else 0

    lp_lcdm   = _logp_uvlf(like, 0.0, 0.0, cosmo=cosmo_lcdm)
    chi2_lcdm = -2.0 * lp_lcdm
    
    print(f'  N_total={N_total}  P={P}  chi2_ΛCDM={chi2_lcdm:.2f}')    


    # ── No free parameters: pure ΛCDM evaluation, skip BOBYQA ───────────────
    if not vary_SHMR and not vary_beta:
        chi2_red = chi2_lcdm / max(N_total, 1)
        print(f'  [{label}] fixed SHMR — no minimisation needed')
        return {
            'label':            label,
            'N_don':            N_don,
            'N_fink':           N_fink,
            'N_total':          N_total,
            'P':                0,
            'lcdm_chi2':        chi2_lcdm,
            'exotic_chi2':      chi2_lcdm,
            'delta_chi2':       0.0,
            'a_best':           0.0,
            'b_best':           0.0,
            'omega_x0':         0.0,
            'shmr_beta_best':   SHMR_BETA,
            'shmr_N_best':      SHMR_N,
            'shmr_log_Mc_best': SHMR_LOG_MC,
            'chi2_red':         chi2_red,
            'elapsed_s':        0.0,
        }

    # ── BOBYQA over SHMR params only ─────────────────────────────────────────
    output_path = os.path.join(output_dir, f'minimise_{label}')

    params = {
        'a_exo':     {'value': 0.0},
        'b_exo':     {'value': 0.0},
        'H0':        {'value': 67.36},
        'omega_b':   {'value': 0.02237},
        'omega_cdm': {'value': 0.1200},
        'n_s':       {'value': 0.9649},
        'logA':      {'value': 3.044, 'drop': True},
        'A_s':       {'value': 'lambda logA: 1e-10*np.exp(logA)'},
        'tau_reio':  {'value': 0.0544},
        'Omega_m':   {'derived': True},
    }

    if vary_SHMR:
        params['shmr_N'] = {
            'prior': {'min': 0.005, 'max': 0.12},
            'ref':   SHMR_N,
            'latex': r'N_{\rm SHMR}',
        }
        params['shmr_log_Mc'] = {
            'prior': {'min': 10.5, 'max': 12.5},
            'ref':   SHMR_LOG_MC,
            'latex': r'\log_{10}(M_c)',
        }
        params['shmr_beta'] = {
            'prior': {'min': 0.1, 'max': 5.0},
            'ref':   SHMR_BETA,
            'latex': r'\beta_{\rm SHMR}',
        }
    elif vary_beta:
        params['shmr_beta'] = {
            'prior': {'min': 0.1, 'max': 5.0},
            'ref':   SHMR_BETA,
            'latex': r'\beta_{\rm SHMR}',
        }

    info = {
        'likelihood': {
            'likelihood.jwst_likelihood_uvlf.UVLFLikelihood': {
                'python_path':           _PROJECT_ROOT,
                'use_volume_correction': use_volume_correction,
                'integrate_bin':         integrate_bin,
                'n_gl':                  2,
                'z_min_donnan':          z_min_don,
                'z_min_finkelstein':     z_min_fink,
                'vary_SHMR':             vary_SHMR,
                'vary_beta':             vary_beta,
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
        'params': params,
        # no prior block — a_exo=0 always satisfies H²≥0
        'sampler': {
            'minimize': {
                'method':    'bobyqa',
                'max_evals': max_evals,
            }
        },
        'output': output_path,
        'debug':  False,
    }

    from cobaya.run import run as cobaya_run
    t0 = time.time()
    upd_info, sampler = cobaya_run(info, resume=resume, force=(not resume))
    elapsed = time.time() - t0

    best   = sampler.products()
    bf     = best['minimum']
    chi2_best  = float(bf['chi2'])
    beta_best  = float(bf['shmr_beta'])   if (vary_SHMR or vary_beta) else SHMR_BETA
    N_best     = float(bf['shmr_N'])      if vary_SHMR                else SHMR_N
    logMc_best = float(bf['shmr_log_Mc']) if vary_SHMR                else SHMR_LOG_MC

    delta_chi2 = chi2_best - chi2_lcdm
    chi2_red   = chi2_best / max(N_total - P, 1)

    print(f'  [{label}] Best  chi2={chi2_best:.2f}  Δchi2={delta_chi2:+.2f}  '
          f'beta={beta_best:.3f}  dt={elapsed:.0f}s')
    if vary_SHMR:
        print(f'  [{label}]       N={N_best:.4f}  logMc={logMc_best:.3f}')
        
        
    result = {
        'label':            label,
        'N_don':            N_don,
        'N_fink':           N_fink,
        'N_total':          N_total,
        'P':                P,
        'lcdm_chi2':        chi2_lcdm,
        'exotic_chi2':      chi2_best,
        'delta_chi2':       delta_chi2,
        'a_best':           0.0,
        'b_best':           0.0,
        'omega_x0':         0.0,
        'shmr_beta_best':   beta_best,
        'shmr_N_best':      N_best,
        'shmr_log_Mc_best': logMc_best,
        'chi2_red':         chi2_red,
        'elapsed_s':        elapsed,
    }
    
    _print_run_result(result)
    
    return result


def run_all_minimisations(output_dir, cosmo_lcdm, max_evals=2000, group=None, resume=False):
    """
    Run all 4 BOBYQA minimisations and return a list of result dicts.
    Failures are caught individually so one crashed run doesn't abort the rest.
    """
    print('\n' + '═' * 72)
    print('  BOBYQA MINIMISATION RUNS  (max_evals={})'.format(max_evals))
    print('═' * 72)
    

    run_specs = [
        # (label, z_don, z_fink, vS, vB, intg, volc, lcdm_only)

        # ── A: no corrections ────────────────────────────────────────────────
        ('A1_full_fixed_SHMR',    9.0,  8.9,  False, False, False, False, False),
        ('A2_restr_fixed_SHMR',   10.0, 10.9, False, False, False, False, False),
        ('A3_full_vary_beta',      9.0,  8.9,  False, True,  False, False, False),
        ('A4_full_vary_SHMR',      9.0,  8.9,  True,  False, False, False, False),

        # ── B: both corrections ───────────────────────────────────────────────
#         ('B1_full_fixed_SHMR',    9.0,  8.9,  False, False, True,  True,  False),
#         ('B2_restr_fixed_SHMR',   10.0, 10.9, False, False, True,  True,  False),
#         ('B3_full_vary_beta',      9.0,  8.9,  False, True,  True,  True,  False),
#         ('B4_full_vary_SHMR',      9.0,  8.9,  True,  False, True,  True,  False),
        ('B5_restr_vary_SHMR',      10.0, 10.9,  True,  False, True,  True,  False),
        ('B6_restr_vary_beta',      10.0, 10.9,  False, True,  True,  True,  False),

        # ── C: volume correction only ─────────────────────────────────────────
        ('C1_full_fixed_SHMR',    9.0,  8.9,  False, False, False, True,  False),
        ('C2_restr_fixed_SHMR',   10.0, 10.9, False, False, False, True,  False),
        ('C3_full_vary_beta',      9.0,  8.9,  False, True,  False, True,  False),
        ('C4_full_vary_SHMR',      9.0,  8.9,  True,  False, False, True,  False),

        # ── D: GL integration only ───────────────────────────────────────────
        ('D1_full_fixed_SHMR',    9.0,  8.9,  False, False, True,  False, False),
        ('D2_restr_fixed_SHMR',   10.0, 10.9, False, False, True,  False, False),
        ('D3_full_vary_beta',      9.0,  8.9,  False, True,  True,  False, False),
        ('D4_full_vary_SHMR',      9.0,  8.9,  True,  False, True,  False, False),

        # ── E: ΛCDM only — a_exo=b_exo=0 fixed, SHMR varies ─────────────────
        ('E1_lcdm_full_fixed',    9.0,  8.9,  False, False, False, False, True),
        ('E2_lcdm_restr_fixed',   10.0, 10.9, False, False, False, False, True),
        ('E3_lcdm_full_beta',      9.0,  8.9,  False, True,  False, False, True),
        ('E4_lcdm_full_SHMR',      9.0,  8.9,  True,  False, False, False, True),
        ('E5_lcdm_vary_beta_both_corrections',      9.0,  8.9,  False,  True, True, True, True),
        ('E6_lcdm_full_SHMR_both_corrections',      9.0,  8.9,  True,  False, True, True, True),
        ('E7_lcdm_vary_beta_both_corrections_restr',      10.0, 10.9,  False,  True, True, True, True),
        ('E8_lcdm_full_SHMR_both_corrections_restr',      10.0, 10.9,  True,  False, True, True, True),
    ]
    
    if group:
        run_specs = [r for r in run_specs if r[0].startswith(group.upper())]

    all_results = []
    for label, z_don, z_fink, vs, vb, intg, volc, lcdm in run_specs:
        try:
            if lcdm:
                r = _run_lcdm_shmr_minimisation(
                    label, z_don, z_fink, vs, vb,
                    intg, volc,
                    output_dir, cosmo_lcdm, max_evals,
                    resume=resume,
                )
            else:
                r = _run_uvlf_minimisation(
                    label, z_don, z_fink, vs, vb,
                    output_dir, cosmo_lcdm, max_evals,
                    integrate_bin=intg,
                    use_volume_correction=volc,
                    resume=resume,
                )
            all_results.append(r)
        except Exception:
            print(f'\n  ❌  [{label}] failed:')
            print(traceback.format_exc())
            all_results.append({'label': label, 'failed': True})

    return all_results


# ══════════════════════════════════════════════════════════════════════════════
#  CONSOLE + LATEX TABLES
# ══════════════════════════════════════════════════════════════════════════════

_RUN_LABELS = {
    'run1_full_fixed':  'Full (fixed SHMR)',
    'run2_restr_fixed': r'Restricted $z\!\geq\!10$ (fixed SHMR)',
    'run3_full_beta':   r'Full (vary $\beta$)',
    'run4_full_shmr':   r'Full (vary $N,M_c,\beta$)',
}

_RUN_LABELS_CONSOLE = {
    'run1_full_fixed':  'Full — fixed SHMR            ',
    'run2_restr_fixed': 'Restricted z≥10 — fixed SHMR ',
    'run3_full_beta':   'Full — vary β                 ',
    'run4_full_shmr':   'Full — vary N, Mc, β          ',
}


def _generate_console_table(all_results):
    """Print the summary table to console."""
    print('\n' + '═' * 100)
    print('  UVLF RESULTS SUMMARY')
    print('═' * 100)
    hdr = (f"  {'Run':<32}  {'N':>5}  {'P':>2}  "
           f"{'χ²_ΛCDM':>10}  {'χ²_best':>10}  "
           f"{'Δχ²':>8}  {'a_exo':>11}  {'b_exo':>11}  "
           f"{'Ω_x0':>10}  {'χ²_red':>7}")
    print(hdr)
    print('─' * 100)

    for r in all_results:
        if r.get('failed'):
            print(f"  {_RUN_LABELS_CONSOLE.get(r['label'], r['label']):<32}"
                  f"  FAILED")
            continue
        print(
                f"  {_RUN_LABELS_CONSOLE.get(r['label'], r['label']):<32}"
                f"  {r['N_total']:>5}"
                f"  {r['P']:>2}"
                f"  {r['lcdm_chi2']:>10.2f}"
                f"  {r['exotic_chi2']:>10.2f}"
                f"  {r['delta_chi2']:>+8.2f}"
                f"  {r['a_best']:>11.3e}"
                f"  {r['b_best']:>11.3e}"
                f"  {r['omega_x0']:>10.3e}"
                f"  {r['chi2_red']:>7.3f}"
            )
    print('═' * 100)


def _generate_latex_table(all_results, output_dir):
    """
    Write table_uvlf_results.tex — the main results table for the PDF.

    Columns: Run | N | P | χ²_ΛCDM | χ²_best | Δχ² | a_exo | Ω_x0 | χ²_red

    χ²_red = χ²_best / (N - P).
    Ω_x0 = a_exo × exp(-z_c² / (2σ_z²)) — the exotic DE density today.
    """
    caption = (
        r'Results of BOBYQA minimisation of the JWST UV luminosity function '
        r'likelihood (Donnan~2024 PRIMER + Finkelstein~2024 CEERS). '
        r'$N$ = number of active data points. '
        r'$P$ = free parameters: 2 for fixed SHMR $(a_{\rm exo}, b_{\rm exo})$; '
        r'3 for vary-$\beta$ (adds $\beta_{\rm SHMR}$); '
        r'5 for vary-SHMR (adds $N_{\rm SHMR}$, $\log M_c$, $\beta_{\rm SHMR}$). '
        r'$\chi^2_{\rm LCDM}$ evaluated analytically at $a_{\rm exo}=b_{\rm exo}=0$ '
        r'(no minimisation). '
        r'$\Delta\chi^2 = \chi^2_{\rm best} - \chi^2_{\rm LCDM}$ (negative = improvement). '
        r'$\Omega_{x,0} = a_{\rm exo}\,\exp(-z_c^2/2\sigma_z^2)$ '
        r'with $z_c=16$, $\sigma_z=3.25$. '
        r'$\chi^2_\nu = \chi^2_{\rm best}/(N - P)$. '
        r'SHMR: Stefanon~2021 double power law; '
        r'reference cosmology: Planck~2018.'
        
    )

    def _row(r):
        if r.get('failed'):
            label = _RUN_LABELS.get(r['label'], r['label'])
            return rf'  {label} & — & — & — & — & — & — & — & — \\'
        label    = _RUN_LABELS.get(r['label'], r['label'])
        lcdm_s   = f"{r['lcdm_chi2']:.1f}"
        best_s   = f"{r['exotic_chi2']:.1f}"
        delta_s  = f"{r['delta_chi2']:+.1f}"
        a_s      = f"{r['a_best']:.1f}"
        ox0_s    = f"{r['omega_x0']:.2e}".replace('e-0', r'\times10^{-').replace('e+0','')
        # format omega_x0 as LaTeX scientific notation
        v        = r['omega_x0']
        exp_str  = f'{v:.2e}'
        mantissa, exp = exp_str.split('e')
        exp_i    = int(exp)
        ox0_tex  = rf'${mantissa}\!\times\!10^{{{exp_i}}}$'
        red_s    = f"{r['chi2_red']:.3f}"
        def _fmt_sci(v):
            if abs(v) < 1e-12:
                return r'$\approx 0$'
            s = f'{v:.2e}'
            mantissa, exp = s.split('e')
            exp_i = int(exp)
            return rf'${mantissa}\!\times\!10^{{{exp_i}}}$'
        
        return (
            rf'  {label} & {r["N_total"]} & {r["P"]} '
            rf'& {r["lcdm_chi2"]:.1f} & {r["exotic_chi2"]:.1f} & {r["delta_chi2"]:+.1f} '
            rf'& {_fmt_sci(r["a_best"])} & {_fmt_sci(r["b_best"])} '
            rf'& {_fmt_sci(r["omega_x0"])} & {r["chi2_red"]:.3f} \\'
        )

    lines = [
        r'\begin{table}[ht]',
        r'\centering',
        r'\small',
        rf'\caption{{{caption}}}',
        r'\label{tab:uvlf_results}',
        r'\begin{tabular}{l|c|c|c|c|c|c|c|c|c}',
        r'\toprule',
        (r'Run & $N$ & $P$ & $\chi^2_{\rm LCDM}$ & $\chi^2_{\rm best}$ '
         r'& $\Delta\chi^2$ & $a_{\rm exo}$ & $b_{\rm exo}$ '
         r'& $\Omega_{x,0}$ & $\chi^2_\nu$ \\'),
        r'\midrule',
    ]

    for i, r in enumerate(all_results):
        lines.append(_row(r))
        # separator after run2 (restricted) and run1 (full fixed)
        if i < len(all_results) - 1:
            curr = all_results[i]['label'][0]
            nxt  = all_results[i+1]['label'][0]
            if curr != nxt:
                lines.append(r'\midrule')

    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]

    tex  = '\n'.join(lines)
    path = os.path.join(output_dir, 'table_uvlf_results.tex')
    with open(path, 'w') as f:
        f.write(tex)
    print(f'\n  LaTeX table → {path}')
    return path


def _generate_perbin_latex(cosmo_lcdm, all_results, output_dir):
    """
    Write table_uvlf_perbin.tex — per-z-bin chi2 at ΛCDM (shared baseline).

    Shows chi2_bin, n_pts, and mean pull direction for each redshift bin.
    Helps Hashim see which bins drive the overall chi2 and where the model
    over- or under-predicts.
    """
    like_full = _build_uvlf_likelihood(9.0, 8.9)
    rows_lcdm = _per_bin_chi2(like_full, cosmo_lcdm)

    caption = (
        r'Per-redshift-bin $\chi^2$ at $\Lambda$CDM ($a_{\rm exo}=b_{\rm exo}=0$, '
        r'Planck~2018 cosmology). '
        r'$n$ = number of $M_{\rm UV}$ bins in each redshift slice. '
        r'Mean pull $= \langle(\phi_{\rm th}-\phi_{\rm obs})/\sigma\rangle$ over the slice: '
        r'positive = overprediction, negative = underprediction. '
        r'The $z=9$ Donnan and $z=8.9$ Finkelstein bins account for the bulk of '
        r'$\chi^2_{\rm LCDM}$ and are excluded in Run~2 (restricted likelihood).'
    )

    lines = [
        r'\begin{table}[ht]',
        r'\centering',
        rf'\caption{{{caption}}}',
        r'\label{tab:uvlf_perbin}',
        r'\begin{tabular}{l|c|c|c|c}',
        r'\toprule',
        r'Dataset & $z_{\rm nom}$ & $n$ & $\chi^2_{\rm bin}$ & Mean pull \\',
        r'\midrule',
    ]

    total_chi2 = 0.0
    total_pts  = 0
    prev_ds    = None
    for (ds, z_nom, chi2_bin, n_pts, pull) in rows_lcdm:
        if prev_ds and prev_ds != ds:
            lines.append(r'\midrule')
        direction = 'over' if pull > 0 else 'under'
        lines.append(
            rf'  {ds} & {z_nom:.1f} & {n_pts} '
            rf'& {chi2_bin:.1f} & {pull:+.2f} ({direction}) \\'
        )
        total_chi2 += chi2_bin
        total_pts  += n_pts
        prev_ds = ds

    lines += [
        r'\midrule',
        rf'  \textbf{{Total}} & — & {total_pts} '
        rf'& \textbf{{{total_chi2:.1f}}} & — \\',
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]

    tex  = '\n'.join(lines)
    path = os.path.join(output_dir, 'table_uvlf_perbin.tex')
    with open(path, 'w') as f:
        f.write(tex)
    print(f'  LaTeX per-bin table → {path}')
    return path


def generate_summary_table(all_results, cosmo_lcdm, output_dir):
    """Generate console table + both LaTeX tables."""
    print('\n' + '═' * 72)
    print('  TABLE GENERATION')
    print('═' * 72)
    _generate_console_table(all_results)
    _generate_latex_table(all_results, output_dir)
    _generate_perbin_latex(cosmo_lcdm, all_results, output_dir)

def _print_run_header(label, z_don, z_fink, vs, vb, intg, volc,
                      lcdm=False, N_total=None, P=None):
    mode = 'ΛCDM only' if lcdm else 'Exotic DE'
    shmr = ('vary N+Mc+β' if vs else 'vary β' if vb else 'fixed SHMR')
    corr = []
    if intg:  corr.append('GL-integrate')
    if volc:  corr.append('vol-correct')
    corr_str = ' + '.join(corr) if corr else 'no corrections'
    n_str = f'  N={N_total}  P={P}' if N_total is not None else ''
    width = 72
    print('\n' + '╔' + '═' * (width - 2) + '╗')
    print(f'║  RUN: {label:<{width - 9}}║')
    print(f'║  {mode} · {shmr} · {corr_str:<{width - len(mode) - len(shmr) - 12}}║')
    print(f'║  z_don ≥ {z_don}  z_fink ≥ {z_fink}{n_str:<{width - 26 - len(str(z_don)) - len(str(z_fink))}}║')
    print('╚' + '═' * (width - 2) + '╝')
    
    
def _print_run_result(r):
    if r.get('failed'):
        print('\n  ❌  ' + r['label'] + ' — FAILED')
        return
    width = 72
    print('\n' + '┌' + '─' * (width - 2) + '┐')
    print(f'│  ✅  {r["label"]:<{width - 7}}│')
    print('├' + '─' * (width - 2) + '┤')
    print(f'│  chi2_ΛCDM  = {r["lcdm_chi2"]:>9.2f}    '
          f'chi2_best = {r["exotic_chi2"]:>9.2f}    '
          f'Δchi2 = {r["delta_chi2"]:>+8.2f}{"":>3}│')
    print(f'│  a_exo      = {r["a_best"]:>9.2f}    '
          f'Ω_x0      = {r["omega_x0"]:>9.3e}{"":>17}│')
    if r['shmr_beta_best'] != SHMR_BETA or r['shmr_N_best'] != SHMR_N:
        print(f'│  β_best     = {r["shmr_beta_best"]:>9.3f}    '
              f'N_best    = {r["shmr_N_best"]:>9.4f}    '
              f'logMc = {r["shmr_log_Mc_best"]:>6.3f}{"":>4}│')
    print(f'│  chi2_red   = {r["chi2_red"]:>9.3f}    '
          f'elapsed   = {r["elapsed_s"]:>6.0f}s{"":>26}│')
    print('└' + '─' * (width - 2) + '┘')
    
# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='UVLF validation suite and BOBYQA minimisation.'
    )
    parser.add_argument('--output_dir', default='test_outputs/uvlf',
        help='Directory for Cobaya output files and LaTeX tables.')
    parser.add_argument('--max_evals', type=int, default=3000,
        help='BOBYQA maximum function evaluations per run (default 2000).')
    
    parser.add_argument('--skip_validation', action='store_true',
    help='Skip U01-U04 validation tests and go straight to minimisation.')
    
    parser.add_argument('--skip_bobyqa', action='store_true',
        help='Run only U01–U04 validation tests, skip minimisation.')
    
    parser.add_argument('--group', default=None,
        help=(
        'Run only one letter group of minimisations:\n'
        '  A — baseline: no corrections, fixed SHMR + vary-beta + vary-SHMR\n'
        '  B — both corrections: GL bin integration + volume correction\n'
        '  C — volume correction only (no GL integration)\n'
        '  D — GL bin integration only (no volume correction)\n'
        '  E — LCDM-only: a_exo=b_exo=0 fixed, SHMR varies to test\n'
        '      whether astrophysics alone explains the JWST tension\n'
        'Default (None) runs all groups sequentially.'
    ))
    parser.add_argument('--resume', '-r', action='store_true',
        help='Resume from previous output instead of overwriting (Cobaya resume=True).')
    
    args = parser.parse_args()
    
    if args.group:
        args.group = str(args.group).upper()

    os.makedirs(args.output_dir, exist_ok=True)

    print('═' * 72)
    print('  UVLF EXOTIC DARK ENERGY — VALIDATION + MINIMISATION')
    print(f'  output_dir = {args.output_dir}')
    print(f'  max_evals  = {args.max_evals}')
    print('═' * 72)

    # ── Shared ΛCDM CLASS instance ────────────────────────────────────────────
    print('\nBuilding ΛCDM CLASS instance...')
    t0 = time.time()
    cosmo_lcdm = _build_cosmo(0.0, 0.0)
    print(f'  CLASS done in {time.time()-t0:.1f}s  '
          f'h={cosmo_lcdm.h():.4f}  '
          f'Omega_m={cosmo_lcdm.Omega_m():.4f}  '
          f'sigma8={cosmo_lcdm.sigma8():.4f}')

    # ── Validation tests ──────────────────────────────────────────────────────
    print('\n' + '═' * 72)
    print('  VALIDATION TESTS  (U01 – U04)')
    print('═' * 72)

    if not args.skip_validation:
        results = _Results()
        test_U01_logp_finite_full(cosmo_lcdm, results)
        test_U02_logp_finite_restricted(cosmo_lcdm, results)
        test_U03_phi_non_negative(cosmo_lcdm, results)
        test_U04_shmr_monotone(cosmo_lcdm, results)

        all_pass = results.summary()

        if not all_pass:
            print('\n  ❌  Fix validation failures before running BOBYQA.')
            if not args.skip_bobyqa:
                cosmo_lcdm.struct_cleanup()
                cosmo_lcdm.empty()
                sys.exit(1)

    # ── BOBYQA runs ───────────────────────────────────────────────────────────
    if args.skip_bobyqa:
        print('\n  [skip_bobyqa] Minimisation skipped.')
    else:
        all_results = run_all_minimisations(args.output_dir, cosmo_lcdm,
                                     args.max_evals, group=args.group,
                                     resume=args.resume)
        generate_summary_table(all_results, cosmo_lcdm, args.output_dir)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cosmo_lcdm.struct_cleanup()
    cosmo_lcdm.empty()
    print('\nDone.')


if __name__ == '__main__':
    main()
