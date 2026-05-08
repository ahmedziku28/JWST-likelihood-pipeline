#!/usr/bin/env python3
"""
tests/test_likelihood.py

Six-test validation suite for JWSTLikelihood before launching Run 1.
Tests 1-5 are standalone (no Cobaya sampler needed — provider is mocked).
Test 6 fires a real Cobaya BOBYQA minimization.

Usage
-----
python tests/test_likelihood.py \
    --phot_path  data/UNCOVER_DR4_SPS_catalog.fits \
    --zspec_path data/UNCOVER_DR4_SPS_zspec_catalog.fits \
    --mode spectroscopic \
    --output_dir tests/likelihood_test_outputs/

Run individual tests:
    python tests/test_likelihood.py ... --tests 1 2 3
Run all:
    python tests/test_likelihood.py ... --tests all
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

# ── Path setup ────────────────────────────────────────────────────────────────
_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from classy import Class
from likelihood.jwst_likelihood import JWSTLikelihood


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# Fixed exotic CLASS params (grid scan)
_Z_C_EXO     = 16.0
_SIGMA_Z_EXO = 3.25

# Physicality polygon bottom edge (same as likelihood)
_POLY_SLOPE     = -0.07202
_POLY_INTERCEPT = -1381.5969

# Planck 2018 base
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

# Test parameter sets
_LCDM          = {'a_exo':    0.0,   'b_exo':   0.0}
_EXOTIC_GOOD   = {'a_exo': -960.0,   'b_exo': 324.0}   # inside polygon, strong
_UNPHYSICAL    = {'a_exo': -5000.0,  'b_exo':   0.0}   # violates H^2 > 0

# Cobaya polygon prior (reproduced here for Test 6 only)
_COBAYA_PRIOR_LAMBDA = (
    "lambda a_samp, s: 0.0 "
    "if s >= (-0.07202 * a_samp - 1381.5969) else -1e500"
)

def _get_total_n_data(like):
    """Calculates total number of data points (N) used in the likelihood. Used to calculate residual likelihood"""
    from pipeline.data_extractor import UNCOVER_SKY_FRACTION
    from pipeline.stellar_mass_function import compute_observed_rho_star
    
    # Use LCDM to get distances for volume calculation
    cosmo = _build_cosmo(0, 0)
    total_n = 0
    for (z_min, z_max, z_mid) in like._zbins:
        chi_lo = cosmo.comoving_distance(z_min)
        chi_hi = cosmo.comoving_distance(z_max)
        V = (4.0 / 3.0) * np.pi * (chi_hi**3 - chi_lo**3) * UNCOVER_SKY_FRACTION
        
        M_thr, _, _, _ = compute_observed_rho_star(like._catalog, z_min, z_max, V)
        total_n += len(M_thr)
        
    cosmo.struct_cleanup(); cosmo.empty()
    return total_n

# ══════════════════════════════════════════════════════════════════════════════
#  MOCK COBAYA PROVIDER
#  Lets us call initialize() and logp() without a running Cobaya sampler.
# ══════════════════════════════════════════════════════════════════════════════

class _MockProvider:
    """
    Minimal stand-in for Cobaya's provider object.
    Wraps a real classy.Class instance and serves the same
    four products that logp() now requests via the adapter.
    """
    def __init__(self, cosmo, params=None):
        self._cosmo  = cosmo
        self._params = params or {}


    def get_comoving_radial_distance(self, z):
        return self._cosmo.comoving_distance(z)          # Mpc

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
    """Minimal logger that writes to stdout."""
    def info(self,  msg): print(f"  [INFO]  {msg}")
    def debug(self, msg): print(f"  [DEBUG] {msg}")
    def warning(self, msg): print(f"  [WARN]  {msg}")
    def error(self,  msg): print(f"  [ERROR] {msg}")


def _build_likelihood(phot_path, zspec_path, mode, vary_shmr=False):
    """
    Instantiate and initialize a JWSTLikelihood without Cobaya.
    Bypasses the Cobaya metaclass machinery by directly setting attributes.
    """
    like = JWSTLikelihood.__new__(JWSTLikelihood)
    like.mode                = mode
    like.vary_SHMR_params    = vary_shmr
    like.phot_path           = phot_path
    like.zspec_path          = zspec_path
    like.survey_area_arcmin2 = 45.0
    like.n_vol_nodes         = 30
    like.log                 = _MockLogger()
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
    """Wire a CLASS cosmo object into the likelihood as a mock provider."""
    params = {'a_samp': cosmo.a_exo() if hasattr(cosmo, 'a_exo') else 0.0}
    if extra_params:
        params.update(extra_params)
    like.provider = _MockProvider(cosmo, params)


def _logp(like, a_exo, b_exo, cosmo=None):
    """
    Call like.logp() at (a_exo, b_exo).
    Builds CLASS if cosmo not supplied.  Returns (logp_value, cosmo).
    """
    if cosmo is None:
        cosmo = _build_cosmo(a_exo, b_exo)
        owns_cosmo = True
    else:
        owns_cosmo = False

    _attach_provider(like, cosmo, {'a_samp': a_exo, 's': a_exo + b_exo})
    result = like.logp(a_samp=a_exo, s=a_exo + b_exo)

    if owns_cosmo:
        cosmo.struct_cleanup(); cosmo.empty()

    return result


def _prior(like, a_exo, b_exo):
    """Call like.prior() at (a_exo, b_exo)."""
    a_samp = a_exo
    s      = a_exo + b_exo
    return like.prior(a_samp=a_samp, s=s)


# ══════════════════════════════════════════════════════════════════════════════
#  RESULT TRACKER
# ══════════════════════════════════════════════════════════════════════════════

class _Results:
    def __init__(self):
        self._rows = []

    def add(self, test_id, name, passed, detail=""):
        icon = "✅ PASS" if passed else "❌ FAIL"
        self._rows.append((test_id, name, icon, detail))
        print(f"  {icon}  {detail}")

    def summary(self):
        print("\n" + "═" * 70)
        print("  SUMMARY")
        print("═" * 70)
        for tid, name, icon, detail in self._rows:
            print(f"  Test {tid}  {icon}  {name}")
            if detail:
                print(f"          └─ {detail}")
        n_pass = sum(1 for *_, icon, _ in self._rows if "PASS" in icon)
        n_fail = len(self._rows) - n_pass
        print("─" * 70)
        print(f"  {n_pass}/{len(self._rows)} passed", end="")
        if n_fail == 0:
            print("  — launch Run 1. بسم الله 🚀")
        else:
            print(f"  — fix {n_fail} failure(s) before launching.")
        print("═" * 70)


# ══════════════════════════════════════════════════════════════════════════════
#  TEST 1 — Does it run at all?
# ══════════════════════════════════════════════════════════════════════════════

def test_1_runs(like, results, N):
    print("\n" + "─" * 70)
    print("TEST 1 — Basic plumbing: initialize + logp(ΛCDM) returns finite")
    print("─" * 70)
    try:
        t0 = time.time()
        lp = _logp(like, **_LCDM)
        dt = time.time() - t0
        finite = np.isfinite(lp)
        chi2_lcdm = -2 * lp
        red_chi2_lcdm = chi2_lcdm / N  # P=0 for LCDM
        
        results.add(
            1, "Basic plumbing", finite,
            f"logp(ΛCDM)={lp:.2f}  chi2={chi2_lcdm:.1f}  red_chi2={red_chi2_lcdm:.3f} (N={N})"
        )
    except Exception:
        results.add(1, "Basic plumbing", False,
                    f"CRASHED:\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════════════════════
#  TEST 2 — ΛCDM vs exotic sign check
# ══════════════════════════════════════════════════════════════════════════════

def test_2_sign_check(like, results):
    print("\n" + "─" * 70)
    print("TEST 2 — Sign check: exotic (a=-960, b=324) should fit better")
    print("         than ΛCDM at high-z bins where JWST tension is largest")
    print("─" * 70)
    try:
        lp_lcdm  = _logp(like, **_LCDM)
        lp_exotic = _logp(like, **_EXOTIC_GOOD)
        delta    = lp_exotic - lp_lcdm
        passed   = (lp_exotic > lp_lcdm)
        results.add(
            2, "ΛCDM vs exotic sign",
            passed,
            f"lnL(ΛCDM)={lp_lcdm:.2f}  "
            f"lnL(exotic)={lp_exotic:.2f}  "
            f"Δ={delta:+.2f}"
            + ("  ← exotic fits better ✓" if passed
               else "  ← ΛCDM fits better — check SHMR inversion or units!")
        )
    except Exception:
        results.add(2, "Sign check", False,
                    f"CRASHED:\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════════════════════
#  TEST 3 — Physicality rejection
# ══════════════════════════════════════════════════════════════════════════════

def test_3_physicality(like, results):
    print("\n" + "─" * 70)
    print("TEST 3 — Physicality: unphysical params (a=-5000, b=0)")
    print("         prior() must return -inf BEFORE CLASS runs")
    print("─" * 70)
    try:
        # prior() is called with (a_samp, s) — does NOT need a CLASS object
        a_exo = _UNPHYSICAL['a_exo']
        b_exo = _UNPHYSICAL['b_exo']
        prior_val = _prior(like, a_exo, b_exo)
        rejected  = not np.isfinite(prior_val)
        results.add(
            3, "Physicality rejection", rejected,
            f"prior(a=-5000, b=0) = {prior_val}"
            + ("  ← correctly -inf ✓" if rejected
               else "  ← should be -inf but got finite — polygon check broken!")
        )
    except Exception:
        results.add(3, "Physicality rejection", False,
                    f"CRASHED:\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════════════════════
#  TEST 4 — Unit consistency
# ══════════════════════════════════════════════════════════════════════════════

def test_4_units(like, results):
    print("\n" + "─" * 70)
    print("TEST 4 — Unit consistency: print theory vs observed rho_star")
    print("         side by side at each z-bin")
    print("─" * 70)
    try:
        from pipeline.hmf import compute_hmf
        from pipeline.stellar_mass_function import (
            compute_theory_rho_star, compute_observed_rho_star
        )
        from pipeline.data_extractor import UNCOVER_SKY_FRACTION

        cosmo = _build_cosmo(**_LCDM)

        dex_diffs  = []
        all_finite = True

        print(f"\n  {'z-bin':<12} {'M_star_thr':>14} "
              f"{'rho_theory':>16} {'rho_obs':>16} {'ratio (T/O)':>14}")
        print(f"  {'-'*12} {'-'*14} {'-'*16} {'-'*16} {'-'*14}")

        for (z_min, z_max, z_mid) in like._zbins:
            # Volume
            chi_lo = cosmo.comoving_distance(z_min)
            chi_hi = cosmo.comoving_distance(z_max)
            V = (4.0 / 3.0) * np.pi * (chi_hi**3 - chi_lo**3) * UNCOVER_SKY_FRACTION

            # Observed
            M_thr, rho_obs, rho_low, rho_high = compute_observed_rho_star(
                like._catalog, z_min, z_max, V
            )
            if len(M_thr) == 0:
                print(f"  [{z_min},{z_max})    no galaxies — skipped")
                continue

            # Theory
            M_h, dndlnm, _ = compute_hmf(cosmo, z_mid)
            rho_theory = compute_theory_rho_star(M_h, dndlnm, M_thr)

            # Sample every 10th galaxy to keep output manageable
            idx = np.round(np.linspace(0, len(M_thr)-1, 5)).astype(int)
            for i in idx:
                if rho_obs[i] > 0 and rho_theory[i] > 0:
                    ratio = rho_theory[i] / rho_obs[i]
                    dex   = np.log10(ratio)
                    dex_diffs.append(abs(dex))
                    finite_flag = "" if np.isfinite(dex) else " ← NaN!"
                    all_finite  = all_finite and np.isfinite(dex)
                    print(f"  [{z_min:.0f},{z_max:.0f})     "
                          f"{M_thr[i]:14.3e}  "
                          f"{rho_theory[i]:16.3e}  "
                          f"{rho_obs[i]:16.3e}  "
                          f"{ratio:14.3e}{finite_flag}")

        cosmo.struct_cleanup(); cosmo.empty()

        # Pass if median ratio is within 5 dex — units consistent
        # (5 dex is generous because theory and data genuinely disagree in ΛCDM)
        if dex_diffs:
            median_dex = float(np.median(dex_diffs))
            unit_ok    = median_dex < 5.0 and all_finite
            results.add(
                4, "Unit consistency", unit_ok,
                f"Median |log10(theory/obs)| = {median_dex:.2f} dex"
                + ("  ← same ballpark ✓" if unit_ok
                   else "  ← >5 dex gap suggests unit contamination (h^3?)!")
            )
        else:
            results.add(4, "Unit consistency", False,
                        "No valid galaxy-theory pairs found to compare.")

    except Exception:
        results.add(4, "Unit consistency", False,
                    f"CRASHED:\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════════════════════
#  TEST 5 — Likelihood surface scan
# ══════════════════════════════════════════════════════════════════════════════

def test_5_surface_scan(like, results, output_dir, n_points=20):
    print("\n" + "─" * 70)
    print(f"TEST 5 — Likelihood surface scan: a_exo in [0, -1500], b_exo=0")
    print(f"         ({n_points} CLASS calls — ~{n_points*3//60+1} min)")
    print("─" * 70)
    try:
        a_exo_grid = np.linspace(0.0, -1500.0, n_points)
        log_likes  = np.full(n_points, np.nan)
        t0 = time.time()

        for k, a_exo in enumerate(a_exo_grid):
            b_exo = 0.0
            s     = a_exo + b_exo
            # Check polygon before calling CLASS
            if a_exo < 0 and s < (_POLY_SLOPE * a_exo + _POLY_INTERCEPT):
                print(f"  [{k+1:02d}/{n_points}] a={a_exo:8.1f}  UNPHYSICAL — skipped")
                continue
            try:
                lp = _logp(like, a_exo, b_exo)
                log_likes[k] = lp
                print(f"  [{k+1:02d}/{n_points}] a={a_exo:8.1f}  "
                      f"lnL={lp:12.2f}  ({time.time()-t0:.0f}s)")
            except Exception as e:
                print(f"  [{k+1:02d}/{n_points}] a={a_exo:8.1f}  "
                      f"FAILED: {e}")

        finite     = np.isfinite(log_likes)
        n_finite   = finite.sum()
        lcdm_idx   = 0
        lcdm_ll    = log_likes[lcdm_idx] if np.isfinite(log_likes[lcdm_idx]) else np.nan

        # ── Plot ──────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(a_exo_grid[finite], log_likes[finite],
                'o-', color='steelblue', ms=5, lw=1.8,
                label=r'$\ln\mathcal{L}(a_{\rm exo})\;[b_{\rm exo}=0]$')
        if np.isfinite(lcdm_ll):
            ax.axhline(lcdm_ll, color='k', ls='--', lw=1.2,
                       label=rf'$\Lambda$CDM baseline $({lcdm_ll:.1f})$')

        # Mark best fit
        if n_finite > 0:
            best_idx = np.nanargmax(log_likes)
            ax.axvline(a_exo_grid[best_idx], color='crimson', ls=':', lw=1.5,
                       label=rf'Peak $a_{{\rm exo}}={a_exo_grid[best_idx]:.1f}$')
            ax.scatter(a_exo_grid[best_idx], log_likes[best_idx],
                       s=100, color='crimson', zorder=6)

        ax.set_xlabel(r'$a_{\rm exo}$', fontsize=12)
        ax.set_ylabel(r'$\ln\mathcal{L}$', fontsize=12)
        ax.set_title(r'Test 5 — Likelihood scan over $a_{\rm exo}$ '
                     r'$(b_{\rm exo}=0)$', fontsize=12)
        ax.legend(fontsize=9, framealpha=0.85)
        ax.grid(True, alpha=0.25, ls='--')
        plt.tight_layout()

        fig_path = os.path.join(output_dir, 'test5_likelihood_scan.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Plot saved → {fig_path}")

        # ── Pass criteria ─────────────────────────────────────────────────
        # Curve should NOT be monotonically decreasing (lnL not always
        # best at a=0) AND should not be flat (lnL must vary).
        if n_finite >= 3:
            ll_vals    = log_likes[finite]
            is_flat    = (np.nanmax(ll_vals) - np.nanmin(ll_vals)) < 1.0
            best_a     = a_exo_grid[np.nanargmax(log_likes)]
            passed     = (not is_flat) and n_finite >= n_points // 2
            results.add(
                5, "Likelihood surface scan", passed,
                f"n_finite={n_finite}/{n_points}  "
                f"best_a={best_a:.1f}  "
                f"range={np.nanmax(ll_vals)-np.nanmin(ll_vals):.1f}"
                + ("  ← curve varies, not flat ✓" if not is_flat
                   else "  ← FLAT — likelihood not seeing cosmology!")
            )
        else:
            results.add(5, "Likelihood surface scan", False,
                        f"Only {n_finite} finite evaluations — too many failures.")

    except Exception:
        results.add(5, "Likelihood surface scan", False,
                    f"CRASHED:\n{traceback.format_exc()}")


def test_6_bobyqa(phot_path, zspec_path, mode, output_dir, a_exo_mode, results, N):
    print("\n" + "─" * 70)
    print(f"TEST 6 — MINIMIZATION (Mode: {mode}, Exotic Test: {a_exo_mode})")
    print("─" * 70)
    
    output_path = os.path.join(output_dir, 'test6_minimize')
    if os.path.exists(output_path):
        import shutil
        for f in os.listdir(output_dir):
            if f.startswith('test6_minimize'):
                full_path = os.path.join(output_dir, f)
                if os.path.isdir(full_path):
                    shutil.rmtree(full_path)
                else:
                    os.remove(full_path)
                    
    try:
        from cobaya.run import run as cobaya_run
        import time # Ensure time is available
        
        # 1. Base Info Setup
        # Use bobyqa for spec-z (faster), scipy/Nelder-Mead for photo-z (more robust)
        sampler_method = 'bobyqa' if mode == 'spectroscopic' else 'scipy'
        
        info = {
            'likelihood': {
                'likelihood.jwst_likelihood.JWSTLikelihood': {
                    'mode': mode,
                    'vary_SHMR_params': False,
                    'phot_path': phot_path,
                    'zspec_path': zspec_path,
                    'python_path': _PROJECT_ROOT,
                }
            },
            'theory': {
                'classy': {
                    'extra_args': {
                        'z_c_exo': _Z_C_EXO,
                        'sigma_z_exo': _SIGMA_Z_EXO,
                        'output': 'mPk',
                        'P_k_max_1/Mpc': 510.0,
                        'z_max_pk': 20.0,
                        'non linear': 'none',
                    },
                    'ignore_obsolete': True,
                }
            },
            'params': {
                'a_samp': {
                    'prior': {'min': -1838.0, 'max': -1e-5},
                    'ref': {'dist': 'norm', 'loc': -20.0, 'scale': 10.0},
                    'drop': True,
                },
                's': {
                    'prior': {'min': -1381.597, 'max': -1e-5},
                    'ref': {'dist': 'norm', 'loc': -30.0, 'scale': 10.0},
                    'drop': True,
                },
                'a_exo': {'value': 'lambda a_samp: a_samp'},
                'b_exo': {'value': 'lambda a_samp, s: s - a_samp'},
                'H0': {'value': 67.36},
                'omega_b': {'value': 0.02237},
                'omega_cdm': {'value': 0.1200},
                'n_s': {'value': 0.9649},
                'logA': {'value': 3.044, 'drop': True},
                'A_s': {'value': 'lambda logA: 1e-10*np.exp(logA)'},
                'tau_reio': {'value': 0.0544},
                'Omega_m': {'derived': True}
            },
            'prior': {'h2_positivity': _COBAYA_PRIOR_LAMBDA},
            'sampler': {
                'minimize': {
                    'method': sampler_method,
                    'max_evals': 4500,
                }
            },
            'output': output_path,
            'debug': False,
        }

        # 2. OVERRIDE if a_exo_mode is True
        if a_exo_mode:
            info['params']['a_samp']['prior'] = {'min': -1838.0, 'max': 1e-4}
            
            print(f"  [INFO] a_exo_mode detected: max a_exo prior set to 1e-4.")

        # 3. Run
        t0 = time.time() # Start the clock!
        upd_info, sampler = cobaya_run(info, force=True)
        dt = time.time() - t0

        # 4. Extract (using the OnePoint bracket access)
        best = sampler.products()
        bf = best.get('minimum', {})
        print(bf)
        
        a_best = bf['a_exo']
        b_best = bf['b_exo']
        chi2_best = bf['chi2']

# 5. Validity Checks
        if not a_exo_mode:
            # Standard constraints for the physical polygon
            s_best = a_best + b_best
            in_poly = (s_best >= (_POLY_SLOPE * a_best + _POLY_INTERCEPT)
                        and s_best <= 0.0
                        and a_best < 0.0)
            passed = (a_best < 0.0) and in_poly and np.isfinite(chi2_best)
        else:
            # In exotic mode, we just care if it's finite and successfully ran
            in_poly = "N/A (Test Mode)"
            passed = np.isfinite(chi2_best)
            
        P = 2 # a_samp and s
        red_chi2_best = chi2_best / (N - P)


        results.add(
            6, "Minimization", passed,
            f" chi2={chi2_best:.1f} red_chi2={red_chi2_best:.3f} "
            f"a_exo={a_best:.1f} b_exo={b_best:.1f} (N={N}, P={P})"
        )
    except Exception:
        results.add(6, "Minimization", False, f"CRASHED:\n{traceback.format_exc()}")

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Six-test likelihood validation before Run 1.'
    )
    parser.add_argument(
        "--phot_path",
        help="Path to UNCOVER_DR4_SPS_catalog.fits",
        default='/home/lustre_p/ahmed.omar/workspace/exo_de_project/data/UNCOVER_DR4_SPS_catalog.fits'
    )
    parser.add_argument(
        "--zspec_path",
        help="Path to UNCOVER_DR4_SPS_zspec_catalog.fits",
        default='/home/lustre_p/ahmed.omar/workspace/exo_de_project/data/UNCOVER_DR4_SPS_zspec_catalog.fits'
    )
    parser.add_argument('--mode', default='spectroscopic',
                        choices=['spectroscopic', 'photometric'])
    parser.add_argument('--output_dir', default='likelihood_test_outputs')
    parser.add_argument('--tests', nargs='+', default=['all'],
                        help='Which tests to run: 1 2 3 4 5 6 or all')
    parser.add_argument('--n_scan', type=int, default=20,
                        help='Number of a_exo points for Test 5')
    parser.add_argument('--a_exo_tester', default='False',
                       choices=['False','True'], help= 'if False, a_exo prior < 0, if True the prior goes up to a_exo <= 10e-5 to see if the model prefers a truly zero a_exo')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    run_all = 'all' in args.tests
    run     = lambda n: run_all or str(n) in args.tests

    print("═" * 70)
    print("  JWST LIKELIHOOD — PRE-LAUNCH VALIDATION SUITE")
    print(f"  mode={args.mode}  phot={args.phot_path}")
    print("═" * 70)

    results = _Results()

    # ── Build likelihood once (shared by Tests 1-5) ───────────────────────
    print("\nBuilding likelihood (initialize)...")
    t0   = time.time()
    like = _build_likelihood(args.phot_path, args.zspec_path, args.mode)
    print(f"  initialize() done in {time.time()-t0:.1f}s")
    
    N_data = _get_total_n_data(like)
    print(f"  [INFO] Total data points (N) = {N_data}")

    # ── Tests 1-5: standalone ─────────────────────────────────────────────
    if run(1): test_1_runs(like, results, N_data)
    if run(2): test_2_sign_check(like, results)
    if run(3): test_3_physicality(like, results)
    if run(4): test_4_units(like, results)
    if run(5): test_5_surface_scan(like, results, args.output_dir, args.n_scan)

    # ── Test 6: needs real Cobaya ─────────────────────────────────────────
    if run(6):
        test_6_bobyqa(
            args.phot_path, args.zspec_path,
            args.mode, args.output_dir, args.a_exo_tester, results, N_data)

    # ── Final verdict ─────────────────────────────────────────────────────
    results.summary()


if __name__ == '__main__':
    main()