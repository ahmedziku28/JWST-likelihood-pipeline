#!/usr/bin/env python3
"""
diagnostic_grid_scan.py

Brute-force χ² grid scan over a_exo + per-bin diagnostic breakdown.
Answers: did the minimiser miss a deep minimum at large |a_exo|?

Place in tests/ alongside test_likelihood_differential.py.

Usage:
    python diagnostic_grid_scan.py \
        --phot_path  data/UNCOVER_DR4_SPS_catalog.fits \
        --zspec_path data/UNCOVER_DR4_SPS_zspec_catalog.fits

Expected runtime: ~3-5 minutes (one CLASS call per grid point).
"""

import argparse
import os
import sys
import time
import numpy as np

# ── Path setup (identical to test_likelihood_differential.py) ────────────────
_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from classy import Class

from pipeline.hmf import compute_hmf, compute_cosmic_variance
from pipeline.stellar_mass_function import shmr_mstar
from pipeline.differential_smf import (
    DEFAULT_LOG10_MSTAR_BINS,
    compute_theory_differential_smf,
    compute_theory_differential_rho,
    compute_observed_differential_smf,
    compute_observed_differential_rho,
)
from pipeline.data_extractor import load_catalogs, UNCOVER_SKY_FRACTION


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

_PLANCK = {
    'h': 0.6736, 'omega_b': 0.02237, 'omega_cdm': 0.1200,
    'n_s': 0.9649, 'A_s': 2.101e-9, 'tau_reio': 0.0544,
    'output': 'mPk', 'P_k_max_1/Mpc': 510.0, 'z_max_pk': 20.0,
    'non linear': 'none', 'z_c_exo': 16.0, 'sigma_z_exo': 3.25,
}

_POLY_SLOPE     = -0.07202
_POLY_INTERCEPT = -1381.5969

ZBINS_SPEC = [
#     (6.0, 8.0, 7.0), 
              (8.0, 10.0, 9.0),
              (10.0, 15.0, 12.5)]
ZBINS_PHOT = [
#     (6.0, 8.0, 7.0),
              (8.0, 10.0, 9.0),
              (10.0, 15.0, 12.5),
              (15.0, 20.0, 17.5)]


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def build_cosmo(a_exo, b_exo):
    p = dict(_PLANCK)
    p['a_exo'] = a_exo
    p['b_exo'] = b_exo
    cosmo = Class()
    cosmo.set(p)
    cosmo.compute()
    return cosmo


def V_survey(cosmo, z_min, z_max):
    chi_min = cosmo.comoving_distance(z_min)
    chi_max = cosmo.comoving_distance(z_max)
    return (4.0 / 3.0) * np.pi * (chi_max**3 - chi_min**3) * UNCOVER_SKY_FRACTION


def is_physical(a_exo, b_exo):
    """Check all physicality constraints in (a_exo, b_exo) space."""
    if a_exo >= 0.0:
        return False
    s = a_exo + b_exo
    if s >= 0.0:
        return False
    if s < _POLY_SLOPE * a_exo + _POLY_INTERCEPT:
        return False
    return True


def physical_b_exo(a_exo):
    """
    Return b_exo = 0 if physical, otherwise the minimum physical b_exo
    with a 5% safety margin above the polygon floor.
    """
    b_floor = -1.07202 * a_exo - 1381.5969      # polygon bottom in (a, b) space
    if b_floor <= 0.0:
        return 0.0                                # b=0 is safely inside
    return b_floor * 1.05                         # 5% above the hard edge


# ══════════════════════════════════════════════════════════════════════════════
#  CORE: compute chi² with full per-bin breakdown
# ══════════════════════════════════════════════════════════════════════════════

def compute_chi2(cosmo, catalog, zbins, observable='smf'):
    """
    Evaluate differential chi² exactly as logp() does, returning the
    per-bin breakdown.

    Returns
    -------
    total_chi2 : float
    details    : list[dict]   one entry per non-empty bin
    """
    edges     = DEFAULT_LOG10_MSTAR_BINS
    total_chi2 = 0.0
    details    = []

    for z_min, z_max, z_mid in zbins:

        V = V_survey(cosmo, z_min, z_max)
        if not np.isfinite(V) or V <= 0.0:
            continue

        M_h, dndlnm, hmf_sigma, Pk = compute_hmf(cosmo, z_mid)

        # ── Theory ────────────────────────────────────────────────────
        if observable == 'smf':
            _, pred = compute_theory_differential_smf(M_h, dndlnm, edges)
        else:
            _, pred = compute_theory_differential_rho(M_h, dndlnm, edges)

        # ── Observation ───────────────────────────────────────────────
        if observable == 'smf':
            _, obs, sigma_poi, sigma_mass, n_gal = \
                compute_observed_differential_smf(catalog, z_min, z_max, V, edges)
        else:
            _, obs, sigma_poi, sigma_mass, n_gal = \
                compute_observed_differential_rho(catalog, z_min, z_max, V, edges)

        # ── Cosmic variance ───────────────────────────────────────────
        sigma_cv = compute_cosmic_variance(
            Pk, M_h, dndlnm, hmf_sigma, V,
            edges, shmr_mstar,
        )

        sigma_tot = np.sqrt(sigma_poi**2 + sigma_mass**2 + (sigma_cv * obs)**2)
        good = (n_gal > 0) & np.isfinite(sigma_tot) & (sigma_tot > 0.0)

        bin_centers = 0.5 * (edges[:-1] + edges[1:])

        for i in range(len(bin_centers)):
            if not good[i]:
                continue
            chi2_i = ((pred[i] - obs[i]) / sigma_tot[i]) ** 2
            total_chi2 += chi2_i
            details.append(dict(
                z_bin     = f'{z_min:.0f}-{z_max:.0f}',
                z_mid     = z_mid,
                logM      = bin_centers[i],
                n_gal     = int(n_gal[i]),
                obs       = obs[i],
                pred      = pred[i],
                ratio     = pred[i] / obs[i] if obs[i] > 0 else np.inf,
                sig_poi   = sigma_poi[i],
                sig_mass  = sigma_mass[i],
                sig_cv_f  = sigma_cv[i],
                sig_cv_a  = sigma_cv[i] * obs[i],
                sig_tot   = sigma_tot[i],
                pull      = (pred[i] - obs[i]) / sigma_tot[i],
                chi2_i    = chi2_i,
            ))

    return total_chi2, details


# ══════════════════════════════════════════════════════════════════════════════
#  PRETTY PRINTERS
# ══════════════════════════════════════════════════════════════════════════════

_HDR = (
    f'  {"z":>5} {"lgM*":>5} {"Ngal":>5} '
    f'{"obs":>11} {"pred":>11} {"pred/obs":>8} '
    f'{"σ_poi":>9} {"σ_mass":>9} {"σ_cv·ϕ":>9} {"σ_tot":>9} '
    f'{"pull":>7} {"χ²_i":>9}'
)

_SEP = (
    f'  {"─"*5} {"─"*5} {"─"*5} '
    f'{"─"*11} {"─"*11} {"─"*8} '
    f'{"─"*9} {"─"*9} {"─"*9} {"─"*9} '
    f'{"─"*7} {"─"*9}'
)


def print_details(details, label):
    print(f'\n{"─"*130}')
    print(f'  {label}')
    print(f'{"─"*130}')
    print(_HDR)
    print(_SEP)

    cur_zbin  = None
    zbin_chi2 = 0.0

    for d in details:
        if d['z_bin'] != cur_zbin:
            if cur_zbin is not None:
                print(f'  {"":>5} {"":>5} {"":>5} '
                      f'{"":>11} {"":>11} {"":>8} '
                      f'{"":>9} {"":>9} {"":>9} {"":>9} '
                      f'{"":>7} {"Σ=":>3}{zbin_chi2:>6.1f}')
                print()
            cur_zbin  = d['z_bin']
            zbin_chi2 = 0.0

        zbin_chi2 += d['chi2_i']

        print(
            f'  {d["z_bin"]:>5} {d["logM"]:>5.1f} {d["n_gal"]:>5d} '
            f'{d["obs"]:>11.3e} {d["pred"]:>11.3e} {d["ratio"]:>8.2f} '
            f'{d["sig_poi"]:>9.2e} {d["sig_mass"]:>9.2e} '
            f'{d["sig_cv_a"]:>9.2e} {d["sig_tot"]:>9.2e} '
            f'{d["pull"]:>+7.1f} {d["chi2_i"]:>9.1f}'
        )

    # last z-bin subtotal
    if cur_zbin is not None:
        print(f'  {"":>5} {"":>5} {"":>5} '
              f'{"":>11} {"":>11} {"":>8} '
              f'{"":>9} {"":>9} {"":>9} {"":>9} '
              f'{"":>7} {"Σ=":>3}{zbin_chi2:>6.1f}')

    n  = len(details)
    tt = sum(d['chi2_i'] for d in details)
    print(f'\n  TOTAL  χ² = {tt:.1f}   over {n} non-empty bins   '
          f'(χ²/bin = {tt / max(n, 1):.1f})\n')


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description='Diagnostic χ² grid scan')
    ap.add_argument('--phot_path',
        default='/home/lustre_p/ahmed.omar/workspace/exo_de_project/'
                'data/UNCOVER_DR4_SPS_catalog.fits')
    ap.add_argument('--zspec_path',
        default='/home/lustre_p/ahmed.omar/workspace/exo_de_project/'
                'data/UNCOVER_DR4_SPS_zspec_catalog.fits')
    ap.add_argument('--observable', default='smf', choices=['smf', 'rho'],
        help='Differential observable (default: smf)')
    args = ap.parse_args()

    print('═' * 80)
    print('  DIAGNOSTIC GRID SCAN — χ² surface + per-bin breakdown')
    print(f'  observable: {args.observable}')
    print('═' * 80)

    # ── Load catalogs ─────────────────────────────────────────────────────
    phot_table, spec_table = load_catalogs(args.phot_path, args.zspec_path)

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 1 — 1D χ² SCAN along a_exo  (b_exo at minimum physical value)
    # ══════════════════════════════════════════════════════════════════════
    #
    # This maps χ²(amplitude) from near-ΛCDM to maximum allowed.
    # One CLASS run per point → ~3s each → ~1 min total.
    #
    a_grid = np.array([
        -1e-5, -1.0, -5.0, -10.0, -20.0, -50.0,
        -100.0, -200.0, -300.0, -500.0, -700.0,
        -1000.0, -1200.0, -1400.0, -1600.0, -1800.0,
    ])

    print('\n' + '═' * 80)
    print('  STEP 1: 1D χ² SCAN (a_exo sweep, b_exo = 0 or polygon floor)')
    print('═' * 80)
    print(f'\n  {"a_exo":>10} {"b_exo":>10} {"Ω_x0":>12} '
          f'{"χ²_spec":>10} {"Δ_spec":>10} '
          f'{"χ²_phot":>10} {"Δ_phot":>10} {"time":>6}')
    print(f'  {"─"*10} {"─"*10} {"─"*12} '
          f'{"─"*10} {"─"*10} '
          f'{"─"*10} {"─"*10} {"─"*6}')

    # ── ΛCDM reference ────────────────────────────────────────────────────
    cosmo0 = build_cosmo(0.0, 0.0)
    chi2_s0, _ = compute_chi2(cosmo0, spec_table, ZBINS_SPEC, args.observable)
    chi2_p0, _ = compute_chi2(cosmo0, phot_table, ZBINS_PHOT, args.observable)

    print(f'  {"ΛCDM":>10} {"0":>10} {"0":>12} '
          f'{chi2_s0:>10.1f} {"ref":>10} '
          f'{chi2_p0:>10.1f} {"ref":>10}')

    scan = []   # (a, b, chi2_spec, chi2_phot)

    for a in a_grid:
        b = physical_b_exo(a)
        if not is_physical(a, b):
            print(f'  {a:>10.1f}   *** outside polygon — skipped ***')
            continue
        try:
            t0 = time.time()
            cosmo = build_cosmo(a, b)
            c2s, _ = compute_chi2(cosmo, spec_table, ZBINS_SPEC, args.observable)
            c2p, _ = compute_chi2(cosmo, phot_table, ZBINS_PHOT, args.observable)
            dt  = time.time() - t0

            omega_x0 = 5.4366e-6 * a
            ds = c2s - chi2_s0
            dp = c2p - chi2_p0

            print(f'  {a:>10.1f} {b:>10.2f} {omega_x0:>12.4e} '
                  f'{c2s:>10.1f} {ds:>+10.1f} '
                  f'{c2p:>10.1f} {dp:>+10.1f} {dt:>5.1f}s')

            scan.append((a, b, c2s, c2p))
            cosmo.struct_cleanup(); cosmo.empty()

        except Exception as e:
            print(f'  {a:>10.1f} {b:>10.2f}   *** CLASS crashed: {e} ***')

    # ── Summary of scan ───────────────────────────────────────────────────
    if scan:
        best_s = min(scan, key=lambda x: x[2])
        best_p = min(scan, key=lambda x: x[3])
        print(f'\n  Best SPEC:  a_exo={best_s[0]:.1f}  b_exo={best_s[1]:.1f}  '
              f'χ²={best_s[2]:.1f}  (Δ={best_s[2]-chi2_s0:+.1f})')
        print(f'  Best PHOT:  a_exo={best_p[0]:.1f}  b_exo={best_p[1]:.1f}  '
              f'χ²={best_p[3]:.1f}  (Δ={best_p[3]-chi2_p0:+.1f})')

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 2 — PER-BIN DIAGNOSTIC at ΛCDM, moderate, strong amplitudes
    # ══════════════════════════════════════════════════════════════════════

    print('\n\n' + '═' * 80)
    print('  STEP 2: PER-BIN χ² BREAKDOWN')
    print('═' * 80)

    diag_points = [
        (0.0,    0.0,                        'ΛCDM (a=0, b=0)'),
        (-500.0, physical_b_exo(-500.0),     'MODERATE (a=-500)'),
        (-1000.0, physical_b_exo(-1000.0),   'STRONG (a=-1000)'),
        (-1500.0, physical_b_exo(-1500.0),   'VERY STRONG (a=-1500)'),
    ]

    # If scan found a clear best, add that too
    if scan:
        a_bs, b_bs = best_s[0], best_s[1]
        if abs(a_bs) > 1.0:   # skip if it's essentially ΛCDM
            diag_points.append((a_bs, b_bs, f'BEST-FIT SPEC (a={a_bs:.0f})'))

    for a, b, label in diag_points:
        if not is_physical(a, b) and (a != 0.0):
            print(f'\n  *** {label} — not physical, skipped ***')
            continue
        try:
            cosmo = build_cosmo(a, b)

            for mode_name, catalog, zbins in [
                ('SPEC', spec_table, ZBINS_SPEC),
                ('PHOT', phot_table, ZBINS_PHOT),
            ]:
                _, details = compute_chi2(
                    cosmo, catalog, zbins, args.observable,
                )
                print_details(details, f'{label}  |  {mode_name}  |  {args.observable}')

            cosmo.struct_cleanup(); cosmo.empty()

        except Exception as e:
            print(f'\n  *** {label} — crashed: {e} ***')

    # ── cleanup ───────────────────────────────────────────────────────────
    cosmo0.struct_cleanup(); cosmo0.empty()
    print('\nDone.')


if __name__ == '__main__':
    main()
