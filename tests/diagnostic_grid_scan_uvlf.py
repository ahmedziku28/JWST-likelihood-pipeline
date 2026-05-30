#!/usr/bin/env python3
"""
diagnostic_grid_scan_uvlf.py

Brute-force chi-squared grid scan over a_exo for the UV luminosity function
(UVLF) likelihood, with a full per-bin diagnostic breakdown.

Answers:
    - Does the UVLF chi2 decrease for exotic DE? Is there a valley?
    - Which redshift / magnitude bins drive chi2?
    - Is the improvement dominated by Donnan or Finkelstein?
    - Does b_exo matter at fixed amplitude?

Data: Donnan et al. 2024 (PRIMER, MNRAS 533) + Finkelstein et al. 2024
      (CEERS, ApJL 969). Both hardcoded in pipeline/uvlf_theory.py.
      No external catalog files needed.

Usage (from project root):
    python diagnostic_grid_scan_uvlf.py

    # With redshift bin integration instead of single-z evaluation:
    python diagnostic_grid_scan_uvlf.py --integrate

Expected runtime: ~2-3 min  (one cosmo.compute() + 8 compute_hmf calls
                               per grid point; 16 grid points).
"""

import argparse
import os
import sys
import time
import numpy as np

# ── Path setup (identical to diagnostic_grid_scan.py) ────────────────────────
_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from classy import Class

from pipeline.hmf import compute_hmf

# NOTE: if your file is named uvlf.py rather than uvlf_theory.py, adjust here.
from pipeline.uvlf import (
    load_donnan,
    load_finkelstein,
    compute_uvlf_theory,
    chi_squared,
    DONNAN_Z_EDGES,
    FINKELSTEIN_Z_EDGES,
)


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# Planck 2018 base cosmology + modified CLASS exotic-DE fields.
# Identical to diagnostic_grid_scan.py — do not change.
_PLANCK = {
    'h'             : 0.6736,
    'omega_b'       : 0.02237,
    'omega_cdm'     : 0.1200,
    'n_s'           : 0.9649,
    'A_s'           : 2.101e-9,
    'tau_reio'      : 0.0544,
    'output'        : 'mPk',
    'P_k_max_1/Mpc' : 510.0,
    'z_max_pk'      : 20.0,
    'non linear'    : 'none',
    'z_c_exo'       : 16.0,
    'sigma_z_exo'   : 3.25,
}

# Prior polygon bottom edge in (a_exo, b_exo) space.
# b_exo >= _POLY_SLOPE * a_exo + _POLY_INTERCEPT  (H2 > 0 constraint).
_POLY_SLOPE     = -0.07202
_POLY_INTERCEPT = -1381.5969

# 1-D scan grid — same values as the SMF diagnostic for direct comparison.
_A_GRID = np.array([
    -1e-5, -1.0, -5.0, -10.0, -20.0, -50.0,
    -100.0, -200.0, -300.0, -500.0, -700.0,
    -1000.0, -1200.0, -1400.0, -1600.0, -1800.0,
])


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS  (identical logic to diagnostic_grid_scan.py)
# ══════════════════════════════════════════════════════════════════════════════

def build_cosmo(a_exo, b_exo):
    """Initialise and compute a CLASS instance at (a_exo, b_exo)."""
    p = dict(_PLANCK)
    p['a_exo'] = a_exo
    p['b_exo'] = b_exo
    cosmo = Class()
    cosmo.set(p)
    cosmo.compute()
    return cosmo


def is_physical(a_exo, b_exo):
    """Return True if (a_exo, b_exo) satisfies all prior polygon constraints."""
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
    Return b_exo = 0 if it is inside the polygon, otherwise the polygon floor
    plus a 5% safety margin.
    """
    b_floor = -1.07202 * a_exo - 1381.5969
    if b_floor <= 0.0:
        return 0.0
    return b_floor * 1.05


# ══════════════════════════════════════════════════════════════════════════════
#  CORE: compute chi2 with full per-bin breakdown
# ══════════════════════════════════════════════════════════════════════════════

def compute_uvlf_chi2(cosmo, donnan_data, finkelstein_data,
                       integrate_bin=False, n_gl=2):
    """
    Evaluate the UVLF chi2 exactly as compute_total_chi2() does, but also
    return the per-bin breakdown needed for the diagnostic tables.

    Parameters
    ----------
    cosmo             : classy.Class — current cosmology (already computed)
    donnan_data       : structured np.ndarray from load_donnan()
    finkelstein_data  : structured np.ndarray from load_finkelstein()
    integrate_bin     : bool — use GL redshift integration (default False)
    n_gl              : int  — GL nodes if integrate_bin=True (default 2)

    Returns
    -------
    chi2_don  : float — Donnan-only chi2
    chi2_fink : float — Finkelstein-only chi2
    details   : list[dict] — one entry per magnitude bin, both datasets
    """
    chi2_don  = 0.0
    chi2_fink = 0.0
    details   = []

    # ── Donnan 2024 — 5 redshift bins ─────────────────────────────────────
    for z_nom in np.unique(donnan_data['z']):

        mask       = donnan_data['z'] == z_nom
        M_UV_bins  = donnan_data['M_UV'][mask]
        phi_obs    = donnan_data['phi'][mask]
        sigma_up   = donnan_data['sigma_up'][mask]
        sigma_down = donnan_data['sigma_down'][mask]

        M_h, dndlnm, _, _ = compute_hmf(cosmo, z_nom)

        z_lo, z_hi = DONNAN_Z_EDGES[z_nom]
        phi_theory  = compute_uvlf_theory(
            M_h, dndlnm, z_nom, M_UV_bins,
            integrate_bin=integrate_bin,
            z_lo=z_lo, z_hi=z_hi,
            cosmo=cosmo, n_gl=n_gl,
        )

        for i in range(len(M_UV_bins)):
            overpred   = phi_theory[i] > phi_obs[i]
            sigma_used = sigma_up[i] if overpred else sigma_down[i]
            pull       = (phi_theory[i] - phi_obs[i]) / sigma_used
            chi2_i     = pull ** 2
            chi2_don  += chi2_i

            details.append(dict(
                dataset    = 'Donnan',
                z_nom      = z_nom,
                M_UV       = M_UV_bins[i],
                phi_obs    = phi_obs[i],
                phi_theory = phi_theory[i],
                ratio      = phi_theory[i] / phi_obs[i],
                sigma_up   = sigma_up[i],
                sigma_down = sigma_down[i],
                sigma_used = sigma_used,
                pull       = pull,
                chi2_i     = chi2_i,
            ))

    # ── Finkelstein 2024 — 3 redshift bins ────────────────────────────────
    for z_nom in np.unique(finkelstein_data['z']):

        mask       = finkelstein_data['z'] == z_nom
        M_UV_bins  = finkelstein_data['M_UV'][mask]
        phi_obs    = finkelstein_data['phi'][mask]
        sigma_up   = finkelstein_data['sigma_up'][mask]
        sigma_down = finkelstein_data['sigma_down'][mask]

        M_h, dndlnm, _, _ = compute_hmf(cosmo, z_nom)

        z_lo, z_hi = FINKELSTEIN_Z_EDGES[z_nom]
        phi_theory  = compute_uvlf_theory(
            M_h, dndlnm, z_nom, M_UV_bins,
            integrate_bin=integrate_bin,
            z_lo=z_lo, z_hi=z_hi,
            cosmo=cosmo, n_gl=n_gl,
        )

        for i in range(len(M_UV_bins)):
            overpred   = phi_theory[i] > phi_obs[i]
            sigma_used = sigma_up[i] if overpred else sigma_down[i]
            pull       = (phi_theory[i] - phi_obs[i]) / sigma_used
            chi2_i     = pull ** 2
            chi2_fink += chi2_i

            details.append(dict(
                dataset    = 'Finkelstein',
                z_nom      = z_nom,
                M_UV       = M_UV_bins[i],
                phi_obs    = phi_obs[i],
                phi_theory = phi_theory[i],
                ratio      = phi_theory[i] / phi_obs[i],
                sigma_up   = sigma_up[i],
                sigma_down = sigma_down[i],
                sigma_used = sigma_used,
                pull       = pull,
                chi2_i     = chi2_i,
            ))

    return chi2_don, chi2_fink, details


# ══════════════════════════════════════════════════════════════════════════════
#  PRETTY PRINTERS
# ══════════════════════════════════════════════════════════════════════════════

# Per-bin breakdown table header and separator.
_HDR = (
    f'  {"dataset":>11} {"z":>6} {"M_UV":>7} '
    f'{"phi_obs":>11} {"phi_thy":>11} {"thy/obs":>8} '
    f'{"sigma_up":>9} {"sig_dn":>9} {"sig_use":>9} '
    f'{"pull":>7} {"chi2_i":>9}'
)
_SEP = (
    f'  {"─"*11} {"─"*6} {"─"*7} '
    f'{"─"*11} {"─"*11} {"─"*8} '
    f'{"─"*9} {"─"*9} {"─"*9} '
    f'{"─"*7} {"─"*9}'
)
_WIDTH = 115


def print_details(details, label):
    """Print the full per-bin chi2 breakdown for one cosmology."""
    print(f'\n{"─"*_WIDTH}')
    print(f'  {label}')
    print(f'{"─"*_WIDTH}')
    print(_HDR)
    print(_SEP)

    cur_dataset = None
    cur_zbin    = None
    zbin_chi2   = 0.0
    don_total   = 0.0
    fink_total  = 0.0

    for d in details:

        # ── Dataset-level separator ────────────────────────────────────────
        if d['dataset'] != cur_dataset:
            if cur_zbin is not None:
                # flush last z-bin subtotal of previous dataset
                _print_zbin_subtotal(zbin_chi2)
                print()
            if cur_dataset is not None:
                # flush dataset total
                ds_tot = don_total if cur_dataset == 'Donnan' else fink_total
                print(f'  {"":>11} {"":>6} {"":>7} {"":>11} {"":>11} {"":>8} '
                      f'{"":>9} {"":>9} {"":>9} {"":>7} '
                      f'{"■ " + cur_dataset + " total =":>9} {ds_tot:>6.1f}')
                print()

            cur_dataset = d['dataset']
            cur_zbin    = None
            zbin_chi2   = 0.0
            print(f'  ── {cur_dataset} ──')

        # ── z-bin separator ───────────────────────────────────────────────
        if d['z_nom'] != cur_zbin:
            if cur_zbin is not None:
                _print_zbin_subtotal(zbin_chi2)
                print()
            cur_zbin  = d['z_nom']
            zbin_chi2 = 0.0

        zbin_chi2 += d['chi2_i']
        if d['dataset'] == 'Donnan':
            don_total  += d['chi2_i']
        else:
            fink_total += d['chi2_i']

        # overprediction flag: * if theory > obs
        flag = '*' if d['phi_theory'] > d['phi_obs'] else ' '

        print(
            f'  {d["dataset"]:>11} {d["z_nom"]:>6.1f} {d["M_UV"]:>7.2f} '
            f'{d["phi_obs"]:>11.3e} {d["phi_theory"]:>11.3e}{flag} {d["ratio"]:>7.3f} '
            f'{d["sigma_up"]:>9.2e} {d["sigma_down"]:>9.2e} {d["sigma_used"]:>9.2e} '
            f'{d["pull"]:>+7.2f} {d["chi2_i"]:>9.2f}'
        )

    # ── Flush final z-bin and dataset ─────────────────────────────────────
    if cur_zbin is not None:
        _print_zbin_subtotal(zbin_chi2)

    if cur_dataset is not None:
        ds_tot = don_total if cur_dataset == 'Donnan' else fink_total
        print(f'\n  {"":>11} {"":>6} {"":>7} {"":>11} {"":>11} {"":>8} '
              f'{"":>9} {"":>9} {"":>9} {"":>7} '
              f'{"■ " + cur_dataset + " total =":>9} {ds_tot:>6.1f}')

    # ── Grand total ───────────────────────────────────────────────────────
    n       = len(details)
    total   = don_total + fink_total
    print(f'\n  {"─"*_WIDTH}')
    print(f'  Donnan chi2     : {don_total:>8.2f}   ({len([d for d in details if d["dataset"]=="Donnan"])} bins)')
    print(f'  Finkelstein chi2: {fink_total:>8.2f}   ({len([d for d in details if d["dataset"]=="Finkelstein"])} bins)')
    print(f'  TOTAL chi2      : {total:>8.2f}   over {n} bins   (chi2/bin = {total/max(n,1):.2f})')
    print(f'  (* = theory overpredicts → using sigma_up)')
    print()


def _print_zbin_subtotal(zbin_chi2):
    """Print the subtotal line for a completed z-bin."""
    print(
        f'  {"":>11} {"":>6} {"":>7} {"":>11} {"":>11} {"":>8} '
        f'{"":>9} {"":>9} {"":>9} {"":>7} '
        f'{"Σ=":>3} {zbin_chi2:>6.2f}'
    )


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description='UVLF diagnostic chi2 grid scan')
    ap.add_argument(
        '--integrate', action='store_true',
        help='Use GL redshift bin integration instead of single-z (slower)',
    )
    ap.add_argument(
        '--n_gl', type=int, default=2,
        help='GL nodes per bin when --integrate is set (default: 2)',
    )
    args = ap.parse_args()

    print('═' * _WIDTH)
    print('  UVLF DIAGNOSTIC GRID SCAN — chi2 surface + per-bin breakdown')
    print(f'  Data: Donnan 2024 (29 pts, 5 z-bins) + Finkelstein 2024 (11 pts, 3 z-bins)')
    print(f'  Mode: {"GL-integrated (n_gl=" + str(args.n_gl) + ")" if args.integrate else "single-z (default)"}')
    print('═' * _WIDTH)

    # ── Load data once — reused at every grid point ───────────────────────
    print('\n  Loading data...')
    donnan_data      = load_donnan()
    finkelstein_data = load_finkelstein()
    print(f'  Donnan:      {len(donnan_data)} rows across '
          f'{len(np.unique(donnan_data["z"]))} z-bins')
    print(f'  Finkelstein: {len(finkelstein_data)} rows across '
          f'{len(np.unique(finkelstein_data["z"]))} z-bins')

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 1 — 1D chi2 SCAN along a_exo  (b_exo at minimum physical value)
    # ══════════════════════════════════════════════════════════════════════

    print('\n' + '═' * _WIDTH)
    print('  STEP 1: 1D chi2 SCAN (a_exo sweep, b_exo = 0 or polygon floor)')
    print('═' * _WIDTH)
    print(f'\n  {"a_exo":>10} {"b_exo":>10} {"Omega_x0":>12} '
          f'{"chi2_don":>10} {"chi2_fink":>10} {"chi2_tot":>10} '
          f'{"Delta_tot":>10} {"time":>6}')
    print(f'  {"─"*10} {"─"*10} {"─"*12} '
          f'{"─"*10} {"─"*10} {"─"*10} '
          f'{"─"*10} {"─"*6}')

    # ── ΛCDM reference ────────────────────────────────────────────────────
    print('  Computing ΛCDM reference...', end='', flush=True)
    t0     = time.time()
    cosmo0 = build_cosmo(0.0, 0.0)
    c2d0, c2f0, _ = compute_uvlf_chi2(
        cosmo0, donnan_data, finkelstein_data,
        integrate_bin=args.integrate, n_gl=args.n_gl,
    )
    c2t0 = c2d0 + c2f0
    dt0  = time.time() - t0
    print(f' done ({dt0:.1f}s)')

    print(f'  {"ΛCDM":>10} {"0":>10} {"0":>12} '
          f'{c2d0:>10.1f} {c2f0:>10.1f} {c2t0:>10.1f} '
          f'{"ref":>10} {dt0:>5.1f}s')

    scan = []   # (a, b, chi2_don, chi2_fink, chi2_total)

    for a in _A_GRID:
        b = physical_b_exo(a)
        if not is_physical(a, b):
            print(f'  {a:>10.1e} {"":>10}   *** outside prior polygon — skipped ***')
            continue
        try:
            t0    = time.time()
            cosmo = build_cosmo(a, b)
            c2d, c2f, _ = compute_uvlf_chi2(
                cosmo, donnan_data, finkelstein_data,
                integrate_bin=args.integrate, n_gl=args.n_gl,
            )
            c2t   = c2d + c2f
            dt    = time.time() - t0

            omega_x0 = 5.4366e-6 * a
            delta    = c2t - c2t0

            print(f'  {a:>10.1f} {b:>10.2f} {omega_x0:>12.4e} '
                  f'{c2d:>10.1f} {c2f:>10.1f} {c2t:>10.1f} '
                  f'{delta:>+10.1f} {dt:>5.1f}s')

            scan.append((a, b, c2d, c2f, c2t))
            cosmo.struct_cleanup()
            cosmo.empty()

        except Exception as e:
            print(f'  {a:>10.1f} {b:>10.2f}   *** CLASS crashed: {e} ***')

    # ── Scan summary ──────────────────────────────────────────────────────
    if scan:
        best = min(scan, key=lambda x: x[4])
        print(f'\n  Best overall:  a_exo = {best[0]:.1f}  b_exo = {best[1]:.2f}'
              f'   chi2 = {best[4]:.1f}  (Delta = {best[4]-c2t0:+.1f})')
        print(f'                 chi2_Donnan = {best[2]:.1f}  '
              f'chi2_Finkelstein = {best[3]:.1f}')

        # Check if chi2 is monotone increasing — no valley
        if all(scan[i][4] >= scan[i-1][4] for i in range(1, len(scan))):
            print('\n  *** WARNING: chi2 is monotonically increasing — no valley found.')
            print('  *** Exotic DE worsens the UVLF fit at all amplitudes.')
            print('  *** Inspect per-bin breakdown (Step 2) to identify dominant bins.')
        else:
            print('\n  *** chi2 valley detected — a_exo < 0 improves the fit.')

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 2 — PER-BIN DIAGNOSTIC at key cosmologies
    # ══════════════════════════════════════════════════════════════════════

    print('\n\n' + '═' * _WIDTH)
    print('  STEP 2: PER-BIN chi2 BREAKDOWN')
    print('  (* after phi_theory = theory overpredicts → sigma_up used)')
    print('═' * _WIDTH)

    diag_points = [
        (0.0,     0.0,                      'ΛCDM  (a=0, b=0)'),
        (-50.0,   physical_b_exo(-50.0),    'WEAK  (a=-50)'),
        (-500.0,  physical_b_exo(-500.0),   'MODERATE  (a=-500)'),
        (-1000.0, physical_b_exo(-1000.0),  'STRONG  (a=-1000)'),
        (-1500.0, physical_b_exo(-1500.0),  'VERY STRONG  (a=-1500)'),
    ]

    # If the scan found a valley, also show the best-fit point
    if scan:
        a_best, b_best = best[0], best[1]
        if abs(a_best) > 1.0 and (a_best, b_best) not in [(d[0], d[1]) for d in diag_points]:
            diag_points.append(
                (a_best, b_best, f'BEST-FIT  (a={a_best:.0f}, b={b_best:.1f})')
            )

    for a, b, label in diag_points:
        if not is_physical(a, b) and a != 0.0:
            print(f'\n  *** {label} — not physical, skipped ***')
            continue
        try:
            cosmo = build_cosmo(a, b)
            _, _, details = compute_uvlf_chi2(
                cosmo, donnan_data, finkelstein_data,
                integrate_bin=args.integrate, n_gl=args.n_gl,
            )
            print_details(details, label)
            cosmo.struct_cleanup()
            cosmo.empty()

        except Exception as e:
            print(f'\n  *** {label} — crashed: {e} ***')

    # ── cleanup ───────────────────────────────────────────────────────────
    cosmo0.struct_cleanup()
    cosmo0.empty()
    print('Done.')


if __name__ == '__main__':
    main()
