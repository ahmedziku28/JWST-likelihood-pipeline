#!/usr/bin/env python3
"""
tests/plot_centroid_effect.py

Shows the effect of the prior centroid (a_exo=-903.588, b_exo=+245.298)
on the theoretical cumulative stellar mass density rho_star(>M_star)
compared to ΛCDM, at each z-bin midpoint.

Pure theory — no catalog needed. 
Assumes classy (class_omx) is installed in the environment.

Usage
-----
python tests/plot_centroid_effect.py --mode spectroscopic --output centroid_effect.png
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Project Path setup ────────────────────────────────────────────────────────
# Keeps ability to find 'pipeline' module from the tests directory
_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pipeline.stellar_mass_function import compute_theory_rho_star
from pipeline.hmf import compute_hmf


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

_A_CENTROID = -903.588
_B_CENTROID = +245.298
_Z_C_EXO     = 16.0
_SIGMA_Z_EXO = 3.25

_ZBINS_SPECZ  = [(6., 8., 7.), (8., 10., 9.), (10., 15., 12.5)]
_ZBINS_PHOTOZ = [(6., 8., 7.), (8., 10., 9.), (10., 15., 12.5), (15., 20., 17.5)]

_M_STAR_FINE = np.logspace(7.5, 12.5, 400)

_PLANCK_BASE = {
    'h':          0.6736,
    'omega_b':    0.02237,
    'omega_cdm':  0.1200,
    'n_s':        0.9649,
    'A_s':        2.101e-9,
    'tau_reio':   0.0544,
    'output':     'mPk',
    'P_k_max_1/Mpc': 510.0,
    'z_max_pk':   20.0,
    'non linear': 'none',
}

_EXO_PARAMS = {
    'z_c_exo':     _Z_C_EXO,
    'sigma_z_exo': _SIGMA_Z_EXO,
}


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _make_cosmo(a_exo=0.0, b_exo=0.0):
    """ Recover ΛCDM if a=b=0, else exotic model. """
    from classy import Class
    p = dict(_PLANCK_BASE)
    p.update(_EXO_PARAMS)
    p['a_exo'] = a_exo
    p['b_exo'] = b_exo
    cosmo = Class()
    cosmo.set(p)
    cosmo.compute()
    return cosmo


def _rho_theory(cosmo, z_mid):
    """ Compute rho_star(>M_star) at z_mid. """
    M_h, dndlnm, _ = compute_hmf(cosmo, z_mid)
    rho = compute_theory_rho_star(M_h, dndlnm, _M_STAR_FINE)
    rho_plot = np.where(rho > 0.0, rho, np.nan)
    return _M_STAR_FINE, rho_plot


def _f_tilde_x(a_exo, b_exo, z_c, omega_m=0.3111, omega_lambda=0.6889):
    """ Peak fractional effect on H^2 (Analytical). """
    rho_x_at_zc = a_exo + b_exo * z_c / (1.0 + z_c)
    E2_zc = omega_m * (1.0 + z_c)**3 + omega_lambda
    return rho_x_at_zc / E2_zc


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PLOT
# ══════════════════════════════════════════════════════════════════════════════

def make_plot(zbins, output_path):
    n_bins = len(zbins)

    print("Initialising CLASS — ΛCDM...")
    cosmo_lcdm = _make_cosmo(a_exo=0.0, b_exo=0.0)

    print(f"Initialising CLASS — centroid (a={_A_CENTROID}, b={_B_CENTROID})...")
    cosmo_exo = _make_cosmo(a_exo=_A_CENTROID, b_exo=_B_CENTROID)

    f_tilde = _f_tilde_x(_A_CENTROID, _B_CENTROID, _Z_C_EXO)

    print("Computing rho_star theory curves...")
    results_lcdm = []
    results_exo  = []
    for (z_min, z_max, z_mid) in zbins:
        _, r_l = _rho_theory(cosmo_lcdm, z_mid)
        _, r_e = _rho_theory(cosmo_exo,  z_mid)
        results_lcdm.append(r_l)
        results_exo.append(r_e)

    cosmo_lcdm.struct_cleanup(); cosmo_lcdm.empty()
    cosmo_exo.struct_cleanup();  cosmo_exo.empty()

    fig = plt.figure(figsize=(4.5 * n_bins + 2, 9))
    gs = fig.add_gridspec(2, n_bins, height_ratios=[2.2, 1.0], hspace=0.08, wspace=0.28,
                          left=0.08, right=0.97, top=0.91, bottom=0.09)

    colors = ['#E63946', '#2A9D8F', '#E9C46A', '#457B9D']
    axes_top = [fig.add_subplot(gs[0, i]) for i in range(n_bins)]
    axes_bot = [fig.add_subplot(gs[1, i]) for i in range(n_bins)]
    log_M = np.log10(_M_STAR_FINE)

    for i, ((z_min, z_max, z_mid), r_l, r_e) in enumerate(zip(zbins, results_lcdm, results_exo)):
        c = colors[i % len(colors)]
        ax_t, ax_b = axes_top[i], axes_bot[i]

        # Top Panel
        ax_t.plot(log_M, np.log10(r_l), '-',  color='k', lw=2.0, label=r'$\Lambda$CDM')
        ax_t.plot(log_M, np.log10(r_e), '--', color=c,   lw=2.0, label='Centroid')
        
        above = r_e > r_l
        if np.any(above):
            ax_t.fill_between(log_M, np.log10(np.where(above, r_l, np.nan)),
                              np.log10(np.where(above, r_e, np.nan)), color=c, alpha=0.15)

        ax_t.set_xlim(7.5, 12.5)
        ax_t.set_title(f'$z = [{z_min:.0f},\\,{z_max:.0f})$', fontsize=11, fontweight='bold')
        ax_t.set_ylabel(r'$\log_{10}(\rho_\star)$')
        ax_t.grid(True, alpha=0.2, ls='--')
        ax_t.legend(fontsize=8, loc='upper right')

        # Bottom Panel (Ratio)
        valid = np.isfinite(r_l) & np.isfinite(r_e) & (r_l > 0.0)
        pct = (r_e / r_l - 1.0) * 100.0
        ax_b.plot(log_M, pct, '-', color=c, lw=1.8)
        ax_b.axhline(0.0, color='k', lw=0.8, ls='--')
        ax_b.set_xlim(7.5, 12.5)
        ax_b.set_xlabel(r'$\log_{10}(M_\star/M_\odot)$')
        ax_b.set_ylabel(r'$\Delta\rho_\star$ [%]')
        ax_b.yaxis.set_major_formatter(ticker.FormatStrFormatter('%+.0f%%'))
        ax_b.grid(True, alpha=0.2, ls='--')

    fig.suptitle(f'Exotic Centroid effect: $a={_A_CENTROID}, b={_B_CENTROID}, \\tilde{{f}}_x={f_tilde:.3f}$', y=0.98)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved → {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='spectroscopic', choices=['spectroscopic', 'photometric'])
    parser.add_argument('--output', default='centroid_effect.png')
    args = parser.parse_args()

    zbins = _ZBINS_SPECZ if args.mode == 'spectroscopic' else _ZBINS_PHOTOZ
    make_plot(zbins, args.output)

if __name__ == '__main__':
    main()