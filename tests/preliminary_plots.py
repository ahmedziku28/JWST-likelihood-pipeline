#!/usr/bin/env python3
"""
plot_preliminary_results.py
===========================
Three figures for the Hashim meeting, all consistent with Group B
(GL integration + volume correction).

    1. UVLF theory vs data  (8-panel, one per z-bin, volume-corrected)
    2. Per-bin chi2 breakdown (bar chart, computed live with volume correction)
    3. Delta-chi2 comparison  (grouped bar chart, Group B + E5-E8 numbers)

Usage:
    source ~/workspace/venvs/exo_DE/bin/activate
    export PYTHONPATH="$HOME/workspace/Modules/class_omx/python:$HOME/workspace/exo_de_project:$PYTHONPATH"
    cd ~/workspace/exo_de_project

    # Full run (all 3 plots):
    python tests/plot_preliminary_results.py --output_dir plots/ \
        --a_exo 0.0 --b_exo -161.0 --beta_E5 1.85

    # Quick run (Plot 3 only, no CLASS needed):
    python tests/plot_preliminary_results.py --output_dir plots/ --skip_theory
"""

import sys, os, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ── Pipeline imports (available on cluster) ──────────────────
from pipeline.uvlf import (
    _build_ref_cosmo, volume_ratio, _phi_single_z,
)
from pipeline.hmf import compute_hmf
from pipeline.uvlf_conversion import (
    SHMR_N, SHMR_LOG_MC, SHMR_BETA, SHMR_GAMMA,
)

# ──────────────────────────────────────────────────────────────
#  DATA — hardcoded from donnan2024.txt and finkelstein2024.txt
#  Units: mag^-1 Mpc^-3 (the 1e-6 factor converts from the
#  "10^-6" convention in the text files to absolute units).
#  These are the RAW published values — volume correction is
#  applied at plot time, never mutating these originals.
# ──────────────────────────────────────────────────────────────

_DONNAN_RAW = {
    9.0: {
        'M_UV':  np.array([-20.75, -20.25, -19.75, -19.25, -18.55, -18.05, -17.55]),
        'phi':   np.array([12, 32, 144, 235, 486, 1110, 1776], dtype=np.float64) * 1e-6,
        'sig_up':np.array([8, 13, 30, 60, 157, 310, 578],     dtype=np.float64) * 1e-6,
        'sig_dn':np.array([5, 10, 28, 49, 139, 310, 510],     dtype=np.float64) * 1e-6,
    },
    10.0: {
        'M_UV':  np.array([-20.75, -20.25, -19.75, -19.25, -18.55, -18.05, -17.55]),
        'phi':   np.array([4, 27, 92, 177, 321, 686, 1278],   dtype=np.float64) * 1e-6,
        'sig_up':np.array([10, 13, 25, 53, 127, 245, 486],    dtype=np.float64) * 1e-6,
        'sig_dn':np.array([4, 10, 20, 45, 111, 223, 432],     dtype=np.float64) * 1e-6,
    },
    11.0: {
        'M_UV':  np.array([-21.25, -20.75, -20.25, -19.75, -19.25, -18.75, -18.25]),
        'phi':   np.array([7, 14, 38, 100, 144, 234, 641],    dtype=np.float64) * 1e-6,
        'sig_up':np.array([9, 11, 16, 37, 81, 118, 361],      dtype=np.float64) * 1e-6,
        'sig_dn':np.array([5, 7, 13, 30, 63, 96, 281],        dtype=np.float64) * 1e-6,
    },
    12.5: {
        'M_UV':  np.array([-21.25, -20.75, -20.25, -19.75, -19.25, -18.75, -18.25]),
        'phi':   np.array([3, 4, 16, 34, 43, 80, 217],        dtype=np.float64) * 1e-6,
        'sig_up':np.array([4, 5, 9, 23, 35, 51, 153],         dtype=np.float64) * 1e-6,
        'sig_dn':np.array([2, 3, 6, 15, 22, 36, 104],         dtype=np.float64) * 1e-6,
    },
    14.5: {
        'M_UV':  np.array([-20.25]),
        'phi':   np.array([3],  dtype=np.float64) * 1e-6,
        'sig_up':np.array([6],  dtype=np.float64) * 1e-6,
        'sig_dn':np.array([2],  dtype=np.float64) * 1e-6,
    },
}

_FINKELSTEIN_RAW = {
    8.9: {
        'M_UV':  np.array([-22.0, -21.0, -20.5, -20.0, -19.5, -19.0]),
        'phi':   np.array([11, 22, 82, 96, 286, 268],   dtype=np.float64) * 1e-6,
        'sig_up':np.array([7, 13, 40, 46, 115, 124],    dtype=np.float64) * 1e-6,
        'sig_dn':np.array([6, 10, 32, 36, 91, 100],     dtype=np.float64) * 1e-6,
    },
    10.9: {
        'M_UV':  np.array([-20.5, -20.0, -19.5]),
        'phi':   np.array([18, 54, 76],    dtype=np.float64) * 1e-6,
        'sig_up':np.array([12, 27, 39],    dtype=np.float64) * 1e-6,
        'sig_dn':np.array([9, 21, 30],     dtype=np.float64) * 1e-6,
    },
    14.0: {
        'M_UV':  np.array([-20.0, -19.5]),
        'phi':   np.array([26, 73],   dtype=np.float64) * 1e-6,
        'sig_up':np.array([33, 69],   dtype=np.float64) * 1e-6,
        'sig_dn':np.array([18, 44],   dtype=np.float64) * 1e-6,
    },
}

# ──────────────────────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────────────────────

PLANCK = {
    'H0': 67.36, 'omega_b': 0.02237, 'omega_cdm': 0.1200,
    'n_s': 0.9649, 'A_s': 2.1e-9, 'tau_reio': 0.0544,
    'z_c_exo': 16.0, 'sigma_z_exo': 3.25,
    'output': 'mPk', 'P_k_max_1/Mpc': 360.0,
    'z_max_pk': 20.0, 'non linear': 'none',
}

DEFAULT_SHMR = dict(N=SHMR_N, log_Mc=SHMR_LOG_MC,
                    beta=SHMR_BETA, gamma=SHMR_GAMMA)


def _init_class(a_exo, b_exo):
    """Initialise CLASS with given exotic DE parameters."""
    from classy import Class
    cosmo = Class()
    pars = dict(PLANCK)
    pars['a_exo'] = a_exo
    pars['b_exo'] = b_exo
    cosmo.set(pars)
    cosmo.compute()
    return cosmo


def _get_volcorr_data(cosmo_mcmc):
    """
    Return DEEP COPIES of Donnan and Finkelstein data with volume
    correction applied.  Uses pipeline.uvlf.volume_ratio — the exact
    same function the likelihood uses during minimisation.
    Never mutates the module-level _DONNAN_RAW / _FINKELSTEIN_RAW.
    """
    import copy
    donnan = copy.deepcopy(_DONNAN_RAW)
    finkelstein = copy.deepcopy(_FINKELSTEIN_RAW)

    cosmo_ref_don  = _build_ref_cosmo('Donnan')
    cosmo_ref_fink = _build_ref_cosmo('Finkelstein')

    for z_nom, d in donnan.items():
        vr = volume_ratio(z_nom, cosmo_ref_don, cosmo_mcmc)
        d['phi']    *= vr
        d['sig_up'] *= vr
        d['sig_dn'] *= vr
        print(f"    Donnan z={z_nom}: V_ref/V_mcmc = {vr:.4f}")

    for z_nom, d in finkelstein.items():
        vr = volume_ratio(z_nom, cosmo_ref_fink, cosmo_mcmc)
        d['phi']    *= vr
        d['sig_up'] *= vr
        d['sig_dn'] *= vr
        print(f"    Finkelstein z={z_nom}: V_ref/V_mcmc = {vr:.4f}")

    cosmo_ref_don.struct_cleanup();  cosmo_ref_don.empty()
    cosmo_ref_fink.struct_cleanup(); cosmo_ref_fink.empty()

    return donnan, finkelstein


def _compute_phi(cosmo, z_nom, M_UV_grid, shmr_kwargs=None):
    """Compute theory phi at a single z on the given M_UV grid."""
    kw = shmr_kwargs if shmr_kwargs is not None else DEFAULT_SHMR
    M_h, dndlnm, _, _ = compute_hmf(cosmo, z_nom)
    return _phi_single_z(M_h, dndlnm, z_nom, M_UV_grid, **kw)


# ──────────────────────────────────────────────────────────────
#  PLOT 1 — UVLF theory vs data (8 panels, volume-corrected)
# ──────────────────────────────────────────────────────────────

def plot_uvlf_panels(output_dir, cosmo_lcdm, donnan, finkelstein,
                     a_exo_B2=0.0, b_exo_B2=-108.0, beta_E5=None):
    """
    8-panel figure, 4 curves per panel:
      1. LCDM              (black dashed)   — baseline
      2. Exotic DE B2       (grey solid)     — conservative, fixed SHMR
      3. Exotic DE B5       (red solid)      — best fit, vary SHMR restricted
      4. LCDM + vary-beta E5 (blue dotted)  — astrophysics competitor
    """
    print("[Plot 1] UVLF theory vs data...")

    panels = [
        ('Finkelstein',  8.9, finkelstein.get(8.9)),
        ('Donnan',       9.0, donnan[9.0]),
        ('Donnan',      10.0, donnan[10.0]),
        ('Finkelstein', 10.9, finkelstein[10.9]),
        ('Donnan',      11.0, donnan[11.0]),
        ('Donnan',      12.5, donnan[12.5]),
        ('Finkelstein', 14.0, finkelstein[14.0]),
        ('Donnan',      14.5, donnan[14.5]),
    ]

    M_UV_fine = np.linspace(-23.0, -16.5, 200)

    # ── Initialise CLASS instances ───────────────────────────
    print("  CLASS: B2 (exotic DE, fixed SHMR)...")
    cosmo_B2 = _init_class(a_exo_B2, b_exo_B2)

    print("  CLASS: B5 (exotic DE, vary SHMR, restricted)...")
    cosmo_B5 = _init_class(-40.72, -123.152)

    # B5 best-fit SHMR parameters
    shmr_B5 = dict(N=2.888226189e-02, log_Mc=1.149731494e+01,
                   beta=1.334903232, gamma=0.01)

    # E5: LCDM + vary beta (uses cosmo_lcdm, different SHMR)
    shmr_E5 = None
    if beta_E5 is not None:
        shmr_E5 = dict(N=SHMR_N, log_Mc=SHMR_LOG_MC,
                       beta=beta_E5, gamma=SHMR_GAMMA)
        print(f"  E5: vary-beta with beta = {beta_E5:.4f}")

    # ── Figure ───────────────────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), dpi=150)
    axes = axes.flatten()

    for i, (dset, z_nom, data) in enumerate(panels):
        ax = axes[i]
        M = M_UV_fine

        # 1. LCDM baseline
        ax.semilogy(M, _compute_phi(cosmo_lcdm, z_nom, M),
                    '--', color='k', lw=1.5, alpha=0.8,
                    label=r'$\Lambda$CDM')

        # 2. B2: exotic DE, fixed SHMR, restricted (conservative)
        ax.semilogy(M, _compute_phi(cosmo_B2, z_nom, M),
                    '-', color='0.50', lw=1.3, alpha=0.7,
                    label='Exotic DE, fixed SHMR (B2), $z\geq10$ ')

        # 3. B5: exotic DE + vary SHMR, restricted (best fit)
        ax.semilogy(M, _compute_phi(cosmo_B5, z_nom, M,
                                     shmr_kwargs=shmr_B5),
                    '-', color='#d62728', lw=2.5,
                    label=r'Exotic DE + vary SHMR (B5), $z\geq10$')

        # 4. E5: LCDM + vary beta, full (astrophysics competitor)
        if shmr_E5 is not None:
            ax.semilogy(M, _compute_phi(cosmo_lcdm, z_nom, M,
                                         shmr_kwargs=shmr_E5),
                        ':', color='#1f77b4', lw=2.2,
                        label=r'$\Lambda$CDM + vary $\beta$ (E5)')

        # Data points (already volume-corrected)
        if data is not None:
            # Prevent lower errorbars from going <=0 on log scale
            # Prevent lower errorbars from reaching <= 0 on log scale
            eps_frac = 0.98

            sig_dn_plot = np.where(
                data['sig_dn'] >= data['phi'],
                data['phi'] * eps_frac,
                data['sig_dn']
            )

            lolims = data['sig_dn'] >= data['phi']

            ax.errorbar(
                data['M_UV'],
                data['phi'],
                yerr=[sig_dn_plot, data['sig_up']],
                lolims=lolims,
                fmt='o',
                color='k',
                ms=5,
                capsize=3,
                lw=1.2,
                zorder=10,
                label=dset
            )

        # ── Panel formatting ─────────────────────────────────
        ax.set_title(f'$z = {z_nom}$  ({dset})', fontsize=12)
        ax.set_xlabel(r'$M_{\rm UV}$', fontsize=11)
        if i % 4 == 0:
            ax.set_ylabel(
                r'$\phi\;[\mathrm{mag}^{-1}\,\mathrm{Mpc}^{-3}]$',
                fontsize=11)
        ax.set_xlim(-23.0, -16.5)
        ax.set_ylim(1e-8, 1e-2)
        ax.invert_xaxis()
        ax.tick_params(labelsize=9)

        # Red wash on overpredicted panels
        if z_nom in [8.9, 9.0]:
            ax.axhspan(1e-8, 1e-2, alpha=0.05, color='red')
            ax.text(0.97, 0.95, 'overpredicted',
                    transform=ax.transAxes, ha='right', va='top',
                    fontsize=8, color='#d62728', style='italic')

    # ── Shared legend (deduplicated) ─────────────────────────
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(),
               loc='lower center', ncol=5, fontsize=10,
               bbox_to_anchor=(0.5, -0.03),
               frameon=True, fancybox=True, edgecolor='0.7')

    fig.suptitle('UVLF: Theory vs JWST Data '
                 '(volume-corrected, Donnan 2024 + Finkelstein 2024)',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()

    path = os.path.join(output_dir, 'uvlf_theory_vs_data.pdf')
    fig.savefig(path, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close(fig)

    # ── Cleanup CLASS instances ──────────────────────────────
    cosmo_B2.struct_cleanup(); cosmo_B2.empty()
    cosmo_B5.struct_cleanup(); cosmo_B5.empty()


# ──────────────────────────────────────────────────────────────
#  PLOT 2 — Per-bin chi2 at LCDM (volume-corrected, computed live)
# ──────────────────────────────────────────────────────────────

def plot_perbin_chi2(output_dir, cosmo_lcdm, donnan, finkelstein):
    """
    Compute split-Gaussian chi2 per z-bin and make the bar chart.
    Data dicts must already be volume-corrected.
    """
    print("[Plot 2] Per-bin chi2 (with volume correction)...")

    results = []

    for tag, dataset in [('Don', donnan), ('Fin', finkelstein)]:
        for z_nom in sorted(dataset.keys()):
            d = dataset[z_nom]
            M_UV_bins = d['M_UV']
            phi_obs = d['phi']
            sig_up  = d['sig_up']
            sig_dn  = d['sig_dn']

            phi_th = _compute_phi(cosmo_lcdm, z_nom, M_UV_bins)

            residual = phi_th - phi_obs
            sigma = np.where(residual > 0, sig_up, sig_dn)
            chi2_bin  = float(np.sum((residual / sigma)**2))
            mean_pull = float(np.mean(residual / sigma))

            results.append((f'{tag} $z{{=}}{z_nom}$', z_nom,
                            len(M_UV_bins), chi2_bin, mean_pull))

    # Print table
    total_chi2 = sum(r[3] for r in results)
    total_n    = sum(r[2] for r in results)
    print(f"\n  {'Bin':<22s} {'n':>3s} {'chi2':>8s} {'pull':>7s}")
    print(f"  {'-'*44}")
    for label, z, n, c2, p in results:
        tag = 'OVER' if p > 0 else 'under'
        print(f"  {label:<22s} {n:3d} {c2:8.1f} {p:+7.2f}  ({tag})")
    print(f"  {'-'*44}")
    print(f"  {'TOTAL':<22s} {total_n:3d} {total_chi2:8.1f}")

    # ── Bar chart ──
    labels = [r[0] for r in results]
    n_pts  = [r[2] for r in results]
    chi2   = [r[3] for r in results]
    pulls  = [r[4] for r in results]
    colors = ['#d62728' if p > 0 else '#1f77b4' for p in pulls]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    x = np.arange(len(results))
    ax.bar(x, chi2, color=colors, edgecolor='k', linewidth=0.5,
           alpha=0.85, width=0.7)

    # Expected chi2 = n
    for i, n in enumerate(n_pts):
        ax.plot([i - 0.35, i + 0.35], [n, n], 'k--', lw=0.8, alpha=0.4)

    # Pull annotations
    for i, (c2, p) in enumerate(zip(chi2, pulls)):
        sign = '+' if p > 0 else ''
        ax.text(i, c2 + 1.0, f'{sign}{p:.2f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold',
                color='#d62728' if p > 0 else '#1f77b4')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(r'$\chi^2_{\rm bin}$', fontsize=13)
    ax.set_title(r'Per-bin $\chi^2$ at $\Lambda$CDM (with volume correction)',
                 fontsize=12)

    legend_elements = [
        Patch(facecolor='#d62728', edgecolor='k', alpha=0.85,
              label='Overpredicted (theory > data)'),
        Patch(facecolor='#1f77b4', edgecolor='k', alpha=0.85,
              label='Underpredicted (theory < data)'),
        Line2D([0], [0], color='k', ls='--', lw=0.8, alpha=0.4,
               label=r'Expected $\chi^2 = n_{\rm bins}$'),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc='upper right')
    ax.set_ylim(0, max(chi2) * 1.15)
    fig.tight_layout()

    path = os.path.join(output_dir, 'perbin_chi2_groupB.pdf')
    fig.savefig(path, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close(fig)

    return results


# ──────────────────────────────────────────────────────────────
#  PLOT 3 — Delta-chi2 comparison (standalone, hardcoded numbers)
# ──────────────────────────────────────────────────────────────

def plot_deltachi2_comparison(output_dir):
    """
    Grouped horizontal bar chart.  All numbers from Group B (both
    corrections) and E5-E8 (LCDM + corrections).

    Baselines:
      Full likelihood with corrections:       chi2_LCDM = 126.2
      Restricted (z>=10) with corrections:    chi2_LCDM =  45.1
    """
    print("[Plot 3] Delta-chi2 comparison...")

    # ── Full likelihood (baseline 126.2) ──────────────────────
    # ── Restricted z>=10 (baseline 45.1) ──────────────────────
    #
    # (label, delta_chi2, color_key)
    #   exo   = exotic DE, full
    #   exo_r = exotic DE, restricted
    #   lcdm  = LCDM + SHMR, full
    #   lcdm_r= LCDM + SHMR, restricted

    runs = [
        # --- Full likelihood block ---
        (r'Exotic DE, fixed SHMR'
         '\n(full, $P{=}2$)',                      -1.2,   'exo'),

        (r'$\Lambda$CDM + vary $\beta$'
         '\n(full, $P{=}1$) [E5]',                -24.2,   'lcdm'),

        (r'Exotic DE + vary $\beta$'
         '\n(full, $P{=}3$)',                      -16.3,   'exo'),

        (r'$\Lambda$CDM + vary all'
         '\n(full, $P{=}3$) [E6]',                -27.4,   'lcdm'),

        (r'Exotic DE + vary all'
         '\n(full, $P{=}5$)',                      -39.0,   'exo'),

        # --- Restricted block ---
        (r'Exotic DE, fixed SHMR'
         '\n($z{\geq}10$, $P{=}2$)',               -6.5,   'exo_r'),

        (r'$\Lambda$CDM + vary $\beta$'
         '\n($z{\geq}10$, $P{=}1$) [E7]',         -7.6,   'lcdm_r'),

        (r'$\Lambda$CDM + vary all'
         '\n($z{\geq}10$, $P{=}3$) [E8]',         -7.2,   'lcdm_r'),
    ]

    labels = [r[0] for r in runs]
    dchi2  = [r[1] for r in runs]
    ckeys  = [r[2] for r in runs]

    color_map = {
        'exo':    '#d62728',
        'exo_r':  '#ff7f0e',
        'lcdm':   '#1f77b4',
        'lcdm_r': '#2ca02c',
    }
    colors = [color_map[k] for k in ckeys]

    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=150)
    x = np.arange(len(runs))
    bars = ax.barh(x, dchi2, color=colors, edgecolor='k',
                   linewidth=0.5, alpha=0.85, height=0.65)

    # Value labels inside bars
    for i, d in enumerate(dchi2):
        ax.text(d - 0.4, i, f'{d:.1f}', ha='right', va='center',
                fontsize=11, fontweight='bold', color='white')

    ax.axvline(0, color='k', lw=1.0)

    # Separator between full and restricted blocks
    ax.axhline(4.5, color='grey', lw=0.6, ls='-', alpha=0.4)
    ax.text(-41, 1.8, 'Full likelihood\n'
            r'($\chi^2_{\Lambda\mathrm{CDM}}=126.2$, 40 bins)',
            fontsize=9, color='grey', ha='left', va='center')
    ax.text(-41, 6.2, r'Restricted $z\geq 10$' '\n'
            r'($\chi^2_{\Lambda\mathrm{CDM}}=45.1$, 27 bins)',
            fontsize=9, color='grey', ha='left', va='center')

    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel(r'$\Delta\chi^2 = \chi^2_{\rm best} - \chi^2_{\Lambda\rm CDM}$'
                  '  (more negative = better)', fontsize=12)
    ax.set_title(r'Minimisation results: Group B (GL + volume correction) '
                 r'+ E5--E8 ($\Lambda$CDM baselines)',
                 fontsize=12)
    ax.set_xlim(-43, 3)
    ax.invert_yaxis()

    legend_elements = [
        Patch(facecolor='#d62728', edgecolor='k', alpha=0.85,
              label='Exotic DE (full)'),
        Patch(facecolor='#ff7f0e', edgecolor='k', alpha=0.85,
              label=r'Exotic DE ($z \geq 10$)'),
        Patch(facecolor='#1f77b4', edgecolor='k', alpha=0.85,
              label=r'$\Lambda$CDM + SHMR (full)'),
        Patch(facecolor='#2ca02c', edgecolor='k', alpha=0.85,
              label=r'$\Lambda$CDM + SHMR ($z \geq 10$)'),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc='lower left')
    fig.tight_layout()

    path = os.path.join(output_dir, 'deltachi2_comparison.pdf')
    fig.savefig(path, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preliminary result plots for Hashim meeting')
    parser.add_argument('--output_dir', default='plots/')
    parser.add_argument('--a_exo', type=float, default=0.0,
        help='Best-fit a_exo from B2 restricted run')
    parser.add_argument('--b_exo', type=float, default=-161.0,
        help='Best-fit b_exo from B2 restricted run')
    parser.add_argument('--beta_E5', type=float, default=None,
        help='Best-fit beta from E5 (LCDM + vary beta + corrections). '
             'Omit to skip the vary-beta curve in Plot 1.')
    parser.add_argument('--skip_theory', action='store_true',
        help='Skip Plots 1 & 2 (need CLASS). Only make Plot 3.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Plot 3 is always made (no CLASS needed, hardcoded numbers)
    plot_deltachi2_comparison(args.output_dir)

    if not args.skip_theory:
        # Shared LCDM CLASS instance for plots 1 & 2
        print("\nInitialising LCDM CLASS...")
        cosmo_lcdm = _init_class(0.0, 0.0)

        # Volume-correct both datasets (deep copies, originals untouched)
        print("Applying volume corrections...")
        donnan_vc, fink_vc = _get_volcorr_data(cosmo_lcdm)

        # Plot 2: per-bin chi2 (computed live with corrected data)
        plot_perbin_chi2(args.output_dir, cosmo_lcdm, donnan_vc, fink_vc)

        # Plot 1: theory vs data panels
        plot_uvlf_panels(args.output_dir, cosmo_lcdm, donnan_vc, fink_vc,
                         a_exo_B2=args.a_exo, b_exo_B2=args.b_exo,
                         beta_E5=args.beta_E5)

        cosmo_lcdm.struct_cleanup(); cosmo_lcdm.empty()
    else:
        print("\n[Plots 1 & 2] Skipped (--skip_theory).")

    print(f"\nDone. All plots in: {args.output_dir}")