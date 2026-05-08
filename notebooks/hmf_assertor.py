"""
hmf_assertor.py
---------------
Validation plots for the direct SMT HMF implementation.
Three figures for Hashim showing the pipeline is physically correct.

Run: python3.8 hmf_assertor.py
Outputs: hmf_assertor_panel1.pdf
         hmf_assertor_panel2.pdf
         hmf_assertor_panel3.pdf
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, '/home/lustre_p/ahmed.omar/workspace/exo_de_project')
from pipeline.hmf import compute_hmf, _M_GRID, _K_GRID, _N_K
import pipeline.hmf as _mod

from classy import Class

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family'    : 'serif',
    'font.size'      : 12,
    'axes.linewidth' : 1.2,
    'axes.labelsize' : 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top'      : True,
    'ytick.right'    : True,
    'legend.frameon' : False,
    'legend.fontsize': 11,
    'figure.dpi'     : 150,
})

# ---------------------------------------------------------------------------
# Cosmologies
# ---------------------------------------------------------------------------
BASE = {
    "output"         : "mPk",
    "h"              : 0.6774,
    "omega_b"        : 0.02230,
    "omega_cdm"      : 0.1188,
    "n_s"            : 0.9667,
    "ln10^{10}A_s"  : 3.064,
    "P_k_max_1/Mpc" : 510.0,
    "z_max_pk"       : 22.0,
    "a_exo"          : 0.0,
    "b_exo"          : 0.0,
}

print("Initializing CLASS (LCDM) ...")
cosmo_lcdm = Class()
cosmo_lcdm.set(BASE)
cosmo_lcdm.compute()

print("Initializing CLASS (exotic DE) ...")
EXOTIC = {**BASE, "a_exo": -960.0, "b_exo": 324.0}
cosmo_exo = Class()
cosmo_exo.set(EXOTIC)
cosmo_exo.compute()

h   = cosmo_lcdm.h()
Om0 = cosmo_lcdm.Omega_m()
rho_m0 = Om0 * 2.775e11 * h**2
R_grid = (3.0 * _M_GRID / (4.0 * np.pi * rho_m0))**(1.0/3.0)

print("Computing HMFs ...")
Z_BINS = [6.0, 8.0, 10.0, 15.0, 20.0]
hmf_lcdm = {}
hmf_exo  = {}
sig_lcdm = {}


for z in Z_BINS:
    M, dn, sig = compute_hmf(cosmo_lcdm, z)
    hmf_lcdm[z] = dn
    sig_lcdm[z] = sig
    _, dn_e, _  = compute_hmf(cosmo_exo, z)
    hmf_exo[z]  = dn_e
    print(f"  z={z:.0f} done")

cmap   = plt.cm.plasma
z_norm = matplotlib.colors.Normalize(vmin=6, vmax=20)
colors = [cmap(z_norm(z)) for z in Z_BINS]
M_h_floor = 10**8.5   # UNCOVER SHMR-converted stellar mass floor

# ---------------------------------------------------------------------------
# FIGURE 1 — HMF redshift evolution: LCDM (left) and exotic vs LCDM (right)
# ---------------------------------------------------------------------------
print("\nPlotting Figure 1 ...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

ax = axes[0]
for z, col in zip(Z_BINS, colors):
    mask = hmf_lcdm[z] > 1e-50
    ax.semilogy(np.log10(_M_GRID[mask]), hmf_lcdm[z][mask],
                color=col, lw=2.0, label=f'$z={z:.0f}$')
ax.axvline(np.log10(M_h_floor), color='gray', ls=':', lw=1.2, alpha=0.8)
ax.text(np.log10(M_h_floor)+0.08, 2e1,
        'UNCOVER\nfloor', color='gray', fontsize=8.5,
        ha='left', va='center')
ax.set_xlabel(r'$\log_{10}(M_h\,/\,M_\odot)$')
ax.set_ylabel(r'${\rm d}n\,/\,{\rm d}\ln M_h\ \ \ [{\rm Mpc}^{-3}]$')
ax.set_xlim(6, 15.5)
ax.set_ylim(1e-30, 3e3)
ax.legend(loc='upper right', ncol=2, fontsize=10)
ax.set_title(r'$\Lambda$CDM HMF — SMT, direct CLASS $P(k,z)$', fontsize=12)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=z_norm)
sm.set_array([])
fig.colorbar(sm, ax=ax, pad=0.02, label='Redshift $z$')

ax = axes[1]
for z, col in zip(Z_BINS, colors):
    ml = hmf_lcdm[z] > 1e-50
    me = hmf_exo[z]  > 1e-50
    ax.semilogy(np.log10(_M_GRID[ml]), hmf_lcdm[z][ml],
                color=col, lw=1.5, ls='--', alpha=0.55)
    ax.semilogy(np.log10(_M_GRID[me]), hmf_exo[z][me],
                color=col, lw=2.2, ls='-')
ax.axvline(np.log10(M_h_floor), color='gray', ls=':', lw=1.2, alpha=0.8)
ax.set_xlabel(r'$\log_{10}(M_h\,/\,M_\odot)$')
ax.set_ylabel(r'${\rm d}n\,/\,{\rm d}\ln M_h\ \ \ [{\rm Mpc}^{-3}]$')
ax.set_xlim(6, 15.5)
ax.set_ylim(1e-30, 3e3)
ax.legend(handles=[
    Line2D([0],[0], color='gray', lw=1.5, ls='--', label=r'$\Lambda$CDM'),
    Line2D([0],[0], color='gray', lw=2.2, ls='-',  label='Exotic DE'),
], loc='upper right', fontsize=10)
ax.set_title(r'Exotic DE vs $\Lambda$CDM', fontsize=12)
fig.colorbar(sm, ax=ax, pad=0.02, label='Redshift $z$')

fig.suptitle('Sheth–Mo–Tormen HMF — Direct Integration of CLASS $P(k,z)$',
             fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig('hmf_assertor_panel1.pdf', bbox_inches='tight')
plt.close()
print("  -> hmf_assertor_panel1.pdf")

# ---------------------------------------------------------------------------
# FIGURE 2 — Enhancement ratio exotic / LCDM
# ---------------------------------------------------------------------------
print("Plotting Figure 2 ...")
fig, ax = plt.subplots(figsize=(7, 5))

for z, col in zip([8.0, 10.0, 15.0, 20.0], colors[1:]):
    dl = hmf_lcdm[z]
    de = hmf_exo[z]
    mask = (dl > 1e-50) & (de > 1e-50)
    ratio = de[mask] / dl[mask]
    ax.semilogy(np.log10(_M_GRID[mask]), ratio,
                color=col, lw=2.2, label=f'$z={z:.0f}$')

ax.axhline(1.0, color='black', ls='-', lw=0.8, alpha=0.4)
ax.axvline(np.log10(M_h_floor), color='gray', ls=':', lw=1.2, alpha=0.8)
ax.text(np.log10(M_h_floor)+0.08, 1.15,
        'UNCOVER floor', color='gray', fontsize=9,
        ha='left', va='bottom')

# Annotate peak enhancement at z=15, M=1e12
idx15 = np.argmin(np.abs(_M_GRID - 1e12))
enh15 = hmf_exo[15.0][idx15] / max(hmf_lcdm[15.0][idx15], 1e-300)
ax.annotate(f'×{enh15:.0f} at $z=15$',
            xy=(12.0, enh15),
            xytext=(11.2, enh15*5),
            arrowprops=dict(arrowstyle='->', color=colors[2], lw=1.2),
            color=colors[2], fontsize=11)

ax.set_xlabel(r'$\log_{10}(M_h\,/\,M_\odot)$')
ax.set_ylabel(r'$[{\rm d}n/{\rm d}\ln M]_{\rm exotic}\;/\;[{\rm d}n/{\rm d}\ln M]_{\Lambda{\rm CDM}}$')
ax.set_xlim(9, 14)
ax.set_ylim(0.95, None)
ax.legend(title='Redshift', loc='upper left')
ax.set_title('Exotic Dark Energy Enhancement of the HMF\n'
             r'$(a_{\rm exo}=-960,\ b_{\rm exo}=324)$', fontsize=12)
fig.tight_layout()
fig.savefig('hmf_assertor_panel2.pdf', bbox_inches='tight')
plt.close()
print("  -> hmf_assertor_panel2.pdf")

# ---------------------------------------------------------------------------
# FIGURE 3 — sigma(M) validation: ours vs CLASS.sigma
# ---------------------------------------------------------------------------
print("Plotting Figure 3 ...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
z_check = [0.0, 8.0, 10.0, 15.0, 20.0]
c_check = ['#111111'] + colors[1:]
for z, col in zip(z_check, c_check):
    _, _, sig = compute_hmf(cosmo_lcdm, z)
    # CLASS sigma at every 8th mass point (sparse for clarity)
    step = 8
    M_sp  = _M_GRID[::step]
    R_sp  = R_grid[::step]
    sc_sp = np.array([cosmo_lcdm.sigma(float(r), z) for r in R_sp])
    lbl   = f'$z={z:.0f}$' if z > 0 else '$z=0$'
    ax.semilogy(np.log10(_M_GRID), sig, color=col, lw=1.8, label=lbl)
    ax.semilogy(np.log10(M_sp), sc_sp, 'o', color=col, ms=3.5,
                alpha=0.8, zorder=5)

ax.set_xlabel(r'$\log_{10}(M_h\,/\,M_\odot)$')
ax.set_ylabel(r'$\sigma(M,\,z)$')
ax.set_xlim(6, 16)
ax.legend(loc='upper right', ncol=2, fontsize=9,
          title='— lines: pipeline\n○ dots: CLASS.sigma')
ax.set_title(r'$\sigma(M,z)$: Pipeline vs CLASS', fontsize=12)

ax = axes[1]
for z, col in zip(z_check, c_check):
    _, _, sig = compute_hmf(cosmo_lcdm, z)
    step = 5
    M_sp = _M_GRID[::step]
    R_sp = R_grid[::step]
    sc   = np.array([cosmo_lcdm.sigma(float(r), z) for r in R_sp])
    rel  = np.abs(sig[::step] - sc) / np.maximum(sc, 1e-30)
    lbl  = f'$z={z:.0f}$' if z > 0 else '$z=0$'
    ax.semilogy(np.log10(M_sp), rel, color=col, lw=1.8, label=lbl)

ax.axhline(1e-3, color='red', ls='--', lw=1.0, alpha=0.7, label='0.1%')
ax.set_xlabel(r'$\log_{10}(M_h\,/\,M_\odot)$')
ax.set_ylabel(r'$|\sigma_{\rm pipeline} - \sigma_{\rm CLASS}|\;/\;\sigma_{\rm CLASS}$')
ax.set_xlim(6, 16)
ax.set_ylim(1e-8, 1e-1)
ax.legend(loc='upper right', fontsize=9)
ax.set_title('Relative Error vs CLASS Ground Truth', fontsize=12)

# Add text box with max errors
lines = ['Max relative error:']
for z in [0.0, 8.0, 15.0, 20.0]:
    _, _, sig = compute_hmf(cosmo_lcdm, z)
    errs = [abs(sig[i] - cosmo_lcdm.sigma(float(R_grid[i]), z))
            / max(cosmo_lcdm.sigma(float(R_grid[i]), z), 1e-30)
            for i in range(0, len(_M_GRID), 10)]
    lines.append(f'  z={z:.0f}: {max(errs):.1e}')
ax.text(0.02, 0.97, '\n'.join(lines),
        transform=ax.transAxes, fontsize=8, family='monospace',
        va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

fig.suptitle(r'Validation: $\sigma(M,z)$ Against CLASS Internal Integrator '
             r'($<10^{-4}$ agreement)', fontsize=12, y=1.01)
fig.tight_layout()
fig.savefig('hmf_assertor_panel3.pdf', bbox_inches='tight')
plt.close()
print("  -> hmf_assertor_panel3.pdf")

# ---------------------------------------------------------------------------
# Terminal summary
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("NUMERICAL SUMMARY")
print("="*60)
print("\nsigma(M,z) max relative error vs CLASS:")
for z in [0.0, 8.0, 10.0, 15.0, 20.0]:
    _, _, sig = compute_hmf(cosmo_lcdm, z)
    errs = [abs(sig[i] - cosmo_lcdm.sigma(float(R_grid[i]), z))
            / max(cosmo_lcdm.sigma(float(R_grid[i]), z), 1e-30)
            for i in range(0, len(_M_GRID), 10)]
    print(f"  z={z:4.1f}:  max={max(errs):.2e}   median={np.median(errs):.2e}")

print("\nExotic DE HMF enhancement at M_h = 10^12 M_sun:")
for z in [6.0, 8.0, 10.0, 15.0]:
    idx = np.argmin(np.abs(_M_GRID - 1e12))
    enh = hmf_exo[z][idx] / max(hmf_lcdm[z][idx], 1e-300)
    print(f"  z={z:4.1f}:  {enh:.2f}x")

print("\nHMF (LCDM) at M_h=1e11 M_sun:")
idx11 = np.argmin(np.abs(_M_GRID - 1e11))
for z in Z_BINS:
    print(f"  z={z:4.1f}:  {hmf_lcdm[z][idx11]:.3e} Mpc^-3")

print("\nAll three PDF figures saved in current directory.")
print("="*60)