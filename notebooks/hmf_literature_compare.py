"""
hmf_literature_compare.py
Compare our SMT HMF against Jiang et al. 2024 Fig 3
and Murray et al. 2013 (hmf package at z=0).
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/lustre_p/ahmed.omar/workspace/exo_de_project')
from pipeline.hmf import compute_hmf, _M_GRID
from classy import Class

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 12,
    'axes.linewidth': 1.2, 'axes.labelsize': 13,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'legend.frameon': False, 'figure.dpi': 150,
})

# ---------------------------------------------------------------------------
# LCDM cosmology matching Jiang et al. 2024
# They use Planck 2018: h=0.6736, Om0=0.3153
# ---------------------------------------------------------------------------
JIANG_PARAMS = {
    "output"        : "mPk",
    "h"             : 0.6736,
    "omega_b"       : 0.02237,
    "omega_cdm"     : 0.1200,
    "n_s"           : 0.9649,
    "ln10^{10}A_s" : 3.044,
    "P_k_max_1/Mpc": 510.0,
    "z_max_pk"      : 22.0,
    "a_exo"         : 0.0,
    "b_exo"         : 0.0,
}

print("Computing CLASS (Jiang+ cosmology) ...")
cosmo = Class()
cosmo.set(JIANG_PARAMS)
cosmo.compute()
h = cosmo.h()

# ---------------------------------------------------------------------------
# Jiang et al. 2024 Figure 3 — digitized reference points
# Units in their paper: log10(M [h^-1 M_sun]) vs dn/dlog10M [h^3 Mpc^-3]
# Read off approximately from Fig 3 at z=8 and z=10 LCDM curves
# These are approximate — use for order-of-magnitude validation only
# ---------------------------------------------------------------------------
# z=8 LCDM reference points from Jiang+ Fig 3
jiang_z8_logM  = np.array([10.0, 10.5, 11.0, 11.5, 12.0, 12.5])  # log10(M/[Msun/h])
jiang_z8_logdn = np.array([-1.5, -2.2, -3.2, -4.6, -6.5, -9.2])  # log10(dn/dlog10M [h^3/Mpc^3])

# z=10 LCDM reference points
jiang_z10_logM  = np.array([10.0, 10.5, 11.0, 11.5, 12.0])
jiang_z10_logdn = np.array([-2.0, -3.0, -4.4, -6.3, -9.0])

# Convert Jiang units to our units:
# M: log10(M [Msun/h]) -> M [Msun]: multiply by h
# dn/dlog10M [h^3/Mpc^3] -> dn/dlnM [Mpc^-3]:
#   dn/dlnM = dn/dlog10M * log10(e) * h^3
# (the h^3 converts h^3/Mpc^3 -> Mpc^-3,
#  the log10(e) converts dlog10M -> dlnM)
log10e = np.log10(np.e)

jiang_z8_M_msun  = 10**(jiang_z8_logM)  * h          # M_sun
jiang_z8_dn      = 10**(jiang_z8_logdn) * log10e * h**3   # Mpc^-3

jiang_z10_M_msun = 10**(jiang_z10_logM) * h
jiang_z10_dn     = 10**(jiang_z10_logdn) * log10e * h**3

# ---------------------------------------------------------------------------
# Compute our HMF at same redshifts
# ---------------------------------------------------------------------------
print("Computing our HMF ...")
M_8,  dn_8,  _ = compute_hmf(cosmo, 8.0)
M_10, dn_10, _ = compute_hmf(cosmo, 10.0)
M_0,  dn_0,  _ = compute_hmf(cosmo, 0.0)

# ---------------------------------------------------------------------------
# hmf package at z=0 as Murray+ 2013 reference
# ---------------------------------------------------------------------------
print("Computing hmf package (Murray+ 2013 reference) ...")
try:
    from hmf import MassFunction
    mf_z0 = MassFunction(
        z            = 0.0,
        cosmo_params = {"H0": h*100, "Om0": cosmo.Omega_m(),
                        "Ob0": cosmo.Omega_b()},
        sigma_8      = cosmo.sigma8(),
        n            = cosmo.n_s(),
        Mmin         = 8.0 - np.log10(h),
        Mmax         = 16.0 - np.log10(h),
        dlog10m      = 0.01,
        hmf_model    = "SMT",
    )
    M_hmf_z0  = mf_z0.m      * h
    dn_hmf_z0 = mf_z0.dndlnm * h**3
    have_hmf  = True
except Exception as e:
    print(f"  hmf not available: {e}")
    have_hmf = False

# ---------------------------------------------------------------------------
# Figure: two panels
# Left:  z=0 — ours vs hmf package (Murray+ 2013)
# Right: z=8,10 — ours vs Jiang+ 2024 Fig 3
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# ---- Left: z=0 ----
ax = axes[0]
ax.semilogy(np.log10(M_0), dn_0,
            color='#E05C2A', lw=2.5, label='This work (Lagrangian SMT)', zorder=3)
if have_hmf:
    mask = dn_hmf_z0 > 1e-30
    ax.semilogy(np.log10(M_hmf_z0[mask]), dn_hmf_z0[mask],
                color='#333333', lw=1.8, ls='--',
                label='hmf pkg, EH transfer\n(Murray+ 2013 ref)', zorder=2)
    # Ratio subplot inset
    from scipy.interpolate import interp1d
    interp_ours = interp1d(np.log10(M_0), np.log10(dn_0 + 1e-400),
                           bounds_error=False, fill_value=-999)
    dn_ours_at_hmf = 10**interp_ours(np.log10(M_hmf_z0[mask]))
    valid = (dn_ours_at_hmf > 1e-30) & (dn_hmf_z0[mask] > 1e-30)
    ratio = dn_ours_at_hmf[valid] / dn_hmf_z0[mask][valid]
    # Print median ratio
    print(f"\n  z=0 median ratio (ours/hmf): {np.median(ratio):.3f}")
    print(f"  z=0 max |ratio-1|: {np.abs(ratio-1).max():.3f}")

ax.set_xlabel(r'$\log_{10}(M_h\,/\,M_\odot)$')
ax.set_ylabel(r'${\rm d}n\,/\,{\rm d}\ln M_h\ \ [{\rm Mpc}^{-3}]$')
ax.set_xlim(8, 16)
ax.set_ylim(1e-20, 1e3)
ax.legend(loc='upper right', fontsize=10)
ax.set_title(r'$z=0$: This work vs Murray et al. (2013)', fontsize=12)
ax.text(0.04, 0.05,
        'Expected 10-20% offset:\nLagrangian (ours) vs EH (hmf)',
        transform=ax.transAxes, fontsize=9, color='gray',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

# ---- Right: z=8,10 vs Jiang+ 2024 ----
ax = axes[1]

# Our curves
mask8  = dn_8  > 1e-50
mask10 = dn_10 > 1e-50
ax.semilogy(np.log10(M_8[mask8]),   dn_8[mask8],
            color='#3A7FD5', lw=2.5, label=r'This work $z=8$', zorder=3)
ax.semilogy(np.log10(M_10[mask10]), dn_10[mask10],
            color='#E05C2A', lw=2.5, label=r'This work $z=10$', zorder=3)

# Jiang+ reference points
ax.semilogy(np.log10(jiang_z8_M_msun),  jiang_z8_dn,
            's', color='#3A7FD5', ms=8, zorder=5,
            label=r'Jiang+ 2024 Fig.3 $z=8$')
ax.semilogy(np.log10(jiang_z10_M_msun), jiang_z10_dn,
            'o', color='#E05C2A', ms=8, zorder=5,
            label=r'Jiang+ 2024 Fig.3 $z=10$')

# Print comparison at reference points
print("\n  z=8 comparison with Jiang+ 2024:")
interp_z8 = interp1d(np.log10(M_8), np.log10(dn_8 + 1e-400),
                     bounds_error=False, fill_value=-999)
for logM, dn_ref in zip(jiang_z8_logM, jiang_z8_dn):
    logM_msun = np.log10(10**logM * h)
    dn_ours   = 10**interp_z8(logM_msun)
    if dn_ours > 1e-300 and dn_ref > 1e-300:
        ratio = dn_ours / dn_ref
        print(f"  log10(M)={logM_msun:.1f}: ours={dn_ours:.2e}  "
              f"Jiang={dn_ref:.2e}  ratio={ratio:.2f}")

ax.set_xlabel(r'$\log_{10}(M_h\,/\,M_\odot)$')
ax.set_ylabel(r'${\rm d}n\,/\,{\rm d}\ln M_h\ \ [{\rm Mpc}^{-3}]$')
ax.set_xlim(9, 14)
ax.set_ylim(1e-15, 1e2)
ax.legend(loc='upper right', fontsize=10)
ax.set_title(r'$z=8,10$: This work vs Jiang et al. (2024) Fig. 3', fontsize=12)
ax.text(0.04, 0.05,
        'Squares/circles: digitized from Fig.3\n(approximate ±0.3 dex)',
        transform=ax.transAxes, fontsize=9, color='gray',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

fig.suptitle('HMF Pipeline Validation Against Published Literature',
             fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig('hmf_literature_compare.pdf', bbox_inches='tight')
plt.close()
print("\nSaved: hmf_literature_compare.pdf")