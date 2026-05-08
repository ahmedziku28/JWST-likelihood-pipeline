"""
grid_ab_scan.py
===============
NxN grid scan over (a_exo, b_exo) parameter space.

Purpose
-------
Map the physically viable region before committing to MCMC priors.
For each (a_exo, b_exo) pair we check:
  1. Does CLASS succeed (no ODE failure)?
  2. Does H²(z) stay positive everywhere?
  3. What is the peak fractional effect on H²?
  4. What is the minimum H²/H²_ΛCDM ratio?

This tells us exactly which prior box to use and where the
"H² goes negative" boundary lies, which is the actual killer
(not the A(z) sign flip, which can't happen with b_exo < |a_exo|).

Run on HPC:
  source ~/workspace/venvs/exo_DE/bin/activate
  export PYTHONPATH="$HOME/workspace/Modules/class_omx/python:$PYTHONPATH"
  python grid_ab_scan.py

Output:
  grid_ab_results.npz  — raw results array
  grid_ab_scan.png     — viability heatmap
"""

import numpy as np
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

# ── PATH TO class_omx ────────────────────────────────────────────────────────
# CLASS_PATH = os.path.expanduser(
#     "~/workspace/Modules/class_omx/python"
# )
# sys.path.insert(0, CLASS_PATH)
from classy import Class, CosmoSevereError, CosmoComputationError


# ── FIXED PLANCK 2018 COSMOLOGY (Table 2, 1807.06209) ────────────────────────
FIXED_COSMO = {
    'omega_b':    0.02237,
    'omega_cdm':  0.1200,
    'H0':         67.4,
    'tau_reio':   0.0544,
    'A_s':        2.101e-9,
    'n_s':        0.9649,
    'm_ncdm':     0.06,
    'N_ncdm':     1,
    'N_ur':       2.0328,
    # Fixed exo window shape
    'z_c_exo':    16.0,
    'sigma_z_exo': 3.25,
    # Minimal output — background only, no Pk needed for this scan
    'output':     '',
}

# ── GRID DEFINITION ──────────────────────────────────────────────────────────
N = 1000

# a_exo: must be negative. Upper limit set by |Omega_x0| <= 0.01 physicality cut.
# Omega_x0 = a_exo * exp(-z_c^2 / 2*sigma_z^2) = a_exo * 5.44e-6
# |a_exo| <= 0.01 / 5.44e-6 ~ 1838
A_MIN, A_MAX = -1838.0, -0.00001        # keep away from 0 to avoid CLASS crash
B_MIN, B_MAX = -2000.0,  1838.0

a_vals = np.linspace(A_MIN, A_MAX, N)
b_vals = np.linspace(B_MIN, B_MAX, N)

# ── RESULT ARRAYS ────────────────────────────────────────────────────────────
# Status codes:
#   0 = prior rejected (b_exo >= |a_exo|)
#   1 = CLASS failed (ODE / severe error)
#   2 = H² went negative (CLASS may or may not catch this)
#   3 = SUCCESS — physically viable

status       = np.zeros((N, N), dtype=int)
omega_x0_arr = np.zeros((N, N))
peak_effect  = np.zeros((N, N))   # |Omega_x0 * W(z_c)| / E^2_LCDM(z_c)
min_h2_ratio = np.full((N, N), np.nan)  # min H²(z)/H²_LCDM(z)

# ── COMPUTE LCDM REFERENCE E²(z_c) ──────────────────────────────────────────
print("Computing ΛCDM reference E²(z_c=16)...")
cosmo_lcdm = Class()
cosmo_lcdm.set(FIXED_COSMO)
cosmo_lcdm.compute()
z_c = 16.0
H0_val = FIXED_COSMO['H0']
H_lcdm_zc = cosmo_lcdm.Hubble(z_c)           # in CLASS units (1/Mpc)
H0_class   = cosmo_lcdm.Hubble(0)
E2_lcdm_zc = (H_lcdm_zc / H0_class)**2
print(f"  E²_ΛCDM(z_c=16) = {E2_lcdm_zc:.5f}")

# Build LCDM H²(z) over a redshift array for ratio comparison
z_arr = np.linspace(0, 60, 720)
H_lcdm_arr = np.array([cosmo_lcdm.Hubble(z) for z in z_arr])
cosmo_lcdm.struct_cleanup()
cosmo_lcdm.empty()

# ── GAUSSIAN SUPPRESSION FACTOR ──────────────────────────────────────────────
sigma_z = FIXED_COSMO['sigma_z_exo']
gauss_factor = np.exp(-z_c**2 / (2 * sigma_z**2))   # ~5.44e-6
print(f"  Gaussian suppression exp(-z_c²/2σ²) = {gauss_factor:.4e}")
print(f"  a_exo = -1838 → Omega_x0 = {-1838 * gauss_factor:.4e}")
print()

# ── MAIN GRID LOOP ────────────────────────────────────────────────────────────
total = N * N
done  = 0

for i, a in enumerate(a_vals):
    for j, b in enumerate(b_vals):
        done += 1
        if done % 50 == 0:
            print(f"  Progress: {done}/{total}  ({100*done/total:.0f}%)")

        # ── Prior cut: b_exo must be < |a_exo| = -a_exo ──────────────────
        if b >= -a:
            status[i, j] = 0
            continue

        # ── Compute Omega_x0 for this a_exo ──────────────────────────────
        Omega_x0 = a * gauss_factor
        omega_x0_arr[i, j] = Omega_x0

        # ── Analytical check: can H² go negative at peak? ─────────────────
        # W(z_c) = A(z_c)/A(0) * exp(z_c^2 / 2*sigma_z^2)
        # A(z_c) = a*(1+z_c) + b*z_c
        # A(0)   = a
        A_zc = a * (1 + z_c) + b * z_c
        A_0  = a
        W_zc = (A_zc / A_0) * np.exp(z_c**2 / (2 * sigma_z**2))

        # Peak contribution to H²/H₀²
        peak_contribution = Omega_x0 * W_zc

        # If peak contribution + E²_LCDM < 0 → H² definitely goes negative
        # Flag analytically before CLASS call to save time
        if peak_contribution + E2_lcdm_zc < 0:
            status[i, j] = 2
            peak_effect[i, j] = abs(peak_contribution) / E2_lcdm_zc
            continue

        # ── CLASS run ────────────────────────────────────────────────────
        params = dict(FIXED_COSMO)
        params['a_exo'] = a
        params['b_exo'] = b

        cosmo = Class()
        cosmo.set(params)

        try:
            cosmo.compute()

            # Compute H²(z)/H²_LCDM(z) over full redshift range
            H_exo_arr = np.array([cosmo.Hubble(z) for z in z_arr])
            ratio_arr  = (H_exo_arr / H_lcdm_arr)**2
            min_ratio  = ratio_arr.min()
            min_h2_ratio[i, j] = min_ratio

            if min_ratio < 0:
                status[i, j] = 2
            else:
                status[i, j] = 3
                peak_effect[i, j] = abs(peak_contribution) / E2_lcdm_zc

        except (CosmoSevereError, CosmoComputationError):
            status[i, j] = 1

        finally:
            cosmo.struct_cleanup()
            cosmo.empty()

print(f"\nGrid complete. Summary:")
print(f"  Prior rejected  (status 0): {np.sum(status == 0)}")
print(f"  CLASS failed    (status 1): {np.sum(status == 1)}")
print(f"  H² negative     (status 2): {np.sum(status == 2)}")
print(f"  Viable          (status 3): {np.sum(status == 3)}")

# ── SAVE RESULTS ─────────────────────────────────────────────────────────────
np.savez('grid_ab_results.npz',
         a_vals=a_vals,
         b_vals=b_vals,
         status=status,
         omega_x0_arr=omega_x0_arr,
         peak_effect=peak_effect,
         min_h2_ratio=min_h2_ratio)
print("\nSaved: grid_ab_results.npz")

# ── PLOT 1: VIABILITY MAP ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Color map for status
cmap_status = mcolors.ListedColormap(['#888888', '#d62728', '#ff7f0e', '#2ca02c'])
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
norm   = mcolors.BoundaryNorm(bounds, cmap_status.N)

AA, BB = np.meshgrid(a_vals, b_vals, indexing='ij')

ax = axes[0]
im = ax.pcolormesh(a_vals, b_vals, status.T,
                   cmap=cmap_status, norm=norm, shading='auto')
# Draw the physicality cut line: b = -a (i.e., b = |a|)
a_line = np.linspace(A_MIN, A_MAX, 200)
ax.plot(a_line, -a_line, 'k--', lw=1.5, label=r'$b_{\rm exo} = |a_{\rm exo}|$')
ax.set_xlabel(r'$a_{\rm exo}$', fontsize=12)
ax.set_ylabel(r'$b_{\rm exo}$', fontsize=12)
ax.set_title('Viability map', fontsize=13)
legend_elements = [
    Patch(facecolor='#888888', label='Prior rejected\n' + r'$(b \geq |a|)$'),
    Patch(facecolor='#d62728', label='CLASS ODE failure'),
    Patch(facecolor='#ff7f0e', label=r'$H^2 < 0$ at peak'),
    Patch(facecolor='#2ca02c', label='Viable'),
]
ax.legend(handles=legend_elements, fontsize=8, loc='upper left')

# ── PLOT 2: PEAK FRACTIONAL EFFECT ON H² ─────────────────────────────────────
ax2 = axes[1]
pe_plot = np.where(status == 3, peak_effect, np.nan)
vmax = np.nanpercentile(pe_plot, 95) if np.any(~np.isnan(pe_plot)) else 1
im2 = ax2.pcolormesh(a_vals, b_vals, pe_plot.T,
                     cmap='plasma', vmin=0, vmax=vmax, shading='auto')
ax2.plot(a_line, -a_line, 'k--', lw=1.5)
plt.colorbar(im2, ax=ax2, label=r'$|\Omega_{x,0} \mathcal{W}(z_c)| / E^2_{\Lambda CDM}(z_c)$')
ax2.set_xlabel(r'$a_{\rm exo}$', fontsize=12)
ax2.set_ylabel(r'$b_{\rm exo}$', fontsize=12)
ax2.set_title(r'Peak fractional effect on $H^2$ (viable only)', fontsize=12)

# ── PLOT 3: Omega_x0 contours in viable region ───────────────────────────────
ax3 = axes[2]
ox_plot = np.where(status == 3, omega_x0_arr, np.nan)
im3 = ax3.pcolormesh(a_vals, b_vals, ox_plot.T,
                     cmap='RdBu_r', shading='auto')
ax3.plot(a_line, -a_line, 'k--', lw=1.5)
plt.colorbar(im3, ax=ax3, label=r'$\Omega_{x,0}$')
ax3.set_xlabel(r'$a_{\rm exo}$', fontsize=12)
ax3.set_ylabel(r'$b_{\rm exo}$', fontsize=12)
ax3.set_title(r'$\Omega_{x,0}$ in viable region', fontsize=12)

plt.tight_layout()
plt.savefig('grid_ab_scan.png', dpi=150, bbox_inches='tight')
print("Saved: grid_ab_scan.png")

# ── PRINT VIABLE REGION STATS ─────────────────────────────────────────────────
viable_mask = status == 3
if np.any(viable_mask):
    print(f"\nViable region statistics:")
    print(f"  a_exo range: [{a_vals[viable_mask.any(axis=1)].min():.1f}, "
          f"{a_vals[viable_mask.any(axis=1)].max():.1f}]")
    print(f"  b_exo range: [{b_vals[viable_mask.any(axis=0)].min():.1f}, "
          f"{b_vals[viable_mask.any(axis=0)].max():.1f}]")
    print(f"  Omega_x0 range: [{omega_x0_arr[viable_mask].min():.4e}, "
          f"{omega_x0_arr[viable_mask].max():.4e}]")
    print(f"  Peak effect range: [{peak_effect[viable_mask].min():.3f}, "
          f"{peak_effect[viable_mask].max():.3f}]")
    print(f"\n  RECOMMENDED PRIOR UPDATE:")
    print(f"  a_exo: [{a_vals[viable_mask.any(axis=1)].min():.0f}, "
          f"{a_vals[viable_mask.any(axis=1)].max():.0f}]")
    print(f"  b_exo: [{b_vals[viable_mask.any(axis=0)].min():.0f}, "
          f"{b_vals[viable_mask.any(axis=0)].max():.0f}]")
else:
    print("\nWARNING: No viable points found. Check CLASS installation and parameters.")
    
# ── PLOT 4: GRADIENT OF PEAK EFFECT (gradual vs cliff) ───────────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

# 4a. min H²/H²_LCDM as continuous variable (not binary)
ax4 = axes2[0]
mhr_plot = np.where(status >= 1, min_h2_ratio, np.nan)
im4 = ax4.pcolormesh(a_vals, b_vals, mhr_plot.T,
                     cmap='RdYlGn', vmin=0, vmax=1.5, shading='auto')
ax4.plot(a_line, -a_line, 'k--', lw=1.5)
plt.colorbar(im4, ax=ax4, label=r'min $H^2(z)/H^2_{\Lambda CDM}(z)$')
ax4.set_xlabel(r'$a_{\rm exo}$', fontsize=12)
ax4.set_ylabel(r'$b_{\rm exo}$', fontsize=12)
ax4.set_title('Smoothness check: min $H^2$ ratio\n(gradual = smooth colour gradient)', fontsize=11)

# 4b. Gradient magnitude of peak_effect — cliff detector
ax5 = axes2[1]
pe_filled = np.where(np.isnan(pe_plot), 0, pe_plot)
grad_a, grad_b = np.gradient(pe_filled, a_vals, b_vals)
grad_mag = np.sqrt(grad_a**2 + grad_b**2)
grad_mag = np.where(status == 3, grad_mag, np.nan)
im5 = ax5.pcolormesh(a_vals, b_vals, grad_mag.T,
                     cmap='hot_r', shading='auto')
ax5.plot(a_line, -a_line, 'k--', lw=1.5)
plt.colorbar(im5, ax=ax5, label='|∇(peak effect)|')
ax5.set_xlabel(r'$a_{\rm exo}$', fontsize=12)
ax5.set_ylabel(r'$b_{\rm exo}$', fontsize=12)
ax5.set_title('Gradient magnitude\n(spikes near boundary = cliff = bad for MCMC)', fontsize=11)

# 4c. Sample rho_x(z) curves across the viable region
ax6 = axes2[2]
z_curves = np.linspace(0, 25, 300)
n_sample = 8
sample_indices = []
for i in range(N):
    for j in range(N):
        if status[i, j] == 3:
            sample_indices.append((i, j))

# Pick evenly spaced samples from the viable region
step = max(1, len(sample_indices) // n_sample)
sampled = sample_indices[::step][:n_sample]

colors = plt.cm.viridis(np.linspace(0, 1, len(sampled)))
for idx, (i, j) in enumerate(sampled):
    a = a_vals[i]
    b = b_vals[j]
    # Compute rho_x(z) analytically (no CLASS needed):
    # rho_x(z) = Omega_x0 * W(z) where
    # W(z) = [a*(1+z)+b*z]/[a] * exp((z_c^2-(z-z_c)^2)/(2*sigma_z^2))
    A_z    = a*(1+z_curves) + b*z_curves
    W_z    = (A_z / a) * np.exp((z_c**2 - (z_curves - z_c)**2) / (2*sigma_z**2))
    Omx    = a * gauss_factor
    rho_x  = Omx * W_z
    ax6.plot(z_curves, rho_x,
             color=colors[idx], alpha=0.8,
             label=f'a={a:.0f}, b={b:.0f}')

ax6.axhline(0, color='k', lw=0.8, ls=':')
ax6.axvline(z_c, color='gray', lw=0.8, ls='--', label=f'$z_c={z_c}$')
ax6.set_xlabel('$z$', fontsize=12)
ax6.set_ylabel(r'$\Omega_{x,0}\,\mathcal{W}(z)$', fontsize=12)
ax6.set_title(r'Sample $\rho_x(z)$ curves from viable region'+'\n(smooth deformation = gradual)',
              fontsize=11)
ax6.legend(fontsize=7, loc='lower right')

plt.tight_layout()
plt.savefig('grid_ab_gradual_check.png', dpi=150, bbox_inches='tight')
print("Saved: grid_ab_gradual_check.png")

print("\nDone.")