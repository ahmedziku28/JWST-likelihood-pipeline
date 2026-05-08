# plot_window_selection.py
#
# Publication figure explaining the choice of z_c and sigma_z.
# Paste into your analysis notebook after loading grid_scan_v2_results.npz.
#
# Two panels:
#   Left:  sigma8 ratio vs redshift at (z_c=16, sigma_z=3.5) for all Omega_x0
#          Shows the enhancement profile and that all amplitudes produce ratio > 1
#   Right: The exotic density window W(z) overlaid on the JWST observation range
#          Shows WHERE the suppression sits relative to the data

import numpy as np
import matplotlib.pyplot as plt

# ---- Load grid scan results ----
data = np.load('../runs/2dGridRun/runs/grid_scan_v2_results.npz')
ratios      = data['ratios']
status      = data['status']
zc_vals     = data['z_c_values']
sz_vals     = data['sigma_z_values']
omx_vals    = data['omega_x0_values']
z_probe     = data['z_probe']

# ---- Chosen values ----
ZC_CHOSEN = 18.0
SZ_CHOSEN = 3.5

bi = np.argmin(np.abs(zc_vals - ZC_CHOSEN))
bj = np.argmin(np.abs(sz_vals - SZ_CHOSEN))

# ---- Figure ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# ==================================================================
#  LEFT PANEL: Enhancement profile at chosen (z_c, sigma_z)
# ==================================================================

# Color map for different Omega_x0 values
# Lighter = weaker amplitude, darker = stronger
colors = [
    '#d0d1e6',  # 1. Very Pale Blue (Weakest)
    '#a6bddb',  # 2. Light Blue
    '#74a9cf',  # 3. Soft Blue
    '#3690c0',  # 4. Medium Blue
    '#0570b0',  # 5. Rich Blue
    '#045a8d',  # 6. Dark Blue
    '#023858',  # 7. Deep Navy
    '#011f3f',  # 8. Midnight Blue
    '#000a18'   # 9. Almost Black (Strongest)
]

for k, ox in enumerate(omx_vals):
    if status[bi, bj, k] != 1:
        continue
    r = ratios[bi, bj, k, :]
    ax1.plot(z_probe, (r - 1) * 100, 'o-',
             color=colors[k], linewidth=1.8, markersize=5,
             label=f'$\\Omega_{{x0}}$ = {ox:.0e}')

ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8)

# Shade the JWST/UNCOVER analysis window
ax1.axvspan(6.0, 14.5, alpha=0.08, color='green')
ax1.text(10.5, ax1.get_ylim()[1] * 0.02 if ax1.get_ylim()[1] > 0 else 0.5,
         'UNCOVER\nanalysis\nwindow',
         ha='center', va='bottom', fontsize=9, color='green', alpha=0.7)

ax1.set_xlabel('Redshift $z$', fontsize=13)
ax1.set_ylabel(r'$\Delta\sigma_8 / \sigma_{8,\Lambda\mathrm{CDM}}$ [%]', fontsize=13)
ax1.set_title(f'Growth enhancement at $z_c={ZC_CHOSEN:.1f}$, '
              f'$\\sigma_z={SZ_CHOSEN:.2f}$', fontsize=13)
ax1.legend(fontsize=9, loc='upper right', framealpha=0.9)
ax1.grid(True, alpha=0.2)
ax1.set_xlim(5.5, 21)

# ==================================================================
#  RIGHT PANEL: The window function W(z) and where it acts
# ==================================================================

z_plot = np.linspace(0, 40, 500)

# Window function (b_exo = 0 for visualization)
a_exo_vis = 1.0  # arbitrary positive for shape visualization
W = np.exp(-(z_plot - ZC_CHOSEN)**2 / (2 * SZ_CHOSEN**2))

# Shade the Gaussian window
ax2.fill_between(z_plot, 0, W, alpha=0.3, color='crimson',
                 label=f'$\\mathcal{{W}}(z)$: $z_c={ZC_CHOSEN:.0f}$, '
                       f'$\\sigma_z={SZ_CHOSEN:.1f}$')
ax2.plot(z_plot, W, '-', color='crimson', linewidth=2)

# Mark the JWST/UNCOVER window
ax2.axvspan(6.0, 20.0, alpha=0.12, color='royalblue',
            label='UNCOVER galaxies ($z=6$–$20$)')

# Mark where enhancement peaks vs where data lives
ax2.annotate('', xy=(10, 0.55), xytext=(ZC_CHOSEN, 0.55),
             arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
ax2.text((10 + ZC_CHOSEN)/2, 0.58, 'Growth\naccumulates',
         ha='center', va='bottom', fontsize=9)

# Mark z_c
ax2.axvline(ZC_CHOSEN, color='crimson', linestyle=':', linewidth=1, alpha=0.7)
ax2.text(ZC_CHOSEN + 0.3, 1.02, f'$z_c={ZC_CHOSEN:.0f}$',
         fontsize=10, color='crimson')

# Mark FWHM
fwhm = 2.355 * SZ_CHOSEN
ax2.annotate('', xy=(ZC_CHOSEN - fwhm/2, 0.5),
             xytext=(ZC_CHOSEN + fwhm/2, 0.5),
             arrowprops=dict(arrowstyle='<->', color='gray', lw=1))
ax2.text(ZC_CHOSEN, 0.43, f'FWHM$\\approx{fwhm:.1f}$',
         ha='center', fontsize=9, color='gray')

# Show that tail at z=0 is negligible
W_at_0 = np.exp(-ZC_CHOSEN**2 / (2 * SZ_CHOSEN**2))
ax2.plot(0, W_at_0, 'kx', markersize=8)
ax2.text(1.5, W_at_0 + 0.02,
         f'$\\mathcal{{W}}(0) = e^{{-z_c^2/2\\sigma_z^2}} \\approx {W_at_0:.1e}$',
         fontsize=8, va='bottom')

ax2.set_xlabel('Redshift $z$', fontsize=13)
ax2.set_ylabel(r'$\mathcal{W}(z)$ (normalized)', fontsize=13)
ax2.set_title('Exotic DE window function', fontsize=13)
ax2.legend(fontsize=9, loc='upper right', framealpha=0.9)
ax2.set_xlim(-1, 35)
ax2.set_ylim(-0.05, 1.15)
ax2.grid(True, alpha=0.2)

plt.tight_layout()
#plt.savefig('fig_window_selection.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'fig_window_selection_{ZC_CHOSEN}_{SZ_CHOSEN}.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nFigure saved.")
print(f"z_c = {ZC_CHOSEN}, sigma_z = {SZ_CHOSEN}")
print(f"FWHM = {fwhm:.1f} redshift units")
print(f"Amplification factor = exp(z_c^2/2*sigma_z^2) = {np.exp(ZC_CHOSEN**2/(2*SZ_CHOSEN**2)):.2e}")
print(f"W(z=0) = {W_at_0:.2e}")
print(f"W(z=1100) = {np.exp(-(1100-ZC_CHOSEN)**2/(2*SZ_CHOSEN**2)):.2e}")