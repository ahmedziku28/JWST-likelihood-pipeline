import numpy as np
import matplotlib.pyplot as plt
import os

def run_labeled_smoothness_check(outdir='scan_out'):
    # 1. Setup the 10x10 grid (100 universes)
    a_vals = np.linspace(-1838, -10, 10)
    b_vals = np.linspace(-1974, 1837, 10)
    z = np.linspace(0, 30, 500)
    
    # 2. Fixed Parameters
    zc, sz = 16.0, 3.25
    g = np.exp(-(z - zc)**2 / (2 * sz**2))
    
    # Baseline LCDM for H(z) panel
    L_z = 0.3 * (1 + z)**3 + 0.7 

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Create a colormap to give each of the 100 lines a unique color
    colors = plt.cm.turbo(np.linspace(0, 1, 100))
    line_idx = 0

    for a in a_vals:
        for b in b_vals:
            # --- THE RAW EQUATION (AS IS) ---
            # No closure subtractions, no normalization
            rho_x = (a + b * z / (z + 1)) * g
            
            # Total Hubble Rate H(z)/H0
            h2_ratio = L_z + rho_x
            h_ratio = np.sqrt(np.where(h2_ratio > 0, h2_ratio, np.nan))

            label_str = f"a={a:.0f}, b={b:.0f}"
            current_color = colors[line_idx]

            ax1.plot(z, rho_x, color=current_color, lw=1.0, label=label_str)
            ax2.plot(z, h_ratio, color=current_color, lw=1.0)
            
            line_idx += 1

    # --- Styling Panel 1: rho_x ---
    ax1.axhline(0, color='k', ls='--', alpha=0.5)
    ax1.set_title(r"Raw Energy Density $\rho_x(z)$", fontsize=14)
    ax1.set_xlabel("$z$", fontsize=12)
    ax1.set_ylabel("Density Contribution", fontsize=12)
    
    # --- Styling Panel 2: H(z) ---
    ax2.plot(z, np.sqrt(L_z), 'k-', lw=2.5, label=r'$\Lambda$CDM Ref', zorder=10)
    ax2.set_title(r"Total Expansion $H(z)/H_0$", fontsize=14)
    ax2.set_xlabel("$z$", fontsize=12)
    ax2.set_ylabel("Hubble Rate", fontsize=12)

    # --- Legend Management ---
    # Placing a 100-item legend is tricky; we'll put it outside or use many columns
    ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1), 
               fontsize=6, ncol=4, title="a, b Combinations", title_fontsize=10)

    plt.suptitle(f"Numerical Smoothness & Gradual Evolution Check\n$N=100$ Unique Modified Gravity Histories", 
                 fontsize=16, y=0.98)
    
    plt.tight_layout(rect=[0, 0, 0.85, 0.95]) # Adjust for the massive legend
    
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, "hashim_labeled_smoothness.png")
    plt.savefig(out_path, dpi=250, bbox_inches='tight')
    plt.close()
    
    print(f"Success! Labeled plot saved to: {out_path}")

run_labeled_smoothness_check()