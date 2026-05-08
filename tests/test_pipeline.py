#!/usr/bin/env python3
"""
tests/test_pipeline.py

Standalone 4-panel diagnostic for the full UNCOVER + SMT pipeline.
No Cobaya. No MCMC. Run this before the chains to verify the full stack.

Plots
-----
1. HMF (SMT) at all z-bin midpoints        — validates CLASS + compute_hmf
2. Theory vs observed rho_star(>M_star)     — validates the full data-theory interface
3. ln L vs omega_m scan                     — validates likelihood is cosmology-sensitive
4. (a_exo, b_exo) prior polygon + centroid  — shows the parameter space before MCMC

Usage
-----
python tests/test_pipeline.py \\
    --phot_path  data/UNCOVER_DR4_SPS_catalog.fits \\
    --zspec_path data/UNCOVER_DR4_SPS_zspec_catalog.fits \\
    --mode spectroscopic \\
    --output pipeline_test.png
"""

import argparse
import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')   # no display needed on cluster; remove for interactive
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

# ── Path setup ────────────────────────────────────────────────────────────────
# Script lives in tests/; project root is one level up.
_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from classy import Class
from pipeline.data_extractor   import load_catalogs, UNCOVER_SKY_FRACTION
from pipeline.stellar_mass_function import (
    compute_theory_rho_star, compute_observed_rho_star
)
from pipeline.hmf import compute_hmf


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# Redshift bins: (z_min, z_max, z_mid)
_ZBINS_PHOTOZ = [(6., 8., 7.), (8., 10., 9.), (10., 15., 12.5), (15., 20., 17.5)]
_ZBINS_SPECZ  = [(6., 8., 7.), (8., 10., 9.), (10., 15., 12.5)]

# Physicality polygon constants (same as likelihood file)
_POLY_SLOPE     = -0.07202
_POLY_INTERCEPT = -1381.5969

# Planck 2018 TT,TE,EE+lowE+lensing (Table 2, Aghanim et al. 2020)
_PLANCK = {
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

# Colour palette — one colour per z-bin, consistent across plots
_BIN_COLORS = ['#E63946', '#2A9D8F', '#E9C46A', '#457B9D']


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _make_cosmo(overrides=None):
    """
    Instantiate, configure, and compute a CLASS cosmology.
    Always call cosmo.struct_cleanup(); cosmo.empty() after use.
    """
    p = dict(_PLANCK)
    if overrides:
        p.update(overrides)
    cosmo = Class()
    cosmo.set(p)
    cosmo.compute()
    return cosmo


def _V_survey(cosmo, z_lo, z_hi):
    """
    Comoving survey volume [Mpc^3] for UNCOVER in [z_lo, z_hi).

    V = (4pi/3) * (chi_hi^3 - chi_lo^3) * UNCOVER_SKY_FRACTION

    cosmo.comoving_distance(z) returns chi(z) in Mpc (plain float, not astropy).
    """
    chi_lo = cosmo.comoving_distance(z_lo)   # Mpc
    chi_hi = cosmo.comoving_distance(z_hi)   # Mpc
    return (4.0 / 3.0) * np.pi * (chi_hi**3 - chi_lo**3) * UNCOVER_SKY_FRACTION


def _loglike_at_cosmo(cosmo, catalog, zbins):
    """
    Compute the split-Gaussian log-likelihood for a given CLASS cosmology.
    Mirrors the logic in JWSTLikelihood.logp() exactly.
    Returns -np.inf on any failure.
    """
    ll = 0.0
    for (z_min, z_max, z_mid) in zbins:
        V = _V_survey(cosmo, z_min, z_max)
        if not np.isfinite(V) or V <= 0.0:
            return -np.inf

        M_thr, rho_obs, rho_low, rho_high = compute_observed_rho_star(
            catalog, z_min, z_max, V
        )
        if len(M_thr) == 0:
            continue

        M_h, dndlnm, _ = compute_hmf(cosmo, z_mid)
        rho_theory      = compute_theory_rho_star(M_h, dndlnm, M_thr)

        sigma = np.where(
            rho_theory > rho_obs,
            rho_high - rho_obs,
            rho_obs  - rho_low,
        )
        bad  = ~np.isfinite(sigma) | (sigma <= 0.0)
        good = ~bad
        if not np.any(good):
            continue

        ll -= 0.5 * np.sum(
            ((rho_theory[good] - rho_obs[good]) / sigma[good])**2
        )
    return ll


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 1 — HMF at all z-bin midpoints
# ══════════════════════════════════════════════════════════════════════════════

def plot1_hmf(cosmo, zbins, ax):
    """
    dn/dlnM vs M_h at each bin midpoint.

    Pass criteria:
      - Curves shift LEFT (cutoff to lower mass) as z increases
      - Normalisation ~ 10^{-3}–10^{-6} Mpc^{-3} at M_h ~ 10^{10-12} M_sun
      - No NaNs, no negatives, no identical curves
    """
    for (z_min, z_max, z_mid), c in zip(zbins, _BIN_COLORS):
        M_h, dndlnm, sigma = compute_hmf(cosmo, z_mid)

        # Mask HMF to where it is numerically meaningful
        valid = dndlnm > 0.0
        ax.loglog(M_h[valid], dndlnm[valid], color=c, lw=1.8,
                  label=f'$z = {z_mid}$')

    ax.set_xlabel(r'$M_h \;[M_\odot]$', fontsize=11)
    ax.set_ylabel(r'$dn/d\ln M_h \;[\mathrm{Mpc}^{-3}]$', fontsize=11)
    ax.set_title('1 — HMF (Sheth-Mo-Tormen) @ Planck 2018', fontsize=11, fontweight='bold')
    ax.set_xlim(1e6, 1e16)
    ax.legend(fontsize=9, framealpha=0.8)
    ax.grid(True, alpha=0.25, which='both', ls='--')
    ax.tick_params(labelsize=9)


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 2 — Theory vs observed rho_star
# ══════════════════════════════════════════════════════════════════════════════

def plot2_rho_star(cosmo, catalog, zbins, ax):
    """
    Cumulative stellar mass density: theory (line) vs UNCOVER (points + errorbars).

    Pass criteria:
      - Theory and observed within ~1-2 dex of each other
      - Observed rho_star DECREASES with increasing M_star threshold
      - Theory not identically zero (SHMR inversion must work)
      - No factor-of-h^3 offset between theory and observed
    """
    # Fine M_star grid for smooth theory curve
    M_star_fine = np.logspace(7.5, 12.5, 300)

    for (z_min, z_max, z_mid), c in zip(zbins, _BIN_COLORS):
        V = _V_survey(cosmo, z_min, z_max)
        M_thr, rho_obs, rho_low, rho_high = compute_observed_rho_star(
            catalog, z_min, z_max, V
        )
        if len(M_thr) == 0:
            print(f"  [WARN] No galaxies in z-bin ({z_min}, {z_max}) — skipping")
            continue

        M_h, dndlnm, _ = compute_hmf(cosmo, z_mid)

        # Theory on fine grid (smooth line)
        rho_theory_fine = compute_theory_rho_star(M_h, dndlnm, M_star_fine)

        # Observed data points in log10 space with asymmetric errorbars
        # d(log10 rho) = d(rho) / (rho * ln 10)
        rho_obs_safe  = np.maximum(rho_obs,  1e-30)
        rho_low_safe  = np.maximum(rho_low,  1e-30)
        rho_high_safe = np.maximum(rho_high, 1e-30)

        log_obs  = np.log10(rho_obs_safe)
        err_down = log_obs - np.log10(rho_low_safe)    # downward bar (16th pct)
        err_up   = np.log10(rho_high_safe) - log_obs   # upward bar  (84th pct)

        # Clip to reasonable range to avoid absurd error bars at tails
        err_down = np.clip(err_down, 0.0, 3.0)
        err_up   = np.clip(err_up,   0.0, 3.0)

        # Smooth theory line
        valid_theory = rho_theory_fine > 0.0
        ax.plot(
            np.log10(M_star_fine[valid_theory]),
            np.log10(rho_theory_fine[valid_theory]),
            '-', color=c, lw=2.0, label=f'Theory $z={z_mid}$', zorder=3
        )

        # Observed data points
        ax.errorbar(
            np.log10(M_thr), log_obs,
            yerr=[err_down, err_up],
            fmt='o', color=c, ms=3.5, alpha=0.7, lw=0.8, capsize=2,
            label=f'UNCOVER $z={z_mid}$', zorder=4
        )

    ax.set_xlabel(r'$\log_{10}(M_\star / M_\odot)$', fontsize=11)
    ax.set_ylabel(r'$\log_{10}(\rho_\star\,[M_\odot\,\mathrm{Mpc}^{-3}])$', fontsize=11)
    ax.set_title(r'2 — Theory vs Observed $\rho_\star(>M_\star)$', fontsize=11,
                 fontweight='bold')
    ax.set_xlim(7.5, 12.5)
    ax.set_ylim(-2, 8)
    ax.legend(fontsize=7, ncol=2, framealpha=0.8)
    ax.grid(True, alpha=0.25, ls='--')
    ax.tick_params(labelsize=9)


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 3 — ln L vs omega_m scan
# ══════════════════════════════════════════════════════════════════════════════

def plot3_likelihood_scan(catalog, zbins, ax, n_points=18):
    """
    Evaluate the split-Gaussian log-likelihood across a grid of omega_m = Omega_m h^2.

    This tests the CLASS -> HMF -> rho_star -> likelihood chain end-to-end.
    omega_m is the proxy for cosmological sensitivity; in the actual chains,
    (a_exo, b_exo) drives the cosmological variation through H(z).

    Pass criteria:
      - ln L is NOT flat (likelihood sees the cosmology change)
      - No systematic -inf values inside a reasonable omega_m range
      - A smooth, well-behaved curve (no discontinuous jumps)
    """
    omega_m_planck = _PLANCK['omega_b'] + _PLANCK['omega_cdm']   # 0.14237
    omega_m_grid   = np.linspace(0.10, 0.20, n_points)
    log_likes      = np.full(n_points, np.nan)

    print(f"  Scanning {n_points} omega_m values — initialising CLASS each time...")
    t0 = time.time()
    for k, omega_m in enumerate(omega_m_grid):
        omega_cdm = omega_m - _PLANCK['omega_b']
        if omega_cdm <= 0.0:
            continue
        try:
            cosmo_k = _make_cosmo({'omega_cdm': omega_cdm})
            log_likes[k] = _loglike_at_cosmo(cosmo_k, catalog, zbins)
            cosmo_k.struct_cleanup()
            cosmo_k.empty()
        except Exception as e:
            print(f"    [WARN] omega_m={omega_m:.4f} failed: {e}")
        print(f"    [{k+1:02d}/{n_points}] omega_m={omega_m:.4f}  "
              f"lnL={log_likes[k]:.2f}  ({time.time()-t0:.1f}s elapsed)")

    finite = np.isfinite(log_likes)
    ax.plot(omega_m_grid[finite], log_likes[finite],
            'o-', color='steelblue', lw=1.8, ms=5, zorder=3,
            label=r'$\ln\mathcal{L}(\omega_m)$')
    ax.axvline(omega_m_planck, color='k', ls='--', lw=1.2,
               label=f'Planck: $\\omega_m = {omega_m_planck:.4f}$')

    # Mark the best-fit point
    if np.any(finite):
        best_idx = np.nanargmax(log_likes)
        ax.axvline(omega_m_grid[best_idx], color='crimson', ls=':', lw=1.2,
                   label=f'Peak: $\\omega_m = {omega_m_grid[best_idx]:.4f}$')

    ax.set_xlabel(r'$\omega_m = \Omega_m h^2$', fontsize=11)
    ax.set_ylabel(r'$\ln \mathcal{L}$', fontsize=11)
    ax.set_title(r'3 — Likelihood cosmological sensitivity ($\omega_m$ scan)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, framealpha=0.8)
    ax.grid(True, alpha=0.25, ls='--')
    ax.tick_params(labelsize=9)


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 4 — (a_exo, b_exo) prior polygon + centroid
# ══════════════════════════════════════════════════════════════════════════════

def plot4_polygon(ax):
    """
    Viable prior region in (a_exo, b_exo) space before any MCMC.

    The polygon is defined by four constraints:
      (i)   a_exo in [-1838, ~0]                  (box)
      (ii)  b_exo <= -a_exo                        (s = a_exo+b_exo <= 0)
      (iii) b_exo >= -1.07202*a_exo - 1381.5969   (physicality H^2 >= 0)
      Note: the lower-s box constraint is strictly subsumed by (iii).

    Vertices of the polygon (computed from the intersection of the boundaries):
      A = (0,    0       )  — ΛCDM limit (top-right)
      B = (0,    -1381.60)  — physicality meets right edge (bottom-right)
      C = (-1838, 588.79 )  — physicality meets left edge (bottom-left)
      D = (-1838, 1838.  )  — s=0 line meets left edge (top-left)
    """
    # Exact polygon vertices
    a_left  = -1838.0
    # Vertex A (top-right): a_exo=0, b_exo=0  (s=0 meets right edge)
    A = np.array([0.0,    0.0])
    # Vertex B (bottom-right): a_exo=0, b_exo = physicality lower bound at a=0
    B = np.array([0.0,    _POLY_SLOPE * 0.0 + _POLY_INTERCEPT])
    # In (a,b) space, physicality is: b >= -1.07202*a - 1381.5969
    # Because b_exo = s - a_exo, and physicality in (a,s) is s >= slope*a + intercept:
    #   b + a >= slope*a + intercept => b >= (slope-1)*a + intercept = -1.07202*a - 1381.5969
    _b_phys = lambda a: (_POLY_SLOPE - 1.0) * a + _POLY_INTERCEPT
    # Vertex C (bottom-left): a_exo=-1838, b_exo = physicality lower bound there
    C = np.array([a_left, _b_phys(a_left)])
    # Vertex D (top-left): a_exo=-1838, b_exo = -a_exo (s=0 line: b = -a)
    D = np.array([a_left, -a_left])

    vertices = np.array([A, D, C, B, A])   # closed polygon

    # Shade the interior
    polygon_patch = MplPolygon(
        np.array([A, D, C, B]), closed=True,
        facecolor='steelblue', alpha=0.18, edgecolor='steelblue', lw=1.5,
        label='Viable prior region'
    )
    ax.add_patch(polygon_patch)
    ax.plot(vertices[:, 0], vertices[:, 1], '-', color='steelblue', lw=1.5)

    # Geometric centroid (area-weighted, computed via shoelace for the quadrilateral)
    # Splitting quadrilateral ADCB into two triangles: ADC and ACB
    def _triangle_centroid_and_area(P1, P2, P3):
        cx = (P1[0] + P2[0] + P3[0]) / 3.0
        cy = (P1[1] + P2[1] + P3[1]) / 3.0
        area = 0.5 * abs(
            (P2[0] - P1[0]) * (P3[1] - P1[1])
          - (P3[0] - P1[0]) * (P2[1] - P1[1])
        )
        return np.array([cx, cy]), area

    c1, a1 = _triangle_centroid_and_area(A, D, C)
    c2, a2 = _triangle_centroid_and_area(A, C, B)
    centroid = (a1 * c1 + a2 * c2) / (a1 + a2)

    ax.scatter(*centroid, s=180, color='crimson', zorder=6, marker='*',
               label=(f'Centroid\n'
                      f'$a_{{\\rm exo}}={centroid[0]:.0f}$, '
                      f'$b_{{\\rm exo}}={centroid[1]:.0f}$'))

    # Mark ΛCDM reference
    ax.scatter(0, 0, s=120, color='k', zorder=7, marker='D',
               label=r'$\Lambda$CDM $(0,\,0)$')

    # Annotate vertices
    for label, pt, ha, va in [
        ('A $(0,0)$',            A, 'left',  'bottom'),
        ('B $(0, -1382)$',       B, 'left',  'top'),
        ('C $(-1838, 589)$',     C, 'right', 'top'),
        ('D $(-1838, 1838)$',    D, 'right', 'bottom'),
    ]:
        ax.annotate(label, pt, fontsize=7, ha=ha, va=va,
                    xytext=(4, 4), textcoords='offset points')

    ax.set_xlabel(r'$a_{\rm exo}$', fontsize=11)
    ax.set_ylabel(r'$b_{\rm exo}$', fontsize=11)
    ax.set_title(r'4 — Prior viable region in $(a_{\rm exo},\,b_{\rm exo})$ space',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.25, ls='--')
    ax.tick_params(labelsize=9)

    # Set sensible axis limits with a little padding
    ax.set_xlim(a_left * 1.05, 200)
    ax.set_ylim(B[1] * 1.05, D[1] * 1.05)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Standalone 4-panel pipeline validation for UNCOVER + SMT.'
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
    parser.add_argument('--output', default='pipeline_test.png',
                        help='Output figure path')
    parser.add_argument('--n_scan', type=int, default=18,
                        help='Number of omega_m points in likelihood scan (Plot 3)')
    args = parser.parse_args()

    zbins = _ZBINS_SPECZ if args.mode == 'spectroscopic' else _ZBINS_PHOTOZ

    # ── Step 1: One CLASS at Planck (used for plots 1 and 2) ─────────────────
    print("=" * 60)
    print("Initialising CLASS @ Planck 2018 (for plots 1 & 2)...")
    t0 = time.time()
    cosmo_planck = _make_cosmo()
    print(f"  Done in {time.time()-t0:.1f}s  |  h={cosmo_planck.h():.4f}  "
          f"Omega_m={cosmo_planck.Omega_m():.4f}")

    # ── Step 2: Load catalogs ─────────────────────────────────────────────────
    print(f"Loading catalogs ({args.mode} mode)...")
    phot_table, spec_table = load_catalogs(args.phot_path, args.zspec_path)
    catalog = spec_table if args.mode == 'spectroscopic' else phot_table
    print(f"  {len(catalog)} galaxies after quality cuts")

    # Per-bin galaxy counts (informational)
    z_arr = np.asarray(catalog['z'])
    for (z_min, z_max, z_mid) in zbins:
        n = np.sum((z_arr >= z_min) & (z_arr < z_max))
        V = _V_survey(cosmo_planck, z_min, z_max)
        print(f"  z=[{z_min},{z_max})  n_gal={n:4d}  V={V:.3e} Mpc^3")

    # ── Step 3: Build figure ─────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(
        f'UNCOVER + SMT Pipeline Validation  |  mode={args.mode}  |  Planck 2018',
        fontsize=13, fontweight='bold', y=1.01
    )

    # ── Plot 1 ───────────────────────────────────────────────────────────────
    print("\n[Plot 1] HMF at all z-bin midpoints...")
    plot1_hmf(cosmo_planck, zbins, axes[0, 0])
    print("  Done.")

    # ── Plot 2 ───────────────────────────────────────────────────────────────
    print("\n[Plot 2] Theory vs observed rho_star...")
    plot2_rho_star(cosmo_planck, catalog, zbins, axes[0, 1])
    print("  Done.")

    # Clean up the Planck CLASS object — no longer needed
    cosmo_planck.struct_cleanup()
    cosmo_planck.empty()
    print("  CLASS object freed.")

    # ── Plot 3 ───────────────────────────────────────────────────────────────
    print(f"\n[Plot 3] Likelihood scan over omega_m ({args.n_scan} points)...")
    plot3_likelihood_scan(catalog, zbins, axes[1, 0], n_points=args.n_scan)
    print("  Done.")

    # ── Plot 4 ───────────────────────────────────────────────────────────────
    print("\n[Plot 4] Prior polygon in (a_exo, b_exo) space...")
    plot4_polygon(axes[1, 1])
    print("  Done.")

    # ── Save ─────────────────────────────────────────────────────────────────
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\nSaved to {args.output}")
    print("=" * 60)
    print("WHAT TO CHECK:")
    print("  Plot 1: curves shift LEFT (lower cutoff mass) as z increases.")
    print("  Plot 2: theory and observed within ~1-2 dex; obs decreasing in M_star.")
    print("  Plot 3: smooth curve, NOT flat — likelihood sees the cosmology.")
    print("  Plot 4: crimson star = prior centroid; black diamond = ΛCDM (0,0).")
    print("=" * 60)


if __name__ == '__main__':
    main()