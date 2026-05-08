#!/usr/bin/env python
"""
grid_scan_v2.py — Fixed-Omega_x0 grid scan
===========================================

WHAT CHANGED FROM v1:
  v1 fixed a_exo = -2.5. This means the peak exotic density was always
  -2.5 regardless of (z_c, sigma_z). At high z_c, matter density is
  ~(1+z_c)^3 >> 2.5, so the fractional H(z) suppression was negligible.
  The scan found z_c ~ 12 simply because that's where matter density is
  smallest, NOT because it's the optimal window location.

  v2 fixes Omega_x0 (the exotic density TODAY) and derives a_exo from it:
    a_exo = Omega_x0 / exp(-z_c^2 / 2*sigma_z^2)

  This way the PEAK density at z_c scales as Omega_x0 * exp(z_c^2/2sigma_z^2),
  which can be enormous while Omega_x0 stays tiny. The exponential amplification
  is the entire point of the transient model — it concentrates a huge effect
  at z_c while being invisible at z=0.

  Five different Omega_x0 values verify the response is linear (perturbative).
  If the efficiency metric is consistent across amplitudes, the optimal
  (z_c, sigma_z) is amplitude-independent and safe to fix for MCMC.

METHOD:
  For each (z_c, sigma_z, Omega_x0):
    1. Derive a_exo = Omega_x0 / exp(-z_c^2 / 2*sigma_z^2)
    2. Set b_exo = 0 (pure Gaussian — tilt is a second-order effect)
    3. Pre-check: will E^2(z_c) > 0?  If not, skip (would crash CLASS)
    4. Run CLASS, compute sigma8_ratio(z) at all probe redshifts
    5. Compute efficiency = (ratio - 1) / |Omega_x0|

OUTPUT:
  runs/grid_scan_v2_results.npz

RUNTIME ESTIMATE:
  74 z_c * 36 sigma_z * 5 Omega_x0 = 13,320 CLASS runs
  At ~3 sec each: ~11 hours
"""

import numpy as np
import os
import sys
import time
from classy import Class


# ==================================================================
#  GRID PARAMETERS
# ==================================================================

# z_c range: 8 to 45
#   Below 8: the dip sits inside the JWST window, Omega_x0 is large,
#            model is constrained to death by BAO/Pantheon at low z.
#   Above 45: dip is so far from z=6-20 that accumulated growth
#             enhancement at JWST redshifts is negligible.
Z_C_VALUES = np.arange(7.0, 45.5, 0.5)     # 75 values

# sigma_z range: 3 to 12
#   Below 3: Gaussian is extremely narrow, only affects a tiny z range.
#            Also sigma_z < z_c/5 or so means the exponential amplification
#            exp(z_c^2/2sigma_z^2) is astronomical — a_exo becomes huge
#            and E^2 can go negative.
#   Above 12: Gaussian is so broad that the exotic component bleeds
#             significantly to z=0, making Omega_x0 constraints bite.
SIGMA_Z_VALUES = np.arange(3.0, 12.5, 0.25)   # 37 values

# Omega_x0 test values — the density parameter TODAY
#   These are all negative (required for H suppression).
#   Span from "barely detectable" to "strong effect" to check linearity.
#   In the MCMC, Omega_x0 = a_exo * exp(-z_c^2/2sigma_z^2) is derived.
OMEGA_X0_VALUES = np.array([-1e-5, -3e-5, -5e-5, -1e-4, -3e-4, -5e-4, -1e-3, -3e-3, -5e-3])

# b_exo = 0 for all runs.
# The tilt parameter shifts the effective center by a fraction of sigma_z.
# For finding the optimal LOCATION (z_c, sigma_z), the pure Gaussian
# shape is sufficient.  b_exo is a refinement explored in the MCMC.
B_EXO_TEST = 0.0

# Redshifts at which we probe structure growth
# Covers the full UNCOVER range: z ~ 6 to z ~ 20
Z_PROBE = np.array([6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5,
                     13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 20.0])

# Standard LCDM parameters (Planck 2018)
LCDM_PARAMS = {
    'h'              : 0.6774,
    'Omega_b'        : 0.0486,
    'Omega_cdm'      : 0.2589,
    'Omega_k'        : 0.0,
    'n_s'            : 0.9667,
    'ln10^{10}A_s'   : 3.064,
    'output'         : 'mPk',
    'P_k_max_1/Mpc'  : 20.0,
    'z_pk'           : ', '.join(map(str, [0.0] + list(Z_PROBE))),
    # Neutrinos: 1 massive (0.06 eV) + 2 massless
    'N_ur'           : 2.0328,
    'N_ncdm'         : 1,
    'm_ncdm'         : 0.06,
    'T_ncdm'         : 0.71611,
}

# Cosmological densities for the pre-flight E^2 check
OMEGA_M = LCDM_PARAMS['Omega_b'] + LCDM_PARAMS['Omega_cdm']
OMEGA_R = 9.15e-5   # radiation density, approximate
OMEGA_LAMBDA = 1.0 - OMEGA_M - OMEGA_R   # flatness


# ==================================================================
#  sigma8(z) FROM CLASS P(k,z)
# ==================================================================

def compute_sigma8_at_z(cosmo, z, R_Mpc_h=8.0, n_k=500):
    """
    Compute sigma_8(z) by integrating CLASS P(k,z) with a top-hat
    window function of radius R = 8 Mpc/h.

    sigma^2(R, z) = (1/2pi^2) * integral[ k^2 P(k,z) W^2(kR) dk ]
    W(x) = 3*(sin(x) - x*cos(x)) / x^3  (Fourier top-hat)
    """
    h = cosmo.h()

    # k grid in CLASS units (Mpc^-1)
    k_min = 1e-4
    k_max = 7.0    # W^2(kR) kills integrand above k ~ 5/R ~ 0.4 Mpc^-1
    k = np.logspace(np.log10(k_min), np.log10(k_max), n_k)

    # Top-hat window: R in Mpc = R_Mpc_h / h
    R = R_Mpc_h / h
    x = k * R
    W = np.where(x < 1e-3, 1.0,
                 3.0 * (np.sin(x) - x * np.cos(x)) / x**3)

    # P(k, z) in Mpc^3 — table lookup after cosmo.compute()
    Pk = np.array([cosmo.pk(ki, z) for ki in k])

    # Integrate in log(k): integral f dk = integral f*k d(ln k)
    integrand = k**2 * Pk * W**2 / (2.0 * np.pi**2)
    sigma2 = np.trapz(integrand * k, np.log(k))

    return np.sqrt(sigma2)


# ==================================================================
#  PRE-FLIGHT SAFETY CHECK
# ==================================================================

def check_E2_positive(z_c, sigma_z, Omega_x0):
    """
    Check that E^2(z) > 0 at the most dangerous point (z = z_c)
    BEFORE calling CLASS.  If E^2 < 0, CLASS would either crash or
    produce garbage.

    At z = z_c with b_exo = 0:
      rho_x(z_c) / rho_crit0 = Omega_x0 * W(z_c)
      W(z_c) = exp(z_c^2 / 2*sigma_z^2)   (since b=0, prefactor = 1)

    E^2(z_c) = Omega_m*(1+z_c)^3 + Omega_r*(1+z_c)^4
               + Omega_Lambda + Omega_x0 * exp(z_c^2 / 2*sigma_z^2)

    We require E^2 > 0 with a safety margin of 10%.
    """
    z = z_c
    matter   = OMEGA_M * (1.0 + z)**3
    rad      = OMEGA_R * (1.0 + z)**4
    lam      = OMEGA_LAMBDA
    # Omega_x0 is negative, W(z_c) is the big amplification factor
    exotic   = Omega_x0 * np.exp(z_c**2 / (2.0 * sigma_z**2))

    E2 = matter + rad + lam + exotic

    # Require at least 10% of the LCDM value to survive
    E2_lcdm = matter + rad + lam
    safe = (E2 > 0.1 * E2_lcdm)

    return safe, E2, E2_lcdm


# ==================================================================
#  MAIN SCAN
# ==================================================================

def run_scan():
    os.makedirs('runs', exist_ok=True)

    # ----------------------------------------------------------
    #  Step 1: LCDM reference (run once)
    # ----------------------------------------------------------
    print("Computing LCDM reference...")
    cosmo_ref = Class()
    cosmo_ref.set(LCDM_PARAMS)
    cosmo_ref.compute()
    sigma8_lcdm = np.array([
        compute_sigma8_at_z(cosmo_ref, z) for z in Z_PROBE
    ])
    cosmo_ref.struct_cleanup()
    cosmo_ref.empty()

    print("LCDM sigma8(z):")
    for z, s in zip(Z_PROBE, sigma8_lcdm):
        print(f"  z={z:5.1f}: sigma8 = {s:.6f}")
    print()

    # ----------------------------------------------------------
    #  Step 2: Allocate storage
    # ----------------------------------------------------------
    n_zc  = len(Z_C_VALUES)
    n_sig = len(SIGMA_Z_VALUES)
    n_omx = len(OMEGA_X0_VALUES)
    n_z   = len(Z_PROBE)

    # ratios[i_zc, j_sig, k_omx, l_z] = sigma8_exo / sigma8_LCDM
    ratios  = np.full((n_zc, n_sig, n_omx, n_z), np.nan)
    # Track what happened at each point
    status  = np.zeros((n_zc, n_sig, n_omx), dtype=np.int8)
    # 0 = not run, 1 = success, -1 = E2 failed preflight, -2 = CLASS crashed
    # Store the derived a_exo for reference
    a_exo_grid = np.full((n_zc, n_sig, n_omx), np.nan)

    # ----------------------------------------------------------
    #  Step 3: Run the scan
    # ----------------------------------------------------------
    t_start = time.time()
    n_total = n_zc * n_sig * n_omx
    n_done  = 0
    n_skip  = 0
    n_fail  = 0
    n_ok    = 0

    # Checkpoint interval: save every 500 runs
    CHECKPOINT_INTERVAL = 500
    OUTFILE = 'runs/grid_scan_v2_results.npz'

    for i, z_c in enumerate(Z_C_VALUES):
        for j, sigma_z in enumerate(SIGMA_Z_VALUES):
            for k, Omega_x0 in enumerate(OMEGA_X0_VALUES):

                n_done += 1

                # --- Derive a_exo from Omega_x0 ---
                # Omega_x0 = a_exo * exp(-z_c^2 / 2*sigma_z^2)
                # => a_exo = Omega_x0 / exp(-z_c^2 / 2*sigma_z^2)
                #          = Omega_x0 * exp(z_c^2 / 2*sigma_z^2)
                exp_factor = np.exp(z_c**2 / (2.0 * sigma_z**2))
                a_exo = Omega_x0 * exp_factor
                a_exo_grid[i, j, k] = a_exo

                # --- Pre-flight: E^2(z_c) > 0? ---
                safe, E2, E2_lcdm = check_E2_positive(
                    z_c, sigma_z, Omega_x0)

                if not safe:
                    status[i, j, k] = -1
                    n_skip += 1
                    if n_done % 1000 == 0:
                        elapsed = time.time() - t_start
                        eta = elapsed / n_done * (n_total - n_done)
                        print(f"  [{n_done}/{n_total}]  "
                              f"z_c={z_c:.1f} sz={sigma_z:.1f} "
                              f"Ox={Omega_x0:.0e} → SKIP (E2={E2:.0f})"
                              f"  [{n_ok} ok, {n_skip} skip, {n_fail} fail]"
                              f"  ETA {eta/3600:.1f}h")
                    continue

                # --- Run CLASS ---
                params = dict(LCDM_PARAMS)
                params['a_exo']       = a_exo
                params['b_exo']       = B_EXO_TEST
                params['z_c_exo']     = z_c
                params['sigma_z_exo'] = sigma_z

                cosmo = Class()
                cosmo.set(params)

                try:
                    cosmo.compute()
                except Exception as e:
                    status[i, j, k] = -2
                    n_fail += 1
                    cosmo.struct_cleanup()
                    cosmo.empty()
                    if n_done % 200 == 0:
                        print(f"  [{n_done}/{n_total}]  "
                              f"z_c={z_c:.1f} sz={sigma_z:.1f} "
                              f"Ox={Omega_x0:.0e} a={a_exo:.1f} → "
                              f"CLASS FAIL: {e}")
                    continue

                # --- Compute sigma8 ratios ---
                sig8_exo = np.array([
                    compute_sigma8_at_z(cosmo, z) for z in Z_PROBE
                ])

                cosmo.struct_cleanup()
                cosmo.empty()

                ratios[i, j, k, :] = sig8_exo / sigma8_lcdm
                status[i, j, k] = 1
                n_ok += 1

                # --- Progress ---
                if n_done % 200 == 0:
                    elapsed = time.time() - t_start
                    eta = elapsed / n_done * (n_total - n_done)
                    r85 = ratios[i, j, k, 2]  # z=8.5
                    print(f"  [{n_done}/{n_total}]  "
                          f"z_c={z_c:.1f} sz={sigma_z:.1f} "
                          f"Ox={Omega_x0:.0e} a={a_exo:.1f} → "
                          f"ratio(z=8.5)={r85:.6f}"
                          f"  [{n_ok} ok, {n_skip} skip, {n_fail} fail]"
                          f"  ETA {eta/3600:.1f}h")

                # --- Checkpoint ---
                if n_done % CHECKPOINT_INTERVAL == 0:
                    _save_results(OUTFILE, ratios, status, a_exo_grid,
                                  sigma8_lcdm)
                    print(f"  → Checkpoint saved at {n_done}/{n_total}")

    # ----------------------------------------------------------
    #  Step 4: Final save
    # ----------------------------------------------------------
    _save_results(OUTFILE, ratios, status, a_exo_grid, sigma8_lcdm)

    elapsed_total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"SCAN COMPLETE")
    print(f"{'='*60}")
    print(f"Total runs:   {n_total}")
    print(f"Succeeded:    {n_ok}")
    print(f"E2 skipped:   {n_skip}")
    print(f"CLASS failed: {n_fail}")
    print(f"Total time:   {elapsed_total/3600:.1f} hours")
    print(f"Saved:        {OUTFILE}")

    # ----------------------------------------------------------
    #  Step 5: Quick summary of best point
    # ----------------------------------------------------------
    _print_quick_summary(ratios, status, sigma8_lcdm)


def _save_results(outfile, ratios, status, a_exo_grid, sigma8_lcdm):
    """Save all results to npz file."""
    np.savez(
        outfile,
        z_c_values      = Z_C_VALUES,
        sigma_z_values  = SIGMA_Z_VALUES,
        omega_x0_values = OMEGA_X0_VALUES,
        z_probe         = Z_PROBE,
        b_exo_test      = B_EXO_TEST,
        ratios          = ratios,        # (n_zc, n_sig, n_omx, n_z)
        status          = status,        # (n_zc, n_sig, n_omx)
        a_exo_grid      = a_exo_grid,   # (n_zc, n_sig, n_omx)
        sigma8_lcdm     = sigma8_lcdm,  # (n_z,)
    )


def _print_quick_summary(ratios, status, sigma8_lcdm):
    """Print the best (z_c, sigma_z) for each Omega_x0 value."""
    print(f"\n{'='*60}")
    print("BEST (z_c, sigma_z) AT EACH Omega_x0")
    print("Criterion: highest ratio at z=8.5, subject to all ratios > 1")
    print(f"{'='*60}")

    iz85 = np.argmin(np.abs(Z_PROBE - 8.5))

    for k, Omega_x0 in enumerate(OMEGA_X0_VALUES):
        # Mask: successful runs with all ratios > 1
        ok = (status[:, :, k] == 1)
        all_above_1 = np.all(ratios[:, :, k, :] > 1.0, axis=2)
        valid = ok & all_above_1

        if valid.sum() == 0:
            # Relax: just require ratio at z=8.5 > 1
            valid = ok & (ratios[:, :, k, iz85] > 1.0)

        if valid.sum() == 0:
            print(f"  Omega_x0 = {Omega_x0:.0e}: NO enhancement found")
            continue

        # Best by ratio at z=8.5
        r85 = np.where(valid, ratios[:, :, k, iz85], -np.inf)
        best_flat = np.argmax(r85)
        bi, bj = np.unravel_index(best_flat, r85.shape)

        print(f"  Omega_x0 = {Omega_x0:.0e}:  z_c={Z_C_VALUES[bi]:.1f}"
              f"  sigma_z={SIGMA_Z_VALUES[bj]:.2f}"
              f"  ratio(z=8.5)={ratios[bi,bj,k,iz85]:.6f}"
              f"  ratio(z=6.5)={ratios[bi,bj,k,0]:.6f}"
              f"  min={ratios[bi,bj,k,:].min():.6f}")


if __name__ == '__main__':
    run_scan()