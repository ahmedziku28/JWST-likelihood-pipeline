# tests/test_growth.py

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from classy import Class
from pipeline.growth_factor import compute_growth_factor


def make_lcdm_cosmo():
    """
    Standard flat ΛCDM with Planck 2018 best-fit parameters.
    No exotic parameters — CLASS will crash if you pass a_exo=0
    because it sets has_exo=FALSE and then finds b_exo unread.
    """
    params = {
        'h'             : 0.6774,
        'Omega_b'       : 0.0486,
        'Omega_cdm'     : 0.2589,
        'Omega_k'       : 0.0,
        'output'        : 'mPk',
        'P_k_max_1/Mpc' : 10.0,
        'z_pk'          : '0',
    }
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()
    return cosmo


def test_matter_domination(cosmo):
    """
    In matter domination D(z) ∝ a = 1/(1+z), so D(z1)/D(z2) = a(z1)/a(z2).

    We test at z = 50, 100, 200 where:
    - Matter dominates (z << z_eq ~ 3400)
    - Dark energy is negligible (z >> z_DE ~ 0.3)
    - Radiation is small but not zero: ~1.5%, ~3%, ~6% of matter density

    Because radiation is not exactly zero, delta ∝ a is approximate.
    We use a 5% tolerance, not 0.1% — this tests qualitative behavior,
    not the exact analytic formula.

    IMPORTANT: Do NOT test at z > 3400 — those redshifts are in radiation
    domination where delta ∝ a does NOT hold at all.
    """
    print("=" * 60)
    print("TEST 1: Matter domination limit  (50 < z < 200)")
    print("        Testing delta ∝ a with 5% tolerance")
    print("        (radiation is 1-6% at these redshifts — not zero)")
    print("=" * 60)

    z_test = np.array([50., 100., 200.])
    D_test = compute_growth_factor(cosmo, z_test)

    all_passed = True
    for i in range(len(z_test) - 1):
        ratio_D = D_test[i] / D_test[i + 1]
        ratio_a = (1 + z_test[i + 1]) / (1 + z_test[i])
        error   = abs(ratio_D / ratio_a - 1)

        # 5% tolerance — accounts for radiation not being exactly zero
        passed     = error < 0.05
        all_passed = all_passed and passed

        # Also show the radiation fraction so we understand the error
        Omega_r = 9e-5   # approximate
        Omega_m = 0.307
        rad_frac_lo = Omega_r * (1 + z_test[i])   / Omega_m * 100
        rad_frac_hi = Omega_r * (1 + z_test[i+1]) / Omega_m * 100

        print(f"  z={z_test[i]:.0f}→{z_test[i+1]:.0f} : "
              f"D ratio={ratio_D:.5f}  a ratio={ratio_a:.5f}  "
              f"error={error:.3f}  "
              f"(rad={rad_frac_lo:.1f}%-{rad_frac_hi:.1f}% of matter)  "
              f"{'PASS' if passed else 'FAIL'}")

    print(f"\n  Overall: {'PASS' if all_passed else 'FAIL'}\n")
    return all_passed


def test_lcdm_shape(cosmo):
    """
    Check that D(z) has the right physical shape:
    1. Monotonically decreasing — structure always grows over time
    2. D(z=0) = 1 exactly — normalization is correct
    3. D(z=1) in the physically expected range for ΛCDM

    We print the full D(z) table so you can inspect the shape visually.
    """
    print("=" * 60)
    print("TEST 2: Physical shape of D(z)")
    print("=" * 60)

    z_test = np.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0])
    D_test = compute_growth_factor(cosmo, z_test)

    # Check 1: monotonically decreasing
    # Structure grows forward in time (decreasing z), so D must
    # decrease as z increases. Any violation means a fundamental bug.
    monotone = all(D_test[i] > D_test[i+1] for i in range(len(D_test)-1))
    print(f"  Monotonically decreasing with z : "
          f"{'PASS' if monotone else 'FAIL'}")

    # Check 2: D(z=0) = 1 exactly by construction
    d0_correct = abs(D_test[0] - 1.0) < 1e-10
    print(f"  D(z=0) = {D_test[0]:.15f} : "
          f"{'PASS' if d0_correct else 'FAIL'}")

    # Print the shape table
    print(f"\n  {'z':>6}   {'D(z)':>10}")
    print(f"  {'-'*20}")
    for z, D in zip(z_test, D_test):
        print(f"  {z:>6.1f}   {D:>10.6f}")

    # Check 3: D(z=1) physically reasonable
    # For Planck parameters (Omega_m ~ 0.307), D(z=1) ~ 0.60-0.62.
    # We use a generous range 0.55-0.70 to be safe.
    d1_reasonable = 0.55 < D_test[2] < 0.70
    print(f"\n  D(z=1) = {D_test[2]:.4f} in range (0.55, 0.70) : "
          f"{'PASS' if d1_reasonable else 'FAIL'}")

    passed = monotone and d0_correct and d1_reasonable
    print(f"\n  Overall: {'PASS' if passed else 'FAIL'}\n")
    return passed


def test_single_redshift(cosmo):
    """
    Regression test: passing a single redshift must not crash.

    This was the original cubic spline error — requesting z=[0.0] gave
    only 2 evaluation points and the cubic spline needs at least 4.
    Fixed by adding a dense internal grid of 500 points in growth_factor.py.
    """
    print("=" * 60)
    print("TEST 3: Single redshift input (regression test)")
    print("=" * 60)

    try:
        D = compute_growth_factor(cosmo, np.array([0.0]))
        print(f"  D(z=0) = {D[0]:.15f}")
        passed = abs(D[0] - 1.0) < 1e-10
        print(f"  {'PASS' if passed else 'FAIL'}\n")
        return passed
    except Exception as e:
        print(f"  CRASH: {e}\n")
        return False


def test_exotic_produces_effect(cosmo_lcdm, cosmo_exo):
    """
    Verify that the exotic dark energy produces a measurable,
    nonzero effect on D(z) compared to ΛCDM.

    We do NOT require D_exo > D_ΛCDM here. The sign of the boost
    depends on (z_c, sigma_z) and will be determined by the 2D grid
    scan. What we verify here is:
    1. The exotic component produces a nonzero change in D(z)
    2. The effect grows with redshift (closer to z_c=30)
    3. The absolute magnitude is physically consistent

    Why the effect can be negative (suppression):
    - Suppressing H near z=30 creates two competing phases:
      * Before z=30 (approach): less friction, more growth (+)
      * After z=30 (recovery): more friction, less growth (-)
    - z=6-12 sits in the recovery phase, so suppression dominates
    - Different (z_c, sigma_z) choices can flip this to enhancement
    - The 2D grid scan finds the optimal values
    """, 2000.
    print("=" * 60)
    print("TEST 4: Exotic dark energy produces nonzero effect on D(z)")
    print("        (sign of effect depends on z_c, sigma_z — see grid scan)")
    print("=" * 60)

    z_high = np.array([5., 6., 8., 10., 12., 14., 16., 18., 20., 22., 30., 40., 50., 70., 100., 200., 500., 1000., 2000.])
    D_lcdm = compute_growth_factor(cosmo_lcdm, z_high)
    D_exo  = compute_growth_factor(cosmo_exo,  z_high)

    boost_values = (D_exo / D_lcdm - 1.0) * 100.0

    print(f"  {'z':>4}   {'D_ΛCDM':>10}   {'D_exo':>10}   {'boost':>10}")
    print(f"  {'-'*45}")
    for z, dl, de, b in zip(z_high, D_lcdm, D_exo, boost_values):
        print(f"  {z:>4.0f}   {dl:>10.6f}   {de:>10.6f}   {b:>+9.4f}%")

    # Check 1: effect is nonzero (code is actually doing something)
    # Use 0.001% as threshold — any smaller is just floating point noise
    effects_are_nonzero = all(abs(b) > 0.001 for b in boost_values)
    print(f"\n  All effects nonzero (|boost| > 0.001%) : "
          f"{'PASS' if effects_are_nonzero else 'FAIL'}")

    # Check 2: effect grows with redshift
    # Whether positive or negative, the magnitude should increase
    # as z approaches z_c=30, because the Gaussian window is larger there
    abs_boosts = np.abs(boost_values)
    effect_grows_with_z = all(
        abs_boosts[i] < abs_boosts[i+1]
        for i in range(len(abs_boosts)-1)
    )
    print(f"  Effect magnitude grows toward z_c=30 : "
          f"{'PASS' if effect_grows_with_z else 'FAIL'}")

    # Check 3: D values are physically valid (positive, less than 1)
    all_valid = all((0 < d < 1) for d in D_exo)
    print(f"  All D_exo values in (0, 1)           : "
          f"{'PASS' if all_valid else 'FAIL'}")

    # Informational: report sign of effect
    sign = "SUPPRESSION" if boost_values[0] < 0 else "ENHANCEMENT"
    print(f"\n  Effect sign at z=6-12: {sign}")
    if boost_values[0] < 0:
        print(f"  This means z_c=30 puts the JWST window in the")
        print(f"  Gaussian recovery phase. The 2D grid scan will")
        print(f"  find which z_c produces enhancement at z=6-12.")

    passed = effects_are_nonzero and effect_grows_with_z and all_valid
    print(f"\n  Overall: {'PASS' if passed else 'FAIL'}\n")
    return passed


if __name__ == '__main__':

    print("\nSetting up ΛCDM cosmology (no exotic parameters)...")
    cosmo_lcdm = make_lcdm_cosmo()
    print("Done.\n")

    print("Setting up exotic dark energy cosmology...")
    print("Using a_exo=-1000 so the boost is large enough to see in the test.")
    print("(Realistic MCMC values will be much smaller — this is just a test.)\n")
    params_exo = {
        'h'            : 0.6774,
        'Omega_b'      : 0.0486,
        'Omega_cdm'    : 0.2589,
        'Omega_k'      : 0.0,
        'output'       : 'mPk',
        'P_k_max_1/Mpc': 10.0,
        'z_pk'         : '0',
        'a_exo'        : -1000.0,
        'b_exo'        : 10.0,
        'z_c_exo'      : 30.0,
        'sigma_z_exo'  : 6.0,
    }
    cosmo_exo = Class()
    cosmo_exo.set(params_exo)
    cosmo_exo.compute()
    print("Done.\n")

    r1 = test_matter_domination(cosmo_lcdm)
    r2 = test_lcdm_shape(cosmo_lcdm)
    r3 = test_single_redshift(cosmo_lcdm)
    r4 = test_exotic_produces_effect(cosmo_lcdm, cosmo_exo)

    cosmo_lcdm.struct_cleanup()
    cosmo_lcdm.empty()
    cosmo_exo.struct_cleanup()
    cosmo_exo.empty()

    print("=" * 60)
    if r1 and r2 and r3 and r4:
        print("ALL TESTS PASSED — growth_factor.py is ready.")
        print("Next step: hmf_plugin.py")
    else:
        print("SOME TESTS FAILED — check output above.")
    print("=" * 60)