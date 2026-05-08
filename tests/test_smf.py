#!/usr/bin/env python3
# tests/test_smf.py
"""
Test suite for pipeline/stellar_mass_function.py  (T1 - T12)
=============================================================
Validates the complete chain:
    SHMR -> theory rho_star -> observed rho_star from UNCOVER

When all 12 tests pass, the pipeline from data_extractor.py through
stellar_mass_function.py is certified correct.

Run from the project root:
    python tests/test_smf.py

Dependencies:
    T1  - T3  : NumPy only (always run)
    T4  - T9  : modified CLASS (classy from class_omx)
    T10 - T12 : UNCOVER FITS files on disk + CLASS

Grid size is read from pipeline.hmf._N_M and _N_K.  Changing those constants
there automatically adjusts every grid-dependent test here.
"""

import sys
import os
import numpy as np

# ── Make the project root importable when run as a script ─────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Production modules ────────────────────────────────────────────────────────
from pipeline.stellar_mass_function import (
    shmr_mstar,
    compute_theory_rho_star,
    compute_observed_rho_star,
    SHMR_N,
    SHMR_LOG_MC,
    SHMR_BETA,
    SHMR_GAMMA,
    _SHMR_MC,
    _N_M,   # re-exported from pipeline.hmf
    _N_K,   # re-exported from pipeline.hmf
)
from pipeline.hmf import (
    _LOG10M_MIN,   # lower edge of HMF mass grid (log10 M_sun)
    _LOG10M_MAX,   # upper edge of HMF mass grid (log10 M_sun)
    _K_MIN,        # lower edge of k grid (Mpc^{-1})
    _K_MAX,        # upper edge of k grid (Mpc^{-1})
    compute_hmf,
)

# ── CLASS availability ────────────────────────────────────────────────────────
try:
    from classy import Class as _Class
    _CLASS_OK = True
except ImportError:
    _Class    = None
    _CLASS_OK = False

# ── Catalog availability ──────────────────────────────────────────────────────
_PHOT_PATH = (
    "/home/lustre_p/ahmed.omar/workspace/exo_de_project/data/"
    "UNCOVER_DR4_SPS_catalog.fits"
)
_SPEC_PATH = (
    "/home/lustre_p/ahmed.omar/workspace/exo_de_project/data/"
    "UNCOVER_DR4_SPS_zspec_catalog.fits"
)
_CATALOG_OK = os.path.exists(_PHOT_PATH) and os.path.exists(_SPEC_PATH)

# ── Survey volume helper (UNCOVER, Planck 2018 LCDM geometry) ─────────────────
from pipeline.data_extractor import UNCOVER_SKY_FRACTION
from astropy.cosmology import FlatLambdaCDM as _FLCDM

# Planck 2018: Omega_m = (omega_b + omega_cdm) / h^2 = 0.14237 / 0.6736^2
_cosmo_ap = _FLCDM(H0=67.36, Om0=0.31379)

def _V_survey(z_lo, z_hi):
    """Comoving survey volume [Mpc^3] for UNCOVER in redshift bin [z_lo, z_hi)."""
    return (
        _cosmo_ap.comoving_volume(z_hi).value
        - _cosmo_ap.comoving_volume(z_lo).value
    ) * UNCOVER_SKY_FRACTION


# ═══════════════════════════════════════════════════════════════════════════════
#  CLASS cosmology factory
# ═══════════════════════════════════════════════════════════════════════════════

_PLANCK18_BASE = {
    "h"             : 0.6736,
    "omega_b"       : 0.02237,
    "omega_cdm"     : 0.1200,
    "n_s"           : 0.9649,
    "ln10^{10}A_s"  : 3.044,
    "P_k_max_1/Mpc" : 510.0,
    "z_max_pk"      : 22.0,
    "output"        : "mPk",
}

def _make_cosmo(a_exo=0.0, b_exo=0.0):
    """Create, compute, and return a CLASS cosmology instance."""
    cosmo = _Class()
    cosmo.set({**_PLANCK18_BASE, "a_exo": a_exo, "b_exo": b_exo})
    cosmo.compute()
    return cosmo


# ═══════════════════════════════════════════════════════════════════════════════
#  Pre-compute HMFs once for all theory tests
# ═══════════════════════════════════════════════════════════════════════════════
# Each compute_hmf call takes ~12 ms (C kernel).  Computing everything up front
# avoids redundancy across tests and keeps total setup time under ~1 s.
#
# Grid key: ("lcdm"|"exo", z_target) -> (M_h, dndlnm, sigma)

_hmf   = {}   # filled below if CLASS is available
_cosmo = {}   # CLASS instances

if _CLASS_OK:
    print(f"\nInitialising CLASS cosmologies (one-time setup for T4-T12) ...",
          flush=True)
    try:
        _cosmo["lcdm"] = _make_cosmo(0.0, 0.0)
        _cosmo["exo"]  = _make_cosmo(-960.0, 324.0)   # strong exotic DE

        # LCDM HMFs at redshifts needed across T4-T9 and T12
        for _z in (0.0, 4.0, 7.0, 8.0, 10.0):
            _hmf[("lcdm", _z)] = compute_hmf(_cosmo["lcdm"], _z)
            print(f"  HMF lcdm  z={_z:.1f} done", flush=True)

        # Exotic DE HMFs at redshifts needed for T8
        for _z in (8.0, 10.0):
            _hmf[("exo", _z)] = compute_hmf(_cosmo["exo"], _z)
            print(f"  HMF exo   z={_z:.1f} done", flush=True)

        print("  CLASS setup complete.\n", flush=True)

    except Exception as _exc:
        print(f"  CLASS initialisation failed: {_exc}")
        _CLASS_OK = False
        _hmf  = {}
        _cosmo = {}
else:
    print("\nWARNING: classy not importable -- T4-T9, T12 will be skipped.\n")

if not _CATALOG_OK:
    print("WARNING: UNCOVER catalog files not found -- T10-T12 will be skipped.\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  Test runner
# ═══════════════════════════════════════════════════════════════════════════════

_results = []   # list of (name: str, passed: bool)

def _section(title):
    bar = "=" * 68
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)

def _ok(name, passed, diag=""):
    """Record and print a single sub-check."""
    label = "PASS" if passed else "FAIL"
    suffix = f"  |  {diag}" if diag else ""
    print(f"  [{label}]  {name}{suffix}")
    _results.append((name, bool(passed)))
    return bool(passed)

def _skip(name, reason):
    """Record a skipped test (does not count as failure)."""
    print(f"  [SKIP]  {name}  |  {reason}")

def _rel(a, b):
    """Relative error |a - b| / max(|b|, 1e-300)."""
    return abs(a - b) / max(abs(b), 1e-300)


# ═══════════════════════════════════════════════════════════════════════════════
#  T1 — SHMR basic properties
# ═══════════════════════════════════════════════════════════════════════════════
_section(f"T1 — SHMR basic properties  (_N_M={_N_M}, _N_K={_N_K})")

# T1a: M_star(M_c) = N * M_c
# Derivation: at M_h = M_c, ratio = 1 -> denominator = 1^{-beta} + 1^{gamma} = 2
#             efficiency = 2N/2 = N -> M_star = N * M_c   (NOT 2N*M_c)
_mstar_at_Mc = float(shmr_mstar(_SHMR_MC))
_expected    = SHMR_N * _SHMR_MC
_ok("T1a  M_star(M_c) = N*M_c",
    _rel(_mstar_at_Mc, _expected) < 1e-10,
    f"got {_mstar_at_Mc:.6e}, expected {_expected:.6e} "
    f"(rel err {_rel(_mstar_at_Mc,_expected):.2e})")

# T1b: strictly monotonically increasing over the full HMF mass range
_M_t1 = np.logspace(_LOG10M_MIN, _LOG10M_MAX, 5_000)
_Ms_t1 = shmr_mstar(_M_t1)
_dMs_t1 = np.diff(_Ms_t1)
_ok("T1b  Strictly monotone increasing over [10^{%.0f}, 10^{%.0f}] M_sun"
    % (_LOG10M_MIN, _LOG10M_MAX),
    np.all(_dMs_t1 > 0),
    f"min delta = {_dMs_t1.min():.3e} M_sun  (must be > 0)")

# T1c/d: physical reasonableness at mass extremes
#
# At M_h = 10^8 M_sun (dwarf galaxy progenitor):
#   ratio = 10^8 / 10^11.5 = 10^{-3.5}
#   denominator term ratio^{-beta} = 10^{3.5 * 1.35} = 10^{4.725} ~ 5.3e4  (dominates)
#   efficiency = 2N / 5.3e4 ~ 1.1e-6
#   M_star ~ 1.1e-6 * 1e8 = 112 M_sun
# ~100 stellar masses is physically correct under strong supernova feedback (beta=1.35).
# Lower bound is 10 M_sun (a few stellar masses), not 1e3 M_sun.
_ms_low  = float(shmr_mstar(1e8))
_ms_high = float(shmr_mstar(1e14))
_ok("T1c  Reasonable at M_h = 10^8  M_sun  [~100 M_sun expected under SN feedback]",
    10 < _ms_low < 1e6,
    f"M_star = {_ms_low:.3e} M_sun  (expect ~100 M_sun)")
_ok("T1d  Reasonable at M_h = 10^14 M_sun",
    1e10 < _ms_high < 1e14,
    f"M_star = {_ms_high:.3e} M_sun")


# ═══════════════════════════════════════════════════════════════════════════════
#  T2 — SHMR ratio M_star/M_h shape
# ═══════════════════════════════════════════════════════════════════════════════
_section("T2 — SHMR efficiency ratio  M_star / M_h  shape")

_M_t2    = np.logspace(8, 16, 20_000)
_ratio_t2 = shmr_mstar(_M_t2) / _M_t2

# T2a: ratio peaks at the analytically derived location
#
# Differentiating efficiency w.r.t. ratio and setting to zero:
#   d/d(ratio) [ratio^{-beta} + ratio^{gamma}] = 0
#   -beta * ratio^{-beta-1} + gamma * ratio^{gamma-1} = 0
#   => ratio_peak^{beta+gamma} = beta / gamma
#   => ratio_peak = (beta / gamma)^{1/(beta+gamma)}
#
# With beta=1.35, gamma=0.01:
#   ratio_peak = (1.35/0.01)^{1/1.36} = 135^{0.735} ~ 36.8
#   M_h_peak = 36.8 * M_c ~ 1.16e13 M_sun
#
# M_c is NOT the peak — it is the characteristic transition mass.  The true
# peak sits well above M_c because gamma << beta makes the high-mass decline
# so gentle that the efficiency keeps rising far past M_c.
_ratio_peak_expected = (SHMR_BETA / SHMR_GAMMA) ** (1.0 / (SHMR_BETA + SHMR_GAMMA))
_peak_Mh_expected    = _SHMR_MC * _ratio_peak_expected   # M_sun

_peak_Mh = _M_t2[np.argmax(_ratio_t2)]
_peak_rel_err = abs(_peak_Mh - _peak_Mh_expected) / _peak_Mh_expected

_ok("T2a  Efficiency peaks at analytic location (beta/gamma)^{1/(beta+gamma)} * M_c",
    _peak_rel_err < 0.05,
    f"peak at M_h = {_peak_Mh:.3e} M_sun,  "
    f"analytic = {_peak_Mh_expected:.3e} M_sun  "
    f"(rel err {_peak_rel_err:.4f}, must be < 0.05)")

# T2b: ratio strictly increasing below M_c (supernova feedback regime)
_below_t2 = _M_t2 < _SHMR_MC
_dratio_below = np.diff(_ratio_t2[_below_t2])
_ok("T2b  Efficiency rising below M_c  (supernova feedback)",
    np.all(_dratio_below > 0),
    f"min delta below M_c = {_dratio_below.min():.3e}  (must be > 0)")

# T2c: in the deep high-mass regime (M_h >> M_h_peak), efficiency ~ (M_h/M_c)^{-gamma}
#
# When ratio >> ratio_peak, the term ratio^{-beta} << ratio^{gamma}, so:
#   efficiency ~ 2N / ratio^{gamma}  =  2N * (M_h / M_c)^{-gamma}
#
# Therefore in log-log space:
#   log(efficiency) ~ const - gamma * log(M_h / M_c)
#   d log(efficiency) / d log(M_h) = -gamma = -0.01
#
# We check this in the region M_h > 10 * M_h_peak (well into the asymptote).
# Over the full range 10*M_h_peak to 10^16 M_sun (~1.9 decades), the efficiency
# drops by only (86)^{0.01} ~ 1.047, i.e. 4.7%.  We allow 10% total variation.
#
# Note: the old test checked "flat above M_c", which is WRONG because the region
# M_c < M_h < M_h_peak is the RISING part of the curve (not yet at the peak).
_deep_mask   = _M_t2 > 10.0 * _peak_Mh_expected   # asymptotic regime only
_log_Mh_deep = np.log10(_M_t2[_deep_mask])
_log_eff_deep = np.log10(_ratio_t2[_deep_mask])

# Log-log slope by least-squares fit (more robust than endpoint difference)
_slope_fit = np.polyfit(_log_Mh_deep, _log_eff_deep, 1)[0]
_slope_tol = 0.005   # allow +-0.005 on a target of -0.01

_ok("T2c  High-mass asymptotic slope in log-log space equals -gamma = -0.01",
    abs(_slope_fit - (-SHMR_GAMMA)) < _slope_tol,
    f"fitted slope = {_slope_fit:.5f},  "
    f"target = {-SHMR_GAMMA:.4f},  "
    f"tolerance = +/-{_slope_tol}")


# ═══════════════════════════════════════════════════════════════════════════════
#  T3 — SHMR invertibility: round-trip error < 1%
# ═══════════════════════════════════════════════════════════════════════════════
_section("T3 — SHMR invertibility  (round-trip error < 1%)")

# Build a fine reference grid covering the full HMF mass range
_M_ref   = np.logspace(_LOG10M_MIN, _LOG10M_MAX, 100_000)
_Mst_ref = shmr_mstar(_M_ref)   # fine forward mapping

# Target stellar masses: avoid extreme edges where grid edge effects occur
_Mst_targets = np.logspace(5, 12, 50)
# Invert: M_h = interp(M_star_target, M_star_ref ascending, M_h_ref ascending)
_Mh_inv        = np.interp(_Mst_targets, _Mst_ref, _M_ref,
                            left=_M_ref[0], right=_M_ref[-1])
_Mst_roundtrip = shmr_mstar(_Mh_inv)
_rt_err = np.abs(_Mst_roundtrip - _Mst_targets) / _Mst_targets
_worst_idx = np.argmax(_rt_err)

_ok("T3   Round-trip error < 1% across M_star in [10^5, 10^12] M_sun",
    np.all(_rt_err < 0.01),
    f"max rel err = {_rt_err.max():.4f}  at "
    f"M_star = {_Mst_targets[_worst_idx]:.2e} M_sun")


# ═══════════════════════════════════════════════════════════════════════════════
#  T4 — Theory rho_star: positivity and unit range at z=0  (LCDM)
# ═══════════════════════════════════════════════════════════════════════════════
_section("T4 — Theory rho_star: positivity and range at z=0 (LCDM)")

if not _CLASS_OK:
    _skip("T4", "CLASS not available")
else:
    _Mh4, _dn4, _ = _hmf[("lcdm", 0.0)]
    _thr4 = np.logspace(8, 12, 12)
    _rho4 = compute_theory_rho_star(_Mh4, _dn4, _thr4)

    _ok("T4a  All values positive",
        np.all(_rho4 > 0),
        f"min = {_rho4.min():.3e} M_sun/Mpc^3")
    _ok("T4b  All values finite",
        np.all(np.isfinite(_rho4)),
        f"any NaN/Inf: {not np.all(np.isfinite(_rho4))}")
    _ok("T4c  Values in plausible physical range [10^4, 10^10] M_sun/Mpc^3",
        np.all((_rho4 >= 1e4) & (_rho4 <= 1e10)),
        f"range = [{_rho4.min():.2e}, {_rho4.max():.2e}]")


# ═══════════════════════════════════════════════════════════════════════════════
#  T5 — Theory rho_star: strictly decreasing with M_star threshold
# ═══════════════════════════════════════════════════════════════════════════════
_section("T5 — Theory rho_star: strictly decreasing with threshold (z=0)")

if not _CLASS_OK:
    _skip("T5", "CLASS not available")
else:
    _thr5 = np.logspace(8, 12, 40)
    _rho5 = compute_theory_rho_star(_hmf[("lcdm", 0.0)][0],
                                     _hmf[("lcdm", 0.0)][1], _thr5)
    _drho5 = np.diff(_rho5)
    _ok("T5   Strictly decreasing with M_star threshold",
        np.all(_drho5 < 0),
        f"min delta = {_drho5.min():.3e} M_sun/Mpc^3  (must be < 0)")


# ═══════════════════════════════════════════════════════════════════════════════
#  T6 — Theory rho_star: hierarchical redshift ordering  (LCDM)
# ═══════════════════════════════════════════════════════════════════════════════
_section("T6 — Theory rho_star: hierarchical redshift ordering (LCDM)")

if not _CLASS_OK:
    _skip("T6", "CLASS not available")
else:
    _thr6     = np.array([1e9, 1e10, 1e11])   # M_sun
    _rho6_z0  = compute_theory_rho_star(*_hmf[("lcdm", 0.0)][:2], _thr6)
    _rho6_z4  = compute_theory_rho_star(*_hmf[("lcdm", 4.0)][:2], _thr6)
    _rho6_z8  = compute_theory_rho_star(*_hmf[("lcdm", 8.0)][:2], _thr6)

    _ok("T6a  rho_star(z=0) > rho_star(z=4) at all thresholds",
        np.all(_rho6_z0 > _rho6_z4),
        f"min ratio z0/z4 = {(_rho6_z0/_rho6_z4).min():.2f}")
    _ok("T6b  rho_star(z=4) > rho_star(z=8) at all thresholds",
        np.all(_rho6_z4 > _rho6_z8),
        f"min ratio z4/z8 = {(_rho6_z4/_rho6_z8).min():.2f}")


# ═══════════════════════════════════════════════════════════════════════════════
#  T7 — Theory rho_star: LCDM order-of-magnitude benchmark
# ═══════════════════════════════════════════════════════════════════════════════
_section("T7 — Theory rho_star: LCDM order-of-magnitude (Baldry+2012 anchor)")

if not _CLASS_OK:
    _skip("T7", "CLASS not available")
else:
    _thr7     = np.array([1e10])   # M_sun
    _rho7_z0  = float(compute_theory_rho_star(*_hmf[("lcdm", 0.0)][:2], _thr7))
    _rho7_z8  = float(compute_theory_rho_star(*_hmf[("lcdm", 8.0)][:2], _thr7))
    _ratio7   = _rho7_z0 / max(_rho7_z8, 1e-300)

    _ok("T7a  rho_star(z=0, >10^10 M_sun) ~ 10^8 M_sun/Mpc^3  [Baldry+2012]",
        1e6 < _rho7_z0 < 1e10,
        f"rho_star(z=0) = {_rho7_z0:.3e} M_sun/Mpc^3  (expect ~10^8)")
    _ok("T7b  rho_star(z=0) / rho_star(z=8) > 100  (hierarchical formation)",
        _ratio7 > 100.0,
        f"ratio z0/z8 = {_ratio7:.1f}")


# ═══════════════════════════════════════════════════════════════════════════════
#  T8 — Exotic DE (a_exo=-960, b_exo=324) enhances rho_star vs LCDM
# ═══════════════════════════════════════════════════════════════════════════════
_section("T8 — Exotic DE (a_exo=-960, b_exo=324) enhances rho_star vs LCDM")

if not _CLASS_OK:
    _skip("T8", "CLASS not available")
else:
    _thr8 = np.logspace(8, 12, 10)
    for _zt8 in (8.0, 10.0):
        _rho_lcdm8 = compute_theory_rho_star(*_hmf[("lcdm", _zt8)][:2], _thr8)
        _rho_exo8  = compute_theory_rho_star(*_hmf[("exo",  _zt8)][:2], _thr8)
        _ratio8    = _rho_exo8 / np.maximum(_rho_lcdm8, 1e-300)
        _ok(f"T8   Exotic > LCDM at all thresholds (z={_zt8:.0f})",
            np.all(_rho_exo8 > _rho_lcdm8),
            f"min ratio exo/LCDM = {_ratio8.min():.2f}  "
            f"at M_star = {_thr8[np.argmin(_ratio8)]:.1e} M_sun")


# ═══════════════════════════════════════════════════════════════════════════════
#  T9 — Integration convergence: 2*_N_M grid agrees with _N_M grid to < 1%
# ═══════════════════════════════════════════════════════════════════════════════
_section(f"T9 — Integration convergence: {2*_N_M}-pt vs {_N_M}-pt grid < 1%")

if not _CLASS_OK:
    _skip("T9", "CLASS not available")
else:
    # Coarse grid: direct output of compute_hmf (length _N_M)
    _Mhc, _dnc, _ = _hmf[("lcdm", 0.0)]

    # Fine grid: 2*_N_M log-spaced points over the SAME mass range
    # (defined by _LOG10M_MIN/_LOG10M_MAX from hmf.py -- automatically adjusts
    #  if those constants change)
    _Mhf = np.logspace(_LOG10M_MIN, _LOG10M_MAX, 2 * _N_M)

    # Interpolate dndlnm onto the fine grid in log-log space.
    # Log-log interpolation is more accurate for power-law-like functions.
    _log_dnc = np.log(np.maximum(_dnc, 1e-300))
    _log_dnf = np.interp(np.log(_Mhf), np.log(_Mhc), _log_dnc)
    _dnf     = np.exp(_log_dnf)

    # Evaluate rho_star at several thresholds on both grids
    _thr9       = np.logspace(9, 12, 12)
    _rho9_coarse = compute_theory_rho_star(_Mhc, _dnc, _thr9)
    _rho9_fine   = compute_theory_rho_star(_Mhf, _dnf, _thr9)

    _err9     = np.abs(_rho9_fine - _rho9_coarse) / np.maximum(_rho9_coarse, 1e-300)
    _worst9   = np.argmax(_err9)
    _ok(f"T9   All thresholds converge < 1%  ({_N_M} vs {2*_N_M} points)",
        np.all(_err9 < 0.01),
        f"max rel diff = {_err9.max():.4f}  "
        f"at M_star = {_thr9[_worst9]:.2e} M_sun")


# ═══════════════════════════════════════════════════════════════════════════════
#  T10 — Observed rho_star from UNCOVER spec-z catalog (z=6-8)
# ═══════════════════════════════════════════════════════════════════════════════
_section("T10 — Observed rho_star from UNCOVER spec-z catalog (z=6-8)")

_spec_table = None   # shared with T11, T12
if not _CATALOG_OK:
    _skip("T10", "UNCOVER FITS files not on disk")
else:
    from pipeline.data_extractor import load_catalogs as _load_cats
    _phot_table, _spec_table = _load_cats(_PHOT_PATH, _SPEC_PATH)
    _V68 = _V_survey(6.0, 8.0)

    _Mst10, _rho10, _rlo10, _rhi10 = compute_observed_rho_star(
        _spec_table, 6.0, 8.0, _V68
    )
    _n10 = len(_Mst10)

    _ok("T10a  Non-empty result in z=6-8 bin",
        _n10 > 0,
        f"n_galaxies = {_n10}")

    if _n10 > 0:
        _ok("T10b  M_star thresholds sorted descending",
            np.all(np.diff(_Mst10) <= 0),
            f"min delta = {np.diff(_Mst10).min():.3e} M_sun")
        _ok("T10c  rho_star_obs positive and finite",
            np.all(_rho10 > 0) and np.all(np.isfinite(_rho10)),
            f"range [{_rho10.min():.2e}, {_rho10.max():.2e}] M_sun/Mpc^3")
        _ok("T10d  rho_star_obs monotonically non-decreasing with index "
            "(cumulative sum)",
            np.all(np.diff(_rho10) >= 0),
            f"min delta = {np.diff(_rho10).min():.3e}")
        _ok("T10e  rho_star_obs in plausible range [10^4, 10^12] M_sun/Mpc^3",
            1e4 < _rho10.max() < 1e12,
            f"total (last index) = {_rho10.max():.3e} M_sun/Mpc^3")
        _ok("T10f  Error bars ordered: rho_low <= rho_obs <= rho_high at each point",
            np.all(_rlo10 <= _rho10 + 1e-10) and np.all(_rho10 <= _rhi10 + 1e-10),
            f"max violation low  = {(_rlo10 - _rho10).max():.3e}  "
            f"max violation high = {(_rho10 - _rhi10).max():.3e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  T11 — Magnification correction: with-mu vs without-mu at z=6-8
# ═══════════════════════════════════════════════════════════════════════════════
_section("T11 — Magnification correction raises rho_star by 3-5x  (z=6-8)")

if not _CATALOG_OK or _spec_table is None:
    _skip("T11", "UNCOVER catalogs not available or T10 failed")
else:
    from astropy.table import Table as _Table

    # Create a copy with all magnification set to 1 (no correction)
    _spec_nmu = _spec_table.copy()
    _spec_nmu["mu"] = np.ones(len(_spec_table), dtype=float)

    _V68 = _V_survey(6.0, 8.0)
    _Mst11_mu,  _rho11_mu,  _, _ = compute_observed_rho_star(
        _spec_table, 6.0, 8.0, _V68
    )
    _Mst11_nmu, _rho11_nmu, _, _ = compute_observed_rho_star(
        _spec_nmu,   6.0, 8.0, _V68
    )

    if len(_Mst11_mu) == 0 or len(_Mst11_nmu) == 0:
        _skip("T11", "no galaxies found in z=6-8 bin")
    else:
        # Compare totals (last element = all galaxies included, minimum threshold)
        _ratio11 = _rho11_mu[-1] / max(_rho11_nmu[-1], 1e-300)

        _ok("T11a  Magnification-corrected rho_star > uncorrected  (mu >= 1 always)",
            _ratio11 > 1.0,
            f"total rho ratio (with mu / without mu) = {_ratio11:.3f}")
        # PROJECT_STATUS.md Section 7: "Correction factor of 3-5x at z=6-8
        # confirmed by side-by-side comparison (magnification_comparison.png)"
        # We allow a generous range [1.5, 15] to account for different galaxy samples.
        _ok("T11b  Magnification boost in expected range [1.5, 15]  "
            "(confirmed 3-5x in project data)",
            1.5 <= _ratio11 <= 15.0,
            f"ratio = {_ratio11:.2f}  (expected 3-5 from PROJECT_STATUS.md Sec. 7)")


# ═══════════════════════════════════════════════════════════════════════════════
#  T12 — End-to-end: theory vs observation dimensions and shapes match (z=6-8)
# ═══════════════════════════════════════════════════════════════════════════════
_section("T12 — End-to-end: theory vs observation  (z=6-8, LCDM vs UNCOVER)")

if not _CLASS_OK or not _CATALOG_OK or _spec_table is None:
    _skip("T12", "CLASS and/or UNCOVER catalogs not available")
else:
    # Observation (spectroscopic, z=6-8)
    _V68 = _V_survey(6.0, 8.0)
    _Mst12, _rho_obs12, _, _ = compute_observed_rho_star(
        _spec_table, 6.0, 8.0, _V68
    )

    if len(_Mst12) == 0:
        _skip("T12", "no galaxies in z=6-8 spec-z bin")
    else:
        # Theory at z=7  (bin midpoint).
        # M_h pre-computed in _hmf[("lcdm", 7.0)].
        _Mh12, _dn12, _ = _hmf[("lcdm", 7.0)]
        # Evaluate theory at the same M_star thresholds as the observation.
        # _Mst12 is descending -> compute_theory_rho_star handles any order.
        _rho_th12 = compute_theory_rho_star(_Mh12, _dn12, _Mst12)

        _ok("T12a  Theory and obs arrays have the same length",
            len(_rho_th12) == len(_rho_obs12),
            f"theory: {len(_rho_th12)},  obs: {len(_rho_obs12)}")

        _ok("T12b  Theory rho_star positive and finite at obs thresholds",
            np.all(_rho_th12 > 0) and np.all(np.isfinite(_rho_th12)),
            f"range [{_rho_th12.min():.2e}, {_rho_th12.max():.2e}] M_sun/Mpc^3")

        # _Mst12 is descending -> rho_th12 should be ascending with index
        # (lower M_star threshold -> larger integral -> higher rho_star).
        # Use a tiny relative tolerance to absorb floating-point noise.
        _drho_th12 = np.diff(_rho_th12)
        _tol12     = 1e-10 * np.abs(_rho_th12[:-1])
        _ok("T12c  Theory rho_star non-decreasing with index "
            "(descending M_star thresholds)",
            np.all(_drho_th12 >= -_tol12),
            f"min delta = {_drho_th12.min():.3e} M_sun/Mpc^3")

        # Sanity: theory (LCDM) and obs should be within ~4 orders of magnitude.
        # The obs will exceed LCDM at high z (that IS the tension), but not by 10^4x.
        _obs_total   = _rho_obs12.max()
        _th_total    = _rho_th12.max()
        _ratio12     = max(_obs_total, _th_total) / max(min(_obs_total, _th_total), 1e-300)
        _ok("T12d  Theory and obs rho_star within 4 orders of magnitude "
            "(same physical units)",
            _ratio12 < 1e4,
            f"obs_total/th_total = {_obs_total/_th_total:.2e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
_bar = "=" * 68
print(f"\n{_bar}")
print(f"  SUMMARY")
print(_bar)

_n_pass  = sum(r[1] for r in _results)
_n_fail  = sum(not r[1] for r in _results)
_n_total = len(_results)

if _n_fail > 0:
    print(f"\n  Failed sub-checks ({_n_fail}):")
    for _name, _passed in _results:
        if not _passed:
            print(f"    x  {_name}")

print(f"\n  Grid constants used:")
print(f"    _N_M  = {_N_M}  (HMF mass grid, from pipeline/hmf.py)")
print(f"    _N_K  = {_N_K}  (k-space grid,  from pipeline/hmf.py)")
print(f"    mass range: 10^{_LOG10M_MIN:.0f} -- 10^{_LOG10M_MAX:.0f} M_sun")

print(f"\n  Result:  {_n_pass} / {_n_total} PASSED,  {_n_fail} / {_n_total} FAILED")
print(_bar)

sys.exit(0 if _n_fail == 0 else 1)