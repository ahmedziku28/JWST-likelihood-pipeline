# pipeline/data_extractor.py
#
# Loads the UNCOVER photometric and spectroscopic catalogs,
# applies quality cuts, and returns TWO separate astropy Tables
# ready for SMF analysis.
#
#
# Output schema (identical for both tables, exactly 5 columns):
#   z         — photo-z: z_50 (Prospector posterior median)
#               spec-z:  z_spec50 (Prospector posterior median)
#   mstar_50  — log10(M_star/M_sun), posterior median
#   mstar_16  — log10(M_star/M_sun), 16th percentile
#   mstar_84  — log10(M_star/M_sun), 84th percentile
#   mu        — lensing magnification (mu_num_50, clipped to >= 1)
#
# Quantities used for quality cuts but NOT propagated to the output:
#   z_16, z_84  (photo-z)  
#   z_spec16, z_spec84    
# (z errors not propagated; only mstar
#  errors enter the likelihood.)
#
# mstar values remain in log10 units throughout this module.
#
#
# Survey parameters — UNCOVER DR4 at Abell 2744
# Area from Bezanson et al. 2024 (survey overview paper)
import numpy as np
import warnings
from astropy.table import Table
from astropy.units import UnitsWarning

# Survey parameters — UNCOVER DR4 at Abell 2744
UNCOVER_AREA_ARCMIN2 = 45.0
UNCOVER_AREA_SR = UNCOVER_AREA_ARCMIN2 * (np.pi / (180.0 * 60.0)) ** 2
UNCOVER_SKY_FRACTION = UNCOVER_AREA_SR / (4.0 * np.pi)


def load_catalogs(phot_path, zspec_path):
    """
    Loads UNCOVER catalogs, applies quality cuts, and returns 5-column tables.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UnitsWarning)
        cat_phot = Table.read(phot_path)
        cat_zspec = Table.read(zspec_path)

    # --- Photometric cuts (Scheme B) ---
    phot_mask = (
        (cat_phot["use_phot"] == 1)
        & (~np.isnan(cat_phot["z_16"]))
        & (~np.isnan(cat_phot["z_84"]))
        & ((cat_phot["z_50"] - cat_phot["z_16"]) <= 1.0)
        & ((cat_phot["z_84"] - cat_phot["z_50"]) <= 1.0)
        & (~np.isnan(cat_phot["mstar_50"]))
    )
    cat_phot = cat_phot[phot_mask]

    # --- Spectroscopic cuts ---
    spec_mask = (
        (cat_zspec["flag_zspec_qual"] >= 2)
        & (cat_zspec["flag_successful_spectrum"] == 1)
        & (~np.isnan(cat_zspec["z_spec16"]))
        & (~np.isnan(cat_zspec["z_spec84"]))
        & ((cat_zspec["z_spec50"] - cat_zspec["z_spec16"]) <= 1.0)
        & ((cat_zspec["z_spec84"] - cat_zspec["z_spec50"]) <= 1.0)
        & (~np.isnan(cat_zspec["mstar_50"]))
    )
    cat_zspec = cat_zspec[spec_mask]

    # --- Build Photometric Table ---
    phot_table = Table()
    phot_table["z"] = np.array(cat_phot["z_50"], dtype=float)
    phot_table["mstar_50"] = np.array(cat_phot["mstar_50"], dtype=float)
    phot_table["mstar_16"] = np.array(cat_phot["mstar_16"], dtype=float)
    phot_table["mstar_84"] = np.array(cat_phot["mstar_84"], dtype=float)
    phot_table["mu"] = np.clip(np.array(cat_phot["mu_num_50"], dtype=float), 1.0, None)

    # --- Build Spectroscopic Table ---
    spec_table = Table()
    spec_table["z"] = np.array(cat_zspec["z_spec50"], dtype=float)
    spec_table["mstar_50"] = np.array(cat_zspec["mstar_50"], dtype=float)
    spec_table["mstar_16"] = np.array(cat_zspec["mstar_16"], dtype=float)
    spec_table["mstar_84"] = np.array(cat_zspec["mstar_84"], dtype=float)
    spec_table["mu"] = np.clip(np.array(cat_zspec["mu_num_50"], dtype=float), 1.0, None)

#        # Print 5 random entries for the Spectroscopy/SPS Catalog
#     print("\n" + "="*60)
#     print("  SPECTROSCOPY CATALOG (spec_table) — 5 RANDOM ENTRIES")
#     print("="*60)
#     spec_table[np.random.choice(len(spec_table), 5, replace=False)].pprint(max_width=-1)
    
    # --- Final finite check ---
    def _clean(tbl):
        finite = (
            np.isfinite(tbl["z"])
            & np.isfinite(tbl["mstar_50"])
            & np.isfinite(tbl["mu"])
        )
        return tbl[finite]

    return _clean(phot_table), _clean(spec_table)

# if __name__ == "__main__":
#     import sys
#     import argparse

#     parser = argparse.ArgumentParser(
#         description="Load and quality-cut UNCOVER DR4 catalogs, report row counts."
#     )
#     parser.add_argument(
#         "--phot_path",
#         help="Path to UNCOVER_DR4_SPS_catalog.fits",
#         default='/home/lustre_p/ahmed.omar/workspace/exo_de_project/data/UNCOVER_DR4_SPS_catalog.fits'
#     )
#     parser.add_argument(
#         "--zspec_path",
#         help="Path to UNCOVER_DR4_SPS_zspec_catalog.fits",
#         default='/home/lustre_p/ahmed.omar/workspace/exo_de_project/data/UNCOVER_DR4_SPS_zspec_catalog.fits'
#     )
#     args = parser.parse_args()

#     print("=" * 60)
#     print("data_extractor.py  —  UNCOVER DR4 catalog loader")
#     print("=" * 60)

#     phot_table, spec_table = load_catalogs(args.phot_path, args.zspec_path)