# explore_catalog.py
# Run this ONCE to understand both catalogs before building the pipeline.
# Tells you: column names, galaxy counts per redshift bin, mass ranges,
# and how many zspec galaxies are available at high redshift.

import numpy as np
from astropy.table import Table, join

CATALOG_DIR = (
    '/home/lustre_p/ahmed.omar/workspace/exo_de_project/data/'
)

# ----------------------------------------------------------------
# BLOCK 1: Photometric catalog
# ----------------------------------------------------------------
print("=" * 60)
print("BLOCK 1: Photometric catalog")
print("=" * 60)

cat_phot = Table.read(CATALOG_DIR + 'UNCOVER_DR4_SPS_catalog.fits')

print(f"Total galaxies in photometric catalog: {len(cat_phot)}")

print(f"\nAll column names:")
for col in cat_phot.colnames:
    print(f"  {col}")

# Quality cut
good_phot = cat_phot['use_phot'] == 1
cat_phot_good = cat_phot[good_phot]
print(f"\nGalaxies passing use_phot=1: {len(cat_phot_good)}")

# Find the stellar mass column name — it might vary slightly
print(f"\nStellar mass columns (any column with 'mstar' or 'mass'):")
stellar_cols = [c for c in cat_phot.colnames
                if 'mstar' in c.lower() or 'mass' in c.lower()]
for col in stellar_cols:
    print(f"  {col}")

# Find magnification column name
print(f"\nMagnification columns (any column with 'mu'):")
mu_cols = [c for c in cat_phot.colnames if 'mu' in c.lower()]
for col in mu_cols:
    print(f"  {col}")

# Redshift distribution at high z
print(f"\nGalaxy counts per redshift bin (photometric, use_phot=1):")
# Updated bins to show 12-15 independently
z_bins = [
    (5,6), (6,7), (7,8), (8,9), (9,10), 
    (10,11), (11,12), (12,13), (13,14), (14,15)
]
for z_lo, z_hi in z_bins:
    mask = ((cat_phot_good['z_50'] >= z_lo) &
            (cat_phot_good['z_50'] <  z_hi))
    n = mask.sum()
    print(f"  z=[{z_lo:.0f}, {z_hi:.0f}) : {n:4d} galaxies")

# Stellar mass range at high z — using whatever the column is called
# Try the most likely column names
for mstar_col in ['mstar_50', 'log_mstar_50', 'logmass_50',
                   'log_mass_50', 'stellar_mass']:
    if mstar_col in cat_phot.colnames:
        break
else:
    # Fall back to first stellar mass column found
    mstar_col = stellar_cols[0] if stellar_cols else None

if mstar_col:
    print(f"\nUsing stellar mass column: '{mstar_col}'")
    high_z = ((cat_phot_good['z_50'] > 5) &
               (cat_phot_good['z_50'] <  18))
    cat_hz = cat_phot_good[high_z]
    masses = cat_hz[mstar_col]
    # Remove non-finite values
    finite = np.isfinite(masses)
    masses = masses[finite]
    if len(masses) > 0:
        print(f"  Stellar mass range at 5<z<18:")
        print(f"  min log10(M*/Msun) = {np.nanmin(masses):.2f}")
        print(f"  max log10(M*/Msun) = {np.nanmax(masses):.2f}")
        print(f"  median             = {np.nanmedian(masses):.2f}")

# Show a few example rows
print(f"\nFirst 5 galaxies with 6 < z_50 < 10 and use_phot=1:")
high_z_mask = ((cat_phot_good['z_50'] >= 6) &
                (cat_phot_good['z_50'] <  10))
sample = cat_phot_good[high_z_mask][:5]
for row in sample:
    mstar_val = row[mstar_col] if mstar_col else 'N/A'
    # Pick magnification column
    mu_val = 'N/A'
    for mc in ['mu_num_50', 'mu_50', 'mu']:
        if mc in cat_phot.colnames:
            mu_val = f"{row[mc]:.2f}"
            break
    print(f"  id={row['id']}  z_50={row['z_50']:.2f}  "
          f"log_mstar={mstar_val:.2f}  mu={mu_val}")


# ----------------------------------------------------------------
# BLOCK 2: Spectroscopic catalog
# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("BLOCK 2: Spectroscopic catalog")
print("=" * 60)

cat_zspec = Table.read(
    CATALOG_DIR + 'UNCOVER_DR4_SPS_zspec_catalog.fits'
)

print(f"Total galaxies in zspec catalog: {len(cat_zspec)}")

print(f"\nAll column names:")
for col in cat_zspec.colnames:
    print(f"  {col}")

# Quality cuts for zspec
# flag_zspec_qual >= 2 means solid or secure
# flag_successful_spectrum = 1 means spectrum was usable
qual_col    = 'flag_zspec_qual'
success_col = 'flag_successful_spectrum'

if qual_col in cat_zspec.colnames and success_col in cat_zspec.colnames:
    good_zspec = ((cat_zspec[qual_col] >= 2) &
                  (cat_zspec[success_col] == 1))
    cat_zspec_good = cat_zspec[good_zspec]
    print(f"\nGalaxies passing flag_zspec_qual>=2 "
          f"AND flag_successful_spectrum=1: {len(cat_zspec_good)}")
else:
    # Try without the success flag — it might not exist
    if qual_col in cat_zspec.colnames:
        good_zspec = cat_zspec[qual_col] >= 2
        cat_zspec_good = cat_zspec[good_zspec]
        print(f"\nGalaxies passing flag_zspec_qual>=2: "
              f"{len(cat_zspec_good)}")
    else:
        cat_zspec_good = cat_zspec
        print(f"\nQuality flag columns not found — using all zspec entries")

# Find redshift column in zspec catalog
# It might be z_spec, z_spec50, or z_50
print(f"\nRedshift-related columns in zspec catalog:")
z_cols_zspec = [c for c in cat_zspec.colnames
                if 'z_' in c.lower() or 'redshift' in c.lower()]
for col in z_cols_zspec:
    print(f"  {col}")

# Use whichever z column exists for the redshift distribution
for z_col in ['z_spec', 'z_spec50', 'z_50', 'z_ml']:
    if z_col in cat_zspec_good.colnames:
        break

print(f"\nUsing redshift column: '{z_col}'")
print(f"\nGalaxy counts per redshift bin (zspec, quality cuts applied):")
for z_lo, z_hi in z_bins:
    mask = ((cat_zspec_good[z_col] >= z_lo) &
            (cat_zspec_good[z_col] <  z_hi))
    n = mask.sum()
    print(f"  z=[{z_lo:.0f}, {z_hi:.0f}) : {n:4d} galaxies")

# Stellar mass range at high z from zspec
zspec_stellar_cols = [c for c in cat_zspec.colnames
                      if 'mstar' in c.lower() or 'mass' in c.lower()]
print(f"\nStellar mass columns in zspec catalog:")
for col in zspec_stellar_cols:
    print(f"  {col}")

for mstar_col_z in ['mstar_50', 'log_mstar_50', 'logmass_50',
                     'log_mass_50']:
    if mstar_col_z in cat_zspec_good.colnames:
        break
else:
    mstar_col_z = zspec_stellar_cols[0] if zspec_stellar_cols else None

if mstar_col_z:
    high_z_zspec = ((cat_zspec_good[z_col] > 5) &
                    (cat_zspec_good[z_col] <  18))
    cat_hz_zspec = cat_zspec_good[high_z_zspec]
    masses_zspec = cat_hz_zspec[mstar_col_z]
    finite_z = np.isfinite(masses_zspec)
    masses_zspec = masses_zspec[finite_z]
    if len(masses_zspec) > 0:
        print(f"\nStellar mass range (zspec, 5<z<18, quality cuts):")
        print(f"  min log10(M*/Msun) = {np.nanmin(masses_zspec):.2f}")
        print(f"  max log10(M*/Msun) = {np.nanmax(masses_zspec):.2f}")
        print(f"  median             = {np.nanmedian(masses_zspec):.2f}")
        print(f"  count              = {len(masses_zspec)}")

# Show example rows
print(f"\nFirst 5 good zspec galaxies with 5 < z < 12:")
hz_zspec_mask = ((cat_zspec_good[z_col] > 5) &
                  (cat_zspec_good[z_col] <  12))
sample_z = cat_zspec_good[hz_zspec_mask][:5]
for row in sample_z:
    mstar_val = row[mstar_col_z] if mstar_col_z else 'N/A'
    qual_val  = row[qual_col] if qual_col in cat_zspec.colnames else 'N/A'
    print(f"  id_msa={row['id_msa']}  "
          f"z={row[z_col]:.3f}  "
          f"log_mstar={mstar_val:.2f}  "
          f"qual={qual_val}")


# ----------------------------------------------------------------
# BLOCK 3: Overlap — zspec galaxies in the photometric catalog
# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("BLOCK 3: Overlap — zspec galaxies in the photometric catalog")
print("=" * 60)

# Match on id_msa
if 'id_msa' in cat_phot.colnames and 'id_msa' in cat_zspec_good.colnames:
    # Use np.ma.compressed() to safely extract non-masked IDs
    phot_ids  = set(np.ma.compressed(cat_phot_good['id_msa']))
    zspec_ids = set(np.ma.compressed(cat_zspec_good['id_msa']))
    
    overlap   = phot_ids & zspec_ids
    
    print(f"\nGalaxies in phot catalog with valid id_msa: {len(phot_ids)}")
    print(f"Good zspec galaxies:                       {len(zspec_ids)}")
    print(f"Overlap (zspec AND in phot catalog):       {len(overlap)}")

    # How many of the overlap are at high z?
    if z_col in cat_zspec_good.colnames:
        # Filter for high-z galaxies first, THEN compress to get IDs
        hz_mask = (cat_zspec_good[z_col] > 5) & (cat_zspec_good[z_col] < 18)
        hz_zspec_ids = set(np.ma.compressed(cat_zspec_good[hz_mask]['id_msa']))
        
        hz_overlap = phot_ids & hz_zspec_ids
        print(f"Of those, with 5 < zspec < 18:             {len(hz_overlap)}")
else:
    print("\nid_msa column not found in one or both catalogs")
    print("Cannot compute overlap — will need a different matching column")

print("\n" + "=" * 60)
print("EXPLORATION COMPLETE")
print("Paste this output and we will build the SMF pipeline.")
print("=" * 60)