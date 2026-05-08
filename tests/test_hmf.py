# test_hmf_audit.py
import numpy as np
from classy import Class
from pipeline.hmf_plugin import compute_hmf
from colossus.cosmology import cosmology as colossus_cosmo
from colossus.lss import mass_function

# --- LCDM CLASS setup ---
params_lcdm = {
    "output"        : "mPk",
    "h"             : 0.6774,
    "Omega_b"       : 0.02230 / 0.6774**2,
    "Omega_cdm"     : 0.1188 / 0.6774**2,
    "n_s"           : 0.9667,
    "sigma8"        : 0.8159,
    "P_k_max_1/Mpc" : 510.0,
    "z_max_pk"      : 22.0,
    "a_exo"         : 0.0,
    "b_exo"         : 0.0,
}

cosmo = Class()
cosmo.set(params_lcdm)
cosmo.compute()

# --- colossus reference ---
colossus_cosmo.setCosmology("planck18")

for z in [8, 10, 15, 20]:
    M_h, dn = compute_hmf(cosmo, z)
    log10_M = np.log10(M_h)

    # colossus SMT at same z
    dn_col = mass_function.massFunction(
        M_h / cosmo.h(),  # M_sun/h for colossus
        z, mdef="fof", model="sheth99",
        q_out="dndlnM"
    ) * cosmo.h()**3     # (Mpc/h)^-3 → Mpc^-3
    # Only compare where both are non-zero and above numerical noise
    mask = (dn > 1e-10) & (dn_col > 1e-10)
    if mask.sum() == 0:
        print(f"z={z}: no overlap in valid mass range")
    else:
        print(f"z={z}: max |ratio-1| = {np.abs(dn[mask]/dn_col[mask] - 1).max():.4f}")
