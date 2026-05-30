"""
exo_de_priors.py — pre-CLASS sanity check for the exotic DE MCMC chains.

This prior runs BEFORE Cobaya calls CLASS (priors are evaluated in
the parameter pipeline before the theory block). It rejects proposals
where the exotic DE component causes excessive suppression of E²(z)
in the active window, which would otherwise crash CLASS with either:

  * background_init / evolver_ndf15: step size too small
  * input_fzero_ridder: root must be bracketed in zriddr

The metric is E²_with_exotic / E²_LCDM at worst-case Ω_m (= h_max² →
smallest matter fraction, largest relative exotic effect). A ratio
below the threshold means the exotic has suppressed E² so much that
CLASS's stiff ODE solver loses step-size resolution near the window.

Catching these in the prior is ~10^4× cheaper than the corresponding
CLASS error AND prevents the chain from accumulating toward the
'stuck for 200 attempts' limit during burn-in.

IMPORTANT — this catches the dominant failure mode (vshmr's
SHMR-exotic degeneracy letting the chain probe extreme-s regions
where the exotic component magnitude blows up) but NOT all CLASS
failures. Some failures occur in benign-looking parameter regions
for reasons internal to CLASS's shooting numerics. Those are
handled by raising sampler.mcmc.max_tries in the YAML.

Threshold is calibrated against the empirical chain data:
  Posterior bulk:               ratio ≈ 0.84  (work)
  Fixed-chain extreme corner:   ratio ≈ 0.67  (work)
  Vshmr crash (extreme-s):      ratio ≈ 0.44  (crash)
  ratio = 0.5 sits between, with safety margin on both sides.
"""

import numpy as np

# Physics constants (fixed CLASS inputs for this campaign)
Z_C       = 16.0
SIGMA_Z   = 3.25
H_MAX     = 0.80

# Photons + 3 ν at fixed m_ncdm = 0.06 eV
OMEGA_R_PHYS = 4.16e-5

# Scan only where exotic window matters: z_c ± 3σ_z ≈ [6.25, 25.75]
_Z_LO = max(Z_C - 3.0 * SIGMA_Z, 1.0)
_Z_HI = Z_C + 3.0 * SIGMA_Z
_Z_SCAN = np.linspace(_Z_LO, _Z_HI, 80)
_ONE_PLUS_Z         = 1.0 + _Z_SCAN
_ONE_PLUS_Z_CUBED   = _ONE_PLUS_Z ** 3
_ONE_PLUS_Z_FOURTH  = _ONE_PLUS_Z ** 4
_W_Z = np.exp(-(_Z_SCAN - Z_C) ** 2 / (2.0 * SIGMA_Z ** 2))
_W0  = float(np.exp(-Z_C ** 2 / (2.0 * SIGMA_Z ** 2)))

# Minimum allowed E²_exo / E²_LCDM. Below this, CLASS reliably crashes.
RATIO_THRESHOLD = 0.5

_REJECT = -1e500


def e2_safety_pre_class(a_samp, s, omega_b, omega_cdm):
    """
    Reject proposals where E²(z) is suppressed by more than (1 - RATIO_THRESHOLD)
    relative to ΛCDM, anywhere in [z_c - 3σ_z, z_c + 3σ_z], at the worst-case
    Ω_m (h = h_max).

    Returns 0.0 if safe, -1e500 if reject.
    """
    h2_worst = H_MAX * H_MAX
    Omega_m = (omega_b + omega_cdm) / h2_worst
    Omega_r = OMEGA_R_PHYS / h2_worst

    a_exo = a_samp
    b_exo = s - a_samp
    Omega_x0 = a_exo * _W0

    matter_rad = Omega_m * _ONE_PLUS_Z_CUBED + Omega_r * _ONE_PLUS_Z_FOURTH
    rho_x = (a_exo + b_exo * _Z_SCAN / _ONE_PLUS_Z) * _W_Z

    # ΛCDM baseline (without exotic): Ω_Λ_lcdm = 1 - Ω_m - Ω_r
    E2_lcdm = matter_rad + (1.0 - Omega_m - Omega_r)
    # Modified: Ω_Λ_exo = 1 - Ω_m - Ω_r - Ω_x0, plus rho_x(z)
    E2_exo = matter_rad + (1.0 - Omega_m - Omega_r - Omega_x0) + rho_x

    ratio_min = (E2_exo / E2_lcdm).min()
    if ratio_min < RATIO_THRESHOLD:
        return _REJECT
    return 0.0
