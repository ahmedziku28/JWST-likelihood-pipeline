#!/usr/bin/env python3
# ============================================================================
# generate_all_runs.py
#
# Generate the complete 112-run MCMC infrastructure for the exotic transient
# dark energy campaign on the HPC cluster.
#
# Builds folder hierarchy under ./runs/ with one .yaml + one .sh + outputs/
# per run, plus runs/README.md and runs/all_runs.csv.
#
# Python 3.8 compatible. Standard library only. No external dependencies.
#
# Run once:
#     python generate_all_runs.py
# ============================================================================

import os
import sys
import csv
from typing import Dict, List, Optional, Any, Tuple


# ============================================================================
#  CONSTANTS — paths, fixed exotic-DE params, polygon
# ============================================================================

RUNS_ROOT              = "runs"
LIKELIHOOD_PYTHON_PATH = "/home/lustre_p/ahmed.omar/workspace/exo_de_project/likelihood/"
VENV_ACTIVATE          = "/home/lustre_p/ahmed.omar/workspace/venvs/exo_DE/bin/activate"
CLIK_PROFILE           = ("/home/lustre_p/ahmed.omar/workspace/Modules/cobaya_stuff/"
                          "cobaya_packages/code/planck/clik-main/bin/clik_profile.sh")
COBAYA_CONFIG_HOME     = "/home/lustre_p/ahmed.omar/workspace/Modules/cobaya_stuff/cobaya_config"
COBAYA_CACHE_HOME      = "/home/lustre_p/ahmed.omar/workspace/Modules/cobaya_stuff/cobaya_cache"
COBAYA_PACKAGES_PATH   = "/home/lustre_p/ahmed.omar/workspace/Modules/cobaya_stuff/cobaya_packages"

# Exotic DE Gaussian window — fixed from prior grid scan
Z_C_EXO     = 16.0
SIGMA_Z_EXO = 3.25

# Polygon constraint: s >= POLY_SLOPE * a_samp + POLY_INTERCEPT
POLY_SLOPE     = -0.07202
POLY_INTERCEPT = -1381.5969


# ============================================================================
#  COMBINATORIAL DEFINITIONS
# ============================================================================

UVLF_DATA_COMBOS = [
    'ceers',         'primer',         'uvlf',
    'ceers_bg',      'primer_bg',      'uvlf_bg',
    'ceers_bg_cmb',  'primer_bg_cmb',  'uvlf_bg_cmb',
]

SHMR_OPTIONS = ['fixed', 'vbeta', 'vshmr']
ZCUT_OPTIONS = ['full', 'restr']
MODELS       = ['exo', 'lcdm']

NON_UVLF_DATA_COMBOS = ['bg', 'bg_cmb']


# ============================================================================
#  ALL_RUNS GENERATION — single source of truth
# ============================================================================

def build_all_runs():
    # type: () -> Dict[str, Dict[str, Any]]
    """Construct the full 112-run dict combinatorially.

    UVLF runs:   2 models * 9 data combos * 3 SHMR * 2 zcuts = 108
    Non-UVLF:    2 models * 2 data combos                    =   4
    Total:                                                     112
    """
    runs = {}  # type: Dict[str, Dict[str, Any]]

    # ── 108 UVLF runs ────────────────────────────────────────────────────
    for model in MODELS:
        for zcut in ZCUT_OPTIONS:
            for shmr in SHMR_OPTIONS:
                for data in UVLF_DATA_COMBOS:
                    run_name = "{}_{}_{}_{}".format(model, data, shmr, zcut)

                    has_bg  = '_bg' in data
                    has_cmb = '_cmb' in data

                    # Survey isolation
                    if data.startswith('ceers'):
                        use_donnan, use_finkelstein = False, True
                    elif data.startswith('primer'):
                        use_donnan, use_finkelstein = True, False
                    else:  # 'uvlf' (combined)
                        use_donnan, use_finkelstein = True, True

                    # Parameter count
                    P_cosmo = 6 if has_cmb else 3   # CMB: theta_s + ωb + ωcdm + τ + logA + ns
                                                    # nonCMB: H0 + ωb + ωcdm
                    P_exo   = 2 if model == 'exo' else 0      # a_samp, s
                    P_shmr  = {'fixed': 0, 'vbeta': 1, 'vshmr': 3}[shmr]
                    P_total = P_cosmo + P_exo + P_shmr

                    # Folder hierarchy: runs/{exotic|lcdm}/{full|restr}/{shmr}/{run_name}/
                    model_dir = 'exotic' if model == 'exo' else 'lcdm'
                    folder    = "{}/{}/{}/{}".format(model_dir, zcut, shmr, run_name)

                    runs[run_name] = {
                        'run_name':              run_name,
                        'model':                 model,
                        'data':                  data,
                        'has_uvlf':              True,
                        'has_bg':                has_bg,
                        'has_cmb':               has_cmb,
                        'use_donnan_bins':       use_donnan,
                        'use_finkelstein_bins':  use_finkelstein,
                        'shmr':                  shmr,
                        'zcut':                  zcut,
                        'n_sampled_params':      P_total,
                        'folder_path':           folder,
                    }

    # ── 4 non-UVLF runs ──────────────────────────────────────────────────
    for model in MODELS:
        for data in NON_UVLF_DATA_COMBOS:
            run_name = "{}_{}".format(model, data)
            has_cmb  = 'cmb' in data
            P_cosmo  = 6 if has_cmb else 3
            P_exo    = 2 if model == 'exo' else 0
            P_total  = P_cosmo + P_exo

            runs[run_name] = {
                'run_name':              run_name,
                'model':                 model,
                'data':                  data,
                'has_uvlf':              False,
                'has_bg':                True,
                'has_cmb':               has_cmb,
                'use_donnan_bins':       False,
                'use_finkelstein_bins':  False,
                'shmr':                  None,
                'zcut':                  None,
                'n_sampled_params':      P_total,
                'folder_path':           "non_uvlf/{}".format(run_name),
            }

    return runs


# ============================================================================
#  YAML BUILDER — formatted strings, no PyYAML
# ============================================================================

def _yaml_bool(b):
    # type: (bool) -> str
    """Return canonical lowercase YAML boolean."""
    return "true" if b else "false"


def make_yaml(cfg):
    # type: (Dict[str, Any]) -> str
    """Build the full YAML string for one run from its configuration."""
    L = []  # accumulator for output lines

    # ── Theory ────────────────────────────────────────────────────────────
    L.append("# ── Theory ─────────────────────────────────────────────────")
    L.append("theory:")
    L.append("  classy:")
    L.append("    path: global")
    L.append("    stop_at_error: False")
    L.append("    extra_args:")
    if cfg['model'] == 'exo':
        L.append("      z_c_exo: {}".format(Z_C_EXO))
        L.append("      sigma_z_exo: {}".format(SIGMA_Z_EXO))
    else:
        L.append("      use_exotic_DE: False")
        
    L.append("      N_ncdm: 1")
    L.append("      N_ur: 2.0328")
    L.append("      P_k_max_1/Mpc: 150.0")
    L.append("      z_max_pk: 16.0")
    if cfg['has_cmb']:
        L.append("      non_linear: halofit")
        
    L.append("")
    # ── Likelihood ───────────────────────────────────────────────────────
    L.append("# ── Likelihood ─────────────────────────────────────────────")
    L.append("likelihood:")

    if cfg['has_uvlf']:
        # z-cut values: full → defaults, restr → tightened low-z exclusion
        if cfg['zcut'] == 'full':
            z_min_don, z_min_fink = 9.0, 8.9
        else:
            z_min_don, z_min_fink = 10.0, 10.9

        vary_SHMR = (cfg['shmr'] == 'vshmr')
        vary_beta = (cfg['shmr'] == 'vbeta')

        L.append("  jwst_likelihood_uvlf.UVLFLikelihood:")
        L.append('    python_path: "{}"'.format(LIKELIHOOD_PYTHON_PATH))
        L.append("    integrate_bin: True")
        L.append("    use_volume_correction: True")
        L.append("    n_gl: 2")
        L.append("    z_min_donnan: {}".format(z_min_don))
        L.append("    z_min_finkelstein: {}".format(z_min_fink))
        L.append("    use_donnan_bins: {}".format(cfg['use_donnan_bins']))
        L.append("    use_finkelstein_bins: {}".format(cfg['use_finkelstein_bins']))
        L.append("    vary_SHMR: {}".format(vary_SHMR))
        L.append("    vary_beta: {}".format(vary_beta))

    if cfg['has_bg']:
        L.append("  H0.riess2020: null")
        L.append("  sn.pantheonplus:")
        L.append("    use_abs_mag: False")
        L.append("  bao.desi_dr2: null")

    if cfg['has_cmb']:
        L.append("  planck_2018_lowl.TT_clik: null")
        L.append("  planck_2018_lowl.EE_clik: null")
        L.append("  planck_2018_highl_plik.TTTEEE_lite: null")
        L.append("  planck_2018_lensing.native: null")

    L.append("")

    # ── Polygon physicality prior (exotic only) ───────────────────────────
    if cfg['model'] == 'exo':
        L.append("# ── Polygon physicality prior (H^2(z) >= 0) ────────────────")
        L.append("prior:")
        # Render intercept sign cleanly: "+ 0.5" or "- 1381.5969", not "+ -1381.5969"
        if POLY_INTERCEPT >= 0:
            intercept_str = "+ {}".format(POLY_INTERCEPT)
        else:
            intercept_str = "- {}".format(abs(POLY_INTERCEPT))
        L.append('  h2_positivity: "lambda a_samp, s: 0.0 if s >= ({} * a_samp {}) else -1e500"'.format(
            POLY_SLOPE, intercept_str))
        # Pre-CLASS E² safety check (rejects worst-case Ω_m proposals that
        # would crash CLASS's background ODE / θ_s shooter). Defined in
        # pipeline/exo_de_priors.py. Only meaningful when ω_b and ω_cdm
        # are sampled (any run with bg or CMB likelihood — i.e. any
        # production exo run except the pure-UVLF ones at fixed cosmology).
        if cfg['has_bg'] or cfg['has_cmb']:
            L.append("  e2_pre_class: \"__import__('likelihood.exo_de_priors', fromlist=['_']).e2_safety_pre_class\"")
        L.append("")


    # ── Params ───────────────────────────────────────────────────────────
    L.append("# ── Parameters ─────────────────────────────────────────────")
    L.append("params:")

    # Exotic DE sampled (exotic only)
    if cfg['model'] == 'exo':
        # ref distribution widens slightly when CMB is included
        if cfg['has_cmb']:
            a_samp_ref = "{min: -100.0, max: -5.0}"
            s_ref      = "{min: -300.0, max: -50.0}"
        else:
            a_samp_ref = "{min: -75.0, max: -5.0}"
            s_ref      = "{min: -250.0, max: -50.0}"

        L.append("  # ── Exotic DE sampled ──")
        L.append("  a_samp:")
        L.append("    prior: {min: -350.0, max: -1.0e-10}")
        L.append("    ref: " + a_samp_ref)
        L.append("    proposal: 6.0")
        L.append("    drop: true")
        L.append("    latex: a_{\\rm samp}")
        L.append("  s:")
        L.append("    prior: {min: -1000.0, max: -1.0e-10}")
        L.append("    ref: " + s_ref)
        L.append("    proposal: 18.0")
        L.append("    drop: true")
        L.append("    latex: \\mathcal{S}")
        L.append("")

        # Exotic DE derived
        L.append("  # ── Exotic DE derived ──")
        L.append("  a_exo:")
        L.append("    value: 'lambda a_samp: a_samp'")
        L.append("    latex: a_{\\rm exo}")
        L.append("  b_exo:")
        L.append("    value: 'lambda a_samp, s: s - a_samp'")
        L.append("    latex: b_{\\rm exo}")
        L.append("  Omega_x0:")
        L.append("    derived: 'lambda a_exo: a_exo * 5.45846e-6'")
        L.append("    latex: \\Omega_{\\mathrm{x},0}")
        L.append("")

    # Cosmological parameters — branch on CMB presence
    if cfg['has_cmb']:
        # θs replaces H0; τ, logA, n_s now sampled
        L.append("  # ── Sampled cosmological (CMB) ──")
        L.append("  theta_s_100:")
        L.append("    prior: {min: 0.9, max: 1.2}")
        L.append("    ref: {dist: norm, loc: 1.0416, scale: 0.0004}")
        L.append("    proposal: 0.0002")
        L.append("    latex: 100\\theta_{\\rm s}")

        # omega_b prior: BBN if bg present; wide uniform if CMB only (theoretical only here)
        if cfg['has_bg']:
            omega_b_prior = "{dist: norm, loc: 0.02235, scale: 0.00037}"
        else:
            omega_b_prior = "{min: 0.005, max: 0.04}"
        L.append("  omega_b:")
        L.append("    prior: " + omega_b_prior)
        L.append("    ref: {dist: norm, loc: 0.02237, scale: 0.00010}")
        L.append("    proposal: 0.00008")
        L.append("    latex: \\Omega_{\\rm b} h^2")

        L.append("  omega_cdm:")
        L.append("    prior: {min: 0.05, max: 0.18}")
        L.append("    ref: {dist: norm, loc: 0.12, scale: 0.001}")
        L.append("    proposal: 0.0005")
        L.append("    latex: \\Omega_{\\rm c} h^2")

        L.append("  tau_reio:")
        L.append("    prior: {min: 0.01, max: 0.8}")
        L.append("    ref: {dist: norm, loc: 0.055, scale: 0.006}")
        L.append("    proposal: 0.003")
        L.append("    latex: \\tau_{\\rm reio}")

        L.append("  logA:")
        L.append("    prior: {min: 1.61, max: 3.91}")
        L.append("    ref: {dist: norm, loc: 3.05, scale: 0.001}")
        L.append("    proposal: 0.001")
        L.append("    drop: true")
        L.append("    latex: \\log(10^{10} A_{\\rm s})")

        L.append("  n_s:")
        L.append("    prior: {min: 0.8, max: 1.2}")
        L.append("    ref: {dist: norm, loc: 0.965, scale: 0.004}")
        L.append("    proposal: 0.002")
        L.append("    latex: n_{\\rm s}")
        L.append("")

    else:
        # No CMB: H0 sampled directly; τ, A_s, n_s fixed
        L.append("  # ── Sampled cosmological (non-CMB) ──")
        L.append("  H0:")
        L.append("    prior: {min: 60.0, max: 80.0}")
        L.append("    ref: {dist: norm, loc: 67.4, scale: 4.0}")
        L.append("    proposal: 0.5")
        L.append("    latex: H_0")

        # omega_b prior: BBN if bg present; tight uniform if no bg, no CMB
        if cfg['has_bg']:
            omega_b_prior = "{dist: norm, loc: 0.02235, scale: 0.00037}"
        else:
            omega_b_prior = "{min: 0.015, max: 0.030}"
        L.append("  omega_b:")
        L.append("    prior: " + omega_b_prior)
        L.append("    ref: {dist: norm, loc: 0.02237, scale: 0.00010}")
        L.append("    proposal: 0.00008")
        L.append("    latex: \\Omega_{\\rm b} h^2")

        L.append("  omega_cdm:")
        L.append("    prior: {min: 0.05, max: 0.18}")
        L.append("    ref: {dist: norm, loc: 0.1200, scale: 0.002}")
        L.append("    proposal: 0.001")
        L.append("    latex: \\Omega_{\\rm c} h^2")
        L.append("")

        L.append("  # ── Fixed primordial / CMB-like parameters ──")
        L.append("  tau_reio:")
        L.append("    value: 0.0544")
        L.append("    latex: \\tau_{\\rm reio}")
        L.append("  logA:")
        L.append("    value: 3.044")
        L.append("    drop: true")
        L.append("    latex: \\log(10^{10} A_{\\rm s})")
        L.append("  A_s:")
        L.append("    value: 'lambda logA: 1e-10*np.exp(logA)'")
        L.append("    latex: A_{\\rm s}")
        L.append("  n_s:")
        L.append("    value: 0.9649")
        L.append("    latex: n_{\\rm s}")
        L.append("")

    # SHMR sampled params (UVLF runs only)
    if cfg['has_uvlf']:
        if cfg['shmr'] == 'vbeta':
            L.append("  # ── SHMR sampled (vary beta only) ──")
            L.append("  shmr_beta:")
            L.append("    prior: {dist: norm, loc: 1.35, scale: 0.26}")
            L.append("    ref: 1.35")
            L.append("    proposal: 0.12")
            L.append("    latex: \\beta_{\\rm SHMR}")
            L.append("")
        elif cfg['shmr'] == 'vshmr':
            L.append("  # ── SHMR sampled (full Stefanon DPL) ──")
            L.append("  shmr_log_Mc:")
            L.append("    prior: {dist: norm, loc: 11.5, scale: 0.2}")
            L.append("    ref: 11.5")
            L.append("    proposal: 0.08")
            L.append("    latex: \\log_{10}(M_c)")
            L.append("  shmr_N:")
            L.append("    prior: {dist: norm, loc: 0.0297, scale: 0.0065}")
            L.append("    ref: 0.0297")
            L.append("    proposal: 0.003")
            L.append("    latex: N_{\\rm SHMR}")
            L.append("  shmr_beta:")
            L.append("    prior: {dist: norm, loc: 1.35, scale: 0.26}")
            L.append("    ref: 1.35")
            L.append("    proposal: 0.12")
            L.append("    latex: \\beta_{\\rm SHMR}")
            L.append("")

    # Neutrino mass — every run
    L.append("  # ── Neutrino mass (fixed) ──")
    L.append("  m_ncdm:")
    L.append("    value: 0.06")
    L.append("    renames: mnu")
    L.append("")

    # Derived parameters — every run
    L.append("  # ── Derived parameters ──")
    L.append("  Omega_m:")
    L.append("    derived: true")
    L.append("    latex: \\Omega_m")
    L.append("  Omega_Lambda:")
    L.append("    derived: true")
    L.append("    latex: \\Omega_\\Lambda")
    L.append("  sigma8:")
    L.append("    latex: \\sigma_8")
    L.append("  Omega_b:")
    L.append("    derived: 'lambda omega_b, H0: omega_b * (100/H0)**2'")
    L.append("    latex: \\Omega_{\\rm b}")
    L.append("  Omega_cdm:")
    L.append("    derived: 'lambda omega_cdm, H0: omega_cdm * (100/H0)**2'")
    L.append("    latex: \\Omega_{\\rm c}")
    L.append("  S8:")
    L.append("    derived: 'lambda sigma8, Omega_m: sigma8*(Omega_m/0.3)**0.5'")
    L.append("    latex: S_8")
    L.append("  age:")
    L.append("    derived: true")
    L.append("    latex: t_0 \\,[\\rm Gyr]")
    L.append("  rs_drag:")
    L.append("    derived: true")
    L.append("    latex: r_{\\rm s,drag} \\,[\\rm Mpc]")

    if cfg['has_cmb']:
        # H0 is now derived from theta_s; A_s is derived from logA; z_reio is a CLASS output
        L.append("  H0:")
        L.append("    latex: H_0")
        L.append("  A_s:")
        L.append("    value: 'lambda logA: 1e-10*np.exp(logA)'")
        L.append("    latex: A_{\\rm s}")
        L.append("  z_reio:")
        L.append("    latex: z_{\\rm re}")

    L.append("")

    # Disable drag/oversampling specifically for exo+UVLF+vshmr. Empirically:
    #   - exo+vbeta+drag+UVLF+CMB:   fine (R-1 dropping at acc=0.34)
    #   - lcdm+vshmr+drag+UVLF+CMB:  fine (R-1=0.07 at acc=0.31)
    #   - exo+vshmr+drag+UVLF:       broken (stalls, severe chain imbalance)
    #   - exo+vshmr+drag+UVLF+CMB:   broken (was the original case)
    # The pathology requires exo + vshmr + drag jointly; CMB amplifies cost
    # but isn't required to trigger. Vbeta's 1 fast param vs vshmr's 3 is
    # presumably the difference in failure threshold.
    is_exo_vshmr = (cfg.get('model') == 'exo' and cfg.get('shmr') == 'vshmr')
    has_fast_block = (
        cfg['has_uvlf']
        and cfg.get('shmr') in ('vbeta', 'vshmr')
        and not is_exo_vshmr
    )
    drag_val           = 'true' if has_fast_block else 'false'
    oversample_pow_val = '0.4'  if has_fast_block else '0'

    L.append("# ── Sampler ────────────────────────────────────────────────")
    L.append("sampler:")
    L.append("  mcmc:")
    L.append("    covmat: auto")
    L.append("    drag: {}".format(drag_val))
    L.append("    oversample_power: {}".format(oversample_pow_val))
    L.append("    proposal_scale: 1.9")
    L.append("    Rminus1_stop: 0.02")
    L.append("    Rminus1_cl_stop: 0.2")
    L.append("    learn_every: '11d'")
    # Survive sporadic CLASS shooting failures during burn-in
    # 2000 gives huge margin without ever
    # masking a real stuck chain because by then the run is unrecoverable
    # for other reasons).
    L.append("    max_tries: 2000")
    L.append("")

    # Output
    L.append("output: outputs/{}".format(cfg['run_name']))

    return "\n".join(L) + "\n"


# ============================================================================
#  SLURM .sh BUILDER
# ============================================================================

def make_sh(cfg):
    # type: (Dict[str, Any]) -> str
    """Build the SLURM submission script string for one run."""
    run_name = cfg['run_name']
    mem      = 8000 if cfg['has_cmb'] else 5000
    clik_line = ("source " + CLIK_PROFILE) if cfg['has_cmb'] else ""

    sh = (
"#!/bin/bash\n"
"#SBATCH --job-name={run_name}\n"
"#SBATCH --nodes=1\n"
"#SBATCH --exclude=lustre,cernnode02,cernnode03,nut01,nut02\n"
"#SBATCH --output={run_name}.log\n"
"#SBATCH --error={run_name}.err\n"
"#SBATCH --ntasks=8\n"
"#SBATCH --time=175:00:00\n"
"#SBATCH --cpus-per-task=1\n"
"#SBATCH --mem-per-cpu={mem}\n"
"\n"
"echo \"======================================================\"\n"
"echo \"Job started on $(hostname) at $(date)\"\n"
"echo \"======================================================\"\n"
"\n"
"module purge\n"
"module load mpi/openmpi-x86_64\n"
"\n"
"source /opt/rh/devtoolset-8/enable\n"
"source {venv}\n"
"\n"
"{clik_line}\n"
"\n"
"export XDG_CONFIG_HOME=\"{cobaya_config}\"\n"
"export XDG_CACHE_HOME=\"{cobaya_cache}\"\n"
"export COBAYA_PACKAGES_PATH='{cobaya_packages}'\n"
"export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK\n"
"\n"
"echo \"Checking which CLASS is loaded...\"\n"
"python -c \"import classy; print('Loaded CLASS from:', classy.__file__)\"\n"
"\n"
"# ── Auto-retry loop ──────────────────────────────────────────────────\n"
"max_retries=50\n"
"retry_count=0\n"
"command=\"mpirun -np $SLURM_NTASKS cobaya-run {run_name}.yaml --resume\"\n"
"\n"
"until $command; do\n"
"  exit_code=$?\n"
"  retry_count=$((retry_count + 1))\n"
"\n"
"  # Segfault / signal kill = transient HPC crash, reset counter\n"
"  if [ $exit_code -eq 139 ] || [ $exit_code -eq 134 ] || [ $exit_code -eq 137 ]; then\n"
"    echo \"Signal kill (exit code $exit_code) at $(date). Resetting retry counter.\"\n"
"    retry_count=0\n"
"  fi\n"
"\n"
"  if [ $retry_count -ge $max_retries ]; then\n"
"    echo \"FATAL: $max_retries consecutive non-signal failures. Giving up.\"\n"
"    exit 1\n"
"  fi\n"
"\n"
"  echo \"Crash #${{retry_count}}/${{max_retries}} (exit code $exit_code). Retrying in 15s...\"\n"
"  sleep 15\n"
"done\n"
"\n"
"echo \"======================================================\"\n"
"echo \"Run converged at $(date) after $retry_count crash(es).\"\n"
"echo \"======================================================\"\n"
    )

    return sh.format(
        run_name        = run_name,
        mem             = mem,
        venv            = VENV_ACTIVATE,
        clik_line       = clik_line,
        cobaya_config   = COBAYA_CONFIG_HOME,
        cobaya_cache    = COBAYA_CACHE_HOME,
        cobaya_packages = COBAYA_PACKAGES_PATH,
    )


# ============================================================================
#  YAML PARAM COUNTER — line-based, no library
# ============================================================================

def count_yaml_params(yaml_str):
    # type: (str) -> int
    """Count sampled parameters by parsing the YAML line by line.

    A parameter is 'sampled' iff its sub-block (at indent 4) contains a
    `prior:` key. The top-level `prior:` block (which holds h2_positivity)
    is explicitly excluded.

    Strategy:
      - Track whether we are inside the top-level `params:` block.
      - Inside it, a new parameter is any line at indent 2 ending in ':'.
      - Within that parameter's sub-block, look for a `prior:` at indent 4.
    """
    lines = yaml_str.split('\n')
    in_params = False
    sampled = 0
    current_param = None
    current_has_prior = False

    for line in lines:
        stripped = line.lstrip()
        # Skip blank lines and comment-only lines
        if not stripped or stripped.startswith('#'):
            continue

        indent = len(line) - len(stripped)

        # Top-level key
        if indent == 0:
            # Finalise the previous parameter before switching blocks
            if current_param is not None and current_has_prior:
                sampled += 1
            current_param = None
            current_has_prior = False
            in_params = line.startswith('params:')
            continue

        if not in_params:
            continue

        if indent == 2 and line.rstrip().endswith(':'):
            # New parameter at depth 2 — finalise previous one
            if current_param is not None and current_has_prior:
                sampled += 1
            current_param = stripped.rstrip(':')
            current_has_prior = False
        elif indent == 4 and current_param is not None and stripped.startswith('prior:'):
            current_has_prior = True
        # other indent-4 keys (ref, proposal, drop, latex, value, derived) are ignored

    # Final parameter
    if current_param is not None and current_has_prior:
        sampled += 1

    return sampled


# ============================================================================
#  README BUILDER
# ============================================================================

def make_readme(n_runs):
    # type: (int) -> str
    return (
"# MCMC Run Infrastructure\n"
"\n"
"This directory contains **{n} predetermined MCMC runs** for the exotic\n"
"transient dark-energy campaign. Generated by `generate_all_runs.py`.\n"
"\n"
"## Folder hierarchy\n"
"\n"
"```\n"
"runs/\n"
"├── exotic/\n"
"│   ├── full/  {{fixed,vbeta,vshmr}}/  <run_name>/  ({{.yaml, .sh, outputs/}})\n"
"│   └── restr/ {{fixed,vbeta,vshmr}}/  <run_name>/\n"
"├── lcdm/\n"
"│   ├── full/  {{fixed,vbeta,vshmr}}/  <run_name>/\n"
"│   └── restr/ {{fixed,vbeta,vshmr}}/  <run_name>/\n"
"└── non_uvlf/\n"
"    └── <run_name>/                    ({{.yaml, .sh, outputs/}})\n"
"```\n"
"\n"
"## Run-name convention\n"
"\n"
"`{{model}}_{{data}}_{{shmr}}_{{zcut}}` for UVLF runs (108), `{{model}}_{{data}}` for\n"
"non-UVLF runs (4).\n"
"\n"
"| Field   | Values |\n"
"|---------|--------|\n"
"| `model` | `exo` (exotic DE active), `lcdm` (exotic DE disabled) |\n"
"| `data`  | `ceers`, `primer`, `uvlf`, `ceers_bg`, `primer_bg`, `uvlf_bg`, `ceers_bg_cmb`, `primer_bg_cmb`, `uvlf_bg_cmb`, `bg`, `bg_cmb` |\n"
"| `shmr`  | `fixed`, `vbeta`, `vshmr` |\n"
"| `zcut`  | `full` (z >= 9.0/8.9), `restr` (z >= 10.0/10.9) |\n"
"\n"
"## Submitting runs\n"
"\n"
"**Recommended workflow** (via `run_manager.py`):\n"
"```\n"
"python run_manager.py test                      # submit the 6 test runs\n"
"python run_manager.py status                    # monitor progress\n"
"# once all 6 are CONVERGED:\n"
"python run_manager.py auto 8                    # daemon: keep 8 jobs active\n"
"```\n"
"Converged runs are automatically skipped by `launch` and `auto`.\n"
"\n"
"**Manual single-run submission**:\n"
"```\n"
"cd runs/<path>/<run_name> && sbatch <run_name>.sh\n"
"```\n"
"\n"
"## Files\n"
"\n"
"- `all_runs.csv` — single source of truth for downstream analysis (run name,\n"
"  flags, parameter count, folder path).\n"
"- `run_manager.log` — created by `run_manager.py auto` when active.\n"
"\n"
"## Diagnostics\n"
"\n"
"```\n"
"python run_manager.py doctor <run_name>         # diagnose a STALLED/FAILED run\n"
"python run_manager.py resubmit <run_name>       # resubmit a STALLED run\n"
"python run_manager.py resubmit --all-stalled    # resubmit all STALLED runs\n"
"```\n"
"\n".format(n=n_runs)
    )


# ============================================================================
#  MAIN
# ============================================================================

def main():
    print("=" * 72)
    print("  generate_all_runs.py — building the 112-run MCMC infrastructure")
    print("=" * 72)

    ALL_RUNS = build_all_runs()

    # Sanity: exactly 112 runs
    assert len(ALL_RUNS) == 112, "Expected 112 runs, got {}".format(len(ALL_RUNS))

    # Write folders + YAML/SH files
    n_files = 0
    for run_name, cfg in ALL_RUNS.items():
        folder = os.path.join(RUNS_ROOT, cfg['folder_path'])
        os.makedirs(folder, exist_ok=True)
        os.makedirs(os.path.join(folder, 'outputs'), exist_ok=True)

        yaml_path = os.path.join(folder, "{}.yaml".format(run_name))
        sh_path   = os.path.join(folder, "{}.sh".format(run_name))

        with open(yaml_path, 'w') as f:
            f.write(make_yaml(cfg))
        with open(sh_path, 'w') as f:
            f.write(make_sh(cfg))
        try:
            os.chmod(sh_path, 0o755)
        except OSError:
            pass
        n_files += 2

    print("\nWrote {} files across {} runs.".format(n_files, len(ALL_RUNS)))

    # ────────────────────────────────────────────────────────────────────
    #  VALIDATION
    # ────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  VALIDATION")
    print("=" * 72)

    all_ok = True

    # 1. Likelihood python_path exists (warn-only; we may be off-cluster)
    if os.path.isdir(LIKELIHOOD_PYTHON_PATH):
        print("  [OK]   Likelihood python_path exists: {}".format(LIKELIHOOD_PYTHON_PATH))
    else:
        print("  [WARN] Likelihood python_path NOT present here (probably off-HPC):")
        print("         {}".format(LIKELIHOOD_PYTHON_PATH))
        print("         (this is fine if you generated the runs on your laptop)")

    # 2. No duplicate run names (dict keys ensure this; assert for paranoia)
    assert len(set(ALL_RUNS.keys())) == len(ALL_RUNS), "duplicate run names detected!"
    print("  [OK]   No duplicate run names ({} unique).".format(len(ALL_RUNS)))

    # 3. Every UVLF run has at least one survey active
    for rn, cfg in ALL_RUNS.items():
        if cfg['has_uvlf']:
            if not (cfg['use_donnan_bins'] or cfg['use_finkelstein_bins']):
                print("  [FAIL] {} has no survey active".format(rn))
                all_ok = False
    print("  [OK]   Every UVLF run has at least one survey active.")

    # 4. By construction, shmr is a single category — vary_SHMR and vary_beta cannot both be true
    print("  [OK]   No run has both vary_SHMR and vary_beta (single-category by construction).")

    # 5. LCDM runs contain no exotic DE params
    forbidden_in_lcdm = ['h2_positivity', 'e2_pre_class', 'a_samp:', 'a_exo:', 'b_exo:', 'Omega_x0:']
    
    for rn, cfg in ALL_RUNS.items():
        if cfg['model'] == 'lcdm':
            with open(os.path.join(RUNS_ROOT, cfg['folder_path'],
                                   "{}.yaml".format(rn))) as f:
                text = f.read()
            for token in forbidden_in_lcdm:
                if token in text:
                    print("  [FAIL] LCDM run {} contains forbidden token: {}".format(rn, token))
                    all_ok = False
            # ' s:' as a bare param name is harder to grep for without false positives,
            # but it's enclosed in the params block only — and forbidden in lcdm
            if "\n  s:\n" in text or "\n  s:\r" in text:
                print("  [FAIL] LCDM run {} contains sampled `s:` param".format(rn))
                all_ok = False
    print("  [OK]   LCDM runs contain no exotic DE parameters.")

    # 6. CMB runs use theta_s_100 (no H0 prior); non-CMB use H0 (no theta_s_100)
    for rn, cfg in ALL_RUNS.items():
        with open(os.path.join(RUNS_ROOT, cfg['folder_path'],
                               "{}.yaml".format(rn))) as f:
            text = f.read()
        if cfg['has_cmb']:
            if 'theta_s_100:' not in text:
                print("  [FAIL] CMB run {} missing theta_s_100".format(rn))
                all_ok = False
            # Make sure H0 isn't sampled (no prior block on H0)
            if "  H0:\n    prior:" in text:
                print("  [FAIL] CMB run {} has H0 sampled".format(rn))
                all_ok = False
        else:
            if 'theta_s_100:' in text:
                print("  [FAIL] Non-CMB run {} contains theta_s_100".format(rn))
                all_ok = False
            if "  H0:\n    prior:" not in text:
                print("  [FAIL] Non-CMB run {} missing H0 sampled".format(rn))
                all_ok = False
    print("  [OK]   CMB runs use theta_s_100; non-CMB runs use H0.")

    # 7. Parameter count check — parse every YAML, compare against expected
    mismatches = []
    for rn, cfg in ALL_RUNS.items():
        with open(os.path.join(RUNS_ROOT, cfg['folder_path'],
                               "{}.yaml".format(rn))) as f:
            text = f.read()
        actual = count_yaml_params(text)
        expected = cfg['n_sampled_params']
        if actual != expected:
            mismatches.append((rn, expected, actual))
    if mismatches:
        print("  [FAIL] {} runs have parameter-count mismatches:".format(len(mismatches)))
        for rn, exp, act in mismatches:
            print("         {} : expected P={}, got P={}".format(rn, exp, act))
        all_ok = False
    else:
        print("  [OK]   All 112 runs match the expected sampled-parameter count.")

    # 8. Volume correction is always True for UVLF runs
    for rn, cfg in ALL_RUNS.items():
        if cfg['has_uvlf']:
            with open(os.path.join(RUNS_ROOT, cfg['folder_path'],
                                   "{}.yaml".format(rn))) as f:
                text = f.read()
            if 'use_volume_correction: True' not in text:
                print("  [FAIL] UVLF run {} missing use_volume_correction: True".format(rn))
                all_ok = False
    print("  [OK]   Every UVLF run has use_volume_correction: True.")

    # 9. CMB runs have CLIK line in .sh; non-CMB runs do not
    for rn, cfg in ALL_RUNS.items():
        with open(os.path.join(RUNS_ROOT, cfg['folder_path'],
                               "{}.sh".format(rn))) as f:
            sh_text = f.read()
        has_clik = 'clik_profile.sh' in sh_text
        if cfg['has_cmb'] and not has_clik:
            print("  [FAIL] CMB run {} missing CLIK line in .sh".format(rn))
            all_ok = False
        if not cfg['has_cmb'] and has_clik:
            print("  [FAIL] Non-CMB run {} has CLIK line in .sh".format(rn))
            all_ok = False
    print("  [OK]   CMB runs include CLIK line; non-CMB runs do not.")

    # 10. Memory: CMB → 8000, non-CMB → 5000
    for rn, cfg in ALL_RUNS.items():
        with open(os.path.join(RUNS_ROOT, cfg['folder_path'],
                               "{}.sh".format(rn))) as f:
            sh_text = f.read()
        expected_mem = 8000 if cfg['has_cmb'] else 5000
        if "mem-per-cpu={}".format(expected_mem) not in sh_text:
            print("  [FAIL] Run {} has wrong mem-per-cpu (expected {})".format(rn, expected_mem))
            all_ok = False
    print("  [OK]   Memory allocation matches CMB/non-CMB rule.")

    # ────────────────────────────────────────────────────────────────────
    #  CSV + README
    # ────────────────────────────────────────────────────────────────────
    csv_path = os.path.join(RUNS_ROOT, 'all_runs.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run_name', 'model', 'data', 'shmr', 'zcut',
                    'has_uvlf', 'has_bg', 'has_cmb',
                    'use_donnan_bins', 'use_finkelstein_bins',
                    'n_sampled_params', 'folder_path'])
        for rn, cfg in ALL_RUNS.items():
            w.writerow([
                rn, cfg['model'], cfg['data'],
                cfg['shmr'] or '', cfg['zcut'] or '',
                cfg['has_uvlf'], cfg['has_bg'], cfg['has_cmb'],
                cfg['use_donnan_bins'], cfg['use_finkelstein_bins'],
                cfg['n_sampled_params'], cfg['folder_path'],
            ])
    print("  [OK]   Wrote {}.".format(csv_path))

    readme_path = os.path.join(RUNS_ROOT, 'README.md')
    with open(readme_path, 'w') as f:
        f.write(make_readme(len(ALL_RUNS)))
    print("  [OK]   Wrote {}.".format(readme_path))

    # Category summary
    cats = {}
    for rn, cfg in ALL_RUNS.items():
        if cfg['has_uvlf']:
            key = "{}/{}/{}".format(cfg['model'], cfg['zcut'], cfg['shmr'])
        else:
            key = "non_uvlf/{}".format(cfg['model'])
        cats[key] = cats.get(key, 0) + 1

    print("\n  Category breakdown:")
    for k in sorted(cats):
        print("    {:30s}  {:3d}".format(k, cats[k]))

    print("\n" + "=" * 72)
    if all_ok:
        print("  Generated 112 runs. Validation: all checks passed.  [OK]")
    else:
        print("  Validation: some checks FAILED — review messages above.")
        sys.exit(1)
    print("=" * 72)


if __name__ == '__main__':
    main()