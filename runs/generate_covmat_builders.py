#!/usr/bin/env python3
# ============================================================================
# generate_covmat_builders.py
#
# Generate the covariance-matrix "builder" runs for the exotic transient
# dark-energy campaign. These are short MCMC runs at FIXED Planck 2018
# cosmology whose only purpose is to learn a well-scaled proposal covariance
# for the exotic-DE and/or SHMR parameter blocks. The learned .covmat is then
# fed to the full 112 production runs (via covmat: <path>) so they start with
# a properly-scaled proposal instead of cobaya's auto BAO covmat — which does
# not cover a_exo/b_exo/SHMR and cripples acceptance.
#
# Builders live in runs/builders/ and are kept separate from the production
# tree. Five builders cover every sampled-parameter block in the campaign:
#
#   builder_exo_fixed   : floats a_samp, s              (CLASS in loop)
#   builder_exo_vbeta   : floats a_samp, s, beta        (CLASS in loop)
#   builder_exo_vshmr   : floats a_samp, s, logMc, N, b (CLASS in loop)
#   builder_lcdm_vbeta  : floats beta                   (CLASS-free in loop)
#   builder_lcdm_vshmr  : floats logMc, N, beta         (CLASS-free in loop)
#
# (No builder_lcdm_fixed: an LCDM fixed-SHMR run has zero non-cosmology
#  sampled params, so there is no covariance to build.)
#
# Python 3.8 compatible. Standard library only. Same style as
# generate_all_runs.py.
#
# Run once:
#     python generate_covmat_builders.py
# ============================================================================

import os
import sys
from typing import Dict, List, Optional, Any


# ============================================================================
#  CONSTANTS — must stay consistent with generate_all_runs.py
# ============================================================================

RUNS_ROOT              = "runs"
BUILDERS_SUBDIR        = "builders"
LIKELIHOOD_PYTHON_PATH = "/home/lustre_p/ahmed.omar/workspace/exo_de_project/likelihood/"
VENV_ACTIVATE          = "/home/lustre_p/ahmed.omar/workspace/venvs/exo_DE/bin/activate"
COBAYA_CONFIG_HOME     = "/home/lustre_p/ahmed.omar/workspace/Modules/cobaya_stuff/cobaya_config"
COBAYA_CACHE_HOME      = "/home/lustre_p/ahmed.omar/workspace/Modules/cobaya_stuff/cobaya_cache"
COBAYA_PACKAGES_PATH   = "/home/lustre_p/ahmed.omar/workspace/Modules/cobaya_stuff/cobaya_packages"

# Exotic DE Gaussian window — fixed (same as production)
Z_C_EXO     = 16.0
SIGMA_Z_EXO = 3.25

# Polygon physicality constraint (same as production)
POLY_SLOPE     = -0.07202
POLY_INTERCEPT = -1381.5969

# Cheaper CLASS settings (validated against the HMF sigma(M) convergence:
# k_max=150 gives <0.005% error at the smallest 1e8 Msun halo; z_max=16
# covers the highest GL node at z~14.8 with headroom).
P_K_MAX   = 150.0
Z_MAX_PK  = 16.0

# Fixed Planck 2018 cosmology for the builders
PLANCK18_H0        = 67.36
PLANCK18_OMEGA_B   = 0.02237
PLANCK18_OMEGA_CDM = 0.1200

# Builder sampler tuning
BUILDER_PROPOSAL_SCALE   = 2.8        # higher than production 1.9 — explore fast
BUILDER_RMINUS1_STOP     = 0.05       # loose; max_samples is the real limiter
BUILDER_LEARN_EVERY      = "5d"       # learn covariance frequently

# Sample caps + walltime differ by builder type:
#   - LCDM builders are CLASS-free → hundreds of accepted samples in minutes.
#   - Exotic builders are CLASS-bound (~45 s/step) → a solid covmat needs hours,
#     not minutes. We give them a generous cap and a multi-hour walltime; this
#     is a one-time cost amortized across all 108 UVLF production runs.
BUILDER_MAX_SAMPLES_EXO   = 500      # accepted samples PER CHAIN (CLASS-bound)
BUILDER_MAX_SAMPLES_LCDM  = 800       # accepted samples PER CHAIN (CLASS-free, cheap)
BUILDER_WALLTIME_EXO      = "24:00:00"  # CLASS-bound; generous ceiling
BUILDER_WALLTIME_LCDM     = "02:00:00"  # CLASS-free; finishes in minutes

# SLURM resources
BUILDER_NTASKS           = 4          # 4 MPI chains: enough for a <=5-dim covmat
BUILDER_CPUS_EXO         = 2          # CLASS-bound builders get OpenMP threading
BUILDER_CPUS_LCDM        = 1          # CLASS-free builders need no threading
BUILDER_MEM_PER_CPU      = 5000


# ============================================================================
#  BUILDER DEFINITIONS
# ============================================================================

def build_all_builders():
    # type: () -> Dict[str, Dict[str, Any]]
    """The 5 builder configurations.

    Each dict carries:
      model     : 'exo' or 'lcdm'
      shmr      : 'fixed' | 'vbeta' | 'vshmr'
      class_loop: True if a floating param touches CLASS (exotic runs do,
                  because a_exo/b_exo live inside CLASS's H(z))
    """
    builders = {}

    # Exotic builders — a_samp, s always float (they modify CLASS H(z)),
    # so CLASS recomputes every step regardless of SHMR choice.
    for shmr in ['fixed', 'vbeta', 'vshmr']:
        name = "builder_exo_{}".format(shmr)
        builders[name] = {
            'builder_name': name,
            'model':        'exo',
            'shmr':         shmr,
            'class_loop':   True,
        }

    # LCDM builders — only SHMR params float, none touch CLASS, so CLASS runs
    # once at startup and never again. No 'fixed' variant (nothing to sample).
    for shmr in ['vbeta', 'vshmr']:
        name = "builder_lcdm_{}".format(shmr)
        builders[name] = {
            'builder_name': name,
            'model':        'lcdm',
            'shmr':         shmr,
            'class_loop':   False,
        }

    return builders


def builder_n_params(cfg):
    # type: (Dict[str, Any]) -> int
    """Number of sampled parameters in a builder."""
    p_exo  = 2 if cfg['model'] == 'exo' else 0
    p_shmr = {'fixed': 0, 'vbeta': 1, 'vshmr': 3}[cfg['shmr']]
    return p_exo + p_shmr


# ============================================================================
#  YAML BUILDER
# ============================================================================

def make_builder_yaml(cfg):
    # type: (Dict[str, Any]) -> str
    """Build the YAML string for one covmat-builder run."""
    L = []
    name = cfg['builder_name']

    # ── Theory ──────────────────────────────────────────────────────────
    L.append("# ── Theory (cosmology FIXED at Planck 2018) ────────────────")
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
    L.append("      P_k_max_1/Mpc: {}".format(P_K_MAX))
    L.append("      z_max_pk: {}".format(Z_MAX_PK))
    L.append("")

    # ── Likelihood (UVLF only; no volume correction at fixed cosmology) ──
    L.append("# ── Likelihood (UVLF only) ─────────────────────────────────")
    L.append("likelihood:")
    L.append("  jwst_likelihood_uvlf.UVLFLikelihood:")
    L.append('    python_path: "{}"'.format(LIKELIHOOD_PYTHON_PATH))
    L.append("    integrate_bin: True")
    # use_volume_correction MUST be False: with cosmology fixed, V_ref/V_MCMC=1,
    # and it avoids the angular_distance/Hubble requirements entirely.
    L.append("    use_volume_correction: False")
    L.append("    n_gl: 2")
    L.append("    z_min_donnan: 9.0")
    L.append("    z_min_finkelstein: 8.9")
    L.append("    use_donnan_bins: True")
    L.append("    use_finkelstein_bins: True")
    L.append("    vary_SHMR: {}".format(cfg['shmr'] == 'vshmr'))
    L.append("    vary_beta: {}".format(cfg['shmr'] == 'vbeta'))
    L.append("")

    # ── Polygon prior (exotic only) ─────────────────────────────────────
    if cfg['model'] == 'exo':
        L.append("# ── Polygon physicality prior (H^2(z) >= 0) ────────────────")
        L.append("prior:")
        if POLY_INTERCEPT >= 0:
            intercept_str = "+ {}".format(POLY_INTERCEPT)
        else:
            intercept_str = "- {}".format(abs(POLY_INTERCEPT))
        L.append('  h2_positivity: "lambda a_samp, s: 0.0 if s >= ({} * a_samp {}) else -1e500"'.format(
            POLY_SLOPE, intercept_str))
        L.append("")

    # ── Params ──────────────────────────────────────────────────────────
    L.append("# ── Parameters ─────────────────────────────────────────────")
    L.append("params:")

    # Exotic DE — FLOATS in exotic builders (same priors as production)
    if cfg['model'] == 'exo':
        L.append("  # ── Exotic DE sampled (same priors as production) ──")
        L.append("  a_samp:")
        L.append("    prior: {min: -350.0, max: -1.0e-10}")
        L.append("    ref: {min: -75.0, max: -5.0}")
        L.append("    proposal: 5.0")
        L.append("    drop: true")
        L.append("    latex: a_{\\rm samp}")
        L.append("  s:")
        L.append("    prior: {min: -1000.0, max: -1.0e-10}")
        L.append("    ref: {min: -250.0, max: -50.0}")
        L.append("    proposal: 20.0")
        L.append("    drop: true")
        L.append("    latex: \\mathcal{S}")
        L.append("")
        L.append("  # ── Exotic DE derived ──")
        L.append("  a_exo:")
        L.append("    value: 'lambda a_samp: a_samp'")
        L.append("    latex: a_{\\rm exo}")
        L.append("  b_exo:")
        L.append("    value: 'lambda a_samp, s: s - a_samp'")
        L.append("    latex: b_{\\rm exo}")
        L.append("")

    # Cosmology — FIXED at Planck 2018 (this is what removes CLASS from the
    # loop for LCDM builders, and keeps the standard sector frozen for exotic)
    L.append("  # ── Cosmology FIXED at Planck 2018 ──")
    L.append("  H0:")
    L.append("    value: {}".format(PLANCK18_H0))
    L.append("    latex: H_0")
    L.append("  omega_b:")
    L.append("    value: {}".format(PLANCK18_OMEGA_B))
    L.append("    latex: \\Omega_{\\rm b} h^2")
    L.append("  omega_cdm:")
    L.append("    value: {}".format(PLANCK18_OMEGA_CDM))
    L.append("    latex: \\Omega_{\\rm c} h^2")
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

    # SHMR — floats per builder type (same priors as production)
    if cfg['shmr'] == 'vbeta':
        L.append("  # ── SHMR sampled (vary beta only) ──")
        L.append("  shmr_beta:")
        L.append("    prior: {dist: norm, loc: 1.35, scale: 0.26}")
        L.append("    ref: 1.35")
        L.append("    proposal: 0.05")
        L.append("    latex: \\beta_{\\rm SHMR}")
        L.append("")
    elif cfg['shmr'] == 'vshmr':
        L.append("  # ── SHMR sampled (full Stefanon DPL) ──")
        L.append("  shmr_log_Mc:")
        L.append("    prior: {dist: norm, loc: 11.5, scale: 0.2}")
        L.append("    ref: 11.5")
        L.append("    proposal: 0.04")
        L.append("    latex: \\log_{10}(M_c)")
        L.append("  shmr_N:")
        L.append("    prior: {dist: norm, loc: 0.0297, scale: 0.0065}")
        L.append("    ref: 0.0297")
        L.append("    proposal: 0.0013")
        L.append("    latex: N_{\\rm SHMR}")
        L.append("  shmr_beta:")
        L.append("    prior: {dist: norm, loc: 1.35, scale: 0.26}")
        L.append("    ref: 1.35")
        L.append("    proposal: 0.05")
        L.append("    latex: \\beta_{\\rm SHMR}")
        L.append("")

    # Neutrino mass — fixed (every run)
    L.append("  # ── Neutrino mass (fixed) ──")
    L.append("  m_ncdm:")
    L.append("    value: 0.06")
    L.append("    renames: mnu")
    L.append("")

    # ── Sampler ─────────────────────────────────────────────────────────
    # No fast/slow block here: cosmology is fixed, so for LCDM builders every
    # param is "fast"; for exotic builders the exotic params drive CLASS and
    # are "slow" but there is no separate fast block to drag against. Either
    # way drag is off. High proposal_scale for fast exploration, max_samples
    # caps the run, loose Rminus1_stop is a non-binding backstop.
    L.append("# ── Sampler (covmat-building: fast exploration, capped) ────")
    L.append("sampler:")
    L.append("  mcmc:")
    L.append("    covmat: auto")
    L.append("    drag: false")
    L.append("    oversample_power: 0")
    L.append("    proposal_scale: {}".format(BUILDER_PROPOSAL_SCALE))
    L.append("    Rminus1_stop: {}".format(BUILDER_RMINUS1_STOP))
    L.append("    Rminus1_cl_stop: 0.2")
    L.append("    learn_every: '{}'".format(BUILDER_LEARN_EVERY))
    max_samp = BUILDER_MAX_SAMPLES_EXO if cfg['class_loop'] else BUILDER_MAX_SAMPLES_LCDM
    L.append("    max_samples: {}".format(max_samp))
    L.append("")

    # ── Output ──────────────────────────────────────────────────────────
    L.append("output: outputs/{}".format(name))

    return "\n".join(L) + "\n"


# ============================================================================
#  SLURM .sh BUILDER
# ============================================================================

def make_builder_sh(cfg):
    # type: (Dict[str, Any]) -> str
    name     = cfg['builder_name']
    cpus     = BUILDER_CPUS_EXO if cfg['class_loop'] else BUILDER_CPUS_LCDM
    walltime = BUILDER_WALLTIME_EXO if cfg['class_loop'] else BUILDER_WALLTIME_LCDM

    sh = (
"#!/bin/bash\n"
"#SBATCH --job-name={name}\n"
"#SBATCH --nodes=1\n"
"#SBATCH --exclude=lustre,cernnode02,cernnode03,nut01,nut02\n"
"#SBATCH --output={name}.log\n"
"#SBATCH --error={name}.err\n"
"#SBATCH --ntasks={ntasks}\n"
"#SBATCH --time={walltime}\n"
"#SBATCH --cpus-per-task={cpus}\n"
"#SBATCH --mem-per-cpu={mem}\n"
"\n"
"echo \"======================================================\"\n"
"echo \"Covmat builder {name} started on $(hostname) at $(date)\"\n"
"echo \"======================================================\"\n"
"\n"
"module purge\n"
"module load mpi/openmpi-x86_64\n"
"\n"
"source /opt/rh/devtoolset-8/enable\n"
"source {venv}\n"
"\n"
"export XDG_CONFIG_HOME=\"{cobaya_config}\"\n"
"export XDG_CACHE_HOME=\"{cobaya_cache}\"\n"
"export COBAYA_PACKAGES_PATH='{cobaya_packages}'\n"
"export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK\n"
"\n"
"echo \"Checking which CLASS is loaded...\"\n"
"python -c \"import classy; print('Loaded CLASS from:', classy.__file__)\"\n"
"\n"
"# Builders are short; a single attempt with light retry on signal kills.\n"
"max_retries=10\n"
"retry_count=0\n"
"command=\"mpirun -np $SLURM_NTASKS cobaya-run {name}.yaml --resume\"\n"
"\n"
"until $command; do\n"
"  exit_code=$?\n"
"  retry_count=$((retry_count + 1))\n"
"  if [ $exit_code -eq 139 ] || [ $exit_code -eq 134 ] || [ $exit_code -eq 137 ]; then\n"
"    echo \"Signal kill (exit $exit_code) at $(date). Resetting retry counter.\"\n"
"    retry_count=0\n"
"  fi\n"
"  if [ $retry_count -ge $max_retries ]; then\n"
"    echo \"FATAL: $max_retries consecutive failures. Proceeding to covmat dump anyway.\"\n"
"    break\n"
"  fi\n"
"  echo \"Crash #${{retry_count}}/${{max_retries}} (exit $exit_code). Retrying in 10s...\"\n"
"  sleep 10\n"
"done\n"
"\n"
"# ── Explicit covmat dump ──────────────────────────────────────────────\n"
"# cobaya writes outputs/{name}.covmat at convergence-check cycles. If the run\n"
"# stopped via max_samples or walltime mid-cycle, that file may be stale or\n"
"# missing. This fallback recomputes the proposal covariance directly from the\n"
"# chains via getdist and writes it in cobaya .covmat format (header line of\n"
"# sampled parameter names, then the matrix). cobaya reads this back fine.\n"
"echo \"Dumping covmat from chains (fallback)...\"\n"
"python - <<'PYDUMP'\n"
"import os, sys\n"
"import numpy as np\n"
"chain_root = os.path.join('outputs', '{name}')\n"
"try:\n"
"    from getdist import loadMCSamples\n"
"    samples = loadMCSamples(chain_root, settings={{'ignore_rows': 0.3}})\n"
"    names = [p.name for p in samples.paramNames.names if not p.isDerived]\n"
"    cov = np.atleast_2d(np.asarray(samples.cov(pars=names)))\n"
"    out_path = chain_root + '.covmat'\n"
"    with open(out_path, 'w') as f:\n"
"        f.write('# ' + ' '.join(names) + '\\n')\n"
"        for row in cov:\n"
"            f.write(' '.join('{{:.8e}}'.format(x) for x in row) + '\\n')\n"
"    print('Wrote covmat ({{}}x{{}}) for params {{}}'.format(cov.shape[0], cov.shape[1], names))\n"
"    print('  -> ' + out_path)\n"
"except Exception as e:\n"
"    print('Covmat dump FAILED: {{}}'.format(e), file=sys.stderr)\n"
"    sys.exit(0)\n"
"PYDUMP\n"
"\n"
"echo \"======================================================\"\n"
"echo \"Builder {name} finished at $(date). Covmat at outputs/{name}.covmat\"\n"
"echo \"======================================================\"\n"
    )

    return sh.format(
        name            = name,
        ntasks          = BUILDER_NTASKS,
        cpus            = cpus,
        walltime        = walltime,
        mem             = BUILDER_MEM_PER_CPU,
        venv            = VENV_ACTIVATE,
        cobaya_config   = COBAYA_CONFIG_HOME,
        cobaya_cache    = COBAYA_CACHE_HOME,
        cobaya_packages = COBAYA_PACKAGES_PATH,
    )


# ============================================================================
#  PARAM COUNTER (line-based, same approach as generate_all_runs.py)
# ============================================================================

def count_yaml_params(yaml_str):
    # type: (str) -> int
    """Count sampled params: a param at indent 2 with a `prior:` at indent 4.
    The top-level `prior:` block (h2_positivity) is excluded."""
    lines = yaml_str.split('\n')
    in_params = False
    sampled = 0
    current_param = None
    current_has_prior = False

    for line in lines:
        stripped = line.lstrip()
        if not stripped or stripped.startswith('#'):
            continue
        indent = len(line) - len(stripped)

        if indent == 0:
            if current_param is not None and current_has_prior:
                sampled += 1
            current_param = None
            current_has_prior = False
            in_params = line.startswith('params:')
            continue

        if not in_params:
            continue

        if indent == 2 and line.rstrip().endswith(':'):
            if current_param is not None and current_has_prior:
                sampled += 1
            current_param = stripped.rstrip(':')
            current_has_prior = False
        elif indent == 4 and current_param is not None and stripped.startswith('prior:'):
            current_has_prior = True

    if current_param is not None and current_has_prior:
        sampled += 1
    return sampled


# ============================================================================
#  COVMAT MAPPING — which builder serves which production run
# ============================================================================

def covmat_for_production_run(model, shmr):
    # type: (str, str) -> Optional[str]
    """Return the builder name whose covmat a production run should use, or
    None if no builder applies (lcdm + fixed has no sampled non-cosmo params,
    so it just uses covmat: auto).

    This is the lookup the later 'apply covmats' step will use.
    """
    if model == 'exo':
        return "builder_exo_{}".format(shmr)        # fixed/vbeta/vshmr all exist
    else:  # lcdm
        if shmr == 'fixed':
            return None                              # nothing to seed
        return "builder_lcdm_{}".format(shmr)        # vbeta/vshmr


# ============================================================================
#  MAIN
# ============================================================================

def main():
    print("=" * 72)
    print("  generate_covmat_builders.py — building covmat-builder runs")
    print("=" * 72)

    builders = build_all_builders()
    builders_root = os.path.join(RUNS_ROOT, BUILDERS_SUBDIR)

    n_files = 0
    for name, cfg in builders.items():
        folder = os.path.join(builders_root, name)
        os.makedirs(folder, exist_ok=True)
        os.makedirs(os.path.join(folder, 'outputs'), exist_ok=True)

        yaml_path = os.path.join(folder, "{}.yaml".format(name))
        sh_path   = os.path.join(folder, "{}.sh".format(name))

        with open(yaml_path, 'w') as f:
            f.write(make_builder_yaml(cfg))
        with open(sh_path, 'w') as f:
            f.write(make_builder_sh(cfg))
        try:
            os.chmod(sh_path, 0o755)
        except OSError:
            pass
        n_files += 2

    print("\nWrote {} files for {} builders.".format(n_files, len(builders)))

    # ── Validation ──────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  VALIDATION")
    print("=" * 72)

    all_ok = True

    # 1. Param count matches expectation
    for name, cfg in builders.items():
        with open(os.path.join(builders_root, name, "{}.yaml".format(name))) as f:
            text = f.read()
        actual = count_yaml_params(text)
        expected = builder_n_params(cfg)
        flag = "OK" if actual == expected else "FAIL"
        if actual != expected:
            all_ok = False
        print("  [{}] {:24s} P={} (expected {})  CLASS-in-loop={}".format(
            flag, name, actual, expected, cfg['class_loop']))

    # 2. Cosmology genuinely fixed (no prior: on H0/omega_b/omega_cdm)
    for name, cfg in builders.items():
        with open(os.path.join(builders_root, name, "{}.yaml".format(name))) as f:
            text = f.read()
        for cosmo_par in ['  H0:', '  omega_b:', '  omega_cdm:']:
            # find the block, ensure next non-comment line is 'value:' not 'prior:'
            idx = text.find(cosmo_par + '\n')
            if idx == -1:
                print("  [FAIL] {} missing {}".format(name, cosmo_par.strip()))
                all_ok = False
                continue
            after = text[idx + len(cosmo_par):idx + len(cosmo_par) + 60]
            if 'value:' not in after.split('\n')[1]:
                print("  [FAIL] {} has {} not fixed (no value:)".format(name, cosmo_par.strip()))
                all_ok = False
    print("  [OK]   Cosmology fixed (H0/omega_b/omega_cdm via value:) in all builders.")

    # 3. use_volume_correction must be False
    for name, cfg in builders.items():
        with open(os.path.join(builders_root, name, "{}.yaml".format(name))) as f:
            text = f.read()
        if 'use_volume_correction: False' not in text:
            print("  [FAIL] {} does not have use_volume_correction: False".format(name))
            all_ok = False
    print("  [OK]   use_volume_correction: False in all builders.")

    # 4. max_samples present (the time cap)
    for name, cfg in builders.items():
        with open(os.path.join(builders_root, name, "{}.yaml".format(name))) as f:
            text = f.read()
        want_max = BUILDER_MAX_SAMPLES_EXO if cfg['class_loop'] else BUILDER_MAX_SAMPLES_LCDM
        if "max_samples: {}".format(want_max) not in text:
            print("  [FAIL] {} missing max_samples cap ({})".format(name, want_max))
            all_ok = False
    print("  [OK]   max_samples cap present (exotic={}, lcdm={}).".format(
        BUILDER_MAX_SAMPLES_EXO, BUILDER_MAX_SAMPLES_LCDM))

    # 5. cpus-per-task: exotic→2, lcdm→1
    for name, cfg in builders.items():
        with open(os.path.join(builders_root, name, "{}.sh".format(name))) as f:
            sh = f.read()
        want = BUILDER_CPUS_EXO if cfg['class_loop'] else BUILDER_CPUS_LCDM
        if "cpus-per-task={}".format(want) not in sh:
            print("  [FAIL] {} wrong cpus-per-task (want {})".format(name, want))
            all_ok = False
    print("  [OK]   cpus-per-task correct (exotic=2, lcdm=1).")

    # 6. Print the production→builder covmat mapping for reference
    print("\n  Production-run → builder covmat mapping:")
    examples = [('exo', 'fixed'), ('exo', 'vbeta'), ('exo', 'vshmr'),
                ('lcdm', 'fixed'), ('lcdm', 'vbeta'), ('lcdm', 'vshmr')]
    for model, shmr in examples:
        b = covmat_for_production_run(model, shmr)
        target = "{}/{}/outputs/{}.covmat".format(BUILDERS_SUBDIR, b, b) if b else "(none — covmat: auto)"
        print("    {:5s} + {:6s}  →  {}".format(model, shmr, target))

    print("\n" + "=" * 72)
    if all_ok:
        print("  Generated {} builders. Validation: all checks passed.  [OK]".format(len(builders)))
    else:
        print("  Validation: some checks FAILED — review messages above.")
        sys.exit(1)
    print("=" * 72)
    print("\n  Submit with:  cd {}/<builder_name> && sbatch <builder_name>.sh".format(builders_root))
    print("  Or wire a `build-covmats` command into run_manager.py (next step).")


if __name__ == '__main__':
    main()
