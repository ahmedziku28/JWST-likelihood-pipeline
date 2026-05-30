#!/usr/bin/env python3
# ============================================================================
# apply_covmats.py
#
# Apply covmat-builder outputs to the 108 UVLF production runs by rewriting
# each YAML's `covmat: auto` → `covmat: <builder_path>`. State-aware:
#
#   - PENDING runs:       always set builder path (warm start on first submit)
#   - CONVERGED runs:     leave alone
#   - RUNNING/QUEUED:     read live R-1 via getdist on the chain files
#                            R-1 below threshold → leave alone (productive)
#                            R-1 above threshold → FLAG for pause/apply/resubmit
#                                                  (do NOT modify YAML unless
#                                                   user passes --apply-running)
#   - STALLED/FAILED:     set builder path (next submission uses it)
#   - ZOMBIE/PAUSED:      set builder path (fresh restart uses it)
#
# Non-UVLF runs (exo_bg, lcdm_bg, exo_bg_cmb, lcdm_bg_cmb) and lcdm_*_fixed_*
# runs have no matching builder — they keep `covmat: auto`.
#
# Idempotent: re-running the same command produces the same result.
#
# Python 3.8 compatible. Stdlib + getdist (already in your venv).
#
# Usage:
#   python apply_covmats.py                  # dry-run report
#   python apply_covmats.py apply            # apply to PENDING/STALLED/FAILED/ZOMBIE/PAUSED
#   python apply_covmats.py apply-running    # ALSO apply to RUNNING with R-1 > threshold
#                                            # (prints names; you pause/apply/resubmit)
#   python apply_covmats.py revert <run>     # reset one YAML back to covmat: auto
#   python apply_covmats.py revert --all     # reset all production YAMLs
#
#   --threshold T   override R-1 threshold for "stuck" runs (default 5.0)
# ============================================================================

import os
import re
import sys
import json
from typing import Dict, List, Optional, Any, Tuple


# ============================================================================
#  CONSTANTS — must stay consistent with generate_all_runs.py / generate_covmat_builders.py
# ============================================================================

RUNS_ROOT       = "runs"
BUILDERS_SUBDIR = "builders"
STATE_FILE      = os.path.join(RUNS_ROOT, ".run_manager_state.json")

# R-1 threshold above which a RUNNING run is considered "stuck" and benefits
# from a fresh start with the builder warm-start covmat.
DEFAULT_RMINUS1_THRESHOLD = 4.0

# Sample-count floor: getdist needs enough samples to compute R-1 meaningfully.
MIN_SAMPLES_FOR_RMINUS1 = 30


# ============================================================================
#  REPLICATE the (model, shmr) → builder mapping
#  Must match covmat_for_production_run() in generate_covmat_builders.py
# ============================================================================

def covmat_for_run(model, shmr):
    # type: (str, Optional[str]) -> Optional[str]
    """Builder name whose covmat seeds this production run, or None."""
    if model == 'exo':
        return "builder_exo_{}".format(shmr)            # fixed / vbeta / vshmr
    # lcdm
    if shmr in (None, 'fixed'):
        return None                                      # no non-cosmo block to seed
    return "builder_lcdm_{}".format(shmr)                # vbeta / vshmr


def builder_covmat_path(builder_name):
    # type: (str) -> str
    """Absolute-style relative path that goes into the YAML (relative to the
    run's working directory, which is the run's folder when sbatch runs).
    Since cobaya resolves covmat: relative to the YAML location, we use an
    absolute path computed from the runs root."""
    abs_runs_root = os.path.abspath(RUNS_ROOT)
    return os.path.join(abs_runs_root, BUILDERS_SUBDIR, builder_name,
                        'outputs', builder_name + '.covmat')


# ============================================================================
#  REPLICATE build_all_runs() — same as in generate_all_runs.py / run_manager.py
# ============================================================================

UVLF_DATA_COMBOS = [
    'ceers', 'primer', 'uvlf', 'ceers_bg', 'primer_bg', 'uvlf_bg',
    'ceers_bg_cmb', 'primer_bg_cmb', 'uvlf_bg_cmb',
]
SHMR_OPTIONS = ['fixed', 'vbeta', 'vshmr']
ZCUT_OPTIONS = ['full', 'restr']
MODELS       = ['exo', 'lcdm']
NON_UVLF_DATA_COMBOS = ['bg', 'bg_cmb']


def build_all_runs():
    # type: () -> Dict[str, Dict[str, Any]]
    runs = {}
    for model in MODELS:
        for zcut in ZCUT_OPTIONS:
            for shmr in SHMR_OPTIONS:
                for data in UVLF_DATA_COMBOS:
                    run_name = "{}_{}_{}_{}".format(model, data, shmr, zcut)
                    model_dir = 'exotic' if model == 'exo' else 'lcdm'
                    runs[run_name] = {
                        'run_name': run_name, 'model': model, 'shmr': shmr,
                        'has_uvlf': True,
                        'folder_path': "{}/{}/{}/{}".format(model_dir, zcut, shmr, run_name),
                    }
    for model in MODELS:
        for data in NON_UVLF_DATA_COMBOS:
            run_name = "{}_{}".format(model, data)
            runs[run_name] = {
                'run_name': run_name, 'model': model, 'shmr': None,
                'has_uvlf': False,
                'folder_path': "non_uvlf/{}".format(run_name),
            }
    return runs


# ============================================================================
#  COLORS
# ============================================================================

USE_COLOR = sys.stdout.isatty()


def _c(text, color):
    if not USE_COLOR:
        return text
    codes = {'green': '32', 'yellow': '33', 'red': '31',
             'cyan': '36', 'gray': '90', 'bold': '1', 'magenta': '35'}
    return '\033[{}m{}\033[0m'.format(codes.get(color, '0'), text)


# ============================================================================
#  STATE-AWARE STATUS CHECK
# ============================================================================

def load_state():
    # type: () -> Dict[str, Dict[str, Any]]
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except (ValueError, IOError):
        return {}


def _count_chain_files(folder, run_name):
    # type: (str, str) -> int
    out = os.path.join(folder, 'outputs')
    if not os.path.isdir(out):
        return 0
    return sum(
        1 for f in os.listdir(out)
        if f.startswith(run_name + '.') and f.endswith('.txt')
        and f[len(run_name) + 1:-4].isdigit()
    )


def _learned_covmat_exists(folder, run_name):
    # type: (str, str) -> bool
    return os.path.exists(os.path.join(folder, 'outputs', run_name + '.covmat'))


def _read_rminus1_live(folder, run_name):
    # type: (str, str) -> Optional[float]
    """Compute current R-1 on means from chain files via getdist.
    Returns None if not enough samples or anything goes wrong.

    Note: after loadMCSamples consolidates the chain files, `samples.chains`
    is None — the per-chain data is already merged. We check total sample
    count instead, requiring at least MIN_SAMPLES_FOR_RMINUS1 * n_chains
    samples (assumed 4 chains for builders, 8 for production)."""
    chain_root = os.path.join(folder, 'outputs', run_name)
    try:
        from getdist import loadMCSamples
        samples = loadMCSamples(chain_root, settings={'ignore_rows': 0.3})
        # Count chain files on disk to estimate per-chain minimum
        out_dir = os.path.join(folder, 'outputs')
        n_chains = sum(1 for f in os.listdir(out_dir)
                       if f.startswith(run_name + '.') and f.endswith('.txt')
                       and f[len(run_name) + 1:-4].isdigit())
        if n_chains == 0:
            return None
        # samples.numrows is total post-burn-in samples across all chains
        if samples.numrows < MIN_SAMPLES_FOR_RMINUS1 * n_chains:
            return None
        try:
            R = float(samples.getGelmanRubin())
            # getdist's small-sample estimator can return R slightly below 1
            # for tiny chains; clamp to 0 since R-1 is non-negative in theory.
            return max(0.0, R - 1.0)
        except Exception:
            return None
    except Exception:
        return None


def get_simple_status(run_name, all_runs, state):
    # type: (str, Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]) -> Dict[str, Any]
    """Lightweight status: enough to decide what covmat: line to write.

    Returns dict with keys:
      coarse:  'PENDING' | 'ACTIVE' | 'CONVERGED' | 'INACTIVE_WITH_CHAINS'
      rminus1: float or None (only for ACTIVE)
      has_learned_covmat: bool
      n_chains: int
    """
    cfg = all_runs[run_name]
    folder = os.path.join(RUNS_ROOT, cfg['folder_path'])
    log_path = os.path.join(folder, run_name + '.log')

    info = {'coarse': 'PENDING', 'rminus1': None,
            'has_learned_covmat': False, 'n_chains': 0}

    info['n_chains'] = _count_chain_files(folder, run_name)
    info['has_learned_covmat'] = _learned_covmat_exists(folder, run_name)

    # CONVERGED check (the cheap one: log epilogue)
    if os.path.exists(log_path):
        try:
            with open(log_path, 'rb') as f:
                size = os.path.getsize(log_path)
                f.seek(max(0, size - 4096))
                tail = f.read().decode('utf-8', errors='ignore')
            if 'Run converged at' in tail:
                info['coarse'] = 'CONVERGED'
                return info
        except (IOError, OSError):
            pass

    entry = state.get(run_name, {})
    job_id = entry.get('job_id')

    # PAUSED counts as INACTIVE_WITH_CHAINS for our purposes
    if entry.get('paused', False):
        if info['n_chains'] > 0:
            info['coarse'] = 'INACTIVE_WITH_CHAINS'
            info['rminus1'] = _read_rminus1_live(folder, run_name)
        else:
            info['coarse'] = 'PENDING'
        return info

    if not job_id:
        # No job tracked. PENDING if no chains, else inactive-with-chains.
        if info['n_chains'] > 0:
            info['coarse'] = 'INACTIVE_WITH_CHAINS'
            info['rminus1'] = _read_rminus1_live(folder, run_name)
        else:
            info['coarse'] = 'PENDING'
        return info

    # We have a job_id. Check if it's still in squeue.
    import subprocess
    try:
        r = subprocess.run(['squeue', '-j', str(job_id), '-h', '-o', '%T'],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           timeout=15)
        sq_state = r.stdout.decode('utf-8', errors='ignore').strip()
    except (subprocess.SubprocessError, OSError):
        sq_state = ''

    if sq_state:
        # Running or queued — ACTIVE
        info['coarse'] = 'ACTIVE'
        if info['n_chains'] > 0:
            info['rminus1'] = _read_rminus1_live(folder, run_name)
        return info

    # Job left queue, didn't converge → INACTIVE_WITH_CHAINS (if any chains)
    if info['n_chains'] > 0:
        info['coarse'] = 'INACTIVE_WITH_CHAINS'
        # ALSO compute R-1 for these — user wants threshold-gating on
        # paused/stalled runs too (they might be paused for cluster-busy
        # reasons, not because they're stuck — so don't auto-modify their YAML
        # unless R-1 actually says they're stuck).
        info['rminus1'] = _read_rminus1_live(folder, run_name)
    else:
        info['coarse'] = 'PENDING'
    return info


# ============================================================================
#  YAML COVMAT REWRITER
# ============================================================================

# Match any of:
#   covmat: auto
#   covmat: null
#   covmat: /path/to/something
#   covmat: 'string'   |   covmat: "string"
# Inside the sampler.mcmc block (indent 4 spaces). We're permissive but only
# act on a single match; if there are two `covmat:` lines we abort that file.
COVMAT_LINE_RE = re.compile(r'^( {4}covmat:\s*)(.*)$', re.MULTILINE)


def read_yaml_covmat(yaml_path):
    # type: (str) -> Optional[str]
    """Return current value of covmat: in the file, or None if no/multiple
    matches found."""
    try:
        with open(yaml_path) as f:
            text = f.read()
    except (IOError, OSError):
        return None
    matches = COVMAT_LINE_RE.findall(text)
    if len(matches) != 1:
        return None
    return matches[0][1].strip()


def write_yaml_covmat(yaml_path, new_value):
    # type: (str, str) -> bool
    """Rewrite the unique `    covmat: ...` line. Returns False if !=1 match."""
    try:
        with open(yaml_path) as f:
            text = f.read()
    except (IOError, OSError):
        return False

    matches = COVMAT_LINE_RE.findall(text)
    if len(matches) != 1:
        return False

    new_text = COVMAT_LINE_RE.sub(
        lambda m: "{}{}".format(m.group(1), new_value),
        text, count=1,
    )
    try:
        with open(yaml_path, 'w') as f:
            f.write(new_text)
        return True
    except (IOError, OSError):
        return False


# ============================================================================
#  DECIDE WHAT TO DO PER RUN
# ============================================================================

def decide_action(run_name, all_runs, state, threshold):
    # type: (str, Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], float) -> Dict[str, Any]
    """Decide what action to take on one production YAML.

    Safety-first design (cobaya may raise compatibility errors if you change
    the YAML's covmat field mid-resume):
      - Modify YAMLs only for runs with NO chain files on disk (truly PENDING).
      - For runs with chain files, leave YAML alone regardless of state.
        Their next --resume picks up the chain's learned covmat anyway.
      - If chains have R-1 above threshold, FLAG the run as stuck so user can
        decide whether to `restart <name>` (which wipes chains AND uses the
        builder via YAML — but only if YAML has been updated). Therefore, for
        flag-stuck runs in apply-running mode, we set the YAML AND tell the
        user they must `restart` (not resume) to pick it up.

    Returns dict with:
      action       : 'skip-non-uvlf' | 'skip-no-builder' | 'skip-converged'
                     | 'set-builder' | 'leave-has-chains' | 'flag-stuck'
      target_value : the YAML covmat value we would write (or current, if leaving)
      reason       : human-readable reason
      rminus1      : current R-1 or None
      builder      : matching builder name or None
      n_chains     : how many chain files exist (for reporting)
    """
    cfg = all_runs[run_name]
    out = {'action': '', 'target_value': '', 'reason': '',
           'rminus1': None, 'builder': None, 'n_chains': 0}

    # Non-UVLF runs don't get a builder
    if not cfg['has_uvlf']:
        out['action'] = 'skip-non-uvlf'
        out['target_value'] = 'auto'
        out['reason'] = 'non-UVLF run (bg / bg+CMB only) — keep covmat: auto'
        return out

    # lcdm + fixed has no sampled non-cosmo params
    builder = covmat_for_run(cfg['model'], cfg['shmr'])
    out['builder'] = builder
    if builder is None:
        out['action'] = 'skip-no-builder'
        out['target_value'] = 'auto'
        out['reason'] = 'no non-cosmology sampled params (lcdm+fixed) — keep covmat: auto'
        return out

    target = builder_covmat_path(builder)
    out['target_value'] = target

    status = get_simple_status(run_name, all_runs, state)
    out['rminus1'] = status['rminus1']
    out['n_chains'] = status['n_chains']

    # CONVERGED → leave alone, no question
    if status['coarse'] == 'CONVERGED':
        out['action'] = 'skip-converged'
        out['reason'] = 'CONVERGED — leave YAML alone'
        return out

    # No chain files anywhere → truly PENDING, safe to rewrite YAML
    if status['n_chains'] == 0:
        out['action'] = 'set-builder'
        out['reason'] = 'PENDING (no chain files) — set builder for warm start'
        return out

    # Has chain files. Determine whether they look stuck.
    rm1 = status['rminus1']
    if rm1 is None:
        out['action'] = 'leave-has-chains'
        out['reason'] = ("has chain files, R-1 not yet computable — "
                         "leave YAML alone (next --resume uses chain's learned covmat)")
        return out

    if rm1 < threshold:
        out['action'] = 'leave-has-chains'
        out['reason'] = ('has chain files, R-1={:.3f} below threshold ({:.1f}) '
                         '— productive, leave YAML alone').format(rm1, threshold)
        return out

    # Stuck — flag, but don't modify YAML automatically. User must explicitly
    # `restart <name>` (which wipes chains + lets new YAML take effect).
    out['action'] = 'flag-stuck'
    out['reason'] = ('has chain files, R-1={:.3f} above threshold ({:.1f}) '
                     '— recommend restart (wipes chains, uses builder)').format(rm1, threshold)
    return out


# ============================================================================
#  MAIN MODES
# ============================================================================

ACTION_COLOR = {
    'skip-non-uvlf':      'gray',
    'skip-no-builder':    'gray',
    'skip-converged':     'green',
    'set-builder':        'cyan',
    'leave-has-chains':   'green',
    'flag-stuck':         'red',
}


def find_yaml_path(cfg):
    # type: (Dict[str, Any]) -> str
    return os.path.join(RUNS_ROOT, cfg['folder_path'], cfg['run_name'] + '.yaml')


def run_dry_or_apply(mode, all_runs, state, threshold):
    # type: (str, Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], float) -> None
    """mode ∈ {'dry-run', 'apply', 'apply-running'}"""

    # Decide actions for every run
    decisions = {}
    for rn in all_runs:
        decisions[rn] = decide_action(rn, all_runs, state, threshold)

    # ── Summary counts ─────────────────────────────────────────────────
    counts = {}
    for d in decisions.values():
        counts[d['action']] = counts.get(d['action'], 0) + 1

    print()
    print(_c("  apply_covmats.py — mode: {}".format(mode), 'bold'))
    print(_c("  R-1 threshold for 'stuck' runs: {:.1f}".format(threshold), 'gray'))
    print("  " + "-" * 70)
    print(_c("  Summary by action:", 'bold'))
    order = ['set-builder', 'flag-stuck', 'leave-has-chains',
             'skip-converged', 'skip-non-uvlf', 'skip-no-builder']
    for action in order:
        if action in counts:
            print("    {:<32s} {:>3d} runs".format(action, counts[action]))
    print()

    # ── Detailed flagged-stuck list ─────────────────────────────────────
    stuck = [rn for rn, d in decisions.items() if d['action'] == 'flag-stuck']
    if stuck:
        print(_c("  ⚠  Runs above R-1 threshold (stuck/slow with chains):", 'red'))
        for rn in sorted(stuck, key=lambda r: -decisions[r]['rminus1']):
            d = decisions[rn]
            print("    {:<35s}  R-1={:.3f}  ({} chain files)".format(
                rn, d['rminus1'], d['n_chains']))
        print()
        print(_c("  To rescue these (chains will be DISCARDED):", 'cyan'))
        print("    1. python apply_covmats.py apply-running   # rewrites stuck YAMLs to builder")
        print("    2. For each stuck run:")
        print("         python run_manager.py restart <run_name>   # WIPES chains, uses builder")
        print("       (DO NOT use `resubmit` — that would --resume from chains with the")
        print("        new YAML, which cobaya may reject as incompatible.)")
        print()

    # ── Dry-run: per-action listing ─────────────────────────────────────
    if mode == 'dry-run':
        print(_c("  Per-run plan (top 20 shown for each non-trivial action):", 'gray'))
        for action in order:
            runs_in_action = [(rn, decisions[rn]) for rn in sorted(decisions)
                              if decisions[rn]['action'] == action]
            if not runs_in_action:
                continue
            print(_c("  {} ({})".format(action, len(runs_in_action)), 'bold'))
            for rn, d in runs_in_action[:20]:
                tag = ""
                if d['rminus1'] is not None:
                    tag = "  R-1={:.3f}".format(d['rminus1'])
                elif d['n_chains'] > 0:
                    tag = "  ({} chains)".format(d['n_chains'])
                print("    {:<35s}{}".format(rn, tag))
            if len(runs_in_action) > 20:
                print("    ... and {} more".format(len(runs_in_action) - 20))
            print()
        print(_c("  This was a dry run. To apply:", 'cyan'))
        print("    python apply_covmats.py apply           # only modifies chain-less PENDING runs (safe)")
        print("    python apply_covmats.py apply-running   # ALSO modifies stuck-with-chains YAMLs")
        print("                                              (you MUST follow with restart, not resubmit)")
        return

    # ── Apply mode ──────────────────────────────────────────────────────
    actions_to_apply = {'set-builder'}
    if mode == 'apply-running':
        actions_to_apply.add('flag-stuck')

    n_modified = 0
    n_unchanged = 0
    n_failed = 0
    modified_stuck = []
    n_skipped_nonauto = 0
    for rn, d in decisions.items():
        if d['action'] not in actions_to_apply:
            continue
        cfg = all_runs[rn]
        yaml_path = find_yaml_path(cfg)
        if not os.path.exists(yaml_path):
            print(_c("    ✗ {} : YAML not found at {}".format(rn, yaml_path), 'red'))
            n_failed += 1
            continue
        current = read_yaml_covmat(yaml_path)
        if current is None:
            print(_c("    ✗ {} : couldn't parse covmat line (0 or multiple matches)".format(rn), 'red'))
            n_failed += 1
            continue
            
        current_stripped = current.strip().strip('"').strip("'")
        if current_stripped == d['target_value']:
            n_unchanged += 1
            continue
        if current_stripped != 'auto':
            if current_stripped.startswith('builder_'):
                n_unchanged += 1
                continue
            print(_c("    — {} : custom covmat '{}' — skipped".format(
                rn, current_stripped), 'yellow'))
            n_skipped_nonauto += 1
            continue
        ok = write_yaml_covmat(yaml_path, d['target_value'])
            
        if ok:
            n_modified += 1
            if d['action'] == 'flag-stuck':
                modified_stuck.append(rn)
                tag = " (STUCK, R-1={:.3f} — restart required)".format(d['rminus1'])
                print(_c("    ✓ {} :  → {}{}".format(rn, d['target_value'], tag), 'red'))
            else:
                print(_c("    ✓ {} :  → {}".format(rn, d['target_value']), 'cyan'))
        else:
            print(_c("    ✗ {} : write failed".format(rn), 'red'))
            n_failed += 1

    print()
    print(_c("  Applied: {} modified, {} already correct, {} skipped (non-auto), {} failed".format(
        n_modified, n_unchanged, n_skipped_nonauto, n_failed), 'bold'))

    if mode == 'apply-running' and modified_stuck:
        print()
        print(_c("  CRITICAL NEXT STEP:", 'red'))
        print(_c("    {} stuck-with-chains YAMLs were modified. Their existing chains".format(
            len(modified_stuck)), 'red'))
        print(_c("    will conflict with the new covmat field if you --resume. You MUST run:", 'red'))
        print()
        for rn in modified_stuck:
            print(_c("      python run_manager.py restart {}".format(rn), 'yellow'))
        print()
        print(_c("    (or revert with: python apply_covmats.py revert <run_name>)", 'gray'))


def run_revert(target, all_runs):
    # type: (str, Dict[str, Dict[str, Any]]) -> None
    """Reset covmat back to `auto`. `target` is a run name or '--all'."""
    if target == '--all':
        targets = list(all_runs.keys())
        print(_c("\n  Reverting ALL {} production YAMLs to covmat: auto".format(len(targets)), 'yellow'))
        ans = input(_c("  Confirm? [y/N]: ", 'cyan'))
        if ans.strip().lower() not in ('y', 'yes'):
            print("  Aborted.")
            return
    else:
        if target not in all_runs:
            print(_c("  Unknown run: {}".format(target), 'red'))
            return
        targets = [target]

    n_done = 0
    for rn in targets:
        cfg = all_runs[rn]
        yaml_path = find_yaml_path(cfg)
        current = read_yaml_covmat(yaml_path)
        if current is None:
            print(_c("    ✗ {} : couldn't parse covmat line".format(rn), 'red'))
            continue
        if current.strip().strip('"').strip("'") == 'auto':
            continue
        if write_yaml_covmat(yaml_path, 'auto'):
            n_done += 1
            print(_c("    ✓ {} : reverted to auto".format(rn), 'green'))
        else:
            print(_c("    ✗ {} : write failed".format(rn), 'red'))
    print()
    print(_c("  Reverted {} YAMLs.".format(n_done), 'bold'))


# ============================================================================
#  CLI
# ============================================================================

USAGE = """\
Usage:
  python apply_covmats.py                          # dry-run report (no changes)
  python apply_covmats.py apply                    # apply to non-active runs
  python apply_covmats.py apply-running            # ALSO apply to stuck-active runs
  python apply_covmats.py revert <run_name>        # reset one YAML to covmat: auto
  python apply_covmats.py revert --all             # reset all production YAMLs

Options:
  --threshold T    R-1 threshold above which a RUNNING run is "stuck" (default 5.0)
"""


def main():
    if not os.path.isdir(RUNS_ROOT):
        print(_c("  [ERROR] {} does not exist. Run generate_all_runs.py first.".format(RUNS_ROOT), 'red'))
        sys.exit(1)

    # Parse optional --threshold
    threshold = DEFAULT_RMINUS1_THRESHOLD
    args = sys.argv[1:]
    if '--threshold' in args:
        i = args.index('--threshold')
        try:
            threshold = float(args[i + 1])
            args = args[:i] + args[i + 2:]
        except (IndexError, ValueError):
            print(_c("  --threshold requires a numeric argument", 'red'))
            sys.exit(1)

    mode = args[0] if args else 'dry-run'

    all_runs = build_all_runs()
    state = load_state()

    if mode == 'dry-run':
        run_dry_or_apply('dry-run', all_runs, state, threshold)
    elif mode == 'apply':
        run_dry_or_apply('apply', all_runs, state, threshold)
    elif mode == 'apply-running':
        run_dry_or_apply('apply-running', all_runs, state, threshold)
    elif mode == 'revert':
        if len(args) < 2:
            print("  Usage: python apply_covmats.py revert <run_name|--all>")
            sys.exit(1)
        run_revert(args[1], all_runs)
    elif mode in ('-h', '--help', 'help'):
        print(USAGE)
    else:
        print(_c("  Unknown mode: {}".format(mode), 'red'))
        print(USAGE)
        sys.exit(1)


if __name__ == '__main__':
    main()
