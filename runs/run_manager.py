#!/usr/bin/env python3
# ============================================================================
# run_manager.py
#
# Orchestrator for the 112-run exotic transient dark-energy MCMC campaign.
# Submits, monitors, and recovers SLURM jobs. Tracks state in a JSON file
# so it can resume mid-campaign across multiple invocations.
#
# Python 3.8 compatible. Standard library only.
#
# Commands:
#     python run_manager.py status
#     python run_manager.py test
#     python run_manager.py launch <N>
#     python run_manager.py auto <N>
#     python run_manager.py resubmit <run_name>
#     python run_manager.py resubmit --all-stalled
#     python run_manager.py restart <run_name>
#     python run_manager.py reset <run_name>
#     python run_manager.py health <run_name>
#     python run_manager.py health <N|all> <state>
#     python run_manager.py doctor <run_name>
# ============================================================================

import os
import re
import sys
import json
import time
import getpass
import subprocess
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import random
import numpy as np
import glob

# ============================================================================
#  CONSTANTS
# ============================================================================

RUNS_ROOT  = "runs"
STATE_FILE = os.path.join(RUNS_ROOT, ".run_manager_state.json")
LOG_FILE   = os.path.join(RUNS_ROOT, "run_manager.log")

POLL_SECONDS_AUTO    = 15 * 60   # default auto-mode poll cadence
PAUSED_POLL_SECONDS  = 60        # tighter cadence while paused
CONTROL_FILE         = os.path.join(RUNS_ROOT, ".run_manager_control.json")
CONVERGENCE_RMINUS1  = 0.02      # must match yaml Rminus1_stop

# ZOMBIE detection: a job in squeue RUNNING but whose chain files haven't been
# written to in this many hours is declared ZOMBIE. Workers update chain .txt
# files independently of master-rank coordination, so chain-file mtime is the
# most trustworthy "is the run alive" signal we have.
ZOMBIE_CHAIN_MTIME_HOURS = 2.0

# Chain-file completeness check: after a run has been RUNNING for this many
# hours, every MPI rank should have written its chain .{i}.txt file at least
# once. If fewer than EXPECTED_CHAIN_COUNT files are present after this grace
# period, `status` flags the run with a warning. This catches the half-dead
# MPI scenario where some ranks crashed silently.
CHAIN_CHECK_AFTER_HOURS = 2.0
EXPECTED_CHAIN_COUNT    = 8   # matches --ntasks=8 in the .sh template

# ============================================================================
#  TEST RUN SET
#  Six runs chosen to exercise every code path. If all six converge cleanly,
#  the remaining 106 runs are permutations of paths already validated.
# ============================================================================

TEST_RUNS = [
    "lcdm_uvlf_fixed_full",          # P=3 — LCDM baseline, both surveys
    "exo_uvlf_fixed_restr",          # P=5 — exotic + restricted z-cut + h2_positivity
    "exo_ceers_vbeta_full",          # P=6 — CEERS-only + vary beta
    "exo_primer_vshmr_full",         # P=8 — PRIMER-only + full SHMR variation
    "exo_uvlf_bg_cmb_vshmr_full",    # P=11 — max complexity: exo+bg+CMB+vshmr
    "lcdm_bg_cmb",                   # P=6 — non-UVLF folder path + LCDM + CMB
]


# ============================================================================
#  REPLICATE build_all_runs() from generate_all_runs.py
#  (Spec: two files only, no shared modules — so we duplicate the dict build.)
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


def build_all_runs():
    # type: () -> Dict[str, Dict[str, Any]]
    runs = {}
    for model in MODELS:
        for zcut in ZCUT_OPTIONS:
            for shmr in SHMR_OPTIONS:
                for data in UVLF_DATA_COMBOS:
                    run_name = "{}_{}_{}_{}".format(model, data, shmr, zcut)
                    has_bg  = '_bg' in data
                    has_cmb = '_cmb' in data
                    P_cosmo = 6 if has_cmb else 3
                    P_exo   = 2 if model == 'exo' else 0
                    P_shmr  = {'fixed': 0, 'vbeta': 1, 'vshmr': 3}[shmr]
                    model_dir = 'exotic' if model == 'exo' else 'lcdm'
                    runs[run_name] = {
                        'run_name': run_name, 'model': model, 'data': data,
                        'has_uvlf': True, 'has_bg': has_bg, 'has_cmb': has_cmb,
                        'shmr': shmr, 'zcut': zcut,
                        'n_sampled_params': P_cosmo + P_exo + P_shmr,
                        'folder_path': "{}/{}/{}/{}".format(model_dir, zcut, shmr, run_name),
                    }
    for model in MODELS:
        for data in NON_UVLF_DATA_COMBOS:
            run_name = "{}_{}".format(model, data)
            has_cmb  = 'cmb' in data
            P_cosmo  = 6 if has_cmb else 3
            P_exo    = 2 if model == 'exo' else 0
            runs[run_name] = {
                'run_name': run_name, 'model': model, 'data': data,
                'has_uvlf': False, 'has_bg': True, 'has_cmb': has_cmb,
                'shmr': None, 'zcut': None,
                'n_sampled_params': P_cosmo + P_exo,
                'folder_path': "non_uvlf/{}".format(run_name),
            }
    return runs


# ============================================================================
#  PRIORITY ORDER — deterministic 112-element list
#
#  Ordering rule:
#    primary:    zcut (full before restr)
#    secondary:  data (most informative first)
#    tertiary:   shmr (fixed → vbeta → vshmr)
#    quaternary: model (exo → lcdm)
#  Non-UVLF runs are appended last in canonical order.
# ============================================================================

_DATA_PRIORITY = [
    'uvlf_bg_cmb',   'uvlf_bg',        'uvlf',
    'primer_bg_cmb', 'ceers_bg_cmb',
    'primer_bg',     'ceers_bg',
    'primer',        'ceers',
]


def _compute_priority_order():
    # type: () -> List[str]
    order = []
    for zcut in ['full', 'restr']:
        for data in _DATA_PRIORITY:
            for shmr in ['fixed', 'vbeta', 'vshmr']:
                for model in ['exo', 'lcdm']:
                    order.append("{}_{}_{}_{}".format(model, data, shmr, zcut))
    # Non-UVLF: CMB before bg-only, exo before lcdm
    for data in ['bg_cmb', 'bg']:
        for model in ['exo', 'lcdm']:
            order.append("{}_{}".format(model, data))
    return order


PRIORITY_ORDER = _compute_priority_order()


# ============================================================================
#  TERMINAL UTILITIES
# ============================================================================

USE_COLOR = sys.stdout.isatty()


def _c(text, color):
    # type: (str, str) -> str
    if not USE_COLOR:
        return text
    codes = {'green': '32', 'yellow': '33', 'red': '31',
             'cyan': '36', 'gray': '90', 'bold': '1', 'magenta': '35'}
    return '\033[{}m{}\033[0m'.format(codes.get(color, '0'), text)


STATUS_SYMBOL = {
    'CONVERGED': ('✓', 'green'),
    'RUNNING':   ('●', 'cyan'),
    'QUEUED':    ('◌', 'yellow'),
    'PENDING':   ('·', 'gray'),
    'STALLED':   ('!', 'yellow'),
    'FAILED':    ('✗', 'red'),
}

INTERESTING_TRANSITIONS = {
    ('RUNNING',   'CONVERGED'): True,
    ('QUEUED',    'CONVERGED'): True,
    ('STALLED',   'CONVERGED'): True,
    ('RUNNING',   'FAILED'):    True,
    ('RUNNING',   'STALLED'):   True,
    ('RUNNING',   'ZOMBIE'):    True,
    ('QUEUED',    'FAILED'):    True,
    ('RUNNING',   'PAUSED'):    True,
    ('QUEUED',    'PAUSED'):    True,
    ('FAILED',    'QUEUED'):    True,
    ('FAILED',    'RUNNING'):   True,
    ('STALLED',   'QUEUED'):    True,
    ('STALLED',   'RUNNING'):   True,
    ('ZOMBIE',    'QUEUED'):    True,
    ('ZOMBIE',    'RUNNING'):   True,
    ('PAUSED',    'QUEUED'):    True,
    ('PAUSED',    'RUNNING'):   True,
}

def _fmt_status(status):
    # type: (str) -> str
    sym, color = STATUS_SYMBOL.get(status, ('?', 'gray'))
    return _c("{} {}".format(sym, status), color)


# ============================================================================
#  STATE FILE
#  Maps run_name → { job_id, submitted_at, attempts }
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


def save_state(state):
    # type: (Dict[str, Dict[str, Any]]) -> None
    tmp = STATE_FILE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(state, f, indent=2, sort_keys=True)
    os.replace(tmp, STATE_FILE)


def log_event(msg):
    # type: (str) -> None
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = "[{}] {}".format(ts, msg)
    try:
        with open(LOG_FILE, 'a') as f:
            f.write(line + '\n')
    except IOError:
        pass

# ── live control file (throttle / pause / resume) ─────────────────────────

def load_control():
    # type: () -> Optional[Dict[str, Any]]
    if not os.path.exists(CONTROL_FILE):
        return None
    try:
        with open(CONTROL_FILE) as f:
            return json.load(f)
    except (ValueError, IOError):
        return None


def save_control(N=None, paused=None):
    # type: (Optional[int], Optional[bool]) -> Dict[str, Any]
    ctrl = load_control() or {}
    if N is not None:
        ctrl['N'] = N
    if paused is not None:
        ctrl['paused'] = paused
    ctrl['updated_at'] = datetime.now().isoformat()
    tmp = CONTROL_FILE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(ctrl, f, indent=2)
    os.replace(tmp, CONTROL_FILE)
    return ctrl


# ============================================================================
#  SLURM HELPERS
# ============================================================================

def _parse_slurm_elapsed(elapsed_str):
    # type: (str) -> Optional[int]
    """Parse SLURM elapsed time format into seconds.
    Formats handled: 'DD-HH:MM:SS', 'HH:MM:SS', 'MM:SS', 'SS'.
    Returns None on parse failure.
    """
    elapsed_str = elapsed_str.strip()
    if not elapsed_str:
        return None
    try:
        days = 0
        if '-' in elapsed_str:
            days_part, time_part = elapsed_str.split('-', 1)
            days = int(days_part)
        else:
            time_part = elapsed_str
        parts = time_part.split(':')
        if len(parts) == 3:
            h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
        elif len(parts) == 2:
            h, m, s = 0, int(parts[0]), int(parts[1])
        elif len(parts) == 1:
            h, m, s = 0, 0, int(parts[0])
        else:
            return None
        return days * 86400 + h * 3600 + m * 60 + s
    except (ValueError, AttributeError):
        return None


def _squeue_state(job_id):
    # type: (str) -> Optional[str]
    """Thin wrapper around _squeue_info for callers that only need state."""
    info = _squeue_info(job_id)
    return info['state'] if info else None


def _squeue_info(job_id):
    # type: (str) -> Optional[Dict[str, Any]]
    """Return {'state': str, 'elapsed_seconds': int} for a job, or None."""
    try:
        r = subprocess.run(
            ['squeue', '-j', str(job_id), '-h', '-o', '%T|%M'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=15,
        )
        out = r.stdout.decode('utf-8', errors='ignore').strip()
        if not out:
            return None
        parts = out.split('|')
        if len(parts) < 2:
            return None
        state = parts[0].strip()
        elapsed = _parse_slurm_elapsed(parts[1])
        return {'state': state, 'elapsed_seconds': elapsed if elapsed is not None else 0}
    except (subprocess.SubprocessError, OSError):
        return None


def _count_chain_files(folder, run_name):
    # type: (str, str) -> int
    """Count .{i}.txt chain files (where i is an integer) in outputs/."""
    output_dir = os.path.join(folder, 'outputs')
    if not os.path.isdir(output_dir):
        return 0
    return sum(
        1 for f in os.listdir(output_dir)
        if f.startswith(run_name + '.') and f.endswith('.txt')
        and f[len(run_name) + 1:-4].isdigit()
    )



def _sacct_state(job_id):
    # type: (str) -> Optional[str]
    """Return SLURM accounting state for a completed/dead job, or None.

    Authoritative source when the job has left the squeue queue.
    sacct may print multiple rows (batch step, extern step, etc.); we take
    the first row's first token. State strings may have a suffix
    (e.g. 'CANCELLED by 12345') — we keep only the leading word.
    """
    try:
        r = subprocess.run(
            ['sacct', '-j', str(job_id), '--format=State', '--noheader', '--parsable2'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=20,
        )
        out = r.stdout.decode('utf-8', errors='ignore').strip()
        if not out:
            return None
        first_row = out.split('\n')[0].strip()
        return first_row.split()[0] if first_row else None
    except (subprocess.SubprocessError, OSError):
        return None


# SLURM accounting states that mean "this job is dead, don't expect a resume"
SLURM_FAILURE_STATES = frozenset({
    'FAILED', 'OUT_OF_MEMORY', 'TIMEOUT', 'NODE_FAIL',
    'BOOT_FAIL', 'DEADLINE', 'PREEMPTED', 'CANCELLED',
})

# Subset of SLURM_FAILURE_STATES that are transient (cluster-side, not the job's
# fault) — eligible for auto-resubmit by the auto daemon.
TRANSIENT_SLURM_FAILURE_STATES = frozenset({
    'NODE_FAIL', 'BOOT_FAIL', 'PREEMPTED',
})

# Cap on consecutive auto-resubmits per run (reset on manual restart/submit)
MAX_AUTO_RESUBMITS = 3

# Application-level failure patterns (checked after the job has left the queue)
FAILURE_PATTERNS = (
    r'FATAL:',
    r'Traceback \(most recent call last\)',
    r'oom-kill|OOMKilled|MemoryError',
    r'CANCELLED AT',
    r'DUE TO TIME LIMIT',
    r'mpirun (detected|noticed|has exited)',
    r'has died|exited on signal|Killed by signal',
    r'No space left on device|disk quota exceeded',
    r'srun: error:',
    r'slurmstepd: error:',
    r'Error in (module |)input_shooting|Could not find correct value of',
    r'ModuleNotFoundError|ImportError',
    r'cobaya\.tools\.HandledException',
    r'cobaya: error',
    r'Could not initialize external likelihood',
    r'MPI_ABORT',
    r'Fatal (MPI )?(error )?in (PMPI|MPI)',
    r'ORTE has lost',
    r'Segmentation fault|SIGSEGV',
    r'has lost contact|lost contact with',
)


def submit_run(cfg):
    # type: (Dict[str, Any]) -> Optional[str]
    """sbatch the .sh; return job_id or None on failure."""
    folder   = os.path.join(RUNS_ROOT, cfg['folder_path'])
    sh_file  = "{}.sh".format(cfg['run_name'])
    sh_path  = os.path.join(folder, sh_file)

    if not os.path.exists(sh_path):
        print(_c("  [ERROR] {} not found".format(sh_path), 'red'))
        return None

    try:
        r = subprocess.run(
            ['sbatch', sh_file],
            cwd=folder,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=30,
        )
        if r.returncode != 0:
            print(_c("  [ERROR] sbatch failed for {}: {}".format(
                cfg['run_name'],
                r.stderr.decode('utf-8', errors='ignore').strip()), 'red'))
            return None

        out = r.stdout.decode('utf-8', errors='ignore').strip()
        # sbatch prints: "Submitted batch job 12345"
        m = re.search(r'Submitted batch job (\d+)', out)
        if not m:
            print(_c("  [ERROR] Could not parse sbatch output: {}".format(out), 'red'))
            return None
        return m.group(1)
    except (subprocess.SubprocessError, OSError) as e:
        print(_c("  [ERROR] sbatch exception for {}: {}".format(cfg['run_name'], e), 'red'))
        return None


def bump_cpus_in_sh(cfg, new_cpus=2):
    # type: (Dict[str, Any], int) -> bool
    """In-place rewrite of #SBATCH --cpus-per-task in the SLURM script."""
    sh_path = os.path.join(RUNS_ROOT, cfg['folder_path'],
                           "{}.sh".format(cfg['run_name']))
    if not os.path.exists(sh_path):
        return False
    with open(sh_path) as f:
        text = f.read()
    new_text = re.sub(
        r'#SBATCH --cpus-per-task=\d+',
        '#SBATCH --cpus-per-task={}'.format(new_cpus),
        text,
        count=1,
    )
    if new_text == text:
        return False
    with open(sh_path, 'w') as f:
        f.write(new_text)
    return True


# ============================================================================
#  STATUS DETECTION — the heart of the orchestrator
# ============================================================================

def _read_tail(path, max_bytes=8000):
    # type: (str, int) -> str
    try:
        size = os.path.getsize(path)
        with open(path, 'rb') as f:
            if size > max_bytes:
                f.seek(-max_bytes, 2)
            return f.read().decode('utf-8', errors='ignore')
    except (OSError, IOError):
        return ""


def _parse_progress(progress_path):
    # type: (str) -> Optional[Tuple[float, float, int]]
    """Return (Rminus1, acceptance_rate, n_samples) from latest entry in
    Cobaya's .progress file, or None if unavailable.

    Cobaya .progress format (whitespace-separated, no fixed columns):
        N   timestamp   acceptance_rate   Rminus1   Rminus1_CL
    Header lines start with '#'.
    """
    if not os.path.exists(progress_path):
        return None
    try:
        with open(progress_path) as f:
            lines = [ln.strip() for ln in f.readlines()
                     if ln.strip() and not ln.startswith('#')]
        if not lines:
            return None
        parts = lines[-1].split()
        if len(parts) < 4:
            return None
        # Cobaya layout: N timestamp acceptance_rate Rminus1 Rminus1_CL
        # We tolerate either with or without timestamp.
        try:
            n_samples = int(float(parts[0]))
        except ValueError:
            return None
        # Try last-column-as-R-1-CL pattern first
        try:
            rminus1 = float(parts[-2])
            acceptance = float(parts[-3])
        except (ValueError, IndexError):
            return None
        return (rminus1, acceptance, n_samples)
    except (IOError, OSError):
        return None


def _is_zombie_by_chains(folder, run_name):
    # type: (str, str) -> bool
    """A run is a zombie if its job is RUNNING in SLURM but no chain file has
    been written in ZOMBIE_CHAIN_MTIME_HOURS hours. Worker MPI ranks update
    their .{i}.txt files independently of master-rank coordination, so this is
    a more trustworthy signal than .progress mtime (which can be legitimately
    stale during early sampling AND unreliable due to Lustre cache).

    Returns False if no chain files exist yet (too early to call it zombie).
    """
    output_dir = os.path.join(folder, 'outputs')
    if not os.path.isdir(output_dir):
        return False
    chain_files = [f for f in os.listdir(output_dir)
                   if f.startswith(run_name + '.') and f.endswith('.txt')]
    if not chain_files:
        return False
    try:
        latest_mtime = max(
            os.path.getmtime(os.path.join(output_dir, f))
            for f in chain_files
        )
    except OSError:
        return False
    age_hours = (time.time() - latest_mtime) / 3600.0
    return age_hours > ZOMBIE_CHAIN_MTIME_HOURS


def _chains_recently_active(folder, run_name, threshold_hours=0.5):
    # type: (str, str, Optional[float]) -> bool
    """True if any chain .txt file was modified within the last threshold_hours.

    Uses ZOMBIE_CHAIN_MTIME_HOURS as default threshold — same signal the zombie
    detector trusts, now also used to override false STALLED when squeue/sacct
    are unreliable.
    """
    if threshold_hours is None:
        threshold_hours = ZOMBIE_CHAIN_MTIME_HOURS
        
    output_dir = os.path.join(folder, 'outputs')
    if not os.path.isdir(output_dir):
        return False
    chain_files = [f for f in os.listdir(output_dir)
                   if f.startswith(run_name + '.') and f.endswith('.txt')
                   and f[len(run_name) + 1:-4].isdigit()]
    if not chain_files:
        return False
    try:
        latest_mtime = max(
            os.path.getmtime(os.path.join(output_dir, f))
            for f in chain_files
        )
    except OSError:
        return False
    age_hours = (time.time() - latest_mtime) / 3600.0
    return age_hours < threshold_hours


def get_status(run_name, all_runs, state):
    # type: (str, Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]) -> Dict[str, Any]
    """Determine status of a single run.

    Returns dict with keys:
      status     : one of {PENDING, QUEUED, RUNNING, CONVERGED, STALLED,
                           FAILED, ZOMBIE, PAUSED}
      rminus1    : float or None
      acceptance : float or None
      n_samples  : int or None
      job_id     : str or None
      chain_count    : str or None
      elapsed_seconds     : str or None
      chain_warning    : str or None

    """
    cfg = all_runs[run_name]
    folder        = os.path.join(RUNS_ROOT, cfg['folder_path'])
    log_path      = os.path.join(folder, run_name + '.log')
    err_path      = os.path.join(folder, run_name + '.err')
    progress_path = os.path.join(folder, 'outputs', run_name + '.progress')

    info = {'status': 'PENDING', 'rminus1': None, 'acceptance': None,
            'n_samples': None, 'job_id': None,
            'chain_count': 0, 'elapsed_seconds': None, 'chain_warning': None}

    entry = state.get(run_name, {})

    # 0. PAUSED — explicit user pause (set by `pause --all`) overrides all else
    if entry.get('paused', False):
        info['status'] = 'PAUSED'
        info['job_id'] = entry.get('job_id')
        return info

    # 1. CONVERGED — strongest signal: the .sh epilogue line
    log_tail = _read_tail(log_path)
    if 'Run converged at' in log_tail:
        info['status'] = 'CONVERGED'
        prog = _parse_progress(progress_path)
        if prog is not None:
            info['rminus1'], info['acceptance'], info['n_samples'] = prog
        return info

    # 2. CONVERGED — fallback #1: progress file's last R-1
    prog = _parse_progress(progress_path)
    if prog is not None:
        info['rminus1'], info['acceptance'], info['n_samples'] = prog
        if prog[0] < CONVERGENCE_RMINUS1:
            info['status'] = 'CONVERGED'
            return info

    # 3. CONVERGED — fallback #2: checkpoint file's Rminus1_last
    checkpoint_path = os.path.join(folder, 'outputs', run_name + '.checkpoint')
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path) as f:
                ckpt_text = f.read()
            m = re.search(r'Rminus1_last:\s*([\d.eE+-]+)', ckpt_text)
            if m:
                try:
                    r_last = float(m.group(1))
                    if info['rminus1'] is None:
                        info['rminus1'] = r_last
                    if r_last < CONVERGENCE_RMINUS1:
                        info['status'] = 'CONVERGED'
                        return info
                except ValueError:
                    pass
        except (IOError, OSError):
            pass

    # 4. No tracked submission yet → PENDING
    job_id = entry.get('job_id')
    info['job_id'] = job_id
    if job_id is None:
        return info

    # 5. Job still in SLURM queue
    sq = _squeue_info(job_id)
    if sq is not None:
        if sq['state'] in ('PENDING', 'CONFIGURING'):
            info['status'] = 'QUEUED'
            return info

        # RUNNING from SLURM's view — populate elapsed and chain count
        info['elapsed_seconds'] = sq['elapsed_seconds']
        info['chain_count'] = _count_chain_files(folder, run_name)

        # Zombie check (chain files all stale)
        if _is_zombie_by_chains(folder, run_name):
            info['status'] = 'ZOMBIE'
            return info

        # Chain-count completeness warning (after grace period)
        elapsed_hours = sq['elapsed_seconds'] / 3600.0
        if (elapsed_hours >= CHAIN_CHECK_AFTER_HOURS
                and info['chain_count'] < EXPECTED_CHAIN_COUNT):
            info['chain_warning'] = "{}/{} chains after {:.1f}h".format(
                info['chain_count'], EXPECTED_CHAIN_COUNT, elapsed_hours)

        info['status'] = 'RUNNING'
        return info

    # 6. Job has left the queue — sacct is authoritative
    sacct_state = _sacct_state(job_id)
    if sacct_state in SLURM_FAILURE_STATES:
        info['status'] = 'FAILED'
        return info

    # 7. Expanded pattern match across .err and .log
    err_tail = _read_tail(err_path, 12000)
    log_tail_full = _read_tail(log_path, 12000)
    combined = err_tail + '\n' + log_tail_full
    for pat in FAILURE_PATTERNS:
        if re.search(pat, combined, flags=re.IGNORECASE):
            info['status'] = 'FAILED'
            return info

    # 8. Chain-liveness override: if squeue/sacct lost track but chain files
    #    are being actively written, the run is almost certainly alive. This
    #    catches the common case on older SLURM where squeue times out.
    if _chains_recently_active(folder, run_name):
        info['status'] = 'RUNNING'
        info['chain_count'] = _count_chain_files(folder, run_name)
        return info

    # 9. Otherwise: STALLED
    info['status'] = 'STALLED'
    return info


def _log_status_transitions(statuses, state):
    # type: (Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]) -> bool
    """Compare current statuses to last_known_status in state. Write interesting
    transitions to run_manager.log. Updates state['last_known_status'] in place.
    Returns True if state was modified (caller should save_state).
    """
    state_modified = False
    for rn, info in statuses.items():
        new_status = info['status']
        entry = state.get(rn, {})
        old_status = entry.get('last_known_status')

        if old_status is None:
            entry['last_known_status'] = new_status
            state[rn] = entry
            state_modified = True
            continue
        if old_status == new_status:
            continue

        if INTERESTING_TRANSITIONS.get((old_status, new_status)):
            detail = ""
            if new_status == 'FAILED':
                jid = entry.get('job_id')
                if jid:
                    sacct = _sacct_state(jid)
                    if sacct:
                        detail = " (sacct={})".format(sacct)
            elif new_status == 'ZOMBIE':
                detail = " (chain files idle > {}h)".format(ZOMBIE_CHAIN_MTIME_HOURS)
            log_event("{} status: {} → {}{}".format(rn, old_status, new_status, detail))

        entry['last_known_status'] = new_status
        state[rn] = entry
        state_modified = True

    return state_modified


def get_all_statuses(all_runs, state, log_transitions=True):
    # type: (Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], bool) -> Dict[str, Dict[str, Any]]
    statuses = {rn: get_status(rn, all_runs, state) for rn in all_runs}
    if log_transitions:
        if _log_status_transitions(statuses, state):
            save_state(state)
    return statuses

# ============================================================================
#  COMMAND: status
# ============================================================================

def cmd_status(all_runs, state):
    # type: (Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]) -> None
    statuses = get_all_statuses(all_runs, state)
    counts = {}
    for s in ('CONVERGED', 'RUNNING', 'QUEUED', 'PENDING', 'STALLED', 'FAILED', 'ZOMBIE', 'PAUSED'):
        counts[s] = sum(1 for v in statuses.values() if v['status'] == s)

    total = len(all_runs)
    conv  = counts['CONVERGED']
    width = 40
    filled = int(width * conv / total)
    bar = '█' * filled + '░' * (width - filled)
    print()
    print(_c("  Campaign progress", 'bold'))
    print("  [{}] {}/{} converged ({:.1f}%)".format(
        bar, conv, total, 100.0 * conv / total))
    print()
    print("  " + "  ".join(
        "{} {}".format(_fmt_status(s)[:30], counts[s])
        for s in ('CONVERGED', 'RUNNING', 'QUEUED', 'PENDING', 'STALLED', 'FAILED', 'ZOMBIE', 'PAUSED')
    ))
    print()

    # Pair display: each exo run paired with its lcdm counterpart
    print(_c("  Pairs (exotic | LCDM)", 'bold'))
    print("  " + "-" * 86)

    # Build canonical pair list in priority order
    seen = set()
    pairs = []  # list of (stem, exo_name, lcdm_name)
    for rn in PRIORITY_ORDER:
        if rn in seen:
            continue
        if rn.startswith('exo_'):
            stem  = rn[len('exo_'):]
            other = 'lcdm_' + stem
        elif rn.startswith('lcdm_'):
            stem  = rn[len('lcdm_'):]
            other = 'exo_' + stem
        else:
            continue
        if other not in all_runs:
            other = None
        pairs.append((stem, rn if rn.startswith('exo_') else other,
                      other if rn.startswith('exo_') else rn))
        seen.add(rn)
        if other:
            seen.add(other)

    # Pre-compute health lookup from cached .health.json files (cheap I/O only)
    health_lookup = {}
    for rn, info in statuses.items():
        if info['status'] == 'RUNNING':
            cfg = all_runs[rn]
            folder = os.path.join(RUNS_ROOT, cfg['folder_path'])
            cached = _read_health_cache(folder, rn)
            if cached is not None:
                health_lookup[rn] = cached

    for stem, exo_name, lcdm_name in pairs:
        exo_str  = _format_pair_cell(exo_name,  statuses, health_lookup) if exo_name  in statuses else " " * 40
        lcdm_str = _format_pair_cell(lcdm_name, statuses, health_lookup) if lcdm_name in statuses else " " * 40
        print("  {} | {}".format(exo_str, lcdm_str))
    print()
    
    # Chain-completeness warnings (RUNNING jobs missing chain files past grace)
    warnings = [(rn, info['chain_warning'])
                for rn, info in statuses.items()
                if info.get('chain_warning')]
    if warnings:
        print(_c("  ⚠  Chain-completeness warnings", 'red'))
        print("    Expected {} chain files after {:.0f}h of running:".format(
            EXPECTED_CHAIN_COUNT, CHAIN_CHECK_AFTER_HOURS))
        for rn, msg in warnings:
            print(_c("    {:<35s}  {}".format(rn, msg), 'red'))
        print(_c("    These may have silently lost MPI ranks. Investigate with `tail` or `doctor`.", 'yellow'))
        print()
        
    # ETA estimate (very rough): assume converged runs averaged 24h wallclock
    running = counts['RUNNING'] + counts['QUEUED']
    pending = counts['PENDING'] + counts['STALLED']
    if running > 0 and pending >= 0:
        # crude: if N slots running in parallel, total remaining ~ pending/N * 24h + (24h for current)
        print(_c("  Rough ETA estimate", 'bold'))
        print("    {} active jobs, {} pending. At ~24 h/run with {} parallel slots,".format(
            running, pending, running))
        if running > 0:
            est_hours = 24.0 * (1.0 + pending / max(running, 1))
            print("    remaining campaign time ≈ {:.1f} h (≈ {:.1f} days)".format(
                est_hours, est_hours / 24.0))
        print()


def _format_pair_cell(run_name, statuses, health_lookup=None):
    # type: (str, Dict[str, Dict[str, Any]], Optional[Dict[str, Optional[Dict[str, Any]]]]) -> str
    info = statuses[run_name]
    base = "{:<30s} {}".format(run_name[:30], _fmt_status(info['status']))
    if info['status'] in ('RUNNING', 'QUEUED') and info['rminus1'] is not None:
        base += "  R-1={:.3f}".format(info['rminus1'])
    elif info['status'] == 'CONVERGED' and info['rminus1'] is not None:
        base += "  R-1={:.3f}".format(info['rminus1'])

    # Health indicator for RUNNING runs older than HEALTH_STATUS_MIN_HOURS
    if info['status'] == 'RUNNING' and health_lookup is not None:
        elapsed_h = (info.get('elapsed_seconds') or 0) / 3600.0
        if elapsed_h >= HEALTH_STATUS_MIN_HOURS and run_name in health_lookup:
            tag = _format_health_tag(health_lookup[run_name])
            if tag:
                base += "  " + tag

    # pad to align column
    visible_len = _visible_length(base)
    if visible_len < 42:
        base += " " * (42 - visible_len)
        
    if info.get('chain_warning'):
        base += "  " + _c("⚠ " + info['chain_warning'], 'red')
    return base

def _visible_length(s):
    # type: (str) -> int
    """Length ignoring ANSI escape codes."""
    return len(re.sub(r'\033\[[0-9;]*m', '', s))


# ============================================================================
#  COMMAND: launch N
# ============================================================================

def cmd_launch(N, all_runs, state):
    # type: (int, Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]) -> None
    statuses = get_all_statuses(all_runs, state)

    # Candidates: PENDING, in priority order. CONVERGED and friends are skipped.
    candidates = [rn for rn in PRIORITY_ORDER
                  if rn in all_runs and statuses[rn]['status'] == 'PENDING']

    if not candidates:
        print(_c("  No PENDING runs available. Use 'resubmit' for STALLED or 'doctor' for FAILED.", 'yellow'))
        return

    to_launch = candidates[:N]
    print(_c("\n  About to submit {} run(s) (top of priority queue):".format(len(to_launch)), 'bold'))
    for rn in to_launch:
        cfg = all_runs[rn]
        flag = " (CMB)" if cfg['has_cmb'] else ""
        print("    {}{} (P={})".format(rn, flag, cfg['n_sampled_params']))

    # Confirm
    cmb_in_set = any(all_runs[rn]['has_cmb'] for rn in to_launch)
    if cmb_in_set:
        print()
        ans = input(_c("  Some runs have CMB likelihood. Bump --cpus-per-task to 2 for those? [y/N]: ", 'cyan'))
        if ans.strip().lower() in ('y', 'yes'):
            for rn in to_launch:
                if all_runs[rn]['has_cmb']:
                    ok = bump_cpus_in_sh(all_runs[rn], new_cpus=2)
                    print("    {} cpus-per-task=2 ({})".format(
                        rn, "applied" if ok else "already set or failed"))

    print()
    ans = input(_c("  Proceed with submission? [y/N]: ", 'cyan'))
    if ans.strip().lower() not in ('y', 'yes'):
        print("  Aborted.")
        return

    n_submitted = 0
    for rn in to_launch:
        job_id = submit_run(all_runs[rn])
        if job_id:
            entry = state.get(rn, {})
            entry.update({
                'job_id': job_id,
                'submitted_at': datetime.now().isoformat(),
                'attempts': entry.get('attempts', 0) + 1,
            })
            state[rn] = entry
            save_state(state)
            log_event("Submitted {} as job {}".format(rn, job_id))
            print(_c("    ✓ {} → job {}".format(rn, job_id), 'green'))
            n_submitted += 1

    print()
    print(_c("  Submitted {}/{} runs.".format(n_submitted, len(to_launch)), 'bold'))


# ============================================================================
#  COMMAND: test
# ============================================================================

def cmd_test(all_runs, state):
    # type: (Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]) -> None
    statuses = get_all_statuses(all_runs, state)

    print()
    print(_c("  TEST SUITE — 6 runs covering all code paths", 'bold'))
    print("  " + "-" * 66)
    for rn in TEST_RUNS:
        if rn not in all_runs:
            print(_c("  [ERROR] Test run not found: {}".format(rn), 'red'))
            return
        cfg = all_runs[rn]
        flag = " (CMB)" if cfg['has_cmb'] else ""
        cur = statuses[rn]['status']
        print("    {:<32s} P={:>2d}{}  [{}]".format(rn, cfg['n_sampled_params'], flag, cur))
    print()

    # Skip ones that are already in flight or done
    submittable = [rn for rn in TEST_RUNS if statuses[rn]['status'] == 'PENDING']
    skipped = [rn for rn in TEST_RUNS if statuses[rn]['status'] != 'PENDING']
    if skipped:
        print(_c("  Skipping (not PENDING): {}".format(', '.join(skipped)), 'yellow'))

    if not submittable:
        print("  All test runs are non-PENDING; nothing to submit.")
        return

    cmb_runs = [rn for rn in submittable if all_runs[rn]['has_cmb']]
    if cmb_runs:
        print(_c("  CMB test runs detected: {}".format(', '.join(cmb_runs)), 'cyan'))
        ans = input(_c("  Bump --cpus-per-task to 2 for those? [y/N]: ", 'cyan'))
        if ans.strip().lower() in ('y', 'yes'):
            for rn in cmb_runs:
                ok = bump_cpus_in_sh(all_runs[rn], new_cpus=2)
                print("    {}: cpus-per-task=2 ({})".format(
                    rn, "applied" if ok else "no change"))

    print()
    ans = input(_c("  Submit all {} test runs at once? [y/N]: ".format(len(submittable)), 'cyan'))
    if ans.strip().lower() not in ('y', 'yes'):
        print("  Aborted.")
        return

    n_submitted = 0
    for rn in submittable:
        job_id = submit_run(all_runs[rn])
        if job_id:
            entry = state.get(rn, {})
            entry.update({
                'job_id': job_id,
                'submitted_at': datetime.now().isoformat(),
                'attempts': entry.get('attempts', 0) + 1,
                'test_run': True,
            })
            state[rn] = entry
            save_state(state)
            log_event("[test] Submitted {} as job {}".format(rn, job_id))
            print(_c("    ✓ {} → job {}".format(rn, job_id), 'green'))
            n_submitted += 1

    print()
    print(_c("  Test set submitted: {}/{}".format(n_submitted, len(submittable)), 'bold'))
    print()
    print("  Next steps:")
    print("    1. Monitor with:   python run_manager.py status")
    print("    2. When all 6 show CONVERGED, the YAML/SLURM/likelihood pipeline is validated.")
    print("    3. Launch the full campaign:  python run_manager.py auto 8")
    print("       (or `launch N` for batch-of-N at a time)")
    print("    Converged test runs will NOT be resubmitted by launch/auto.")
    print()

# ============================================================================
#  COMMANDS: throttle / pause / resume / drain (live control)
# ============================================================================

def cmd_throttle(N):
    # type: (int) -> None
    save_control(N=N)
    print(_c("  Throttle set: N = {}".format(N), 'green'))
    print("  Any running `auto` daemon will pick this up at its next poll cycle.")
    log_event("Throttle set to N={}".format(N))


def cmd_pause():
    save_control(paused=True)
    print(_c("  Paused. No new jobs will be submitted by auto.", 'yellow'))
    print("  Currently RUNNING/QUEUED jobs continue. Use `drain` to scancel them.")
    log_event("Paused")

def cmd_pause_one(run_name, all_runs, state):
    # type: (str, Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]) -> None
    """Pause a single named run: scancel its job and mark it PAUSED in state.
    Chain files preserved; bring back via `resubmit --all-paused` or
    `restart <name>` if you want fresh chains."""
    if run_name not in all_runs:
        print(_c("  Unknown run: {}".format(run_name), 'red'))
        return

    entry = state.get(run_name, {})
    job_id = entry.get('job_id')
    if not job_id:
        print(_c("  {} has no tracked job_id — nothing to pause.".format(run_name), 'yellow'))
        return

    sq = _squeue_state(job_id)
    if sq is None:
        print(_c("  {} job {} no longer in queue — marking as paused anyway.".format(
            run_name, job_id), 'yellow'))
    else:
        try:
            r = subprocess.run(['scancel', str(job_id)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15)
            if r.returncode != 0:
                err = r.stderr.decode('utf-8', errors='ignore').strip()
                print(_c("  scancel failed for {}: {}".format(run_name, err), 'red'))
                return
        except (subprocess.SubprocessError, OSError) as e:
            print(_c("  scancel exception: {}".format(e), 'red'))
            return

    entry['paused'] = True
    entry.pop('job_id', None)
    state[run_name] = entry
    save_state(state)
    log_event("pause {}: scancelled (job {}) → PAUSED".format(run_name, job_id))
    print(_c("  ⏸ {} (job {}) paused. Chain files preserved.".format(run_name, job_id), 'yellow'))
    print(_c("  To bring it back: python run_manager.py resubmit --all-paused", 'cyan'))
    print(_c("  Or for a fresh start with new covmat: python run_manager.py restart {}".format(run_name), 'cyan'))
    

def cmd_resume():
    save_control(paused=False)
    print(_c("  Resumed.", 'green'))
    log_event("Resumed")


def cmd_drain(all_runs, state):
    # type: (Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]) -> None
    statuses = get_all_statuses(all_runs, state)
    targets = []
    for rn in all_runs:
        if statuses[rn]['status'] in ('RUNNING', 'QUEUED'):
            jid = state.get(rn, {}).get('job_id')
            if jid:
                targets.append((rn, jid))

    if not targets:
        print("  No RUNNING/QUEUED jobs to drain.")
        return

    print(_c("\n  DRAIN — will scancel {} active job(s):".format(len(targets)), 'red'))
    for rn, jid in targets:
        print("    {:<35s}  job {}".format(rn, jid))
    print()
    print(_c("  Chains are preserved on disk; cobaya --resume will pick up where", 'yellow'))
    print(_c("  these runs left off when they are resubmitted.", 'yellow'))
    ans = input(_c("  Confirm scancel? [y/N]: ", 'cyan'))
    if ans.strip().lower() not in ('y', 'yes'):
        print("  Aborted.")
        return

    n_cancelled = 0
    for rn, jid in targets:
        try:
            r = subprocess.run(['scancel', str(jid)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               timeout=15)
            if r.returncode == 0:
                if rn in state:
                    state[rn].pop('job_id', None)
                    save_state(state)
                log_event("Drained {} (cancelled job {})".format(rn, jid))
                print(_c("    ✓ scancelled {} (job {})".format(rn, jid), 'green'))
                n_cancelled += 1
            else:
                err = r.stderr.decode('utf-8', errors='ignore').strip()
                print(_c("    ✗ scancel failed for {}: {}".format(rn, err), 'red'))
        except (subprocess.SubprocessError, OSError) as e:
            print(_c("    ✗ scancel exception for {}: {}".format(rn, e), 'red'))

    print()
    print(_c("  Drained {}/{} jobs. They are now PENDING (re-runnable with --resume).".format(
        n_cancelled, len(targets)), 'bold'))
    
def cmd_pause_all(all_runs, state):
    # type: (Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]) -> None
    """Pause the daemon AND scancel every RUNNING/QUEUED job, marking them
    PAUSED in state. Chain files are preserved on disk; cobaya --resume picks
    up where they left off when later released via `resubmit --all-paused`.
    """
    statuses = get_all_statuses(all_runs, state)
    targets = []
    for rn in all_runs:
        if statuses[rn]['status'] in ('RUNNING', 'QUEUED'):
            jid = state.get(rn, {}).get('job_id')
            if jid:
                targets.append((rn, jid))

    print()
    print(_c("  PAUSE --ALL — will pause the daemon AND scancel {} active job(s):".format(
        len(targets)), 'yellow'))
    for rn, jid in targets:
        print("    {:<35s}  job {}".format(rn, jid))
    if not targets:
        print("    (no active jobs; only the daemon will be paused)")
    print()
    print(_c("  Chain files are preserved. To bring these back later, run:", 'cyan'))
    print(_c("    python run_manager.py resubmit --all-paused", 'cyan'))
    print()
    ans = input(_c("  Confirm? [y/N]: ", 'cyan'))
    if ans.strip().lower() not in ('y', 'yes'):
        print("  Aborted.")
        return

    save_control(paused=True)
    log_event("pause --all: daemon paused")

    n_paused = 0
    for rn, jid in targets:
        try:
            r = subprocess.run(['scancel', str(jid)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               timeout=15)
            if r.returncode == 0:
                entry = state.get(rn, {})
                entry['paused'] = True
                entry.pop('job_id', None)
                state[rn] = entry
                save_state(state)
                log_event("pause --all: {} scancelled (job {}) → PAUSED".format(rn, jid))
                print(_c("    ⏸ {} (job {})".format(rn, jid), 'yellow'))
                n_paused += 1
            else:
                err = r.stderr.decode('utf-8', errors='ignore').strip()
                print(_c("    ✗ scancel failed for {}: {}".format(rn, err), 'red'))
        except (subprocess.SubprocessError, OSError) as e:
            print(_c("    ✗ scancel exception for {}: {}".format(rn, e), 'red'))

    print()
    print(_c("  Paused {}/{} runs. Daemon will not submit new jobs.".format(
        n_paused, len(targets)), 'bold'))
    print(_c("  Use `resubmit --all-paused` to bring them back. Use `resume` to un-pause the daemon.", 'cyan'))


def cmd_resubmit_all_paused(all_runs, state):
    # type: (Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]) -> None
    """Submit every PAUSED run (cobaya --resume from chain files). Clears the
    paused flag and resets auto_resubmit_attempts.
    """
    targets = [rn for rn in all_runs if state.get(rn, {}).get('paused', False)]
    if not targets:
        print("  No PAUSED runs to resubmit.")
        return

    print(_c("\n  Resubmitting {} PAUSED runs:".format(len(targets)), 'bold'))
    for rn in targets:
        print("    {}".format(rn))
    ans = input(_c("  Confirm? [y/N]: ", 'cyan'))
    if ans.strip().lower() not in ('y', 'yes'):
        print("  Aborted.")
        return

    n_done = 0
    for rn in targets:
        job_id = submit_run(all_runs[rn])
        if job_id:
            entry = state.get(rn, {})
            entry.update({
                'job_id': job_id,
                'submitted_at': datetime.now().isoformat(),
                'attempts': entry.get('attempts', 0) + 1,
                'auto_resubmit_attempts': 0,
            })
            entry.pop('paused', None)
            state[rn] = entry
            save_state(state)
            log_event("resubmit --all-paused: {} → job {}".format(rn, job_id))
            print(_c("    ✓ {} → job {}".format(rn, job_id), 'green'))
            n_done += 1

    print()
    print(_c("  Resubmitted {}/{} paused runs.".format(n_done, len(targets)), 'bold'))
    print(_c("  Note: daemon is still paused if you ran `pause --all`. Use `resume` to let auto submit fresh runs again.", 'cyan'))
    
# ============================================================================
#  COMMAND: auto N (daemon)
# ============================================================================

def cmd_auto(N, all_runs, state, poll_seconds=POLL_SECONDS_AUTO):
    # type: (int, Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], int) -> None
    print(_c("\n  AUTO MODE: maintaining {} concurrent active jobs.".format(N), 'bold'))
    print("  Poll interval: {} s ({} min).  Ctrl+C to stop.".format(
        poll_seconds, poll_seconds // 60))
    print("  Live control from another shell:")
    print("    python run_manager.py throttle <M>   # change N on the fly")
    print("    python run_manager.py pause          # stop new submissions")
    print("    python run_manager.py resume         # un-pause")
    print("    python run_manager.py drain          # scancel active jobs (destructive)")
    print()

    # Seed the control file with the starting state
    save_control(N=N, paused=False)

    cmb_remaining = any(
        all_runs[rn]['has_cmb'] and state.get(rn, {}).get('job_id') is None
        for rn in all_runs
    )
    if cmb_remaining:
        ans = input(_c("  Bump --cpus-per-task to 2 for ALL remaining CMB runs (one-time)? [y/N]: ", 'cyan'))
        if ans.strip().lower() in ('y', 'yes'):
            n_bumped = 0
            for rn, cfg in all_runs.items():
                if cfg['has_cmb'] and state.get(rn, {}).get('job_id') is None:
                    if bump_cpus_in_sh(cfg, new_cpus=2):
                        n_bumped += 1
            print(_c("  Bumped cpus on {} CMB scripts.".format(n_bumped), 'green'))
            print()

    iteration = 0
    try:
        while True:
            iteration += 1

            # Re-read control every iteration so throttle/pause take effect
            ctrl = load_control() or {}
            current_N = ctrl.get('N', N)
            paused    = ctrl.get('paused', False)

            statuses = get_all_statuses(all_runs, state)
            active = sum(1 for s in statuses.values()
                         if s['status'] in ('RUNNING', 'QUEUED'))
            converged = sum(1 for s in statuses.values()
                            if s['status'] == 'CONVERGED')
            pending = [rn for rn in PRIORITY_ORDER
                       if rn in all_runs and statuses[rn]['status'] == 'PENDING']

            ts = datetime.now().strftime('%H:%M:%S')
            tag = _c(" [PAUSED]", 'yellow') if paused else ""
            print(_c("  [{}] iter {}  N={}  conv={}/{}  active={}  pending={}{}".format(
                ts, iteration, current_N, converged, len(all_runs),
                active, len(pending), tag), 'gray'))

            if not paused:
                
                # First: auto-recover any FAILED runs with transient SLURM verdicts
                n_recovered = _attempt_auto_recovery(all_runs, state, statuses)
                if n_recovered > 0:
                    # Re-derive active so recovered jobs count toward the slot fill
                    statuses = get_all_statuses(all_runs, state)
                    active = sum(1 for s in statuses.values()
                                 if s['status'] in ('RUNNING', 'QUEUED'))

                slots = current_N - active
                if slots > 0 and pending:
                    to_launch = pending[:slots]
                    for rn in to_launch:
                        job_id = submit_run(all_runs[rn])
                        if job_id:
                            entry = state.get(rn, {})
                            entry.update({
                                'job_id': job_id,
                                'submitted_at': datetime.now().isoformat(),
                                'attempts': entry.get('attempts', 0) + 1,
                            })
                            state[rn] = entry
                            save_state(state)
                            log_event("[auto] Submitted {} as job {}".format(rn, job_id))
                            print(_c("    ✓ auto-launched {} (job {})".format(rn, job_id), 'green'))

            if converged == len(all_runs):
                print(_c("\n  All 112 runs converged. Campaign complete.", 'green'))
                return

            sleep_for = PAUSED_POLL_SECONDS if paused else poll_seconds
            time.sleep(sleep_for)
    except KeyboardInterrupt:
        print(_c("\n  Auto mode stopped by user.", 'yellow'))


# ============================================================================
#  COMMAND: resubmit
# ============================================================================

def cmd_resubmit(arg, all_runs, state):
    # type: (str, Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]) -> None
    statuses = get_all_statuses(all_runs, state)

    if arg == '--all-stalled':
        targets = [rn for rn in all_runs if statuses[rn]['status'] == 'STALLED']
        if not targets:
            print("  No STALLED runs to resubmit.")
            return
        print(_c("\n  Resubmitting {} STALLED runs:".format(len(targets)), 'bold'))
        
        for rn in targets:
            print("    {}".format(rn))
        ans = input(_c("  Confirm? [y/N]: ", 'cyan'))
        if ans.strip().lower() not in ('y', 'yes'):
            print("  Aborted.")
            return
    else:
        if arg not in all_runs:
            print(_c("  Unknown run: {}".format(arg), 'red'))
            return
        targets = [arg]

    for rn in targets:
        job_id = submit_run(all_runs[rn])
        if job_id:
            entry = state.get(rn, {})
            entry.update({
                'job_id': job_id,
                'submitted_at': datetime.now().isoformat(),
                'attempts': entry.get('attempts', 0) + 1,
                'auto_resubmit_attempts': 0,   # ← ADD THIS LINE
            })
            state[rn] = entry
            save_state(state)
            log_event("Resubmitted {} as job {}".format(rn, job_id))
            print(_c("    ✓ {} → job {}".format(rn, job_id), 'green'))
            
# ============================================================================
#  COMMAND: restart <run_name> (destructive wipe + fresh submit)
# ============================================================================

def cmd_restart(run_name, all_runs, state):
    """Wipe a run's outputs/log/err + state entry, then submit fresh.

    Use this when the chain files are poisoned (bad sampler config, corrupted
    checkpoint, etc.) and you want a clean restart, NOT a resume.
    """
    if run_name not in all_runs:
        print(_c("  Unknown run: {}".format(run_name), 'red'))
        return

    cfg = all_runs[run_name]
    folder = os.path.join(RUNS_ROOT, cfg['folder_path'])
    output_dir = os.path.join(folder, 'outputs')
    info = get_status(run_name, all_runs, state)

    print()
    print(_c("  RESTART — {}".format(run_name), 'bold'))
    print("  Current status: " + _fmt_status(info['status']))
    print(_c("  This will:", 'yellow'))
    if info['status'] in ('RUNNING', 'QUEUED'):
        print("    1. scancel job {}".format(info['job_id']))
    print("    2. delete all chain/output files in {}/".format(output_dir))
    print("    3. delete {}.log and {}.err".format(run_name, run_name))
    print("    4. clear state-file entry for this run")
    print("    5. submit a fresh run (NO --resume; starts from scratch)")
    print()
    ans = input(_c("  Confirm? [y/N]: ", 'cyan'))
    if ans.strip().lower() not in ('y', 'yes'):
        print("  Aborted.")
        return

    # 1. scancel
    if info['status'] in ('RUNNING', 'QUEUED') and info['job_id']:
        try:
            r = subprocess.run(['scancel', str(info['job_id'])],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               timeout=15)
            if r.returncode == 0:
                print(_c("    ✓ scancelled job {}".format(info['job_id']), 'green'))
                time.sleep(2)  # let SLURM release before resubmit
            else:
                err = r.stderr.decode('utf-8', errors='ignore').strip()
                print(_c("    ! scancel reported: {}".format(err), 'yellow'))
        except (subprocess.SubprocessError, OSError) as e:
            print(_c("    ! scancel exception: {}".format(e), 'yellow'))

    # 2. wipe chain files
    n_deleted = 0
    if os.path.isdir(output_dir):
        for fname in os.listdir(output_dir):
            if fname.startswith(run_name + '.') or fname == run_name:
                try:
                    os.remove(os.path.join(output_dir, fname))
                    n_deleted += 1
                except OSError:
                    pass
    print(_c("    ✓ wiped {} output file(s)".format(n_deleted), 'green'))

    # 3. wipe .log / .err
    for ext in ('.log', '.err'):
        path = os.path.join(folder, run_name + ext)
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
    print(_c("    ✓ wiped .log and .err", 'green'))

    # 4. clear state entry
    state.pop(run_name, None)
    save_state(state)
    print(_c("    ✓ cleared state-file entry", 'green'))

    # 5. submit fresh
    job_id = submit_run(cfg)
    if job_id:
        state[run_name] = {
            'job_id': job_id,
            'submitted_at': datetime.now().isoformat(),
            'attempts': 1,
            'auto_resubmit_attempts': 0,
        }
        save_state(state)
        log_event("Restarted {} as job {} (fresh, chains wiped)".format(run_name, job_id))
        print(_c("    ✓ submitted fresh as job {}".format(job_id), 'green'))


# ============================================================================
#  COMMAND: reset <run_name>
#  Dump covmat from existing chains → edit YAML → wipe outputs → optionally
#  resubmit. Designed for: YAML changed via generate_all_runs.py but cobaya
#  --resume refuses because settings mismatch. Preserves the chain-learned
#  proposal as a warm start while allowing a fresh run with new YAML settings.
# ============================================================================

def _dump_covmat_from_chains(run_name, all_runs):
    # type: (str, Dict[str, Any]) -> Optional[str]
    """Load chains via getdist, compute covmat of sampled params, write to
    {folder}/{run_name}.covmat in cobaya format. Returns path, or None."""
    try:
        from getdist import loadMCSamples
    except ImportError:
        print(_c("  [ERROR] getdist not installed (pip install getdist)", 'red'))
        return None

    cfg = all_runs[run_name]
    folder = os.path.join(RUNS_ROOT, cfg['folder_path'])
    chain_root = os.path.join(folder, 'outputs', run_name)

    try:
        samples = loadMCSamples(chain_root, settings={'ignore_rows': 0.3})
    except Exception as e:
        print(_c("  [ERROR] Failed to load chains: {}".format(e), 'red'))
        return None

    names = [p.name for p in samples.paramNames.names if not p.isDerived]
    if not names:
        print(_c("  [ERROR] No sampled parameters found in chains", 'red'))
        return None

    try:
        cov = np.atleast_2d(np.asarray(samples.cov(pars=names)))
    except Exception as e:
        print(_c("  [ERROR] Failed to compute covariance: {}".format(e), 'red'))
        return None

    eigvals = np.linalg.eigvalsh(cov)
    if np.any(eigvals <= 0):
        print(_c("  [WARN] Covmat has non-positive eigenvalue (min={:.3e}). "
                 "Chains may be too short.".format(eigvals.min()), 'yellow'))

    out_path = os.path.join(folder, run_name + '.covmat')
    try:
        with open(out_path, 'w') as f:
            f.write('# ' + ' '.join(names) + '\n')
            for row in cov:
                f.write(' '.join('{:.18e}'.format(x) for x in row) + '\n')
    except (IOError, OSError) as e:
        print(_c("  [ERROR] Failed to write covmat: {}".format(e), 'red'))
        return None

    return out_path


def cmd_reset(run_name, all_runs, state):
    """Dump covmat from existing chains, edit YAML to use it, wipe outputs/
    and log/err, then optionally resubmit."""
    if run_name not in all_runs:
        print(_c("  Unknown run: {}".format(run_name), 'red'))
        return

    cfg = all_runs[run_name]
    folder = os.path.join(RUNS_ROOT, cfg['folder_path'])
    output_dir = os.path.join(folder, 'outputs')
    info = get_status(run_name, all_runs, state)

    print()
    print(_c("  RESET — {}".format(run_name), 'bold'))
    print("  Current status: " + _fmt_status(info['status']))

    # ── Guard: need chain files to dump from ────────────────────────────
    n_chains = _count_chain_files(folder, run_name)
    if n_chains == 0:
        print(_c("  [ERROR] No chain files found in {}/".format(output_dir), 'red'))
        print("  Nothing to dump. Use 'restart' for a clean start without covmat,")
        print("  or 'submit' to resubmit with the current YAML as-is.")
        return

    # ── Warn if CONVERGED ───────────────────────────────────────────────
    if info['status'] == 'CONVERGED':
        print(_c("  WARNING: this run appears CONVERGED (R-1={}).".format(
            info.get('rminus1', '?')), 'yellow'))
        ans = input(_c("  Reset a converged run? [y/N]: ", 'cyan'))
        if ans.strip().lower() not in ('y', 'yes'):
            print("  Aborted.")
            return

    print("  Found {} chain file(s). Dumping covmat...".format(n_chains))

    # ── Step 1: dump covmat from chains ─────────────────────────────────
    covmat_path = _dump_covmat_from_chains(run_name, all_runs)
    if covmat_path is None:
        print(_c("  Aborting reset — covmat dump failed.", 'red'))
        return
    print(_c("    ✓ Wrote {}".format(covmat_path), 'green'))

    # ── Step 2: edit YAML covmat: line ──────────────────────────────────
    yaml_path = os.path.join(folder, run_name + '.yaml')
    covmat_re = re.compile(r'^( {4}covmat:\s*)(.*)$', re.MULTILINE)
    try:
        with open(yaml_path) as f:
            yaml_text = f.read()
    except (IOError, OSError) as e:
        print(_c("  [ERROR] Cannot read YAML: {}".format(e), 'red'))
        return

    matches = covmat_re.findall(yaml_text)
    if len(matches) != 1:
        print(_c("  [ERROR] Expected exactly 1 'covmat:' line in YAML, found {}".format(
            len(matches)), 'red'))
        return

    new_yaml = covmat_re.sub(
        lambda m: "{}{}.covmat".format(m.group(1), run_name),
        yaml_text, count=1,
    )
    with open(yaml_path, 'w') as f:
        f.write(new_yaml)
    print(_c("    ✓ YAML covmat: → {}.covmat".format(run_name), 'green'))

    # ── Step 3: scancel if active ───────────────────────────────────────
    if info['status'] in ('RUNNING', 'QUEUED') and info.get('job_id'):
        try:
            r = subprocess.run(['scancel', str(info['job_id'])],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               timeout=15)
            if r.returncode == 0:
                print(_c("    ✓ scancelled job {}".format(info['job_id']), 'green'))
                time.sleep(2)
            else:
                err = r.stderr.decode('utf-8', errors='ignore').strip()
                print(_c("    ! scancel reported: {}".format(err), 'yellow'))
        except (subprocess.SubprocessError, OSError) as e:
            print(_c("    ! scancel exception: {}".format(e), 'yellow'))

    # ── Step 4: delete outputs/ ─────────────────────────────────────────
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    print(_c("    ✓ deleted outputs/", 'green'))

    # ── Step 5: delete .log and .err (prevents false CONVERGED) ─────────
    for ext in ('.log', '.err'):
        p = os.path.join(folder, run_name + ext)
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass
    print(_c("    ✓ deleted .log and .err", 'green'))

    # ── Step 6: clear state entry ───────────────────────────────────────
    state.pop(run_name, None)
    save_state(state)
    print(_c("    ✓ cleared state-file entry", 'green'))

    # ── Step 7: prompt resubmit ─────────────────────────────────────────
    print()
    ans = input(_c("  Submit now? [y/N]: ", 'cyan'))
    if ans.strip().lower() in ('y', 'yes'):
        job_id = submit_run(cfg)
        if job_id:
            state[run_name] = {
                'job_id': job_id,
                'submitted_at': datetime.now().isoformat(),
                'attempts': 1,
                'auto_resubmit_attempts': 0,
            }
            save_state(state)
            log_event("Reset {} -> fresh submit as job {} (chain-learned covmat)".format(
                run_name, job_id))
            print(_c("    ✓ submitted as job {}".format(job_id), 'green'))
    else:
        log_event("Reset {} -> PENDING (chain-learned covmat applied, not submitted)".format(
            run_name))
        print("  Run is now PENDING with the new YAML + chain-learned covmat.")
        print("  Submit later with: python run_manager.py submit {}".format(run_name))


# ============================================================================
#  COMMAND: submit <run_name>  (canonical name; `resubmit` is now an alias)
# ============================================================================

def cmd_submit(arg, all_runs, state):
    """Submit a single run by name, bypassing the priority queue.

    Preserves chain files on disk; cobaya --resume picks up where they left off.
    Also handles --all-stalled batch operation for resuming STALLED runs.
    """
    statuses = get_all_statuses(all_runs, state)

    if arg not in all_runs:
        print(_c("  Unknown run: {}".format(arg), 'red'))
        return
    targets = [arg]

    for rn in targets:
        job_id = submit_run(all_runs[rn])
        if job_id:
            entry = state.get(rn, {})
            
            entry.update({
                'job_id': job_id,
                'submitted_at': datetime.now().isoformat(),
                'attempts': entry.get('attempts', 0) + 1,
                'auto_resubmit_attempts': 0,  # manual submit resets the counter
            })
            entry.pop('paused', None)  # ← ADD: manual submit clears paused state
            state[rn] = entry
            save_state(state)
            log_event("Submitted {} as job {}".format(rn, job_id))
            print(_c("    ✓ {} → job {}".format(rn, job_id), 'green'))


# ============================================================================
#  COMMAND: tail <N|all> <state>
# ============================================================================

_TAIL_STATE_MAP = {
    'running':   'RUNNING',
    'queued':    'QUEUED',
    'failed':    'FAILED',
    'converged': 'CONVERGED',
    'stalled':   'STALLED',
    'pending':   'PENDING',
    'zombie':    'ZOMBIE',
    'paused':    'PAUSED',
}


def _read_log_tail_lines(log_path, n_lines=100):
    if not os.path.exists(log_path):
        return []
    try:
        with open(log_path) as f:
            lines = f.readlines()
        return lines[-n_lines:]
    except (IOError, OSError):
        return []


def _print_getdist_summary(run_name, all_runs):
    """Load chains via getdist and print marginalised 1D stats as text."""
    try:
        from getdist import loadMCSamples
    except ImportError:
        print(_c("    [getdist not available — pip install getdist]", 'red'))
        return

    cfg = all_runs[run_name]
    chain_root = os.path.join(RUNS_ROOT, cfg['folder_path'], 'outputs', run_name)

    try:
        samples = loadMCSamples(chain_root, settings={'ignore_rows': 0.3})
    except Exception as e:
        print(_c("    [Failed to load chains: {}]".format(e), 'red'))
        return

    try:
        marge = samples.getMargeStats()
    except Exception as e:
        print(_c("    [Failed to compute marginalised stats: {}]".format(e), 'red'))
        return

    sampled = [p for p in samples.paramNames.names if not p.isDerived]
    derived = [p for p in samples.paramNames.names if p.isDerived]

    def _emit(label, params):
        if not params:
            return
        print(_c("    " + label, 'bold'))
        print("    " + "─" * 70)
        print("      {:<18s} {:>14s}   {:<14s} {:>22s}".format(
            "Parameter", "Mean", "σ", "68% interval"))
        print("    " + "─" * 70)
        for p in params:
            try:
                ps = marge.parWithName(p.name)
                mean, err = ps.mean, ps.err
                lim = ps.limits[0]  # 68% by default
                lo, hi = lim.lower, lim.upper
                print("      {:<18s} {:>14.5g}   {:<14.5g} [{:>9.5g}, {:>9.5g}]".format(
                    p.name, mean, err, lo, hi))
            except Exception:
                print("      {:<18s} (failed to compute stats)".format(p.name))
        print()

    print()
    _emit("Sampled parameters", sampled)
    _emit("Derived parameters", derived)


def cmd_tail(N_arg, state_arg, all_runs, state):
    """Show last 100 lines of the .log for N random runs in the requested state."""
    target_status = _TAIL_STATE_MAP.get(state_arg.lower())
    if target_status is None:
        print(_c("  Unknown state: {}".format(state_arg), 'red'))
        print("  Valid states: {}".format(', '.join(sorted(_TAIL_STATE_MAP.keys()))))
        return

    statuses = get_all_statuses(all_runs, state)
    matching = [rn for rn in all_runs if statuses[rn]['status'] == target_status]

    if not matching:
        print(_c("  No runs in state {}.".format(target_status), 'yellow'))
        return

    if N_arg.lower() == 'all':
        N = len(matching)
    else:
        try:
            N = int(N_arg)
        except ValueError:
            print(_c("  N must be an integer or 'all'.", 'red'))
            return
        if N <= 0:
            print(_c("  N must be > 0.", 'red'))
            return
        N = min(N, len(matching))

    selected = random.sample(matching, N) if N < len(matching) else list(matching)
    selected.sort()

    bar = "═" * 70

    for rn in selected:
        info = statuses[rn]
        cfg  = all_runs[rn]
        log_path = os.path.join(RUNS_ROOT, cfg['folder_path'], rn + '.log')
        _, color = STATUS_SYMBOL[info['status']]

        extras = []
        if info['rminus1']    is not None: extras.append("R-1={:.4f}".format(info['rminus1']))
        if info['n_samples']  is not None: extras.append("samples={}".format(info['n_samples']))
        if info['acceptance'] is not None: extras.append("accept={:.3f}".format(info['acceptance']))
        extras_str = "  ".join(extras)

        print()
        print(_c(bar, color))
        print(_c("  {}  [{}]  {}".format(rn, info['status'], extras_str), color))
        print(_c(bar, color))

        lines = _read_log_tail_lines(log_path, 100)
        if not lines:
            print(_c("    (no .log on disk yet)", 'gray'))
        else:
            for line in lines:
                print("  " + line.rstrip())

    if target_status == 'CONVERGED':
        print()
        ans = input(_c("  Generate getdist summary for {} converged run(s)? [y/N]: ".format(
            len(selected)), 'cyan'))
        if ans.strip().lower() in ('y', 'yes'):
            for rn in selected:
                print()
                print(_c(bar, 'magenta'))
                print(_c("  getdist summary: {}".format(rn), 'magenta'))
                if statuses[rn]['rminus1'] is not None:
                    print(_c("  Rminus1_last: {:.5f}".format(statuses[rn]['rminus1']), 'magenta'))
                print(_c(bar, 'magenta'))
                _print_getdist_summary(rn, all_runs)


# ============================================================================
#  AUTO TRANSIENT-FAILURE RECOVERY
# ============================================================================

def _attempt_auto_recovery(all_runs, state, statuses):
    """Auto-resubmit FAILED runs whose sacct verdict is NODE_FAIL / BOOT_FAIL /
    PREEMPTED — the transient cluster-side failures that cobaya --resume can
    handle. Cap per-run attempts at MAX_AUTO_RESUBMITS to avoid loops on truly
    sick nodes.

    Returns the number of runs successfully auto-resubmitted this iteration.
    """
    n_recovered = 0
    for rn, cfg in all_runs.items():
        if statuses[rn]['status'] != 'FAILED':
            continue
        entry = state.get(rn, {})
        job_id = entry.get('job_id')
        if not job_id:
            continue
        sacct = _sacct_state(job_id)
        if sacct not in TRANSIENT_SLURM_FAILURE_STATES:
            continue
        n_attempts = entry.get('auto_resubmit_attempts', 0)
        if n_attempts >= MAX_AUTO_RESUBMITS:
            continue

        new_job_id = submit_run(cfg)
        if new_job_id:
            entry.update({
                'job_id': new_job_id,
                'submitted_at': datetime.now().isoformat(),
                'attempts': entry.get('attempts', 0) + 1,
                'auto_resubmit_attempts': n_attempts + 1,
                'last_auto_resubmit_at': datetime.now().isoformat(),
                'last_auto_resubmit_reason': sacct,
            })
            state[rn] = entry
            save_state(state)
            log_event("[auto-recover] {} ({}) → job {} (attempt {}/{})".format(
                rn, sacct, new_job_id, n_attempts + 1, MAX_AUTO_RESUBMITS))
            print(_c("    ⟲ auto-recovered {} ({}) → job {} [{}/{}]".format(
                rn, sacct, new_job_id, n_attempts + 1, MAX_AUTO_RESUBMITS), 'magenta'))
            n_recovered += 1
    return n_recovered


# ============================================================================
#  COMMAND: doctor
# ============================================================================

DOCTOR_PATTERNS = [
    # (regex pattern, label, suggested fix)
    (r'oom-kill|OOMKilled|MemoryError',
     'Out of memory',
     "Bump `--mem-per-cpu` in the .sh (e.g. 5000 → 8000, or 8000 → 12000). "
     "CMB runs hold large Planck arrays; UVLF-only runs rarely OOM."),
    (r'clik|libclik',
     'CLIK / Planck likelihood load failure',
     "Re-source the CLIK profile script. Confirm Planck data is in "
     "$COBAYA_PACKAGES_PATH/data/. Try: `source $clik_profile.sh && cobaya-install`."),
    (r"Could not find correct value of cs2_fld|shooting failed|root finding",
     'CLASS shooting / root finding failure',
     "An exotic-DE parameter combo pushed CLASS out of its solver range. "
     "Check the `ref` distributions in the YAML are not too wide; if the polygon "
     "constraint is tight, the initial guess may be on the boundary."),
    (r'mpirun (detected|noticed|has exited)|exited on signal|Killed by signal',
     'MPI rank crash (transient)',
     "Usually a node-level glitch. The .sh has an auto-retry loop that catches "
     "signal kills and resets the counter. Just resubmit."),
    (r'No space left on device|disk quota exceeded',
     'Disk full',
     "Check `lustre` quota with `lfs quota -u $USER /lustre`. Clear stale outputs."),
    (r'ModuleNotFoundError|ImportError|cannot import',
     'Python module not found',
     "The virtualenv may not be activated. Check `source` line in .sh points "
     "to the right venv. Verify `python -c \"import classy, cobaya\"` works there."),
    (r'Could not initialize external likelihood|UVLFLikelihood',
     'Cobaya likelihood load error',
     "Check `python_path` in the YAML resolves and the .py file imports cleanly. "
     "Try: `python -c \"from jwst_likelihood_uvlf import UVLFLikelihood\"` "
     "after setting PYTHONPATH."),
    (r'DUE TO TIME LIMIT|CANCELLED AT',
     'SLURM time limit',
     "Wall-clock 54h was insufficient. R-1 is likely still > 0.02. "
     "Resubmit — cobaya-run with --resume picks up where it stopped. "
     "If repeated: increase `--time` in the .sh, or relax `Rminus1_stop` to 0.05 "
     "for an exploratory chain."),
]

# ============================================================================
#  COMMAND: storage  (disk usage report)
# ============================================================================

def _bytes_to_human(n):
    # type: (int) -> str
    if n < 1024:
        return "{} B".format(n)
    if n < 1024 * 1024:
        return "{:.1f} KB".format(n / 1024.0)
    if n < 1024 ** 3:
        return "{:.1f} MB".format(n / (1024.0 ** 2))
    return "{:.2f} GB".format(n / (1024.0 ** 3))


def _du_run(folder):
    # type: (str) -> int
    """Recursive sum of file sizes in a run folder."""
    total = 0
    if not os.path.isdir(folder):
        return 0
    try:
        for dirpath, _, filenames in os.walk(folder):
            for fname in filenames:
                fp = os.path.join(dirpath, fname)
                try:
                    total += os.path.getsize(fp)
                except OSError:
                    pass
    except OSError:
        pass
    return total


def _lfs_quota():
    # type: () -> Optional[Dict[str, int]]
    """Returns {'used': bytes, 'limit': bytes} from `lfs quota -u $USER .`,
    or None if not on Lustre or the command fails."""
    user = os.environ.get('USER', '')
    if not user:
        return None
    try:
        r = subprocess.run(['lfs', 'quota', '-u', user, '.'],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           timeout=10)
        if r.returncode != 0:
            return None
        out = r.stdout.decode('utf-8', errors='ignore')
        for line in out.split('\n'):
            parts = line.split()
            if len(parts) >= 4 and parts[0].startswith('/'):
                try:
                    used_kb  = int(parts[1].rstrip('*'))
                    limit_kb = int(parts[3])
                    return {'used': used_kb * 1024,
                            'limit': limit_kb * 1024 if limit_kb > 0 else None}
                except (ValueError, IndexError):
                    continue
        return None
    except (subprocess.SubprocessError, OSError):
        return None

def cmd_reset_failed(all_runs, state):
    # type: (Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]) -> None
    """Wipe FAILED runs' chains + state so they appear as PENDING again.
    Logs and chain files are renamed with a .failed-backup suffix for forensics,
    NOT deleted. Does not resubmit — daemon/launch picks them up later."""
    import time
    statuses = get_all_statuses(all_runs, state)
    failed = [rn for rn in all_runs if statuses[rn]['status'] == 'FAILED']

    if not failed:
        print(_c("  No FAILED runs to reset.", 'green'))
        return

    print()
    print(_c("  Will reset {} FAILED runs to PENDING:".format(len(failed)), 'yellow'))
    for rn in failed:
        print("    {}".format(rn))
    print()
    print(_c("  This wipes chain files and renames .log/.err to .failed-backup-<ts>.", 'gray'))
    print(_c("  Runs will NOT be resubmitted — daemon/launch handles that later.", 'gray'))
    ans = input(_c("  Confirm? [y/N]: ", 'cyan'))
    if ans.strip().lower() not in ('y', 'yes'):
        print("  Aborted.")
        return

    ts = int(time.time())
    n_reset = 0
    for rn in failed:
        cfg = all_runs[rn]
        folder = os.path.join(RUNS_ROOT, cfg['folder_path'])
        out_dir = os.path.join(folder, 'outputs')

        # Wipe chain artefacts (cobaya files named <run>.*)
        if os.path.isdir(out_dir):
            for f in os.listdir(out_dir):
                if f.startswith(rn + '.'):
                    try:
                        os.remove(os.path.join(out_dir, f))
                    except OSError as e:
                        print(_c("    ! couldn't remove {}: {}".format(f, e), 'red'))

        # Rename .log / .err to backup so failure-pattern matching no longer fires
        for ext in ('.log', '.err'):
            src = os.path.join(folder, rn + ext)
            if os.path.exists(src):
                bak = "{}.failed-backup-{}".format(src, ts)
                try:
                    os.rename(src, bak)
                except OSError as e:
                    print(_c("    ! couldn't rename {}: {}".format(src, e), 'red'))

        # Clear state entry entirely (removes job_id, paused flag, attempts)
        if rn in state:
            del state[rn]

        log_event("reset-failed: {} → PENDING (chains wiped, log backed up @ {})".format(rn, ts))
        n_reset += 1

    save_state(state)
    print()
    print(_c("  Reset {}/{} FAILED runs to PENDING.".format(n_reset, len(failed)), 'green'))
    print(_c("  Forensic backups: <run_folder>/<run_name>.{{log,err}}.failed-backup-{}".format(ts), 'gray'))
    
    
def cmd_storage(arg, all_runs, state):
    # type: (Optional[str], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]) -> None
    """Disk usage report.

    Usage:
      storage           → summary by status + grand total + projection
      storage --top     → also show top-10 biggest runs
      storage --all     → also list every run by size
    """
    statuses = get_all_statuses(all_runs, state)

    sizes = {}
    for rn, cfg in all_runs.items():
        folder = os.path.join(RUNS_ROOT, cfg['folder_path'])
        sizes[rn] = _du_run(folder)

    by_status = {}
    for rn, st_info in statuses.items():
        by_status.setdefault(st_info['status'], []).append(rn)

    print()
    print(_c("  Storage report", 'bold'))
    print("  " + "-" * 65)

    grand = 0
    for st in ('CONVERGED', 'RUNNING', 'QUEUED', 'ZOMBIE', 'PAUSED',
               'STALLED', 'FAILED', 'PENDING'):
        runs = by_status.get(st, [])
        if not runs:
            continue
        total = sum(sizes[rn] for rn in runs)
        grand += total
        avg = total / len(runs)
        nonzero = [s for s in (sizes[rn] for rn in runs) if s > 0]
        if nonzero:
            print("    {:<10s} {:>3d} runs  total {:>10s}   avg {:>9s}".format(
                st, len(runs), _bytes_to_human(total), _bytes_to_human(avg)))
        else:
            print("    {:<10s} {:>3d} runs  (no output yet)".format(st, len(runs)))

    print("  " + "-" * 65)
    print("    {:<10s} {:>3d} runs  total {:>10s}".format(
        "TOTAL", len(all_runs), _bytes_to_human(grand)))

    converged_runs = by_status.get('CONVERGED', [])
    if converged_runs:
        conv_total = sum(sizes[rn] for rn in converged_runs)
        conv_avg = conv_total / len(converged_runs)
        clean_projection = len(all_runs) * conv_avg
        print()
        print(_c("  Projection (extrapolating from {} converged run(s)):".format(
            len(converged_runs)), 'cyan'))
        print("    avg per converged run: {}".format(_bytes_to_human(conv_avg)))
        print("    projected campaign total: {}".format(_bytes_to_human(clean_projection)))

    quota = _lfs_quota()
    if quota:
        used = quota['used']
        limit = quota.get('limit')
        if limit:
            pct = 100.0 * used / limit
            color = 'red' if pct > 80 else ('yellow' if pct > 60 else 'green')
            print(_c("\n  Lustre quota: {} / {} ({:.1f}%)".format(
                _bytes_to_human(used), _bytes_to_human(limit), pct), color))
        else:
            print("\n  Lustre usage: {}".format(_bytes_to_human(used)))

    if arg in ('--all', '--top'):
        n_show = len(all_runs) if arg == '--all' else 10
        ranked = sorted(sizes.items(), key=lambda kv: kv[1], reverse=True)[:n_show]
        print()
        print(_c("  Largest {} runs:".format(len(ranked)), 'bold'))
        for rn, sz in ranked:
            if sz == 0:
                continue
            st = statuses[rn]['status']
            print("    {:<35s} {:<12s} {}".format(rn, st, _bytes_to_human(sz)))
    print()
    
# ============================================================================
#  COMMAND: follow <run_name>  (live tail -f of a run's .log)
# ============================================================================

def cmd_follow(run_name, all_runs, state):
    # type: (str, Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]) -> None
    """Stream a run's .log file like tail -f. Ctrl+C to stop."""
    if run_name not in all_runs:
        print(_c("  Unknown run: {}".format(run_name), 'red'))
        return

    cfg = all_runs[run_name]
    log_path = os.path.join(RUNS_ROOT, cfg['folder_path'], run_name + '.log')

    if not os.path.exists(log_path):
        print(_c("  No .log file yet at {}".format(log_path), 'yellow'))
        print("  Run may not have started, or Lustre cache hasn't synced.")
        return

    info = get_status(run_name, all_runs, state)
    print()
    print(_c("  Following {}  [{}]".format(run_name, info['status']), 'bold'))
    print(_c("  Ctrl+C to stop.", 'gray'))
    print("  " + "─" * 70)

    # Print recent backlog (~last 20 lines) for context
    try:
        with open(log_path) as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 4096))
            recent = f.read()
        tail_lines = recent.split('\n')[-21:]
        for line in tail_lines:
            print("  " + line)
    except (IOError, OSError) as e:
        print(_c("  Error reading {}: {}".format(log_path, e), 'red'))
        return

    # Follow appended bytes; handle log rotation
    try:
        with open(log_path) as f:
            f.seek(0, 2)
            while True:
                line = f.readline()
                if line:
                    print("  " + line.rstrip())
                else:
                    try:
                        cur_size = os.path.getsize(log_path)
                        if cur_size < f.tell():
                            f.close()
                            f = open(log_path)
                            print(_c("  [log file truncated — re-opened]", 'yellow'))
                        else:
                            time.sleep(1.0)
                    except OSError:
                        time.sleep(1.0)
    except KeyboardInterrupt:
        print()
        print(_c("  Stopped following.", 'gray'))
        

def cmd_doctor(run_name, all_runs, state):
    # type: (str, Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]) -> None
    if run_name not in all_runs:
        print(_c("  Unknown run: {}".format(run_name), 'red'))
        return
    cfg = all_runs[run_name]
    folder        = os.path.join(RUNS_ROOT, cfg['folder_path'])
    err_path      = os.path.join(folder, run_name + '.err')
    log_path      = os.path.join(folder, run_name + '.log')
    progress_path = os.path.join(folder, 'outputs', run_name + '.progress')

    info = get_status(run_name, all_runs, state)

    print()
    print(_c("  DOCTOR — {}".format(run_name), 'bold'))
    print("  Status: " + _fmt_status(info['status']))
    if info['job_id']:
        print("  Last job ID: {}".format(info['job_id']))
    if info['rminus1'] is not None:
        print("  Last R-1: {:.4f}  (target < {})".format(info['rminus1'], CONVERGENCE_RMINUS1))
        print("  Acceptance: {:.3f}".format(info['acceptance']) if info['acceptance'] else "")
        print("  Samples: {}".format(info['n_samples']))
    print()

    if not os.path.exists(err_path) and not os.path.exists(log_path):
        print(_c("  No log or err file yet. Has the job run at all?", 'yellow'))
        return

    err_tail = _read_tail(err_path, 12000)
    log_tail = _read_tail(log_path, 12000)
    combined = err_tail + "\n" + log_tail

    matches = []
    for pat, label, fix in DOCTOR_PATTERNS:
        if re.search(pat, combined, flags=re.IGNORECASE):
            matches.append((label, fix))

    if not matches:
        print(_c("  No known error pattern matched.", 'yellow'))
        print("  Tail of .err ({}):".format(err_path))
        print(_c("  " + "-" * 60, 'gray'))
        for line in err_tail.split('\n')[-20:]:
            print("  " + line)
        return

    print(_c("  Detected issues:", 'bold'))
    for i, (label, fix) in enumerate(matches, 1):
        print()
        print(_c("  [{}] {}".format(i, label), 'red' if i == 1 else 'yellow'))
        print("      Fix: " + fix)
    print()





# ============================================================================
#  HEALTH MONITORING — chain-based convergence diagnostics
#
#  Bypasses .progress entirely. Computes Gelman-Rubin R-1 directly from
#  chain .txt files at multiple fractions of the chain length to produce
#  a convergence trajectory from a single snapshot.
# ============================================================================

HEALTH_CACHE_FILE      = '.health.json'
HEALTH_CACHE_MAX_AGE_H = 2.0       # stale threshold for status display
HEALTH_MIN_CHAINS      = 2         # need at least 2 chains for GR
HEALTH_MIN_SAMPLES     = 50        # per chain, post-burn-in
HEALTH_BURN_FRAC       = 0.3       # discard first 30%
HEALTH_CHECKPOINTS     = [0.2, 0.4, 0.6, 0.8, 1.0]
HEALTH_STATUS_MIN_HOURS = 3.0      # only show health in status after this





def _gelman_rubin_per_param(weights_list, params_list):
    # type: (List[Any], List[Any]) -> Any
    """Compute Gelman-Rubin R-1 per sampled parameter.

    Uses weighted statistics to properly handle cobaya's multiplicity weights.
    Returns numpy array of R-1 values, one per parameter.
    """
    M = len(weights_list)
    P = params_list[0].shape[1]

    chain_means = np.zeros((M, P))
    chain_vars = np.zeros((M, P))
    chain_weights_total = np.zeros(M)

    for j in range(M):
        w = weights_list[j]
        theta = params_list[j]
        W_total = w.sum()
        chain_weights_total[j] = W_total
        # Weighted mean
        mean_j = np.average(theta, weights=w, axis=0)
        chain_means[j] = mean_j
        # Weighted variance
        diff = theta - mean_j
        chain_vars[j] = np.average(diff ** 2, weights=w, axis=0)

    W_bar = chain_weights_total.mean()
    grand_mean = np.average(chain_means, weights=chain_weights_total, axis=0)

    # Between-chain variance
    B = W_bar / (M - 1) * np.sum(
        chain_weights_total[:, None] / chain_weights_total.sum()
        * (chain_means - grand_mean) ** 2 * M,
        axis=0
    )

    # Within-chain variance (weighted average of per-chain variances)
    W = np.average(chain_vars, weights=chain_weights_total, axis=0)

    # Avoid division by zero
    W_safe = np.where(W > 0, W, 1e-30)

    V_hat = (W_bar - 1) / W_bar * W + B / W_bar
    R_minus_1 = V_hat / W_safe - 1.0

    # Clamp negative values (can happen with very short/correlated chains)
    R_minus_1 = np.maximum(R_minus_1, 0.0)

    return R_minus_1


def _compute_health(folder, run_name):
    # type: (str, str) -> Optional[Dict[str, Any]]
    """Full health computation for one run. Returns health dict or None."""

    try:
        from getdist import loadMCSamples
    except ImportError:
        return None

    chain_root = os.path.join(folder, 'outputs', run_name)

    # getdist handles .paramnames, chain discovery, everything
    try:
        samples_gd = loadMCSamples(chain_root, settings={'ignore_rows': 0})
    except Exception:
        return None

    sampled_params = [p for p in samples_gd.paramNames.names if not p.isDerived]
    sampled_names = [p.name for p in sampled_params]
    if not sampled_names:
        return None

    # Map sampled param names → chain-file column indices
    # Chain columns: 0=weight, 1=-logpost, 2+i = param i (paramnames order)
    all_param_names = [p.name for p in samples_gd.paramNames.names]
    sampled_cols = [2 + all_param_names.index(n) for n in sampled_names]

    # Read per-chain raw data (getdist merges; we need them separate for GR)
    chain_files = sorted(glob.glob(chain_root + '.*.txt'))
    chain_files = [f for f in chain_files
                   if f.rsplit('.', 2)[-2].isdigit()]

    weights_list = []
    params_list = []
    chain_lengths_raw = []

    for cf in chain_files:
        try:
            data = np.loadtxt(cf)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            if data.shape[0] < 5:
                continue
            chain_lengths_raw.append(data.shape[0])
            weights_list.append(data[:, 0])
            params_list.append(data[:, sampled_cols])
        except Exception:
            continue

    M = len(weights_list)
    if M < HEALTH_MIN_CHAINS:
        return None
    

    # Apply burn-in: discard first HEALTH_BURN_FRAC of each chain
    burned_weights = []
    burned_params = []

    for w, p in zip(weights_list, params_list):
        n = len(w)
        start = int(n * HEALTH_BURN_FRAC)
        if n - start < 10:
            continue
        burned_weights.append(w[start:])
        burned_params.append(p[start:])

    if len(burned_weights) < HEALTH_MIN_CHAINS:
        return None

    # Find minimum post-burn-in chain length
    chain_lengths_post = [len(w) for w in burned_weights]
    n_min = min(chain_lengths_post)

    if n_min < HEALTH_MIN_SAMPLES:
        return {'verdict': 'EARLY', 'n_samples': int(sum(w.sum() for w in burned_weights)),
                'chain_lengths': chain_lengths_raw,
                'rminus1': None, 'trajectory': [], 'per_param': {},
                'bottleneck': None, 'n_eff': 0,
                'acceptance': None, 'timestamp': datetime.now().isoformat()}

    # ── Trajectory: R-1 at each checkpoint fraction ─────────────────────
    trajectory = []
    for frac in HEALTH_CHECKPOINTS:
        n_use = max(20, int(frac * n_min))
        trunc_w = [w[:n_use] for w in burned_weights]
        trunc_p = [p[:n_use] for p in burned_params]
        r1_per_param = _gelman_rubin_per_param(trunc_w, trunc_p)
        trajectory.append((frac, float(r1_per_param.max())))

    # ── Current R-1 (from full post-burn-in chains) ─────────────────────
    r1_per_param = _gelman_rubin_per_param(burned_weights, burned_params)
    rminus1 = float(r1_per_param.max())

    per_param = {}
    bottleneck = None
    bottleneck_val = -1.0
    for name, val in zip(sampled_names, r1_per_param):
        per_param[name] = round(float(val), 4)
        if val > bottleneck_val:
            bottleneck_val = val
            bottleneck = name

    # ── Acceptance rate ──────────────────────────────────────────────────
    total_rows = sum(len(w) for w in weights_list)
    total_weight = sum(float(w.sum()) for w in weights_list)
    acceptance = total_rows / total_weight if total_weight > 0 else 0.0

    # ── N_eff (Kish ESS from multiplicity weights, summed across chains) ──
    n_eff = 0.0
    for w in burned_weights:
        sum_w = float(w.sum())
        sum_w2 = float((w ** 2).sum())
        if sum_w2 > 0:
            n_eff += (sum_w ** 2) / sum_w2
    n_eff = int(n_eff)

    # ── Total post-burn-in weighted sample count ────────────────────────
    n_samples = int(sum(w.sum() for w in burned_weights))

    # ── Verdict ─────────────────────────────────────────────────────────
    verdict = _health_verdict(trajectory, rminus1, acceptance)

    return {
        'timestamp': datetime.now().isoformat(),
        'rminus1': round(rminus1, 4),
        'trajectory': [[f, round(r, 4)] for f, r in trajectory],
        'per_param': per_param,
        'bottleneck': bottleneck,
        'n_samples': n_samples,
        'n_eff': n_eff,
        'acceptance': round(acceptance, 4),
        'chain_lengths': chain_lengths_raw,
        'verdict': verdict,
    }


def _health_verdict(trajectory, rminus1, acceptance):
    # type: (List[Tuple[float, float]], float, float) -> str
    """Classify convergence health from trajectory and current diagnostics."""
    if rminus1 is None:
        return 'EARLY'
    if rminus1 < 0.05:
        return 'ALMOST'

    # Trend: slope of R-1 over the last 3+ trajectory points
    if len(trajectory) >= 3:
        recent = trajectory[-3:]
        r_vals = [r for _, r in recent]
        decreasing = all(r_vals[i] > r_vals[i + 1] for i in range(len(r_vals) - 1))
        flat_or_rising = r_vals[-1] >= r_vals[0] * 0.95  # within 5% = flat
    else:
        decreasing = False
        flat_or_rising = True

    if acceptance < 0.10:
        return 'STUCK'
    if flat_or_rising and rminus1 > 0.5:
        return 'STUCK'
    if decreasing and acceptance < 0.15:
        return 'SLOW'
    if decreasing:
        return 'CONVERGING'

    # Not clearly decreasing but not obviously stuck
    if rminus1 < 0.3:
        return 'CONVERGING'
    return 'SLOW'


def _read_health_cache(folder, run_name):
    # type: (str, str) -> Optional[Dict[str, Any]]
    """Read cached .health.json, return None if missing or unparseable."""
    path = os.path.join(folder, HEALTH_CACHE_FILE)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (ValueError, IOError, OSError):
        return None


def _write_health_cache(folder, health):
    # type: (str, Dict[str, Any]) -> None
    """Write health dict to .health.json in the run's folder."""
    path = os.path.join(folder, HEALTH_CACHE_FILE)
    try:
        with open(path, 'w') as f:
            json.dump(health, f, indent=2)
    except (IOError, OSError):
        pass


def _health_age_hours(health):
    # type: (Dict[str, Any]) -> float
    """Hours since the health cache was computed."""
    try:
        ts = datetime.fromisoformat(health['timestamp'])
        return (datetime.now() - ts).total_seconds() / 3600.0
    except (KeyError, ValueError, TypeError):
        return 999.0


VERDICT_DISPLAY = {
    'CONVERGING': ('↓CONV',   'green'),
    'ALMOST':     ('↓ALMOST', 'green'),
    'SLOW':       ('→SLOW',   'yellow'),
    'STUCK':      ('✗STUCK',  'red'),
    'EARLY':      ('…EARLY',  'gray'),
}


def _format_health_tag(health):
    # type: (Optional[Dict[str, Any]]) -> str
    """Short health indicator for the status table."""
    if health is None:
        return ''
    verdict = health.get('verdict', '')
    display, color = VERDICT_DISPLAY.get(verdict, ('?', 'gray'))
    age = _health_age_hours(health)
    stale = '?' if age > HEALTH_CACHE_MAX_AGE_H else ''
    r1 = health.get('rminus1')
    if r1 is not None:
        return _c('R-1={:.2f} {}{}'.format(r1, display, stale), color)
    return _c('{}{}'.format(display, stale), color)


def _print_health_report(run_name, health):
    # type: (str, Dict[str, Any]) -> None
    """Pretty-print full health report for one run."""
    print()
    print(_c("  HEALTH — {}".format(run_name), 'bold'))
    print("  " + "─" * 65)

    verdict = health.get('verdict', '?')
    v_display, v_color = VERDICT_DISPLAY.get(verdict, ('?', 'gray'))

    # Summary line
    n_samp = health.get('n_samples', 0)
    acc = health.get('acceptance')
    chains = health.get('chain_lengths', [])
    n_chains = len(chains)
    acc_str = "{:.2f}".format(acc) if acc is not None else "?"
    print("  Samples: {:,} (post-burn-in)    Acceptance: {}    Chains: {}/{}".format(
        n_samp, acc_str, n_chains, EXPECTED_CHAIN_COUNT))

    # Chain length consistency
    if chains and len(chains) > 1:
        min_c, max_c = min(chains), max(chains)
        if max_c > 0 and (max_c - min_c) / max_c > 0.3:
            print(_c("  ⚠  Chain lengths vary widely: {} — some ranks may have crashed".format(
                chains), 'yellow'))

    # Trajectory
    traj = health.get('trajectory', [])
    if traj:
        print()
        print(_c("  R-1 trajectory:", 'bold'))
        max_r1 = max(r for _, r in traj) if traj else 1.0
        bar_max = 28
        for frac, r1 in traj:
            bar_len = int(bar_max * r1 / max(max_r1, 0.001))
            bar_len = max(bar_len, 1)
            bar = '█' * bar_len
            marker = '  ← current' if frac >= 0.99 else ''
            print("   {:>4.0f}%  {:<{w}s}  {:.3f}{}".format(
                frac * 100, bar, r1, marker, w=bar_max))

    # Per-parameter R-1
    per_param = health.get('per_param', {})
    if per_param:
        bottleneck = health.get('bottleneck', '')
        print()
        print(_c("  Per-parameter R-1:", 'bold'))
        sorted_params = sorted(per_param.items(), key=lambda x: -x[1])
        for name, val in sorted_params:
            marker = _c(" ◄ bottleneck", 'yellow') if name == bottleneck else ""
            print("    {:<20s}  {:.4f}{}".format(name, val, marker))

    # N_eff
    neff = health.get('n_eff')
    if neff is not None and neff > 0:
        print()
        print("  N_eff (Kish ESS): {:,}".format(neff))

    # Verdict
    print()
    verdict_messages = {
        'CONVERGING': 'R-1 dropping steadily. Be patient.',
        'ALMOST':     'Nearly converged — just needs a bit more time.',
        'SLOW':       'R-1 decreasing but slowly. Acceptance may be low — proposal could be poor.',
        'STUCK':      'R-1 is flat or rising. Consider restart with a fresh covmat (reset command).',
        'EARLY':      'Not enough post-burn-in samples to assess convergence yet.',
    }
    msg = verdict_messages.get(verdict, '')
    print("  Verdict: {} — {}".format(_c(v_display, v_color), msg))
    print()


def cmd_health(args, all_runs, state):
    # type: (List[str], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]) -> None
    """Compute and display chain-based convergence health diagnostics.

    Usage:
      health <run_name>            — one specific run
      health <N> running           — N random RUNNING runs
      health all running           — all RUNNING runs
    """
    statuses = get_all_statuses(all_runs, state, log_transitions=False)

    # Parse args
    if len(args) == 1:
        # Single run name
        run_name = args[0]
        if run_name not in all_runs:
            print(_c("  Unknown run: {}".format(run_name), 'red'))
            return
        targets = [run_name]
    elif len(args) == 2:
        # N <state> or "all" <state>
        state_filter = args[1].lower()
        target_status = _TAIL_STATE_MAP.get(state_filter)
        if target_status is None:
            print(_c("  Unknown state filter: {}".format(args[1]), 'red'))
            return
        pool = [rn for rn in all_runs if statuses[rn]['status'] == target_status]
        if not pool:
            print("  No {} runs.".format(target_status))
            return
        if args[0].lower() == 'all':
            targets = pool
        else:
            try:
                N = int(args[0])
            except ValueError:
                print(_c("  First argument must be a run name, a number, or 'all'.", 'red'))
                return
            targets = random.sample(pool, min(N, len(pool)))
    else:
        print("  Usage: python run_manager.py health <run_name>")
        print("         python run_manager.py health <N|all> <running|failed|stalled|...>")
        return

    print()
    print(_c("  Computing health for {} run(s)...".format(len(targets)), 'bold'))

    for run_name in targets:
        cfg = all_runs[run_name]
        folder = os.path.join(RUNS_ROOT, cfg['folder_path'])
        n_chains = _count_chain_files(folder, run_name)

        if n_chains == 0:
            print()
            print(_c("  HEALTH — {}".format(run_name), 'bold'))
            print(_c("  No chain files found.", 'yellow'))
            continue

        health = _compute_health(folder, run_name)
        if health is None:
            print()
            print(_c("  HEALTH — {}".format(run_name), 'bold'))
            print(_c("  Could not compute health (chains too short or missing .paramnames).", 'yellow'))
            continue

        _write_health_cache(folder, health)
        _print_health_report(run_name, health)


# ============================================================================
#  ENTRY POINT
# ============================================================================

USAGE = """\
Usage:
  python run_manager.py status
  python run_manager.py test
  python run_manager.py launch <N>
  python run_manager.py auto <N> [--poll-seconds S]
  # to activate auto even when logged out do 
  
  tmux new -s campaign
  python run_manager.py auto 8
  # Ctrl+B then D to detach; close JupyterLab freely
  # later: tmux attach -t campaign

  # In the terminal
  
  python run_manager.py throttle <N>             # change auto's N live
  python run_manager.py pause                  # daemon: stop new submissions
  python run_manager.py pause <run_name>       # scancel + mark one run PAUSED
  python run_manager.py pause --all            # daemon + scancel all active jobs
  python run_manager.py resume                   # un-pause daemon (paused runs stay paused; use resubmit)
  python run_manager.py drain                    # scancel active jobs (back to PENDING)
  python run_manager.py submit <run_name>        # sbatch a single run (preserves chains)
  python run_manager.py resubmit <run_name>      # alias of submit, intended for STALLED runs
  python run_manager.py resubmit --all-stalled   # batch-resume all STALLED runs
  python run_manager.py resubmit --all-paused    # bring all PAUSED runs back (cobaya --resume)
  python run_manager.py restart <run_name>       # WIPE chains + fresh submit
  python run_manager.py reset <run_name>         # dump covmat from chains + wipe outputs + fresh start
  python run_manager.py reset-failed             # wipe chains+state of all FAILED runs → PENDING
  python run_manager.py health <run_name>        # chain-based convergence diagnostics (bypasses .progress)
  python run_manager.py health <N|all> <state>   # health for N random (or all) runs in given state
  python run_manager.py tail <N|all> <state>     # last 100 .log lines of N runs
                                                 # state: running|queued|failed|converged|stalled|pending|zombie|paused
                                                 # converged also offers a getdist summary
  python run_manager.py storage                  # disk usage by status + projection
  python run_manager.py storage --top            # also list top-10 biggest runs
  python run_manager.py storage --all            # also list every run by size
  python run_manager.py follow <run_name>        # tail -f the run's .log
  python run_manager.py doctor <run_name>        # diagnose a failed run
"""


def main():
    if len(sys.argv) < 2:
        print(USAGE)
        sys.exit(1)

    if not os.path.isdir(RUNS_ROOT):
        print(_c("  [ERROR] {} does not exist. Run generate_all_runs.py first.".format(RUNS_ROOT), 'red'))
        sys.exit(1)

    all_runs = build_all_runs()
    if len(all_runs) != 112:
        print(_c("  [ERROR] build_all_runs() returned {} runs (expected 112).".format(len(all_runs)), 'red'))
        sys.exit(1)

    state = load_state()
    cmd = sys.argv[1]

    if cmd == 'status':
        cmd_status(all_runs, state)
    elif cmd == 'test':
        cmd_test(all_runs, state)
    elif cmd == 'launch':
        if len(sys.argv) < 3:
            print("  Usage: python run_manager.py launch <N>")
            sys.exit(1)
        try:
            N = int(sys.argv[2])
        except ValueError:
            print(_c("  N must be an integer.", 'red'))
            sys.exit(1)
        cmd_launch(N, all_runs, state)
        
    elif cmd == 'auto':
        if len(sys.argv) < 3:
            print("  Usage: python run_manager.py auto <N> [--poll-seconds S]")
            sys.exit(1)
        try:
            N = int(sys.argv[2])
        except ValueError:
            print(_c("  N must be an integer.", 'red'))
            sys.exit(1)
        # Optional --poll-seconds
        poll_seconds = POLL_SECONDS_AUTO
        if '--poll-seconds' in sys.argv:
            idx = sys.argv.index('--poll-seconds')
            try:
                poll_seconds = int(sys.argv[idx + 1])
                if poll_seconds < 10:
                    print(_c("  --poll-seconds must be >= 10.", 'red'))
                    sys.exit(1)
            except (IndexError, ValueError):
                print(_c("  --poll-seconds requires an integer argument.", 'red'))
                sys.exit(1)
        cmd_auto(N, all_runs, state, poll_seconds=poll_seconds)
    elif cmd == 'throttle':
        if len(sys.argv) < 3:
            print("  Usage: python run_manager.py throttle <N>")
            sys.exit(1)
        try:
            N = int(sys.argv[2])
        except ValueError:
            print(_c("  N must be an integer.", 'red'))
            sys.exit(1)
        cmd_throttle(N)
        
    elif cmd == 'pause':
        if len(sys.argv) >= 3:
            arg = sys.argv[2]
            if arg == '--all':
                cmd_pause_all(all_runs, state)
            else:
                cmd_pause_one(arg, all_runs, state)
        else:
            cmd_pause()
    elif cmd == 'resume':
        cmd_resume()
        
    elif cmd == 'reset-failed':
        cmd_reset_failed(all_runs, state)
        
    elif cmd == 'drain':
        cmd_drain(all_runs, state)
    elif cmd == 'submit':
        if len(sys.argv) < 3:
            print("  Usage: python run_manager.py submit <run_name>")
            sys.exit(1)
        cmd_submit(sys.argv[2], all_runs, state)
    elif cmd == 'resubmit':
        if len(sys.argv) < 3:
            print("  Usage: python run_manager.py resubmit <run_name|--all-stalled|--all-paused>")
            sys.exit(1)
            
        if sys.argv[2] == '--all-paused':
            cmd_resubmit_all_paused(all_runs, state)
        else:
            cmd_resubmit(sys.argv[2], all_runs, state)
            
            
    elif cmd == 'restart':
        if len(sys.argv) < 3:
            print("  Usage: python run_manager.py restart <run_name>")
            sys.exit(1)
        cmd_restart(sys.argv[2], all_runs, state)

    elif cmd == 'reset':
        if len(sys.argv) < 3:
            print("  Usage: python run_manager.py reset <run_name>")
            sys.exit(1)
        cmd_reset(sys.argv[2], all_runs, state)

    elif cmd == 'health':
        if len(sys.argv) < 3:
            print("  Usage: python run_manager.py health <run_name>")
            print("         python run_manager.py health <N|all> <running|failed|stalled|...>")
            sys.exit(1)
        cmd_health(sys.argv[2:], all_runs, state)

    elif cmd == 'tail':
        if len(sys.argv) < 4:
            print("  Usage: python run_manager.py tail <N|all> <running|queued|failed|converged|stalled|pending>")
            sys.exit(1)
        cmd_tail(sys.argv[2], sys.argv[3], all_runs, state)
        
    elif cmd == 'doctor':
        if len(sys.argv) < 3:
            print("  Usage: python run_manager.py doctor <run_name>")
            sys.exit(1)
        cmd_doctor(sys.argv[2], all_runs, state)
    elif cmd == 'storage':
        sub = sys.argv[2] if len(sys.argv) >= 3 else None
        cmd_storage(sub, all_runs, state)
    elif cmd == 'follow':
        if len(sys.argv) < 3:
            print("  Usage: python run_manager.py follow <run_name>")
            sys.exit(1)
        cmd_follow(sys.argv[2], all_runs, state)
    else:
        print(_c("  Unknown command: {}".format(cmd), 'red'))
        print(USAGE)
        sys.exit(1)


if __name__ == '__main__':
    main()