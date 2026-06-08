#!/usr/bin/env python3
# ============================================================================
# apply_covmats.py
#
# Campaign-wide covmat manager for the exotic-DE MCMC project.
#
# Modes:
#   plan                      Planning report: per-receiver best current plan,
#                             ideal-donor lookup, wait-vs-proceed verdict.
#   apply                     Commit best plans for chain-less PENDING runs.
#   apply --skip-wait         Commit only "proceed" verdicts; leave "wait"
#                             receivers PENDING until upstream catches up.
#   apply-running             Also rewrite YAMLs of stuck-with-chains runs
#                             (R-1 above --stuck-r1 threshold). User must
#                             follow with `run_manager.py restart <name>`.
#   revert <run_name>         Reset one receiver's YAML to `covmat: auto`.
#   revert --all              Reset all 108 UVLF receivers' YAMLs.
#
# Architecture (in order of preference per receiver):
#   1. Best production donor (single-donor fast path, coverage >= 90%, score >= 20)
#   2. Block-diagonal splice from multiple production donors
#   3. Builder fallback (V1's `builder_<model>_<shmr>` covmat)
#   4. `covmat: auto`
#
# Donor pool = CONVERGED runs + near-converged runs (health-cache
# R-1_means < DONOR_R1M_CUTOFF AND R-1_CL < DONOR_R1CL_CUTOFF).
#
# Idempotent: each invocation recomputes the best source per receiver.
# Upgrades (builder -> production donor, or weaker -> better production donor)
# happen automatically as new donors qualify.
#
# Safety: chain-less PENDING receivers are modified freely. RUNNING-and-mixing
# receivers are never touched. RUNNING-but-stuck receivers are flagged in
# `apply-running` mode and require an explicit `run_manager restart` to take
# effect (rewriting the YAML alone isn't enough — cobaya has already loaded
# the old one).
#
# Python 3.8 compatible. Depends on numpy (for covmat algebra) and
# run_manager.py being importable from the same directory (for ETA).
# ============================================================================

import os
import re
import sys
import json
import glob
import argparse
import datetime as _dt
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set


# ============================================================================
#  CONSTANTS
# ============================================================================

RUNS_ROOT       = "runs"
BUILDERS_SUBDIR = "builders"
STATE_FILE      = os.path.join(RUNS_ROOT, ".run_manager_state.json")
HEALTH_CACHE    = ".health.json"   # per-run cache file

# Donor eligibility cutoffs (match cobaya's stop criteria exactly)
DONOR_R1M_CUTOFF  = 0.02   # health-cache R-1 (means) ceiling for donor pool
DONOR_R1CL_CUTOFF = 0.20   # health-cache R-1 (CL)    ceiling for donor pool

# Single-donor fast path
SINGLE_DONOR_COVERAGE_MIN = 0.90  # >= 90% of receiver params covered
SINGLE_DONOR_SCORE_MIN    = 20.0  # absolute score floor

# Scoring score-ceiling reference (max possible from score_donor under perfect
# match: +10 model + +5 data + +2 zcut + +5 R-1 + +10 coverage = 32).
# Used by wait_verdict to interpolate the f(s) speedup curve.
SCORE_MAX = 32.0

# Wait-verdict parameters (math justified in module docstring above main).
WAIT_F_MIN          = 0.45   # empirical: best-warm-start runtime / cold runtime
WAIT_DELTA_MIN      = 5.0    # minimum quality gap for "wait" to be worth it
WAIT_ETA_HARD_CAP_H = 168.0  # ideal donor more than a week out -> always proceed

# Stuck-run threshold for `apply-running` mode
STUCK_R1_THRESHOLD = 4.0


# ============================================================================
#  CAMPAIGN GROUND TRUTH — mirrors generate_all_runs.py
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
    """Construct the full 112-run dict combinatorially. Must stay in sync
    with generate_all_runs.py::build_all_runs."""
    runs = {}  # type: Dict[str, Dict[str, Any]]

    # 108 UVLF runs
    for model in MODELS:
        for zcut in ZCUT_OPTIONS:
            for shmr in SHMR_OPTIONS:
                for data in UVLF_DATA_COMBOS:
                    run_name = "{}_{}_{}_{}".format(model, data, shmr, zcut)
                    has_bg  = '_bg' in data
                    has_cmb = '_cmb' in data
                    if data.startswith('ceers'):
                        use_donnan, use_finkelstein = False, True
                    elif data.startswith('primer'):
                        use_donnan, use_finkelstein = True, False
                    else:
                        use_donnan, use_finkelstein = True, True
                    model_dir = 'exotic' if model == 'exo' else 'lcdm'
                    folder = "{}/{}/{}/{}".format(model_dir, zcut, shmr, run_name)
                    runs[run_name] = {
                        'run_name': run_name, 'model': model, 'data': data,
                        'has_uvlf': True, 'has_bg': has_bg, 'has_cmb': has_cmb,
                        'use_donnan_bins': use_donnan,
                        'use_finkelstein_bins': use_finkelstein,
                        'shmr': shmr, 'zcut': zcut,
                        'folder_path': folder,
                    }

    # 4 non-UVLF (bg-only / bg+cmb-only) runs
    for model in MODELS:
        for data in NON_UVLF_DATA_COMBOS:
            run_name = "{}_{}".format(model, data)
            has_cmb = 'cmb' in data
            runs[run_name] = {
                'run_name': run_name, 'model': model, 'data': data,
                'has_uvlf': False, 'has_bg': True, 'has_cmb': has_cmb,
                'use_donnan_bins': False, 'use_finkelstein_bins': False,
                'shmr': None, 'zcut': None,
                'folder_path': "non_uvlf/{}".format(run_name),
            }
    return runs


def params_of_run(cfg):
    # type: (Dict[str, Any]) -> List[str]
    """Return the sampled-parameter list for a run, IN YAML-DECLARATION ORDER.
    Order matters because covmats are written with rows/cols in this order."""
    out = []
    # Exotic block (sampled first in the YAML)
    if cfg['model'] == 'exo':
        out += ['a_samp', 's']
    # Cosmology block
    if cfg['has_cmb']:
        out += ['theta_s_100', 'omega_b', 'omega_cdm',
                'tau_reio', 'logA', 'n_s']
    else:
        out += ['H0', 'omega_b', 'omega_cdm']
    # SHMR block (UVLF runs only)
    if cfg.get('has_uvlf'):
        if cfg['shmr'] == 'vbeta':
            out += ['shmr_beta']
        elif cfg['shmr'] == 'vshmr':
            out += ['shmr_log_Mc', 'shmr_N', 'shmr_beta']
    # CMB nuisance: A_planck is auto-injected by cobaya's planck_2018_*
    # likelihoods at runtime — it is NOT declared in the YAML's params: block
    # (generate_all_runs.py never lists it), but the chain DOES sample it and
    # the donor's .covmat WILL contain its row + its (A_planck, logA) and
    # (A_planck, tau) cross-correlations. Including it here preserves those
    # correlations in the receiver's seeded covmat, avoiding the relearn cost
    # for the (A_planck, logA) degeneracy ridge that otherwise stalls
    # CMB-bearing warm-starts.
    if cfg['has_cmb']:
        out += ['A_planck']
    return out


def parameter_blocks(cfg):
    # type: (Dict[str, Any]) -> "OrderedDict[str, List[str]]"
    """Return blocks as an ordered dict: block_name -> list of param names.
    Block ordering matches params_of_run's interleaving order."""
    from collections import OrderedDict
    blocks = OrderedDict()  # type: ignore
    if cfg['model'] == 'exo':
        blocks['exotic'] = ['a_samp', 's']
    if cfg['has_cmb']:
        blocks['cosmology'] = ['theta_s_100', 'omega_b', 'omega_cdm',
                               'tau_reio', 'logA', 'n_s']
    else:
        blocks['cosmology'] = ['H0', 'omega_b', 'omega_cdm']
    if cfg.get('has_uvlf'):
        if cfg['shmr'] == 'vbeta':
            blocks['shmr'] = ['shmr_beta']
        elif cfg['shmr'] == 'vshmr':
            blocks['shmr'] = ['shmr_log_Mc', 'shmr_N', 'shmr_beta']
    # Nuisance block: A_planck is auto-injected by cobaya's Planck likelihoods
    # at runtime but DOES appear in donor .covmat files. Include it here so
    # the matchmaker preserves its (A_planck, logA) and (A_planck, tau)
    # cross-correlations in the seeded covmat.
    if cfg['has_cmb']:
        blocks['nuisance'] = ['A_planck']
    return blocks


def donor_datasets(cfg):
    # type: (Dict[str, Any]) -> Set[str]
    """Set of dataset tokens a run uses. Used by data-combo scoring."""
    s = set()  # type: Set[str]
    if cfg.get('has_uvlf'):
        if cfg.get('use_donnan_bins'):
            s.add('uvlf_donnan')
        if cfg.get('use_finkelstein_bins'):
            s.add('uvlf_finkelstein')
    if cfg.get('has_bg'):
        s.add('bg')
    if cfg.get('has_cmb'):
        s.add('cmb')
    return s


# ============================================================================
#  COLORS (lifted from V1 — keeps stdout style consistent across scripts)
# ============================================================================

USE_COLOR = sys.stdout.isatty()


def _c(text, color):
    # type: (str, str) -> str
    if not USE_COLOR:
        return text
    codes = {'green': '32', 'yellow': '33', 'red': '31',
             'cyan': '36', 'gray': '90', 'bold': '1', 'magenta': '35'}
    return '\033[{}m{}\033[0m'.format(codes.get(color, '0'), text)


# ============================================================================
#  STATE + HEALTH CACHE I/O
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


def load_health(folder):
    # type: (str) -> Optional[Dict[str, Any]]
    """Read the per-run .health.json cache from run_manager."""
    path = os.path.join(folder, HEALTH_CACHE)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (ValueError, IOError):
        return None


def _count_chain_files(folder, run_name):
    # type: (str, str) -> int
    """Count chain .txt files in outputs/. Mirrors run_manager."""
    out = os.path.join(folder, 'outputs')
    if not os.path.isdir(out):
        return 0
    return sum(
        1 for f in os.listdir(out)
        if f.startswith(run_name + '.') and f.endswith('.txt')
        and f[len(run_name) + 1:-4].isdigit()
    )


# ============================================================================
#  DONOR POOL DISCOVERY
# ============================================================================

def is_donor_eligible(run_name, cfg, state, r1m_cutoff, r1cl_cutoff):
    # type: (str, Dict[str, Any], Dict[str, Any], float, float) -> Tuple[bool, Optional[float], Optional[float]]
    """A run is donor-eligible if status=CONVERGED OR its health-cache R-1
    is below both cutoffs. Returns (eligible, r1m, r1cl). r1m/r1cl come from
    health cache when available, else (None, None) and the CONVERGED branch
    is the only acceptance path."""
    folder = os.path.join(RUNS_ROOT, cfg['folder_path'])
    health = load_health(folder)
    r1m  = health.get('rminus1')    if health else None
    r1cl = health.get('rminus1_cl') if health else None

    # CONVERGED — accept regardless of (possibly stale) health-cache values
    entry = state.get(run_name, {})
    if entry.get('last_known_status') == 'CONVERGED':
        return True, r1m, r1cl

    # Near-converged via health cache
    if r1m is None or r1cl is None:
        return False, r1m, r1cl
    if r1m < r1m_cutoff and r1cl < r1cl_cutoff:
        return True, r1m, r1cl

    return False, r1m, r1cl


def covmat_path_for_run(cfg):
    # type: (Dict[str, Any]) -> str
    folder = os.path.join(RUNS_ROOT, cfg['folder_path'])
    safe_path = os.path.join(folder, 'outputs', cfg['run_name'] + '.covmat')
    if os.path.exists(safe_path):
        return safe_path
    return os.path.join(folder, cfg['run_name'] + '.covmat')


def discover_donor_pool(all_runs, state, r1m_cutoff, r1cl_cutoff):
    # type: (Dict[str, Dict[str, Any]], Dict[str, Any], float, float) -> Dict[str, Dict[str, Any]]
    """Return dict run_name -> donor_info, including only runs that
    (a) are donor-eligible AND (b) have an on-disk covmat to copy."""
    donors = {}  # type: Dict[str, Dict[str, Any]]
    for rn, cfg in all_runs.items():
        eligible, r1m, r1cl = is_donor_eligible(rn, cfg, state,
                                                r1m_cutoff, r1cl_cutoff)
        if not eligible:
            continue
        cm_path = covmat_path_for_run(cfg)
        if not os.path.exists(cm_path):
            continue
        donors[rn] = {
            'cfg':        cfg,
            'params':     params_of_run(cfg),
            'r1m':        r1m if r1m is not None else 0.02,  # converged-ish
            'r1cl':       r1cl if r1cl is not None else 0.20,
            'covmat_path': cm_path,
        }
    return donors


# ============================================================================
#  COVMAT FILE I/O
# ============================================================================

def read_covmat(path):
    # type: (str) -> Tuple[List[str], "np.ndarray"]
    """Parse cobaya's .covmat text format.

    Format:
        # p1 p2 p3 ...
        m11 m12 m13 ...
        m21 m22 m23 ...
        ...

    Returns (param_names, NxN symmetric matrix).
    """
    with open(path) as f:
        first = f.readline().strip()
        if first.startswith('#'):
            names = first[1:].split()
        else:
            # No header — try to recover from .paramnames sibling
            names = []
        rest = f.read()
    if not names:
        raise ValueError(
            "covmat at {} has no header — refusing to guess names".format(path))
    matrix = np.fromstring(rest, sep=' ')
    n = len(names)
    if matrix.size != n * n:
        # Maybe it's whitespace-tabular instead of space-delimited
        matrix = np.loadtxt(path, comments='#')
    matrix = matrix.reshape(n, n)
    # Symmetrize against numerical asymmetry in legacy files
    matrix = 0.5 * (matrix + matrix.T)
    return names, matrix


def write_covmat(path, param_names, matrix):
    # type: (str, List[str], "np.ndarray") -> None
    """Write a covmat in cobaya's format."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('# ' + ' '.join(param_names) + '\n')
        # Match cobaya's precision (it writes %.7e by default)
        for row in matrix:
            f.write(' '.join('{: .8e}'.format(v) for v in row) + '\n')


def extract_submatrix(donor_names, donor_matrix, wanted_names):
    # type: (List[str], "np.ndarray", List[str]) -> Tuple[List[str], "np.ndarray", List[str]]
    """Pull out the rows/cols of donor_matrix that correspond to wanted_names,
    preserving the order in wanted_names. Returns
    (matched_names, submatrix, missing_names_from_donor)."""
    pos = {p: i for i, p in enumerate(donor_names)}
    matched = []  # type: List[str]
    indices = []  # type: List[int]
    missing = []  # type: List[str]
    for p in wanted_names:
        if p in pos:
            matched.append(p)
            indices.append(pos[p])
        else:
            missing.append(p)
    if not indices:
        return [], np.zeros((0, 0)), missing
    sub = donor_matrix[np.ix_(indices, indices)]
    return matched, sub, missing


def is_pos_def(matrix, eps=1e-12):
    # type: ("np.ndarray", float) -> bool
    """Cholesky-based PSD check. eps regularizes tiny negative eigenvalues
    from rounding."""
    if matrix.size == 0:
        return True
    try:
        np.linalg.cholesky(matrix + eps * np.eye(matrix.shape[0]))
        return True
    except np.linalg.LinAlgError:
        return False


# ============================================================================
#  SHMR FILTER + SCORING
# ============================================================================

def shmr_usable(donor_cfg, receiver_cfg):
    # type: (Dict[str, Any], Dict[str, Any]) -> bool
    """Hard filter from the spec:
      - receiver SHMR variant must equal donor's, EXCEPT:
      - fixed (or None) receivers accept any donor (non-SHMR rows extracted),
      - non-UVLF receivers (shmr=None) accept any donor.
    """
    r_shmr = receiver_cfg.get('shmr')
    d_shmr = donor_cfg.get('shmr')
    if r_shmr is None or r_shmr == 'fixed':
        return True
    return d_shmr == r_shmr


def score_donor(donor_cfg, donor_params, donor_r1m, receiver_cfg):
    # type: (Dict[str, Any], List[str], float, Dict[str, Any]) -> Tuple[float, float]
    """Score a donor for a given receiver. Returns (score, coverage).

    Components (from the project spec):
      (a) Model match:        +10 if same, +0 otherwise
      (b) Data-combo match:   +5 same; +3 donor-subset; +0 donor-superset;
                              +1 partial overlap
      (c) Zcut match:         +2 same; +1 full<->restr; +0 donor has no zcut
      (d) R-1 quality:        5 * min(1, 0.02 / max(r1m, 1e-6)), capped at 5
      (e) Parameter coverage: 10 * |donor ∩ receiver| / |receiver|
    """
    score = 0.0

    # (a) Model
    if donor_cfg['model'] == receiver_cfg['model']:
        score += 10.0

    # (b) Data-combo
    d_sets = donor_datasets(donor_cfg)
    r_sets = donor_datasets(receiver_cfg)
    if d_sets == r_sets:
        score += 5.0
    elif d_sets.issubset(r_sets):
        # Donor has FEWER datasets than receiver — its covmat is wider, safe.
        score += 3.0
    elif r_sets.issubset(d_sets):
        # Donor has MORE datasets — covmat too tight, traps receiver chains
        # (covmat-trap pathology in PROJECT_STATUS).
        score += 0.0
    else:
        # Disjoint or partially overlapping
        score += 1.0

    # (c) Zcut
    d_zcut = donor_cfg.get('zcut')
    r_zcut = receiver_cfg.get('zcut')
    if d_zcut == r_zcut and d_zcut is not None:
        score += 2.0
    elif d_zcut is not None and r_zcut is not None and d_zcut != r_zcut:
        score += 1.0  # full<->restr
    # else (one or both is None): +0 — no zcut info to match on

    # (d) R-1 quality (capped at 5)
    score += 5.0 * min(1.0, 0.02 / max(donor_r1m, 1e-6))

    # (e) Coverage
    r_params = set(params_of_run(receiver_cfg))
    d_params = set(donor_params)
    coverage = (len(r_params & d_params) / max(len(r_params), 1))
    score += 10.0 * coverage

    return score, coverage


def score_hypothetical_converged(donor_cfg, receiver_cfg):
    # type: (Dict[str, Any], Dict[str, Any]) -> Tuple[float, float]
    """Score a donor AS IF it were converged at the cobaya-stop R-1. Used to
    compute the 'ideal donor score' for the wait verdict — the score-ceiling
    for the receiver, scoring every campaign run against the receiver as if
    that run had already finished."""
    return score_donor(donor_cfg, params_of_run(donor_cfg),
                       0.02, receiver_cfg)


# ============================================================================
#  MATCHMAKER — single-donor fast path + block-diagonal splice
# ============================================================================

def matchmake(receiver_cfg, donors):
    # type: (Dict[str, Any], Dict[str, Dict[str, Any]]) -> Dict[str, Any]
    """Find the best plan for one receiver from the current donor pool.

    Returns a 'plan dict' with keys:
      kind         : 'single' | 'splice' | 'none'
      donor        : (single)  donor name
      block_donors : (splice)  {block_name: donor_name or None}
      score        : composite quality score for the plan
      coverage     : fraction of receiver's params covered by the plan
      candidates   : top-5 candidates with their scores (for verbose output)
    """
    r_params  = params_of_run(receiver_cfg)
    r_set     = set(r_params)
    r_name    = receiver_cfg['run_name']

    # SHMR-filter + score every donor
    scored = []  # type: List[Tuple[str, Dict[str, Any], float, float]]
    for dn, dinfo in donors.items():
        if dn == r_name:
            continue  # don't self-donate
        if not shmr_usable(dinfo['cfg'], receiver_cfg):
            continue
        s, cov = score_donor(dinfo['cfg'], dinfo['params'],
                             dinfo['r1m'], receiver_cfg)
        scored.append((dn, dinfo, s, cov))

    scored.sort(key=lambda t: -t[2])

    if not scored:
        return {'kind': 'none', 'score': 0.0, 'coverage': 0.0,
                'candidates': []}

    # Single-donor fast path
    top_dn, top_dinfo, top_score, top_cov = scored[0]
    if top_cov >= SINGLE_DONOR_COVERAGE_MIN and top_score >= SINGLE_DONOR_SCORE_MIN:
        return {
            'kind':       'single',
            'donor':      top_dn,
            'score':      top_score,
            'coverage':   top_cov,
            'candidates': [(dn, s, c) for dn, _, s, c in scored[:5]],
        }

    # Splicing — per block, pick best donor whose params cover that block
    # (or, for the cosmology block on a CMB<->non-CMB mismatch, the donor
    # that covers the overlap subset).
    blocks = parameter_blocks(receiver_cfg)
    block_donors = {}  # type: Dict[str, Optional[str]]
    block_coverage = {}  # type: Dict[str, float]

    for block_name, block_params in blocks.items():
        block_set = set(block_params)
        # Tier 1: donors that cover the ENTIRE block
        full_cov = [(dn, s) for dn, di, s, _ in scored
                    if block_set.issubset(set(di['params']))]
        if full_cov:
            full_cov.sort(key=lambda t: -t[1])
            block_donors[block_name] = full_cov[0][0]
            block_coverage[block_name] = 1.0
            continue
        # Tier 2: best PARTIAL coverage (e.g. cross-CMB cosmology overlap
        # has only {omega_b, omega_cdm})
        best_dn, best_score, best_cov = None, -1.0, 0
        for dn, di, s, _ in scored:
            inter = len(block_set & set(di['params']))
            if inter == 0:
                continue
            if inter > best_cov or (inter == best_cov and s > best_score):
                best_dn, best_score, best_cov = dn, s, inter
        block_donors[block_name] = best_dn
        block_coverage[block_name] = best_cov / len(block_params) if best_dn else 0.0

    # Composite score for splice = receiver-weighted average of contributing
    # donors' scores, with uncovered blocks contributing zero score.
    total_weight = float(len(r_params))
    splice_score = 0.0
    splice_cov_num = 0
    for block_name, block_params in blocks.items():
        dn = block_donors.get(block_name)
        if dn is None:
            continue
        # Donor's match-score (look it up from `scored`)
        ds = next(s for d2, _, s, _ in scored if d2 == dn)
        # How many of THIS block's params the donor actually covers
        dset = set(donors[dn]['params'])
        covered = len(set(block_params) & dset)
        weight = covered / total_weight
        splice_score += ds * weight
        splice_cov_num += covered

    splice_coverage = splice_cov_num / total_weight

    if splice_score <= 0.0:
        return {'kind': 'none', 'score': 0.0, 'coverage': 0.0,
                'candidates': [(dn, s, c) for dn, _, s, c in scored[:5]]}

    # Cross-block correlation donors: for each pair of blocks the receiver has,
    # find the best-scoring donor whose chain covers params from BOTH blocks.
    # The splice assembly will transfer that donor's correlation structure
    # (unitless rho values) into the splice covmat, rescaled to match our
    # chosen per-block diagonal scales. Net effect: receiver chain starts
    # with realistic cross-block degeneracy orientations baked in, instead of
    # spending thousands of samples relearning them via learn_proposal.
    cross_donors = {}  # type: Dict[Tuple[str, str], str]
    block_names = list(blocks.keys())
    for i in range(len(block_names)):
        for j in range(i + 1, len(block_names)):
            b1, b2 = block_names[i], block_names[j]
            cross_dn = _find_cross_donor(blocks[b1], blocks[b2], scored, donors)
            if cross_dn is not None:
                cross_donors[(b1, b2)] = cross_dn

    return {
        'kind':           'splice',
        'block_donors':   block_donors,
        'block_coverage': block_coverage,
        'cross_donors':   cross_donors,
        'score':          splice_score,
        'coverage':       splice_coverage,
        'candidates':     [(dn, s, c) for dn, _, s, c in scored[:5]],
    }




def _extract_cross_submatrix(donor_names, donor_matrix, row_params, col_params):
    # type: (List[str], "np.ndarray", List[str], List[str]) -> Tuple[List[str], List[str], "np.ndarray"]
    """Extract a possibly-non-square submatrix where rows are row_params and
    columns are col_params, from a donor's covmat. Missing params on either
    axis are silently dropped (matched lists carry the resolved labels)."""
    pos = {p: i for i, p in enumerate(donor_names)}
    row_matched, row_idx = [], []
    for p in row_params:
        if p in pos:
            row_matched.append(p)
            row_idx.append(pos[p])
    col_matched, col_idx = [], []
    for p in col_params:
        if p in pos:
            col_matched.append(p)
            col_idx.append(pos[p])
    if not row_idx or not col_idx:
        return [], [], np.zeros((0, 0))
    sub = donor_matrix[np.ix_(row_idx, col_idx)]
    return row_matched, col_matched, sub


def _find_cross_donor(block_a_params, block_b_params, scored, donors):
    # type: (List[str], List[str], List[Any], Dict[str, Any]) -> Optional[str]
    """Best-scoring donor whose chain covers params from BOTH block_a and
    block_b — i.e., one whose joint posterior carries genuine cross-block
    correlations. Returns donor name or None.

    Selection criterion: maximum combined coverage of params from both blocks
    (so donors with 5/5 in B1 + 3/3 in B2 beat donors with 5/5 + 1/3),
    breaking ties by match-score."""
    a_set, b_set = set(block_a_params), set(block_b_params)
    best_dn, best_score, best_coverage = None, -1.0, -1
    for dn, _di, s, _cov in scored:
        dset = set(donors[dn]['params'])
        cov_a = len(a_set & dset)
        cov_b = len(b_set & dset)
        if cov_a == 0 or cov_b == 0:
            continue  # cross-correlations need overlap on BOTH sides
        total = cov_a + cov_b
        if total > best_coverage or (total == best_coverage and s > best_score):
            best_dn, best_score, best_coverage = dn, s, total
    return best_dn



# ============================================================================
#  BUILDER FALLBACK (V1 logic preserved)
# ============================================================================

def builder_name_for(cfg):
    # type: (Dict[str, Any]) -> Optional[str]
    """Map (model, shmr) -> builder name, matching V1's covmat_for_run."""
    if not cfg.get('has_uvlf') or cfg.get('shmr') is None:
        return None
    if cfg['model'] == 'exo':
        return "builder_exo_{}".format(cfg['shmr'])
    # lcdm
    if cfg['shmr'] in ('fixed',):
        return None
    return "builder_lcdm_{}".format(cfg['shmr'])


def builder_covmat_path(builder):
    # type: (str) -> str
    folder = os.path.join(RUNS_ROOT, BUILDERS_SUBDIR, builder)
    safe_path = os.path.abspath(os.path.join(folder, builder + '.covmat'))
    if os.path.exists(safe_path):
        return safe_path
    return os.path.abspath(os.path.join(folder, 'outputs', builder + '.covmat'))


def score_builder_for(receiver_cfg):
    # type: (Dict[str, Any]) -> Tuple[Optional[str], float, float]
    """Return (builder_name, score, coverage) of the builder-fallback plan.
    Builders sample exotic+shmr at fixed Planck18 cosmology. Score them as
    'subset donors' with R-1 ~= 0.025 (typical builder final R-1)."""
    bn = builder_name_for(receiver_cfg)
    if bn is None:
        return None, 0.0, 0.0
    bpath = builder_covmat_path(bn)
    if not os.path.exists(bpath):
        return bn, 0.0, 0.0
    # Build a virtual builder cfg
    builder_cfg = {
        'model':    receiver_cfg['model'],   # +10 model match
        'data':     'uvlf',                  # builder uses UVLF-only
        'has_uvlf': True, 'has_bg': False, 'has_cmb': False,
        'use_donnan_bins': True, 'use_finkelstein_bins': True,
        'shmr':     receiver_cfg['shmr'],    # SHMR variant matches
        'zcut':     None,                    # no zcut
    }
    # Builder params = exotic + shmr only (no cosmology)
    builder_params = []  # type: List[str]
    if builder_cfg['model'] == 'exo':
        builder_params += ['a_samp', 's']
    if builder_cfg['shmr'] == 'vbeta':
        builder_params += ['shmr_beta']
    elif builder_cfg['shmr'] == 'vshmr':
        builder_params += ['shmr_log_Mc', 'shmr_N', 'shmr_beta']
    s, cov = score_donor(builder_cfg, builder_params, 0.025, receiver_cfg)
    return bn, s, cov


# ============================================================================
#  WAIT VERDICT — proceed-now vs wait-for-ideal-donor
# ============================================================================

def find_ideal_donor(receiver_cfg, all_runs):
    # type: (Dict[str, Any], Dict[str, Dict[str, Any]]) -> Tuple[Optional[str], float]
    """Score every campaign run AGAINST the receiver as if that run were
    converged at R-1=0.02. Return (best_run_name, hypothetical_score).
    Excludes the receiver itself."""
    r_name = receiver_cfg['run_name']
    best_dn, best_score = None, -1.0
    for dn, dcfg in all_runs.items():
        if dn == r_name:
            continue
        if not shmr_usable(dcfg, receiver_cfg):
            continue
        s, _ = score_hypothetical_converged(dcfg, receiver_cfg)
        if s > best_score:
            best_dn, best_score = dn, s
    return best_dn, best_score


def _try_import_run_manager():
    """Import run_manager's ETA helpers lazily — they're heavyweight."""
    try:
        sys.path.insert(0, RUNS_ROOT)
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import run_manager   # type: ignore
        return run_manager
    except Exception:
        return None


def ideal_donor_eta_hours(ideal_run_name, all_runs, state):
    # type: (str, Dict[str, Dict[str, Any]], Dict[str, Any]) -> Optional[float]
    """ETA in hours for the ideal donor to reach R-1 = 0.02. None if the
    ideal donor isn't running or its .progress fit fails."""
    cfg = all_runs[ideal_run_name]
    folder = os.path.join(RUNS_ROOT, cfg['folder_path'])
    progress = os.path.join(folder, 'outputs', ideal_run_name + '.progress')
    if not os.path.exists(progress):
        return None
    # Don't run the ETA pipeline on already-converged donors
    if state.get(ideal_run_name, {}).get('last_known_status') == 'CONVERGED':
        return 0.0

    rm = _try_import_run_manager()
    if rm is None:
        return None
    try:
        eta = rm._compute_eta_from_progress(progress, for_calibration=False)
    except Exception:
        return None
    if eta.get('means_status') == 'ok':
        return eta.get('means_eta_h')
    if eta.get('means_status') == 'already_below_threshold':
        return 0.0
    return None


def wait_verdict(s_cur, s_ideal, eta_h, t_cold_h,
                 f_min=WAIT_F_MIN, delta_min=WAIT_DELTA_MIN,
                 s_max=SCORE_MAX, eta_cap=WAIT_ETA_HARD_CAP_H):
    # type: (float, float, Optional[float], Optional[float], float, float, float, float) -> Dict[str, Any]
    """Decide whether to wait for the ideal donor or proceed now.

    Math (justified in module docstring):
        f(s) = 1 - (1 - f_min) * s / s_max
        savings_h = t_cold_h * (f(s_cur) - f(s_ideal))
        wait <==> eta_h < savings_h AND (s_ideal - s_cur) >= delta_min
                  AND eta_h <= eta_cap

    Returns dict with: verdict ('proceed'|'wait'|'wait-but-unknown'),
                       savings_h, gap, reasons.
    """
    out = {'verdict': 'proceed', 'savings_h': None, 'gap': s_ideal - s_cur,
           'reasons': []}

    gap = s_ideal - s_cur
    if gap < delta_min:
        out['reasons'].append("gap {:.1f} < {:.1f}".format(gap, delta_min))
        return out

    if eta_h is None:
        out['reasons'].append("ideal donor has no ETA (not running or fit failed)")
        return out

    if eta_h > eta_cap:
        out['reasons'].append("ETA {:.0f}h > hard cap {:.0f}h".format(eta_h, eta_cap))
        return out

    if t_cold_h is None or t_cold_h <= 0:
        # No calibration — fall back to gap-only heuristic
        out['reasons'].append("no cold-runtime calibration; using gap-only heuristic")
        if gap >= 2 * delta_min and eta_h < 48.0:
            out['verdict'] = 'wait'
        return out

    f_cur   = 1.0 - (1.0 - f_min) * s_cur   / s_max
    f_ideal = 1.0 - (1.0 - f_min) * s_ideal / s_max
    savings = max(0.0, t_cold_h * (f_cur - f_ideal))
    out['savings_h'] = savings

    if eta_h < savings:
        out['verdict'] = 'wait'
        out['reasons'].append(
            "ETA {:.1f}h < savings {:.1f}h (gap {:.1f}, t_cold {:.0f}h)".format(
                eta_h, savings, gap, t_cold_h))
    else:
        out['reasons'].append(
            "ETA {:.1f}h >= savings {:.1f}h".format(eta_h, savings))
    return out


def estimate_t_cold_for(receiver_cfg, all_runs, state):
    # type: (Dict[str, Any], Dict[str, Dict[str, Any]], Dict[str, Any]) -> Optional[float]
    """Estimate cold-start wall-clock to convergence by taking the median
    elapsed-time of CONVERGED runs in the same 'class' (same model, same
    has_cmb, same shmr). Returns None if no comparable converged runs."""
    candidates = []
    for rn, cfg in all_runs.items():
        if state.get(rn, {}).get('last_known_status') != 'CONVERGED':
            continue
        if cfg['model'] != receiver_cfg['model']:
            continue
        if cfg['has_cmb'] != receiver_cfg['has_cmb']:
            continue
        if cfg.get('shmr') != receiver_cfg.get('shmr'):
            continue
        # Take the run's converged elapsed time from state if recorded
        entry = state.get(rn, {})
        elapsed = entry.get('converged_elapsed_h')
        if elapsed is None:
            # Fall back to .progress last timestamp - first timestamp
            folder = os.path.join(RUNS_ROOT, cfg['folder_path'])
            progress = os.path.join(folder, 'outputs', rn + '.progress')
            elapsed = _progress_wallclock_hours(progress)
        if elapsed is not None and elapsed > 0:
            candidates.append(elapsed)
    if not candidates:
        # Widen: drop shmr matching
        for rn, cfg in all_runs.items():
            if state.get(rn, {}).get('last_known_status') != 'CONVERGED':
                continue
            if cfg['model'] != receiver_cfg['model']:
                continue
            if cfg['has_cmb'] != receiver_cfg['has_cmb']:
                continue
            folder = os.path.join(RUNS_ROOT, cfg['folder_path'])
            progress = os.path.join(folder, 'outputs', rn + '.progress')
            elapsed = _progress_wallclock_hours(progress)
            if elapsed is not None and elapsed > 0:
                candidates.append(elapsed)
    if not candidates:
        return None
    candidates.sort()
    return candidates[len(candidates) // 2]


def _progress_wallclock_hours(progress_path):
    # type: (str) -> Optional[float]
    """First-to-last timestamp delta in hours from a .progress file."""
    if not os.path.exists(progress_path):
        return None
    try:
        with open(progress_path) as f:
            ts = []
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) < 4:
                    continue
                try:
                    t = _dt.datetime.fromisoformat(parts[1])
                    ts.append(t)
                except (ValueError, IndexError):
                    continue
    except (IOError, OSError):
        return None
    if len(ts) < 2:
        return None
    return (ts[-1] - ts[0]).total_seconds() / 3600.0


# ============================================================================
#  PLAN COMPILER — combine matchmaker + builder + ideal-donor + verdict
# ============================================================================

def compile_plan(receiver_cfg, donors, all_runs, state):
    # type: (Dict[str, Any], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Any]) -> Dict[str, Any]
    """End-to-end plan for one receiver. Combines:
      - matchmaker on production donors
      - builder fallback comparison
      - ideal-donor lookup + wait verdict
    Returns the unified plan dict the rest of the script consumes."""
    plan = matchmake(receiver_cfg, donors)
    bn, b_score, b_cov = score_builder_for(receiver_cfg)
    plan['builder']          = bn
    plan['builder_score']    = b_score
    plan['builder_coverage'] = b_cov

    # If matchmaker has no plan, builder takes over (if available).
    # If matchmaker has a plan, builder competes on score.
    if plan['kind'] == 'none':
        if bn is not None and b_score > 0:
            plan['kind']     = 'builder'
            plan['score']    = b_score
            plan['coverage'] = b_cov
        else:
            plan['kind']     = 'auto'
            plan['score']    = 0.0
            plan['coverage'] = 0.0
    else:
        if bn is not None and b_score > plan['score']:
            # Builder beats the spliced plan — use builder
            plan['kind']     = 'builder'
            plan['score']    = b_score
            plan['coverage'] = b_cov

    # Ideal-donor lookup + wait verdict
    ideal_dn, ideal_score = find_ideal_donor(receiver_cfg, all_runs)
    plan['ideal_donor']       = ideal_dn
    plan['ideal_donor_score'] = ideal_score
    plan['ideal_donor_status'] = state.get(ideal_dn, {}).get(
        'last_known_status', 'PENDING') if ideal_dn else None

    eta_h = (ideal_donor_eta_hours(ideal_dn, all_runs, state)
             if ideal_dn else None)
    t_cold = estimate_t_cold_for(receiver_cfg, all_runs, state)
    verdict = wait_verdict(plan['score'], ideal_score, eta_h, t_cold)
    plan['ideal_donor_eta_h'] = eta_h
    plan['t_cold_h']          = t_cold
    plan['wait_verdict']      = verdict

    return plan


# ============================================================================
#  COVMAT WRITE-OUT — execute a plan and produce receiver's covmat file
# ============================================================================

def output_covmat_path_for(receiver_cfg):
    # type: (Dict[str, Any]) -> str
    """Writes the new spliced/copied covmat safely OUTSIDE the outputs/ folder. to prevent cobaya deleting it when launched"""
    return os.path.abspath(os.path.join(
        RUNS_ROOT, receiver_cfg['folder_path'],
        receiver_cfg['run_name'] + '.covmat'))


def execute_plan(receiver_cfg, plan, donors, verbose=False):
    # type: (Dict[str, Any], Dict[str, Any], Dict[str, Dict[str, Any]], bool) -> Tuple[Optional[str], List[str]]
    """Materialize a plan to disk. Returns (covmat_yaml_value, log_lines).

    For kind=='single':  copy donor's covmat (extract submatrix matching
                          receiver's params).
    For kind=='splice':  build block-diagonal covmat from per-block donors.
    For kind=='builder': point to the existing builder covmat file.
    For kind=='auto':    return 'auto' (no file written).

    The YAML's `covmat:` line will be set to the returned string.
    """
    log = []  # type: List[str]
    r_params = params_of_run(receiver_cfg)
    out_path = output_covmat_path_for(receiver_cfg)

    if plan['kind'] == 'auto':
        log.append("auto: no donor available; cobaya will diagonal-init")
        return 'auto', log

    if plan['kind'] == 'builder':
        # V1 behavior: point YAML at the builder covmat without rewriting it.
        bn = plan['builder']
        bpath = builder_covmat_path(bn)
        if not os.path.exists(bpath):
            log.append("builder covmat missing at {} — falling back to auto".format(bpath))
            return 'auto', log
        log.append("builder: pointing YAML at {}".format(bpath))
        return bpath, log

    if plan['kind'] == 'single':
        dn = plan['donor']
        dpath = donors[dn]['covmat_path']
        dnames, dmat = read_covmat(dpath)
        matched, sub, missing = extract_submatrix(dnames, dmat, r_params)
        if not matched:
            log.append("single: donor {} provided no overlapping params".format(dn))
            return 'auto', log
        if not is_pos_def(sub):
            log.append("single: extracted submatrix from {} not PSD; will fall to splice/builder/auto".format(dn))
            # Caller should detect this and re-run with that donor removed.
            return None, log
        write_covmat(out_path, matched, sub)
        if missing and verbose:
            log.append("single: donor {} missing {} -> proposal_scale defaults".format(
                dn, missing))
        log.append("single: donor {} -> wrote {} params to {}".format(
            dn, len(matched), out_path))
        return out_path, log

    if plan['kind'] == 'splice':
        block_donors = plan['block_donors']
        cross_donors = plan.get('cross_donors', {})  # backward-compat for old plans
        blocks       = parameter_blocks(receiver_cfg)
        N = len(r_params)
        idx_of = {p: i for i, p in enumerate(r_params)}
        full = np.zeros((N, N))
        placed = []  # type: List[int]
        placed_by_block = {}  # type: Dict[str, List[str]]

        # ── Phase 1: per-block diagonals (current behaviour) ──────────────────
        for block_name, block_params in blocks.items():
            dn = block_donors.get(block_name)
            if dn is None:
                continue
            dnames, dmat = read_covmat(donors[dn]['covmat_path'])
            matched, sub, missing = extract_submatrix(dnames, dmat, block_params)
            if not matched:
                continue
            if not is_pos_def(sub):
                log.append("splice: block {} from {} not PSD; skipped".format(
                    block_name, dn))
                continue
            for i, p_i in enumerate(matched):
                for j, p_j in enumerate(matched):
                    full[idx_of[p_i], idx_of[p_j]] = sub[i, j]
                placed.append(idx_of[p_i])
            placed_by_block[block_name] = matched
            log.append("splice[{}]: from {} (+{} params{})".format(
                block_name, dn, len(matched),
                "" if not missing else ", missing " + ",".join(missing)))

        # ── Phase 2: cross-block correlations from cross-donors ───────────────
        # For each pair (B1, B2) of blocks that both placed >=1 param, pull the
        # cross-donor's correlation matrix block (unitless rho), then rescale
        # by OUR placed diagonals (sigmas from the per-block donors). This is
        # the rho-transfer construction: shape from cross-donor, magnitude
        # from local per-block donors.
        full_with_cross = full.copy()
        cross_log = []
        for (b1, b2), cross_dn in cross_donors.items():
            placed_b1 = placed_by_block.get(b1, [])
            placed_b2 = placed_by_block.get(b2, [])
            if not placed_b1 or not placed_b2:
                continue  # one of the blocks didn't place anything
            dnames, dmat = read_covmat(donors[cross_dn]['covmat_path'])
            row_matched, col_matched, cross_sub = _extract_cross_submatrix(
                dnames, dmat, placed_b1, placed_b2)
            if not row_matched or not col_matched:
                continue
            # Donor-side standard deviations (for normalization to rho)
            pos = {p: i for i, p in enumerate(dnames)}
            sigma_row_donor = np.array(
                [np.sqrt(dmat[pos[p], pos[p]]) for p in row_matched])
            sigma_col_donor = np.array(
                [np.sqrt(dmat[pos[p], pos[p]]) for p in col_matched])
            if np.any(sigma_row_donor <= 0) or np.any(sigma_col_donor <= 0):
                continue
            rho = cross_sub / np.outer(sigma_row_donor, sigma_col_donor)
            # Our placed diagonals (sigmas from the chosen per-block donors)
            sigma_row_ours = np.array(
                [np.sqrt(full[idx_of[p], idx_of[p]]) for p in row_matched])
            sigma_col_ours = np.array(
                [np.sqrt(full[idx_of[p], idx_of[p]]) for p in col_matched])
            new_cross = np.outer(sigma_row_ours, sigma_col_ours) * rho
            # Place symmetrically
            for ri, p_r in enumerate(row_matched):
                for ci, p_c in enumerate(col_matched):
                    full_with_cross[idx_of[p_r], idx_of[p_c]] = new_cross[ri, ci]
                    full_with_cross[idx_of[p_c], idx_of[p_r]] = new_cross[ri, ci]
            cross_log.append("splice-cross[{},{}]: from {} ({}x{} rho-block)".format(
                b1, b2, cross_dn, len(row_matched), len(col_matched)))

        # ── Phase 3: PSD check, fall back to block-diagonal if cross broke it ─
        # Cross-correlations from one donor combined with diagonals from
        # different donors are not guaranteed to produce a PSD matrix —
        # different posteriors have geometrically incompatible shapes. If the
        # cross-augmented matrix isn't PSD, fall back to the block-diagonal.
        placed = sorted(set(placed))
        if not placed:
            log.append("splice: no blocks placed; falling to auto")
            return 'auto', log
        out_params = [r_params[i] for i in placed]
        out_mat_cross = full_with_cross[np.ix_(placed, placed)]
        out_mat_diag  = full[np.ix_(placed, placed)]

        if is_pos_def(out_mat_cross):
            out_mat = out_mat_cross
            log.extend(cross_log)
            log.append("splice: cross-block correlations applied for {} pair(s)".format(
                len(cross_log)))
        elif is_pos_def(out_mat_diag):
            out_mat = out_mat_diag
            log.append("splice: cross-block correlations broke PSD; "
                       "fell back to block-diagonal")
        else:
            log.append("splice: assembled covmat not PSD even block-diagonal; "
                       "falling to auto")
            return 'auto', log

        write_covmat(out_path, out_params, out_mat)
        log.append("splice: assembled {}/{} param rows -> {}".format(
            len(placed), N, out_path))
        return out_path, log
    

    log.append("unknown plan kind: {}".format(plan['kind']))
    return 'auto', log


# ============================================================================
#  YAML I/O
# ============================================================================

YAML_COVMAT_RE = re.compile(r"^\s*covmat:\s*(.*?)\s*$", re.MULTILINE)


def find_yaml_path(cfg):
    # type: (Dict[str, Any]) -> str
    return os.path.join(RUNS_ROOT, cfg['folder_path'],
                        cfg['run_name'] + '.yaml')


def read_yaml_covmat(path):
    # type: (str) -> Optional[str]
    """Return the current value of the YAML's covmat: field, or None on
    parse failure (zero or multiple matches)."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        text = f.read()
    matches = YAML_COVMAT_RE.findall(text)
    if len(matches) != 1:
        return None
    return matches[0]


def write_yaml_covmat(path, new_value):
    # type: (str, str) -> bool
    """Rewrite the YAML's covmat: line. new_value is the raw string that
    goes after `covmat:` (e.g. 'auto' or an absolute path)."""
    if not os.path.exists(path):
        return False
    with open(path) as f:
        text = f.read()
    matches = list(YAML_COVMAT_RE.finditer(text))
    if len(matches) != 1:
        return False
    m = matches[0]
    new_line = "    covmat: {}".format(new_value)
    new_text = text[:m.start()] + new_line + text[m.end():]
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        f.write(new_text)
    os.replace(tmp, path)
    return True


# ============================================================================
#  PLANNING REPORT — human-readable summary
# ============================================================================

def _fmt_plan_line(rn, plan, chain_count, current_yaml_val):
    # type: (str, Dict[str, Any], int, Optional[str]) -> str
    kind  = plan['kind']
    score = plan['score']
    cov   = plan['coverage']
    verdict = plan.get('wait_verdict', {}).get('verdict', 'proceed')

    kind_str = {
        'single':  _c("single ", 'green'),
        'splice':  _c("splice ", 'cyan'),
        'builder': _c("builder", 'yellow'),
        'auto':    _c("auto   ", 'gray'),
    }.get(kind, kind)

    v_str = {
        'proceed':           _c("PROCEED", 'green'),
        'wait':              _c("WAIT   ", 'yellow'),
        'wait-but-unknown':  _c("WAIT?  ", 'gray'),
    }.get(verdict, verdict)

    chain_tag = "" if chain_count == 0 else _c(" ({}ch)".format(chain_count), 'gray')

    ideal = plan.get('ideal_donor') or '—'
    ideal_score = plan.get('ideal_donor_score') or 0.0
    ideal_status = plan.get('ideal_donor_status') or '—'

    return ("  {:<32s} {} s={:5.1f} cov={:.0%}  ideal={:<32s} "
            "ideal_s={:.1f} ({:>9s})  {}{}").format(
        rn, kind_str, score, cov, ideal, ideal_score, ideal_status,
        v_str, chain_tag)


def cmd_plan(all_runs, state, donors, args):
    # type: (Dict, Dict, Dict, argparse.Namespace) -> Dict[str, Dict[str, Any]]
    """Build and print the planning report. Returns the plan dict keyed by
    receiver name (the caller may pass it to cmd_apply)."""
    plans = {}  # type: Dict[str, Dict[str, Any]]
    receivers = [rn for rn, cfg in sorted(all_runs.items())
                 if rn not in donors]  # everything that isn't already a donor

    print()
    print(_c("  CAMPAIGN COVMAT PLAN", 'bold'))
    print("  Donor pool: {} runs (CONVERGED + near-converged with R-1 below {}/{})".format(
        len(donors), args.r1m_cutoff, args.r1cl_cutoff))
    print("  Receivers: {} (everything not in donor pool)".format(len(receivers)))
    print()
    print("  " + _c("kind     score cov   ideal donor                      "
                    "ideal-score (status)   verdict", 'gray'))

    n_single = n_splice = n_builder = n_auto = 0
    n_proceed = n_wait = 0

    for rn in receivers:
        cfg = all_runs[rn]
        plan = compile_plan(cfg, donors, all_runs, state)
        plans[rn] = plan

        folder = os.path.join(RUNS_ROOT, cfg['folder_path'])
        chains = _count_chain_files(folder, rn)
        yaml_val = read_yaml_covmat(find_yaml_path(cfg))

        line = _fmt_plan_line(rn, plan, chains, yaml_val)
        print(line)

        n_single  += plan['kind'] == 'single'
        n_splice  += plan['kind'] == 'splice'
        n_builder += plan['kind'] == 'builder'
        n_auto    += plan['kind'] == 'auto'
        v = plan.get('wait_verdict', {}).get('verdict', 'proceed')
        if v == 'wait':
            n_wait += 1
        else:
            n_proceed += 1

    print()
    print("  " + _c("Plan distribution: ", 'bold')
          + "single={}  splice={}  builder={}  auto={}".format(
              n_single, n_splice, n_builder, n_auto))
    print("  " + _c("Wait verdict:      ", 'bold')
          + "{} PROCEED  /  {} WAIT".format(n_proceed, n_wait))
    print()
    if n_wait:
        print(_c("  Receivers flagged WAIT are blocked on an ideal donor that's", 'gray'))
        print(_c("  still running. Use `apply --skip-wait` to commit only PROCEED.", 'gray'))
    print()
    print(_c("  To commit:", 'cyan'))
    print("    python apply_covmats.py apply              # commit ALL receivers")
    print("    python apply_covmats.py apply --skip-wait  # commit only PROCEED verdicts")
    print()
    return plans


# ============================================================================
#  APPLY MODE — write covmats + rewrite YAMLs
# ============================================================================

def _format_covmat_yaml_value(s):
    # type: (str) -> str
    """The YAML `covmat:` field can be `auto` (bare) or a path (we keep
    absolute paths bare; YAML accepts that)."""
    return s


def cmd_apply(all_runs, state, donors, plans, args):
    # type: (Dict, Dict, Dict, Dict, argparse.Namespace) -> None
    """Execute committed plans. By default, only chain-less PENDING
    receivers are modified. With --apply-running, RUNNING-but-stuck receivers
    also get their YAMLs rewritten (user must `restart` to take effect)."""
    n_written = 0
    n_unchanged = 0
    n_skipped_wait = 0
    n_skipped_chains = 0
    n_skipped_nonauto = 0
    n_failed = 0
    stuck_modified = []  # type: List[str]

    for rn, plan in sorted(plans.items()):
        cfg = all_runs[rn]
        verdict = plan.get('wait_verdict', {}).get('verdict', 'proceed')

        if args.skip_wait and verdict == 'wait':
            n_skipped_wait += 1
            continue

        folder = os.path.join(RUNS_ROOT, cfg['folder_path'])
        chains = _count_chain_files(folder, rn)

        if chains > 0 and not args.apply_running:
            n_skipped_chains += 1
            continue

        if chains > 0 and args.apply_running:
            # Stuck check
            health = load_health(folder)
            r1m = health.get('rminus1') if health else None
            if r1m is None or r1m <= args.stuck_r1:
                n_skipped_chains += 1
                continue
            stuck_modified.append(rn)

        # Optional: skip if current covmat already points at the right thing
        yaml_path = find_yaml_path(cfg)
        current = read_yaml_covmat(yaml_path)
        if current is None:
            print(_c("    ✗ {} : couldn't parse covmat line".format(rn), 'red'))
            n_failed += 1
            continue
        current_clean = current.strip().strip('"').strip("'")

        # Execute the plan with PSD-aware fallbacks
        new_value, log = execute_plan(cfg, plan, donors, verbose=args.verbose)
        if new_value is None:
            # PSD failure on single-donor — degrade to splice
            plan_fallback = dict(plan)
            plan_fallback['kind'] = 'splice'
            new_value, log2 = execute_plan(cfg, plan_fallback, donors,
                                           verbose=args.verbose)
            log += log2

        if new_value is None:
            new_value = 'auto'

        if current_clean == new_value or (current_clean == 'auto' and new_value == 'auto'):
            n_unchanged += 1
            if args.verbose:
                print("    = {} : already {}".format(rn, new_value))
            continue

        # Safety: never overwrite a non-auto/non-builder covmat without --force
        if (current_clean != 'auto'
                and not current_clean.startswith(os.path.abspath(RUNS_ROOT))
                and not args.force):
            print(_c("    — {} : custom covmat '{}' — skipped (use --force to override)".format(
                rn, current_clean), 'yellow'))
            n_skipped_nonauto += 1
            continue

        ok = write_yaml_covmat(yaml_path, new_value)
        if not ok:
            print(_c("    ✗ {} : YAML write failed".format(rn), 'red'))
            n_failed += 1
            continue

        n_written += 1
        kind_label = plan['kind']
        tag = ""
        if rn in stuck_modified:
            tag = _c(" [STUCK — needs restart]", 'red')
        print(_c("    ✓ {} : {} -> {}{}".format(
            rn, kind_label, new_value, tag),
            'green' if rn not in stuck_modified else 'cyan'))
        if args.verbose:
            for line in log:
                print("      | {}".format(line))

    print()
    print(_c(
        "  Applied: {} written, {} unchanged, {} skipped (chains), "
        "{} skipped (wait), {} skipped (non-auto), {} failed".format(
            n_written, n_unchanged, n_skipped_chains, n_skipped_wait,
            n_skipped_nonauto, n_failed), 'bold'))

    if stuck_modified:
        print()
        print(_c("  CRITICAL: {} stuck-with-chains YAML(s) were modified. ".format(
            len(stuck_modified)), 'red'))
        print(_c("  Chains conflict with new covmat on --resume. You MUST run:", 'red'))
        for rn in stuck_modified:
            print(_c("    python run_manager.py restart {}".format(rn), 'yellow'))


# ============================================================================
#  REVERT MODE
# ============================================================================

def cmd_revert(all_runs, target):
    # type: (Dict[str, Dict[str, Any]], str) -> None
    if target == '--all':
        targets = [rn for rn, cfg in all_runs.items() if cfg.get('has_uvlf')]
        print(_c("\n  Revert ALL {} UVLF receivers to covmat: auto".format(
            len(targets)), 'yellow'))
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
        path = find_yaml_path(cfg)
        cur = read_yaml_covmat(path)
        if cur is None:
            print(_c("    ✗ {} : couldn't parse covmat line".format(rn), 'red'))
            continue
        if cur.strip().strip('"').strip("'") == 'auto':
            continue
        if write_yaml_covmat(path, 'auto'):
            n_done += 1
            print(_c("    ✓ {} : reverted to auto".format(rn), 'green'))
        else:
            print(_c("    ✗ {} : YAML write failed".format(rn), 'red'))
    print()
    print(_c("  Reverted {} YAMLs.".format(n_done), 'bold'))


# ============================================================================
#  CLI
# ============================================================================

USAGE_EPILOG = """\
WAIT VERDICT MATH

  Per receiver, savings from waiting for the ideal donor:
      f(s) = 1 - (1 - f_min) * s / s_max
      savings_h = t_cold_h * (f(s_current) - f(s_ideal))
      wait iff ETA_ideal_h < savings_h AND gap >= delta_min AND ETA <= eta_cap

  Defaults (tunable via flags):
      f_min       = {f_min}   (empirical: best-warm-start runtime / cold runtime)
      delta_min   = {dmin}    (minimum quality gap for waiting to matter)
      eta_cap     = {ecap:.0f}h    (hard cap on willingness to wait)
      s_max       = {smax:.0f}     (score ceiling for normalization)

EXAMPLES

  python apply_covmats.py                   # planning report (default)
  python apply_covmats.py plan              # same
  python apply_covmats.py apply             # commit all (PROCEED + WAIT)
  python apply_covmats.py apply --skip-wait # commit only PROCEED
  python apply_covmats.py apply-running     # also rewrite stuck-with-chains YAMLs
  python apply_covmats.py revert <name>     # one YAML back to auto
  python apply_covmats.py revert --all      # all 108 UVLF YAMLs back to auto
""".format(f_min=WAIT_F_MIN, dmin=WAIT_DELTA_MIN,
           ecap=WAIT_ETA_HARD_CAP_H, smax=SCORE_MAX)


def build_argparser():
    # type: () -> argparse.ArgumentParser
    p = argparse.ArgumentParser(
        prog='apply_covmats.py',
        description='Campaign-wide covmat manager (production donors + builders + wait verdict)',
        epilog=USAGE_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mode', nargs='?', default='plan',
                   choices=['plan', 'apply', 'apply-running', 'revert'],
                   help='Operation mode (default: plan)')
    p.add_argument('target', nargs='?',
                   help='Run name for revert, or --all')
    # Donor pool cutoffs
    p.add_argument('--r1m-cutoff', type=float, default=DONOR_R1M_CUTOFF,
                   help='Donor R-1 means ceiling (default {})'.format(DONOR_R1M_CUTOFF))
    p.add_argument('--r1cl-cutoff', type=float, default=DONOR_R1CL_CUTOFF,
                   help='Donor R-1 CL ceiling (default {})'.format(DONOR_R1CL_CUTOFF))
    # Apply behavior
    p.add_argument('--skip-wait', action='store_true',
                   help='Skip receivers with WAIT verdict')
    p.add_argument('--apply-running', action='store_true',
                   help='Also modify YAMLs of stuck-with-chains runs')
    p.add_argument('--stuck-r1', type=float, default=STUCK_R1_THRESHOLD,
                   help='R-1 above which a chain-having run is "stuck"')
    p.add_argument('--force', action='store_true',
                   help='Override the skip on custom (non-auto) YAML covmats')
    # Wait-verdict tuning
    p.add_argument('--wait-f-min', type=float, default=WAIT_F_MIN,
                   help='Best-warm-start runtime fraction (default {})'.format(WAIT_F_MIN))
    p.add_argument('--wait-delta-min', type=float, default=WAIT_DELTA_MIN,
                   help='Min score gap for "wait" verdict (default {})'.format(WAIT_DELTA_MIN))
    p.add_argument('--wait-eta-cap', type=float, default=WAIT_ETA_HARD_CAP_H,
                   help='Hard cap on willingness to wait, hours (default {:.0f})'.format(WAIT_ETA_HARD_CAP_H))
    # Verbose
    p.add_argument('-v', '--verbose', action='store_true',
                   help='Per-receiver detail (top candidates, splice logs)')
    return p


def main():
    parser = build_argparser()
    args = parser.parse_args()

    if not os.path.isdir(RUNS_ROOT):
        print(_c("  [ERROR] {} not found. Run generate_all_runs.py first.".format(
            RUNS_ROOT), 'red'))
        sys.exit(1)

    # Update module-level constants from CLI flags so functions pick them up
    global WAIT_F_MIN, WAIT_DELTA_MIN, WAIT_ETA_HARD_CAP_H
    WAIT_F_MIN          = args.wait_f_min
    WAIT_DELTA_MIN      = args.wait_delta_min
    WAIT_ETA_HARD_CAP_H = args.wait_eta_cap

    all_runs = build_all_runs()
    state    = load_state()

    if args.mode == 'revert':
        if args.target is None:
            print("  Usage: apply_covmats.py revert <run_name|--all>")
            sys.exit(1)
        cmd_revert(all_runs, args.target)
        return

    donors = discover_donor_pool(all_runs, state,
                                 args.r1m_cutoff, args.r1cl_cutoff)
    plans  = cmd_plan(all_runs, state, donors, args)

    if args.mode in ('apply', 'apply-running'):
        if args.mode == 'apply-running':
            args.apply_running = True
        cmd_apply(all_runs, state, donors, plans, args)


if __name__ == '__main__':
    main()