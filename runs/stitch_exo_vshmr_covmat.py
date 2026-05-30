#!/usr/bin/env python3
# ============================================================================
# stitch_exo_vshmr_covmat.py
#
# Build a block-diagonal 5x5 covariance matrix for the exo+vshmr production
# runs by combining:
#   - 2x2 covmat from builder_exo_fixed     (a_samp, s)
#   - 3x3 covmat from builder_lcdm_vshmr    (shmr_log_Mc, shmr_N, shmr_beta)
#
# Off-diagonal cross-blocks are set to zero (assumed independence). Cobaya
# learns the true cross-correlations during the production run via learn_every.
#
# The output 5x5 covmat is written to:
#   runs/builders/builder_exo_vshmr/outputs/builder_exo_vshmr.covmat
#
# After running this, `apply_covmats.py` picks up the new file automatically.
#
# Python 3.8 compatible. Standard library + numpy.
#
# Usage:
#   python stitch_exo_vshmr_covmat.py
# ============================================================================

import os
import sys
import numpy as np
from typing import List, Tuple

# ============================================================================
#  CONSTANTS
# ============================================================================

RUNS_ROOT          = "runs"
BUILDERS_SUBDIR    = "builders"

EXO_BLOCK_BUILDER  = "builder_exo_fixed"    # provides (a_samp, s)
SHMR_BLOCK_BUILDER = "builder_lcdm_vshmr"   # provides (shmr_log_Mc, shmr_N, shmr_beta)
TARGET_BUILDER     = "builder_exo_vshmr"    # output goes here

# Final parameter order — MUST match how the production exo+vshmr YAMLs list
# their sampled parameters (cobaya matches covmat entries by name, but writing
# them in the canonical order makes the file readable).
FINAL_PARAM_ORDER = ['a_samp', 's', 'shmr_log_Mc', 'shmr_N', 'shmr_beta']
EXPECTED_EXO_PARAMS  = {'a_samp', 's'}
EXPECTED_SHMR_PARAMS = {'shmr_log_Mc', 'shmr_N', 'shmr_beta'}


# ============================================================================
#  COVMAT I/O (cobaya format: '# name1 name2 ...' header + matrix rows)
# ============================================================================

def read_covmat(path):
    # type: (str) -> Tuple[List[str], np.ndarray]
    """Parse a cobaya-format .covmat file. Returns (param_names, cov_matrix).

    Raises ValueError on any structural problem. Validates symmetry and
    positive-definiteness before returning.
    """
    if not os.path.exists(path):
        raise FileNotFoundError("Covmat not found: {}".format(path))
    with open(path) as f:
        text = f.read()

    lines = [l for l in text.split('\n') if l.strip()]
    if not lines:
        raise ValueError("{} is empty".format(path))

    header = lines[0].strip()
    if not header.startswith('#'):
        raise ValueError("{}: expected header line starting with '#', got '{}'".format(
            path, header[:60]))
    names = header.lstrip('#').strip().split()
    if not names:
        raise ValueError("{}: no parameter names in header line".format(path))

    matrix_rows = []
    for ln in lines[1:]:
        ln = ln.strip()
        if not ln or ln.startswith('#'):
            continue
        try:
            row = [float(x) for x in ln.split()]
        except ValueError:
            raise ValueError("{}: cannot parse row as floats: '{}'".format(path, ln[:60]))
        matrix_rows.append(row)

    cov = np.asarray(matrix_rows, dtype=float)
    n = len(names)
    if cov.shape != (n, n):
        raise ValueError("{}: matrix shape {} doesn't match {} names".format(
            path, cov.shape, n))
    if not np.allclose(cov, cov.T, atol=1e-12, rtol=1e-9):
        raise ValueError("{}: covariance matrix is not symmetric".format(path))
    eigvals = np.linalg.eigvalsh(cov)
    if np.any(eigvals <= 0):
        raise ValueError(
            "{}: covariance matrix not positive-definite (min eigval={:.3e})".format(
                path, eigvals.min()))

    return names, cov


def write_covmat(path, names, cov):
    # type: (str, List[str], np.ndarray) -> None
    """Write a cobaya-format .covmat file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('# ' + ' '.join(names) + '\n')
        for row in cov:
            f.write(' '.join('{:.8e}'.format(x) for x in row) + '\n')


# ============================================================================
#  COLORS (matches the style of the other scripts)
# ============================================================================

USE_COLOR = sys.stdout.isatty()


def _c(text, color):
    if not USE_COLOR:
        return text
    codes = {'green': '32', 'yellow': '33', 'red': '31',
             'cyan': '36', 'gray': '90', 'bold': '1'}
    return '\033[{}m{}\033[0m'.format(codes.get(color, '0'), text)


# ============================================================================
#  STITCHING LOGIC
# ============================================================================

def reorder_block(names_in, cov_in, names_desired):
    # type: (List[str], np.ndarray, List[str]) -> np.ndarray
    """Reorder a sub-covmat's rows/cols so they match `names_desired`.

    All names in `names_desired` must be present in `names_in`. Names in
    `names_in` not in `names_desired` are dropped.
    """
    idx = [names_in.index(n) for n in names_desired]
    return cov_in[np.ix_(idx, idx)]


def stitch_block_diagonal(exo_names, exo_cov, shmr_names, shmr_cov):
    # type: (List[str], np.ndarray, List[str], np.ndarray) -> np.ndarray
    """Build the 5x5 block-diagonal covmat in FINAL_PARAM_ORDER. Cross-block
    off-diagonals are zero (assumed independence — cobaya learns the real
    cross-correlations during sampling)."""
    # Reorder each input block to match FINAL_PARAM_ORDER
    exo_order  = [n for n in FINAL_PARAM_ORDER if n in EXPECTED_EXO_PARAMS]
    shmr_order = [n for n in FINAL_PARAM_ORDER if n in EXPECTED_SHMR_PARAMS]
    exo_block  = reorder_block(exo_names,  exo_cov,  exo_order)
    shmr_block = reorder_block(shmr_names, shmr_cov, shmr_order)

    n = len(FINAL_PARAM_ORDER)
    out = np.zeros((n, n))
    n_exo = len(exo_order)
    out[:n_exo, :n_exo] = exo_block
    out[n_exo:, n_exo:] = shmr_block
    return out


# ============================================================================
#  MAIN
# ============================================================================

def main():
    exo_path = os.path.join(RUNS_ROOT, BUILDERS_SUBDIR, EXO_BLOCK_BUILDER,
                            'outputs', EXO_BLOCK_BUILDER + '.covmat')
    shmr_path = os.path.join(RUNS_ROOT, BUILDERS_SUBDIR, SHMR_BLOCK_BUILDER,
                             'outputs', SHMR_BLOCK_BUILDER + '.covmat')
    target_path = os.path.join(RUNS_ROOT, BUILDERS_SUBDIR, TARGET_BUILDER,
                               'outputs', TARGET_BUILDER + '.covmat')

    print()
    print(_c("  stitch_exo_vshmr_covmat.py", 'bold'))
    print("  " + "-" * 65)

    # Refuse to overwrite if a real builder finished and dumped a covmat. Be
    # conservative — the user can rm the file if they want to overwrite.
    if os.path.exists(target_path):
        try:
            existing_names, existing_cov = read_covmat(target_path)
            if existing_cov.shape == (5, 5):
                print(_c("  Target already exists at:", 'yellow'))
                print("    {}".format(target_path))
                print(_c("  Shape: {}, params: {}".format(
                    existing_cov.shape, existing_names), 'yellow'))
                ans = input(_c("  Overwrite with stitched block-diagonal? [y/N]: ", 'cyan'))
                if ans.strip().lower() not in ('y', 'yes'):
                    print("  Aborted — existing covmat preserved.")
                    return
        except Exception:
            # Existing file is broken; we're better off overwriting silently
            pass

    # ── Read inputs ─────────────────────────────────────────────────────
    try:
        print(_c("\n  Reading exotic block: {}".format(exo_path), 'cyan'))
        exo_names, exo_cov = read_covmat(exo_path)
        print("    params: {}, shape: {}".format(exo_names, exo_cov.shape))

        print(_c("\n  Reading SHMR block: {}".format(shmr_path), 'cyan'))
        shmr_names, shmr_cov = read_covmat(shmr_path)
        print("    params: {}, shape: {}".format(shmr_names, shmr_cov.shape))
    except (FileNotFoundError, ValueError) as e:
        print(_c("  [ERROR] {}".format(e), 'red'))
        print(_c("  Both source builders must have completed and produced a "
                 ".covmat before stitching.", 'red'))
        sys.exit(1)

    # ── Validate parameter sets ─────────────────────────────────────────
    if set(exo_names) != EXPECTED_EXO_PARAMS:
        print(_c("\n  [ERROR] exotic block has unexpected params.", 'red'))
        print("    got:      {}".format(sorted(exo_names)))
        print("    expected: {}".format(sorted(EXPECTED_EXO_PARAMS)))
        sys.exit(1)
    if set(shmr_names) != EXPECTED_SHMR_PARAMS:
        print(_c("\n  [ERROR] SHMR block has unexpected params.", 'red'))
        print("    got:      {}".format(sorted(shmr_names)))
        print("    expected: {}".format(sorted(EXPECTED_SHMR_PARAMS)))
        sys.exit(1)

    # ── Stitch ───────────────────────────────────────────────────────────
    out_cov = stitch_block_diagonal(exo_names, exo_cov, shmr_names, shmr_cov)

    # Sanity checks on the result
    if not np.allclose(out_cov, out_cov.T):
        print(_c("\n  [ERROR] stitched covmat is not symmetric (bug)", 'red'))
        sys.exit(1)
    eigvals = np.linalg.eigvalsh(out_cov)
    if np.any(eigvals <= 0):
        print(_c("\n  [ERROR] stitched covmat is not positive-definite "
                 "(min eigval={:.3e})".format(eigvals.min()), 'red'))
        sys.exit(1)

    # ── Report ──────────────────────────────────────────────────────────
    print(_c("\n  Stitched 5x5 covmat in order: {}".format(FINAL_PARAM_ORDER), 'bold'))
    print()
    # Pretty-print the matrix
    col_width = 14
    print("    " + " " * col_width + "".join("{:>{w}s}".format(n, w=col_width)
                                              for n in FINAL_PARAM_ORDER))
    for i, name in enumerate(FINAL_PARAM_ORDER):
        print("    {:>{w}s}".format(name, w=col_width)
              + "".join("{:>{w}.4e}".format(out_cov[i, j], w=col_width)
                        for j in range(len(FINAL_PARAM_ORDER))))

    print(_c("\n  Per-parameter standard deviations (sqrt of diagonal):", 'cyan'))
    for name, sigma in zip(FINAL_PARAM_ORDER, np.sqrt(np.diag(out_cov))):
        print("    {:<15s}  sigma = {:.4e}".format(name, sigma))

    # ── Write ───────────────────────────────────────────────────────────
    write_covmat(target_path, FINAL_PARAM_ORDER, out_cov)
    print(_c("\n  [OK] Wrote {}".format(target_path), 'green'))

    # ── Next steps ──────────────────────────────────────────────────────
    print(_c("\n  Next:", 'bold'))
    print("    1. python apply_covmats.py             # dry-run to confirm")
    print("    2. python apply_covmats.py apply       # propagate to the 36 exo+vshmr PENDING runs")
    print()
    print(_c("  Note: the off-diagonal cross-blocks (a_samp/s) ↔ (SHMR) are zero.", 'gray'))
    print(_c("  Cobaya will learn the real cross-correlations during the production run", 'gray'))
    print(_c("  via learn_every. The stitched covmat is a warm start, not a final proposal.", 'gray'))


if __name__ == '__main__':
    main()
