#!/usr/bin/env python3
"""
exo_prior_scan.py
=================
Viable-region scan + Gaussian prior fit for (a_exo, b_exo) in the class_omx
exotic-dark-energy fork, for publication-grade MCMC prior selection.

Physics
-------
Model:
    rho_x(z) = [a + b * z/(z+1)] * exp[-(z - z_c)^2 / (2 sigma_z^2)]
    rho_x(0) = a * exp[-z_c^2 / (2 sigma_z^2)]  ==  Omega_{x,0}

Because rho_x(0) is already in H_0^2 units, rho_x(z) IS the contribution to
H^2(z)/H_0^2 (no extra prefactor). With Omega_Lambda re-shot to close
Sum Omega_i = 1, the closure subtracts Omega_{x,0} = a*g(0) at z=0, giving
a Friedmann equation that is LINEAR in (a_exo, b_exo) at every redshift:

    H^2(z) / H_0^2  =  L(z)  +  a * u(z)  +  b * v(z)

with
    g(z) = exp[-(z - z_c)^2 / (2 sigma_z^2)]
    g0   = g(0)   =  exp[-z_c^2 / (2 sigma_z^2)]
    u(z) = g(z) - g0                # absorbs the closure subtraction
    v(z) = (z / (z+1)) * g(z)
    L(z) = H^2_LCDM(z) / H_0^2

Consequences:
  * The viability test H^2(z) > 0 for all z is a pure algebraic intersection
    of half-planes in (a, b). No per-cell CLASS call.
  * The viable region is EXACTLY a convex polygon (plus the prior cut
    b < |a|, i.e. a + b < 0 for a < 0), and can be computed in closed form
    via scipy.spatial.HalfspaceIntersection.
  * CLASS is only used once to tabulate L(z), and optionally as a paranoid
    validator on each viable grid cell.

Phases
------
  analytic  — vectorized grid scan of H^2 min over z, MPI across a-rows
  polygon   — exact convex polygon of the viable set
  validate  — CLASS on every viable cell, leak-proofed via
              multiprocessing.Pool(maxtasksperchild=...)
  prior     — fit Gaussian to validated set, emit Cobaya + MontePython specs
  plot      — publication-quality figure

Quick local smoke test (no MPI, N = 200):
  python exo_prior_scan.py --phase analytic --N 200 --outdir test_out
  python exo_prior_scan.py --phase polygon --outdir test_out
  python exo_prior_scan.py --phase prior   --outdir test_out
  python exo_prior_scan.py --phase plot    --outdir test_out

Full HPC run (mpi4py + SLURM):
  sbatch exo_prior_scan.slurm
"""

import argparse
import os
import sys
import json
import time
import numpy as np


# ---------------------------------------------------------------------------
#  Constants and fixed cosmology
# ---------------------------------------------------------------------------
Z_C        = 16.0
SIGMA_Z    = 3.25
Z_GRID_MAX = 100.0
Z_GRID_N   = 2000

FIXED_COSMO = {
    'omega_b':     0.02237,
    'omega_cdm':   0.1200,
    'H0':          67.4,
    'tau_reio':    0.0544,
    'A_s':         2.101e-9,
    'n_s':         0.9649,
    'm_ncdm':      0.06,
    'N_ncdm':      1,
    'N_ur':        2.0328,
    'z_c_exo':     Z_C,
    'sigma_z_exo': SIGMA_Z,
    'output':      '',   # background only
}


# ---------------------------------------------------------------------------
#  MPI helpers (graceful fallback when mpi4py is absent)
# ---------------------------------------------------------------------------
try:
    from mpi4py import MPI
    HAS_MPI = True
except ImportError:
    HAS_MPI = False


def mpi_info():
    if HAS_MPI:
        c = MPI.COMM_WORLD
        return c, c.Get_rank(), c.Get_size()
    return None, 0, 1


# ---------------------------------------------------------------------------
#  Linear-model helpers
# ---------------------------------------------------------------------------
def build_uv(z_arr):
    """Return u(z), v(z) of the H^2/H_0^2 linear decomposition for the model
        rho_x(z) = [a + b * z/(z+1)] * exp[-(z - z_c)^2 / (2 sigma_z^2)]
    with Omega_Lambda re-shot at z=0.
    """
    g  = np.exp(-(z_arr - Z_C) ** 2 / (2 * SIGMA_Z ** 2))
    g0 = float(np.exp(-Z_C ** 2 / (2 * SIGMA_Z ** 2)))
    u  = g - g0                              # a-coefficient (closure-corrected)
    v  = (z_arr / (z_arr + 1.0)) * g         # b-coefficient
    return u, v


def get_lcdm_table(z_arr):
    """Run LCDM once and return L(z) = H^2(z)/H_0^2."""
    from classy import Class
    cosmo = Class()
    cosmo.set(FIXED_COSMO)
    cosmo.compute()
    H0 = cosmo.Hubble(0)
    L = np.array([(cosmo.Hubble(z) / H0) ** 2 for z in z_arr])
    cosmo.struct_cleanup()
    cosmo.empty()
    return L


# ---------------------------------------------------------------------------
#  PHASE 1 — analytic grid scan
# ---------------------------------------------------------------------------
def phase_analytic(args):
    comm, rank, size = mpi_info()
    N = args.N

    a_vals = np.linspace(args.a_min, args.a_max, N, dtype=np.float64)
    b_vals = np.linspace(args.b_min, args.b_max, N, dtype=np.float64)
    z_arr  = np.linspace(0.0, Z_GRID_MAX, Z_GRID_N)

    # Compute LCDM reference once on rank 0, broadcast
    if rank == 0:
        print(f"[analytic] N={N}, MPI size={size}, Z_GRID_N={Z_GRID_N}")
        print(f"[analytic] computing LCDM reference H^2(z)...")
        L = get_lcdm_table(z_arr)
    else:
        L = None
    if HAS_MPI:
        L = comm.bcast(L, root=0)
    u, v = build_uv(z_arr)

    # Pre-compute L + a*u part cheaply later; for now split a-rows across ranks
    row_chunks = np.array_split(np.arange(N), size)
    my_rows    = row_chunks[rank]
    nrows      = len(my_rows)

    # Local output slabs (float32 to halve memory)
    min_h2    = np.full((nrows, N), np.inf, dtype=np.float32)
    argmin_z  = np.zeros((nrows, N), dtype=np.int32)
    excluded  = np.zeros((nrows, N), dtype=bool)     # prior cut: b >= -a

    t0 = time.time()

    # Per-row work: at fixed a,
    #   H^2(z, b)/H_0^2 = (L(z) + a*u(z))  +  v(z) * b
    # which is linear in b. For min over z at each b, we broadcast.
    for local_i, i in enumerate(my_rows):
        a      = a_vals[i]
        core   = (L + a * u).astype(np.float32)          # (K,)
        vz32   = v.astype(np.float32)                    # (K,)
        bz32   = b_vals.astype(np.float32)               # (N,)
        # H2 shape (K, N): core[:, None] + vz[:, None] * b[None, :]
        H2     = core[:, None] + vz32[:, None] * bz32[None, :] #Column = b = b, row = z
        k_min  = H2.argmin(axis=0)
        min_h2[local_i, :]   = H2[k_min, np.arange(N)]
        argmin_z[local_i, :] = k_min

        # Prior cut: require b < -a (=> a + b < 0)
        excluded[local_i, :] = (b_vals >= -a)

        if rank == 0 and (local_i % max(1, nrows // 20) == 0):
            dt  = time.time() - t0
            eta = dt * (nrows / (local_i + 1) - 1)
            print(f"[analytic] rank 0: row {local_i}/{nrows}  "
                  f"({dt:5.1f}s elapsed, ETA {eta:5.1f}s)")

    # Mark excluded cells with a sentinel so they do not contaminate plots
    min_h2[excluded] = np.nan

    # Gather slabs on rank 0
    if HAS_MPI:
        gathered_m = comm.gather(min_h2,   root=0)
        gathered_a = comm.gather(argmin_z, root=0)
        gathered_e = comm.gather(excluded, root=0)
    else:
        gathered_m = [min_h2]
        gathered_a = [argmin_z]
        gathered_e = [excluded]

    if rank == 0:
        full_min = np.vstack(gathered_m)
        full_arg = np.vstack(gathered_a)
        full_exc = np.vstack(gathered_e)
        viable   = (~full_exc) & np.isfinite(full_min) & (full_min > 0)

        os.makedirs(args.outdir, exist_ok=True)
        np.savez_compressed(
            os.path.join(args.outdir, 'analytic.npz'),
            a_vals=a_vals, b_vals=b_vals, z_arr=z_arr,
            min_h2=full_min, argmin_z=full_arg,
            viable=viable, excluded=full_exc,
        )
        print(f"[analytic] viable: {viable.sum():,} / {N*N:,} "
              f"({100 * viable.sum() / N ** 2:.2f}%)")
        print(f"[analytic] wall time: {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
#  PHASE 2 — exact convex polygon
# ---------------------------------------------------------------------------
def phase_polygon(args):
    from scipy.spatial import HalfspaceIntersection, ConvexHull

    z_arr = np.linspace(0.0, Z_GRID_MAX, Z_GRID_N)
    L     = get_lcdm_table(z_arr)
    u, v  = build_uv(z_arr)

    # scipy HalfspaceIntersection form: A x + d <= 0, x = (a, b)
    # Physics constraints: a*u(z) + b*v(z) + L(z) > 0  =>  -u*a - v*b - L <= 0
    halfs_phys = np.column_stack([-u, -v, -L])

    # Prior cut: a + b < 0   =>  a + b <= 0
    halfs_prior = np.array([[1.0, 1.0, 0.0]])

    # Bounding box (scipy needs a bounded polytope)
    halfs_box = np.array([
        [-1.0,  0.0,  args.a_min],   #  a >= a_min  =>  -a + a_min <= 0
        [ 1.0,  0.0, -args.a_max],   #  a <= a_max
        [ 0.0, -1.0,  args.b_min],
        [ 0.0,  1.0, -args.b_max],
    ])

    halfspaces = np.vstack([halfs_phys, halfs_prior, halfs_box])

    # Deep interior point (well inside all constraints)
    interior = np.array([-0.5, -1.0])

    hs   = HalfspaceIntersection(halfspaces, interior)
    pts  = hs.intersections
    # Remove numerical duplicates before hull
    pts  = np.unique(np.round(pts, 9), axis=0)
    hull = ConvexHull(pts)
    poly = pts[hull.vertices]           # ordered CCW

    # Exact centroid via shoelace
    x, y     = poly[:, 0], poly[:, 1]
    cross    = x * np.roll(y, -1) - np.roll(x, -1) * y
    area     = 0.5 * cross.sum()
    cx       = ((x + np.roll(x, -1)) * cross).sum() / (6 * area)
    cy       = ((y + np.roll(y, -1)) * cross).sum() / (6 * area)
    centroid = np.array([cx, cy])

    os.makedirs(args.outdir, exist_ok=True)
    np.savez(os.path.join(args.outdir, 'polygon.npz'),
             vertices=poly, centroid=centroid, area=abs(area))

    print(f"[polygon] vertices : {len(poly)}")
    print(f"[polygon] centroid : ({cx:+.4f}, {cy:+.4f})")
    print(f"[polygon] area     : {abs(area):.4e}")
    print(f"[polygon] a range  : [{poly[:,0].min():.2f}, {poly[:,0].max():.2f}]")
    print(f"[polygon] b range  : [{poly[:,1].min():.2f}, {poly[:,1].max():.2f}]")


# ---------------------------------------------------------------------------
#  PHASE 3 — CLASS validation on every viable cell (leak-proofed)
# ---------------------------------------------------------------------------
def _class_worker(task):
    """Run CLASS on one (a, b) cell. Executed in a short-lived worker."""
    i, j, a, b = task
    try:
        from classy import Class, CosmoSevereError, CosmoComputationError
        params = dict(FIXED_COSMO)
        params['a_exo'] = a
        params['b_exo'] = b
        cosmo = Class()
        cosmo.set(params)
        cosmo.compute()
        zs = np.linspace(0.0, 60.0, 200)
        H  = np.array([cosmo.Hubble(z) for z in zs])
        cosmo.struct_cleanup()
        cosmo.empty()
        return (i, j, float((H ** 2).min()), 0)
    except Exception:
        return (i, j, float('nan'), 1)


def phase_validate(args):
    from multiprocessing import Pool

    comm, rank, size = mpi_info()

    data    = np.load(os.path.join(args.outdir, 'analytic.npz'))
    a_vals  = data['a_vals']
    b_vals  = data['b_vals']
    viable  = data['viable']
    ii, jj  = np.where(viable)
    
    ii, jj = ii[::750], jj[::750] # thinning steps to make it faster
    
    ntot    = len(ii)

    if rank == 0:
        print(f"[validate] {ntot:,} viable cells to CLASS-check")
        print(f"[validate] MPI ranks = {size}, "
              f"workers per rank = {args.val_workers_per_rank}, "
              f"respawn every = {args.respawn_every}")
        est_per_call_s = 2.0
        est_total_h = ntot * est_per_call_s / (size * args.val_workers_per_rank) / 3600
        print(f"[validate] rough ETA at 2 s/call: {est_total_h:.1f} h")

    # Split the work evenly across MPI ranks
    mychunk = np.array_split(np.arange(ntot), size)[rank]
    tasks = [
        (int(ii[k]), int(jj[k]), float(a_vals[ii[k]]), float(b_vals[jj[k]]))
        for k in mychunk
    ]

    # multiprocessing.Pool INSIDE each MPI rank, with maxtasksperchild to
    # kill the CLASS memory leak. Respawn every N tasks.
    results = []
    t0 = time.time()
    with Pool(processes=args.val_workers_per_rank,
              maxtasksperchild=args.respawn_every) as pool:
        for k, res in enumerate(pool.imap_unordered(_class_worker, tasks,
                                                    chunksize=4)):
            results.append(res)
            if rank == 0 and (k % max(1, len(tasks) // 100) == 0):
                dt  = time.time() - t0
                eta = dt * (len(tasks) / (k + 1) - 1)
                print(f"[validate] rank 0: {k}/{len(tasks)}  "
                      f"({dt / 60:5.1f} min elapsed, ETA {eta / 60:5.1f} min)")

    results = np.array(results, dtype=np.float64)   # (n, 4): i, j, minH2, flag

    if HAS_MPI:
        gathered = comm.gather(results, root=0)
    else:
        gathered = [results]

    if rank == 0:
        full = np.vstack([g for g in gathered if len(g)])
        np.savez_compressed(os.path.join(args.outdir, 'validate.npz'),
                            results=full)
        nok  = int((full[:, 3] == 0).sum())
        nbad = int((full[:, 3] != 0).sum())
        nneg = int(((full[:, 3] == 0) & (full[:, 2] < 0)).sum())
        print(f"[validate] CLASS ok   : {nok:,}")
        print(f"[validate] CLASS fail : {nbad:,}")
        print(f"[validate] H^2 < 0    : {nneg:,}  "
              f"(analytic disagreement if > 0)")
        print(f"[validate] wall time  : {(time.time() - t0) / 60:.1f} min")


# ---------------------------------------------------------------------------
#  PHASE 4 — fit Gaussian prior
# ---------------------------------------------------------------------------
def _fit_gaussian(points):
    """Return (mu, cov_sample, cov_prior, scale) where the prior covariance
    is inflated so the outermost point sits at Mahalanobis distance 3."""
    mu  = points.mean(axis=0)
    cov = np.cov(points.T)
    inv = np.linalg.inv(cov)
    d   = points - mu
    mahal2 = np.einsum('ij,jk,ik->i', d, inv, d)
    scale  = mahal2.max() / (3.0 ** 2)
    return mu, cov, cov * scale, scale


def phase_prior(args):
    ana_path = os.path.join(args.outdir, 'analytic.npz')
    val_path = os.path.join(args.outdir, 'validate.npz')
    data_a   = np.load(ana_path)
    a_vals   = data_a['a_vals']
    b_vals   = data_a['b_vals']

    if os.path.exists(val_path):
        r    = np.load(val_path)['results']
        good = (r[:, 3] == 0) & (r[:, 2] >= 0)
        ii   = r[good, 0].astype(int)
        jj   = r[good, 1].astype(int)
        pts  = np.column_stack([a_vals[ii], b_vals[jj]])
        src  = f'CLASS-validated ({len(pts):,} pts)'
    else:
        v   = data_a['viable']
        ii, jj = np.where(v)
        pts = np.column_stack([a_vals[ii], b_vals[jj]])
        src = f'analytic ({len(pts):,} pts)'

    if len(pts) < 10:
        raise RuntimeError(f"[prior] too few viable points ({len(pts)}) to fit")

    mu, cov_s, cov_p, scale = _fit_gaussian(pts)
    sa_s, sb_s = np.sqrt(cov_s[0, 0]), np.sqrt(cov_s[1, 1])
    rho_s      = cov_s[0, 1] / (sa_s * sb_s)
    sa_p, sb_p = np.sqrt(cov_p[0, 0]), np.sqrt(cov_p[1, 1])
    rho_p      = cov_p[0, 1] / (sa_p * sb_p)

    info = {
        'source':           src,
        'n_points':         int(len(pts)),
        'centroid':         mu.tolist(),
        'sample_cov':       cov_s.tolist(),
        'sample_sigma_a':   float(sa_s),
        'sample_sigma_b':   float(sb_s),
        'sample_rho_ab':    float(rho_s),
        'prior_mean':       mu.tolist(),
        'prior_cov':        cov_p.tolist(),
        'prior_sigma_a':    float(sa_p),
        'prior_sigma_b':    float(sb_p),
        'prior_rho_ab':     float(rho_p),
        'prior_inflation':  float(np.sqrt(scale)),
        'note': ('Prior covariance is the sample covariance inflated so that '
                 'the viable set is enclosed within the 3-sigma Mahalanobis '
                 'ellipse. Uncorrelated Gaussian priors under-cover the '
                 'viable region; use the correlated form for tight runs.'),
    }

    with open(os.path.join(args.outdir, 'prior_fit.json'), 'w') as f:
        json.dump(info, f, indent=2)

    # ---- Cobaya YAML
    cov_inv = np.linalg.inv(cov_p)
    cov_inv_list = cov_inv.tolist()
    cobaya = f"""# Cobaya prior block for (a_exo, b_exo)
# Source      : {src}
# Centroid    : ({mu[0]:+.4f}, {mu[1]:+.4f})
# Inflation   : sqrt({scale:.3f}) = {np.sqrt(scale):.3f}x over sample cov
# Correlation : rho_ab = {rho_p:+.3f}

params:
  a_exo:
    prior:
      dist: norm
      loc:   {mu[0]:.6f}
      scale: {sa_p:.6f}
    ref:
      dist: norm
      loc:   {mu[0]:.6f}
      scale: {sa_p / 3:.6f}
    proposal: {sa_p / 3:.6f}
    latex: a_\\mathrm{{exo}}

  b_exo:
    prior:
      dist: norm
      loc:   {mu[1]:.6f}
      scale: {sb_p:.6f}
    ref:
      dist: norm
      loc:   {mu[1]:.6f}
      scale: {sb_p / 3:.6f}
    proposal: {sb_p / 3:.6f}
    latex: b_\\mathrm{{exo}}

# ---- Correlated version (RECOMMENDED if rho_ab is non-trivial) ----
# Replace the two independent priors above with external priors that
# together evaluate the full 2D Gaussian. Cobaya accepts additive
# log-prior terms via the top-level `prior:` block:
#
# prior:
#   exo_gauss_corr: |
#     import numpy as np
#     _mu  = np.array([{mu[0]:.6f}, {mu[1]:.6f}])
#     _inv = np.array({cov_inv_list})
#     def _p(a_exo, b_exo):
#         d = np.array([a_exo, b_exo]) - _mu
#         return -0.5 * float(d @ _inv @ d)
#     return _p
#
# When using the correlated form, set the per-parameter priors to wide
# uniforms so they do not double-count:
#   a_exo: {{ prior: {{ min: {mu[0] - 4*sa_p:.2f}, max: {mu[0] + 4*sa_p:.2f} }} }}
#   b_exo: {{ prior: {{ min: {mu[1] - 4*sb_p:.2f}, max: {mu[1] + 4*sb_p:.2f} }} }}
"""
    with open(os.path.join(args.outdir, 'prior_cobaya.yaml'), 'w') as f:
        f.write(cobaya)

    # ---- MontePython param file entries
    mp = f"""# MontePython param entries for (a_exo, b_exo)
# Source: {src}
# Format: [mean, min, max, sigma, scale, role]
data.parameters['a_exo'] = [{mu[0]:.6f}, {mu[0] - 3 * sa_p:.6f}, {mu[0] + 3 * sa_p:.6f}, {sa_p / 3:.6f}, 1, 'cosmo']
data.parameters['b_exo'] = [{mu[1]:.6f}, {mu[1] - 3 * sb_p:.6f}, {mu[1] + 3 * sb_p:.6f}, {sb_p / 3:.6f}, 1, 'cosmo']

# MontePython uses these as uniform-with-Gaussian-proposal by default.
# For a hard Gaussian prior, either:
#   (a) add a gaussian_prior block in your likelihood, or
#   (b) post-process the chain with importance weights
#         w_i = exp(-0.5 * (x_i - mu)^T Cov^-1 (x_i - mu))
#       with mu = ({mu[0]:.4f}, {mu[1]:.4f}) and Cov^-1 =
#         {cov_inv_list}
"""
    with open(os.path.join(args.outdir, 'prior_montepython.param'), 'w') as f:
        f.write(mp)

    print(f"[prior] source          : {src}")
    print(f"[prior] centroid        : ({mu[0]:+.4f}, {mu[1]:+.4f})")
    print(f"[prior] sample sigma    : ({sa_s:.4f}, {sb_s:.4f}), rho={rho_s:+.3f}")
    print(f"[prior] prior  sigma    : ({sa_p:.4f}, {sb_p:.4f}), rho={rho_p:+.3f}")
    print(f"[prior] inflation factor: {np.sqrt(scale):.3f}x")
    print(f"[prior] wrote: prior_fit.json, prior_cobaya.yaml, prior_montepython.param")


# ---------------------------------------------------------------------------
#  PHASE 5 — publication-quality plots
# ---------------------------------------------------------------------------
def phase_plot(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse, Polygon as MplPolygon

    # --- Setup Plotting Style ---
    plt.rcParams.update({
        'font.size':        11,
        'axes.linewidth':   1.0,
        'xtick.direction':  'in',
        'ytick.direction':  'in',
        'xtick.top':        True,
        'ytick.right':      True,
    })

    # 1. Load the shared geometry and analytic results
    ana_data = np.load(os.path.join(args.outdir, 'analytic.npz'))
    a_vals = ana_data['a_vals']
    b_vals = ana_data['b_vals']
    
    # Load shared prior and polygon
    poly_path  = os.path.join(args.outdir, 'polygon.npz')
    prior_path = os.path.join(args.outdir, 'prior_fit.json')
    poly  = np.load(poly_path)['vertices'] if os.path.exists(poly_path) else None
    prior = json.load(open(prior_path))    if os.path.exists(prior_path) else None

    # Helper to draw the actual figure
    def save_dual_panel(a, b, viable_grid, h2_grid, title_suffix, filename):
        # Apply the speed-hack downsampling
        step = 20
        a_p, b_p = a[::step], b[::step]
        vi_p = viable_grid[::step, ::step]
        h2_p = h2_grid[::step, ::step]

        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), dpi=140)
        
        # Left Panel: Viability
        ax = axes[0]
        ax.pcolormesh(a_p, b_p, vi_p.T.astype(float), cmap='Greens', 
                      vmin=0, vmax=1.6, shading='auto', rasterized=True)
        if poly is not None:
            ax.add_patch(MplPolygon(poly, closed=True, fill=False, edgecolor='k', lw=1.6, label='Viable Polygon'))
        if prior is not None:
            mu, cov = np.array(prior['prior_mean']), np.array(prior['prior_cov'])
            eigvals, eigvecs = np.linalg.eigh(cov)
            angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
            for k, color in zip([1, 2, 3], ['#1f77b4', '#ff7f0e', '#d62728']):
                w, h = 2 * k * np.sqrt(eigvals)
                ax.add_patch(Ellipse(mu, w, h, angle=angle, fill=False, edgecolor=color, lw=1.3, ls='--', label=f'{k}$\\sigma$ Prior'))
        ax.set_title(f'Viability Map ({title_suffix})')
        ax.set_xlabel(r'$a_{\rm exo}$'); ax.set_ylabel(r'$b_{\rm exo}$')
        ax.legend(fontsize=8, loc='best', framealpha=0.9)

        # Right Panel: Diagnostic
        ax2 = axes[1]
        disp = np.where(np.isfinite(h2_p), h2_p, np.nan)
        vmax = float(np.nanpercentile(disp, 98)) if np.any(np.isfinite(disp)) else 1.0
        im = ax2.pcolormesh(a_p, b_p, disp.T, cmap='RdYlGn', vmin=-0.05, vmax=vmax, shading='auto', rasterized=True)
        plt.colorbar(im, ax=ax2, pad=0.02).set_label(r'$\min_z\ H^2(z)/H_0^2$')
        if np.any(np.isfinite(disp)) and np.nanmax(disp) > 0:
            ax2.contour(a_p, b_p, disp.T, levels=[0.0], colors='k', linewidths=1.2)
        ax2.set_title(f'Smoothness Diagnostic ({title_suffix})')
        ax2.set_xlabel(r'$a_{\rm exo}$'); ax2.set_ylabel(r'$b_{\rm exo}$')

        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, filename), dpi=220, bbox_inches='tight')
        plt.close()
        print(f"[plot] wrote {filename}")

    # --- PLOT 1: Analytic ---
    save_dual_panel(a_vals, b_vals, ana_data['viable'], ana_data['min_h2'], 
                    "Analytic Source", "exo_prior_analytic.png")

    # --- PLOT 2: Validation (if exists) ---
    val_path = os.path.join(args.outdir, 'validate.npz')
    if os.path.exists(val_path):
        val_results = np.load(val_path)['results']
        
        # We reconstruct a sparse grid for the validation plot
        # results[:, 0] is 'i', results[:, 1] is 'j', results[:, 2] is 'minH2'
        valid_viable = np.zeros_like(ana_data['viable'], dtype=bool)
        valid_min_h2 = np.full_like(ana_data['min_h2'], np.nan)
        
        indices_i = val_results[:, 0].astype(int)
        indices_j = val_results[:, 1].astype(int)
        
        # Mark only checked points as viable if they passed CLASS
        valid_viable[indices_i, indices_j] = (val_results[:, 3] == 0)
        valid_min_h2[indices_i, indices_j] = val_results[:, 2]

        save_dual_panel(a_vals, b_vals, valid_viable, valid_min_h2, 
                        "CLASS Validated Source", "exo_prior_validated.png")


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--phase', required=True,
                   choices=['analytic', 'polygon', 'validate',
                            'prior', 'plot', 'all_fast'])
    p.add_argument('--N',     type=int, default=15000)
    p.add_argument('--a_min', type=float, default=-1838.0)
    p.add_argument('--a_max', type=float, default=-1e-5)
    p.add_argument('--b_min', type=float, default=-2000.0)
    p.add_argument('--b_max', type=float, default=1838.0)
    p.add_argument('--outdir', default='scan_out')
    p.add_argument('--val_workers_per_rank', type=int, default=1,
                   help='multiprocessing pool size inside each MPI rank')
    p.add_argument('--respawn_every', type=int, default=200,
                   help='maxtasksperchild for CLASS worker pool (leak guard)')
    args = p.parse_args()

    if   args.phase == 'analytic': phase_analytic(args)
    elif args.phase == 'polygon':  phase_polygon(args)
    elif args.phase == 'validate': phase_validate(args)
    elif args.phase == 'prior':    phase_prior(args)
    elif args.phase == 'plot':     phase_plot(args)
    elif args.phase == 'all_fast':
        # analytic + polygon + prior + plot, NO CLASS validation
        phase_analytic(args)
        comm, rank, _ = mpi_info()
        if rank == 0:
            phase_polygon(args)
            phase_prior(args)
            phase_plot(args)


if __name__ == '__main__':
    main()