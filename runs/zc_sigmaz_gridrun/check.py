import numpy as np
import os

fpath = 'runs/grid_scan_v2_results.npz'
print(f"Looking for: {os.path.abspath(fpath)}")
print(f"Exists: {os.path.exists(fpath)}")

if not os.path.exists(fpath):
    # Try current directory
    fpath = 'grid_scan_v2_results.npz'
    print(f"Trying: {os.path.abspath(fpath)}")
    print(f"Exists: {os.path.exists(fpath)}")

if not os.path.exists(fpath):
    # List what's here
    print("\nFiles in current directory:")
    for f in os.listdir('.'):
        print(f"  {f}")
    if os.path.isdir('runs'):
        print("\nFiles in runs/:")
        for f in os.listdir('runs'):
            print(f"  {f}")
    import sys
    sys.exit(1)

data = np.load(fpath)
print(f"\nLoaded. Keys: {list(data.keys())}")

ratios  = data['ratios']
status  = data['status']
z_c_vals    = data['z_c_values']
sig_z_vals  = data['sigma_z_values']
omx_vals    = data['omega_x0_values']
z_probe     = data['z_probe']

print(f"Ratios shape: {ratios.shape}")
print(f"Available probe redshifts: {z_probe}")
print(f"Omega_x0 values: {omx_vals}")
print(f"Status counts: ok={np.sum(status==1)}, skip={np.sum(status==-1)}, fail={np.sum(status==-2)}")

for z_target in [8.5, 10.5, 12.5, 14.5]:
    iz = np.argmin(np.abs(z_probe - z_target))
    k = min(4, len(omx_vals) - 1)
    ok = (status[:, :, k] == 1)
    r = np.where(ok, ratios[:, :, k, iz], -np.inf)

    if np.all(r == -np.inf):
        print(f"\n  Probe z = {z_probe[iz]:.1f}: NO valid runs for Omega_x0 = {omx_vals[k]:.0e}")
        continue

    best_flat = np.argmax(r)
    bi, bj = np.unravel_index(best_flat, r.shape)

    iz85 = np.argmin(np.abs(z_probe - 8.5))
    print(f"\n  Probe z = {z_probe[iz]:.1f}:")
    print(f"    Best z_c     = {z_c_vals[bi]:.1f}")
    print(f"    Best sigma_z = {sig_z_vals[bj]:.2f}")
    print(f"    Ratio        = {ratios[bi, bj, k, iz]:.6f}")
    print(f"    Ratio(z=8.5) = {ratios[bi, bj, k, iz85]:.6f}")