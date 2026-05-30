#!/bin/bash
#SBATCH --job-name=builder_lcdm_vshmr
#SBATCH --nodes=1
#SBATCH --exclude=lustre,cernnode02,cernnode03,nut01,nut02
#SBATCH --output=builder_lcdm_vshmr.log
#SBATCH --error=builder_lcdm_vshmr.err
#SBATCH --ntasks=4
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5000

echo "======================================================"
echo "Covmat builder builder_lcdm_vshmr started on $(hostname) at $(date)"
echo "======================================================"

module purge
module load mpi/openmpi-x86_64

source /opt/rh/devtoolset-8/enable
source /home/lustre_p/ahmed.omar/workspace/venvs/exo_DE/bin/activate

export XDG_CONFIG_HOME="/home/lustre_p/ahmed.omar/workspace/Modules/cobaya_stuff/cobaya_config"
export XDG_CACHE_HOME="/home/lustre_p/ahmed.omar/workspace/Modules/cobaya_stuff/cobaya_cache"
export COBAYA_PACKAGES_PATH='/home/lustre_p/ahmed.omar/workspace/Modules/cobaya_stuff/cobaya_packages'
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Checking which CLASS is loaded..."
python -c "import classy; print('Loaded CLASS from:', classy.__file__)"

# Builders are short; a single attempt with light retry on signal kills.
max_retries=10
retry_count=0
command="mpirun -np $SLURM_NTASKS cobaya-run builder_lcdm_vshmr.yaml --resume"

until $command; do
  exit_code=$?
  retry_count=$((retry_count + 1))
  if [ $exit_code -eq 139 ] || [ $exit_code -eq 134 ] || [ $exit_code -eq 137 ]; then
    echo "Signal kill (exit $exit_code) at $(date). Resetting retry counter."
    retry_count=0
  fi
  if [ $retry_count -ge $max_retries ]; then
    echo "FATAL: $max_retries consecutive failures. Proceeding to covmat dump anyway."
    break
  fi
  echo "Crash #${retry_count}/${max_retries} (exit $exit_code). Retrying in 10s..."
  sleep 10
done

# ── Explicit covmat dump ──────────────────────────────────────────────
# cobaya writes outputs/builder_lcdm_vshmr.covmat at convergence-check cycles. If the run
# stopped via max_samples or walltime mid-cycle, that file may be stale or
# missing. This fallback recomputes the proposal covariance directly from the
# chains via getdist and writes it in cobaya .covmat format (header line of
# sampled parameter names, then the matrix). cobaya reads this back fine.
echo "Dumping covmat from chains (fallback)..."
python - <<'PYDUMP'
import os, sys
import numpy as np
chain_root = os.path.join('outputs', 'builder_lcdm_vshmr')
try:
    from getdist import loadMCSamples
    samples = loadMCSamples(chain_root, settings={'ignore_rows': 0.3})
    names = [p.name for p in samples.paramNames.names if not p.isDerived]
    cov = np.atleast_2d(np.asarray(samples.cov(pars=names)))
    out_path = chain_root + '.covmat'
    with open(out_path, 'w') as f:
        f.write('# ' + ' '.join(names) + '\n')
        for row in cov:
            f.write(' '.join('{:.8e}'.format(x) for x in row) + '\n')
    print('Wrote covmat ({}x{}) for params {}'.format(cov.shape[0], cov.shape[1], names))
    print('  -> ' + out_path)
except Exception as e:
    print('Covmat dump FAILED: {}'.format(e), file=sys.stderr)
    sys.exit(0)
PYDUMP

echo "======================================================"
echo "Builder builder_lcdm_vshmr finished at $(date). Covmat at outputs/builder_lcdm_vshmr.covmat"
echo "======================================================"
