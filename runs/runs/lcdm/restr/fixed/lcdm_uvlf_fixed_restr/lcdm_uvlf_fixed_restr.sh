#!/bin/bash
#SBATCH --job-name=lcdm_uvlf_fixed_restr
#SBATCH --nodes=1
#SBATCH --exclude=lustre,cernnode02,cernnode03,nut01,nut02
#SBATCH --output=lcdm_uvlf_fixed_restr.log
#SBATCH --error=lcdm_uvlf_fixed_restr.err
#SBATCH --ntasks=8
#SBATCH --time=175:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5000

echo "======================================================"
echo "Job started on $(hostname) at $(date)"
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

# ── Auto-retry loop ──────────────────────────────────────────────────
max_retries=50
retry_count=0
command="mpirun -np $SLURM_NTASKS cobaya-run lcdm_uvlf_fixed_restr.yaml --resume"

until $command; do
  exit_code=$?
  retry_count=$((retry_count + 1))

  # Segfault / signal kill = transient HPC crash, reset counter
  if [ $exit_code -eq 139 ] || [ $exit_code -eq 134 ] || [ $exit_code -eq 137 ]; then
    echo "Signal kill (exit code $exit_code) at $(date). Resetting retry counter."
    retry_count=0
  fi

  if [ $retry_count -ge $max_retries ]; then
    echo "FATAL: $max_retries consecutive non-signal failures. Giving up."
    exit 1
  fi

  echo "Crash #${retry_count}/${max_retries} (exit code $exit_code). Retrying in 15s..."
  sleep 15
done

echo "======================================================"
echo "Run converged at $(date) after $retry_count crash(es)."
echo "======================================================"
