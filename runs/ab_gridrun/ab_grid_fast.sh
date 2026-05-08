#!/bin/bash
#SBATCH --job-name=exo_scan
#SBATCH --output=exo_scan.log
#SBATCH --error=exo_scan.err
#SBATCH --nodelist=nut06
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=2000

# =============================================================================
# exo_prior_scan.sh
# -----------------------------------------------------------------------------
# Runs the 5-phase viability scan + prior fit pipeline.
#
# Phases that use this script:
#   analytic   — vectorized grid scan         (MPI, fast: ~1 min at N=10k)
#   validate   — CLASS on every viable cell   (MPI, slow: hours at N=10k)
#   all_fast   — analytic + polygon + prior + plot, NO CLASS validation
#
# Invocation:
#   PHASE=all_fast NGRID=10000 sbatch exo_prior_scan.slurm
#   PHASE=analytic NGRID=10000 sbatch exo_prior_scan.slurm
#   PHASE=validate             sbatch exo_prior_scan.slurm
#
# After `validate`, on the head node:
#   python exo_prior_scan.py --phase prior --outdir scan_out
#   python exo_prior_scan.py --phase plot  --outdir scan_out
# =============================================================================

set -euo pipefail

module purge
PHASE="${PHASE:-all_fast}"
NGRID="${NGRID:-10000}"
OUTDIR="${OUTDIR:-scan_out}"

module load mpi/openmpi-x86_64
source /home/lustre_p/ahmed.omar/workspace/venvs/exo_DE/bin/activate

# Threading: keep MKL/OpenBLAS quiet so MPI ranks don't oversubscribe cores.
# The analytic phase is single-threaded numpy; CLASS itself is single-threaded.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mkdir -p logs "$OUTDIR"

echo "===================================================================="
echo " PHASE  = $PHASE"
echo " NGRID  = $NGRID"
echo " OUTDIR = $OUTDIR"
echo " NODES  = $SLURM_JOB_NUM_NODES"
echo " TASKS  = $SLURM_NTASKS"
echo " CPUS/T = $SLURM_CPUS_PER_TASK"
echo "===================================================================="

case "$PHASE" in
  analytic | all_fast)
    # mpirun will now correctly use all 16 tasks
    mpirun -np "$SLURM_NTASKS" python3.8 -u ab_grid_fast.py \
      --phase "$PHASE" --N "$NGRID" --outdir "$OUTDIR"
    ;;
  validate)
    # This also needs mpirun for the heavy lift later
    mpirun -np "$SLURM_NTASKS" python3.8 -u ab_grid_fast.py \
      --phase validate --outdir "$OUTDIR" \
      --val_workers_per_rank "$SLURM_CPUS_PER_TASK" \
      --respawn_every 200
    ;;
  *)
    echo "Unknown PHASE=$PHASE." >&2
    exit 1
    ;;
esac

echo "Done: $PHASE"
