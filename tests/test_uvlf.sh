#!/bin/bash
#SBATCH --job-name=test_uvlf
#SBATCH --nodelist=nut03
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=1G
#SBATCH --time=12:00:00
#SBATCH --output=test_uvlf.log


source /home/lustre_p/ahmed.omar/workspace/venvs/exo_DE/bin/activate


# ── Environment ───────────────────────────────────────────────────────────────
module purge
module load mpi/openmpi-x86_64

 
PROJECT_ROOT=/home/lustre_p/ahmed.omar/workspace/exo_de_project
OUTPUT_DIR=$PROJECT_ROOT/tests/likelihood_test_outputs

# ── OpenMP: give the C kernel (hmf_sigma.so) all available cores ─────────────
# This parallelises the sigma(M) integration over the 750-point halo mass grid.
# Each likelihood evaluation is serial but the inner C loop is OMP-parallel.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=close
export OMP_PLACES=cores
 
# ── CLASS thread safety ───────────────────────────────────────────────────────
# CLASS is not thread-safe across Python processes, but we run one minimiser
# process at a time so this is fine.  If you later run MPI MCMC, use
# mpirun -n N python ... instead and set OMP_NUM_THREADS=total_cores/N.
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
 
mkdir -p $PROJECT_ROOT/tests/likelihood_test_outputs
 
echo "============================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Cores:     $SLURM_CPUS_PER_TASK"
echo "OMP:       $OMP_NUM_THREADS threads"
echo "Started:   $(date)"
echo "============================================"
 
# ── Run ───────────────────────────────────────────────────────────────────────
# Remove --skip_bobyqa to run the full minimisation suite.
# Add --skip_bobyqa during development to run only the 13 validation tests.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK   # = 4

mpirun -n 4 python3.8 test_uvlf.py \
    --output_dir ${OUTPUT_DIR}/${GROUP:-all}_boby \
    --skip_validation \
    ${GROUP:+--group $GROUP}
#    \ -r
 
echo "============================================"
echo "Finished:  $(date)"
echo "============================================"
 
