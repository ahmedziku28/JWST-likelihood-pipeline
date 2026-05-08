#!/bin/bash
#SBATCH --job-name=2d_grid
#SBATCH --nodelist=nut03
#SBATCH --output=scan_%j.out
#SBATCH --error=scan_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=2000
echo "======================================================"
echo "Job started on $(hostname) at $(date)"
echo "======================================================"
module purge
source /home/lustre_p/ahmed.omar/workspace/venvs/exo_DE/bin/activate
echo "Using: $(which python)"
python -u grid_run.py
echo "Job finished at $(date)"