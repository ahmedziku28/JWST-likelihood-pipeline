#!/bin/bash
#SBATCH --job-name=ab_grid
#SBATCH --nodelist=nut03
#SBATCH --output=ab_grid.log
#SBATCH --error=ab_grid.err
#SBATCH --ntasks=1
#SBATCH --time=54:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2000


source /home/lustre_p/ahmed.omar/workspace/venvs/exo_DE/bin/activate


python3.8 ab_grid.py