#!/bin/bash
#SBATCH --job-name=background
#SBATCH --nodelist=nut06
#SBATCH --output=background.log
#SBATCH --error=background.err
#SBATCH --ntasks=8
#SBATCH --time=54:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5000

echo "======================================================"
echo "Job started on $(hostname) at $(date)"
echo "======================================================"


source /opt/rh/devtoolset-9/enable

module purge
module load mpi/openmpi-x86_64

# ONLY use the compute-node-shared path
source /home/lustre_p/ahmed.omar/workspace/venvs/exo_DE/bin/activate

# echo "Python = $(which python)"

export XDG_CONFIG_HOME="/home/lustre_p/ahmed.omar/workspace/Modules/cobaya_stuff/cobaya_config"
export XDG_CACHE_HOME="/home/lustre_p/ahmed.omar/workspace/Modules/cobaya_stuff/cobaya_cache"
export COBAYA_PACKAGES_PATH='/home/lustre_p/ahmed.omar/workspace/Modules/cobaya_stuff/cobaya_packages'
# mkdir -p $XDG_CONFIG_HOME $XDG_CACHE_HOME




#remove -f if resume and add -r
#keep/add -f if force overwrite and remove -r
# Add this inside your .sh file to force it to check
echo "Checking which CLASS is loaded..."
python -c "import classy; print('Loaded CLASS from:', classy.__file__)"

# mpirun -n 4 cobaya-run background.yaml -f --debug # smoke test run

mpirun -n 8 cobaya-run background.yaml -f --debug

echo "Job finished at $(date)"
