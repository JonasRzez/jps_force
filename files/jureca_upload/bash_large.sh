#!/bin/bash -x
#SBATCH -J db
#SBATCH --account=jias70
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --output=out.%j
#SBATCH --error=err.%j
#SBATCH --time=15:00:00
#SBATCH --mail-user=j.rzezonka@fz-juelich.de
#SBATCH --mail-type=ALL

module load CMake
module load GCC
module load OpenMPI
module load Boost
module load Python
module load GCCcore/.8.3.0
module load SciPy-Stack/2019a-Python-3.6.8

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

rm -r logfile
mkdir logfile

python ini_mc_large_i.py
wait
srun sh -c 'python mc_l_$(($SLURM_PROCID+1)).py'
#wait
#sbatch bash_py.sh

rm mc_l_*
