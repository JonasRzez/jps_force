#!/bin/bash -x
#SBATCH -J db
#SBATCH --account=jias70
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=12
#SBATCH --output=out.%j
#SBATCH --error=err.%j
#SBATCH --time=00:08:00
#SBATCH --mail-user=j.rzezonka@fz-juelich.de
#SBATCH --mail-type=ALL

module load CMake
module load GCC
module load ParaStationMPI
module load Boost
module load Python
module load GCCcore/.8.3.0
module load SciPy-Stack/2019a-Python-3.6.8

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

python folder_ini.py
srun sh -c 'python mc_$(($SLURM_PROCID+1)).py'
python trajectory_modify.py


