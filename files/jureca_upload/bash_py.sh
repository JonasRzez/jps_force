#!/bin/bash -x
#SBATCH -J db
#SBATCH --account=jias70
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --output=out.%j
#SBATCH --error=err.%j
#SBATCH --time=00:45:00
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

python waiting_time_err.py 
#python density_map.py 
#python trajectory_vornoi.py
#python angles.py 
#python graph.py
#python flow.py
