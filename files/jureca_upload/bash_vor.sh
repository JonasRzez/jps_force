#!/bin/bash -x
#SBATCH -J db
#SBATCH --account=jias70
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=12
#SBATCH --output=out.%j
#SBATCH --error=err.%j
#SBATCH --time=24:00:00
#SBATCH --mail-user=j.rzezonka@fz-juelich.de
#SBATCH --mail-type=ALL

module load Python
module load SciPy-Stack/2019a-Python-3.6.8

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

python ini_voronoi.py
wait
srun sh -c 'python vor_$(($SLURM_PROCID+1)).py'
wait
python errorplot.py

rm vor_*
