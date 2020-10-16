python ini_mc.py
wait
python mc_$(($SLURM_PROCID+1)).py &
wait
sbatch bash_py.sh

rm mc_*



