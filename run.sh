#! /bin/bash



mkdir -p ./job-outs/

if [ -f ./bash/sleep.sh ]; then
	rm ./bash/sleep.sh
fi

mkdir -p ./bash/

touch ./bash/sleep.sh

tee -a ./bash/sleep.sh <<EOF
#! /bin/bash

######## login
#SBATCH --job-name=sleep
#SBATCH --output=./job-outs/sleep.out
#SBATCH --error=./job-outs/sleep.err

#SBATCH --account=pi-lhansen
#SBATCH --partition=standard
#SBATCH --cpus-per-task=5
#SBATCH --mem=10G
#SBATCH --time=7-00:00:00
#SBATCH --exclude=mcn53

####### load modules
module load python/booth/3.8  gcc/9.2.0

echo "\$SLURM_JOB_NAME"

echo "Program starts \$(date)"
start_time=\$(date +%s)
# perform a task

python3 -u /home/bcheng4/LocAsset/benchmark.py --train_from_scratch
python3 -u /home/bcheng4/LocAsset/benchmark2.py 

echo "Program ends \$(date)"
end_time=\$(date +%s)

# elapsed time with second resolution
elapsed=\$((end_time - start_time))

eval "echo Elapsed time: \$(date -ud "@\$elapsed" +'\$((%s/3600/24)) days %H hr %M min %S sec')"

EOF
# count=$(($count + 1))
sbatch ./bash/sleep.sh
