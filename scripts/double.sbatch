#!/bin/bash

#SBATCH --job-name=double
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --mem=64GB
#SBATCH --priority=4294967293
#SBATCH --account=class
#SBATCH --out=./out/double.out


module purge;
module load python/intel/3.8.6;

## activate
source venv/bin/activate

## install required packages
pip install -r requirements.txt

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;

python main.py --batch-size=64 --lr=0.00025 --discount-factor=0.99 --num-epochs=50 --memory-size=10000 --frame-repeat=12 --steps-per-epoch=2000 --epsilon-decay=0.9996 --model='double_dqn' --checkpoint="double"
