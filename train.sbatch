#!/bin/bash
#SBATCH --output="train.log"
#SBATCH --job-name="denoise"
#SBATCH --time 07:00:00
#SBATCH --partition=vulcan-ampere
#SBATCH --gres=gpu:rtxa4000:4
#SBATCH --qos=vulcan-high
#SBATCH --account=vulcan-abhinav
#SBATCH --mem=32GB

export WORKDIR=/vulcanscratch/daniekpo/repos/CMSC848B_Project

cd $WORKDIR
conda activate task-planning
python ./main.py --img-list-path="./data/imagenet_files_vulcan.txt" --subset-size 10000 --batch-size 64 --n-epochs 10 --log-every 1 
