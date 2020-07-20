#!/bin/bash
#SBATCH --job-name=reddit_lm
#SBATCH --output=train_logs/relationships_output.txt
#SBATCH --mail-user=schneider@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL
#SBATCH --partition=students
#SBATCH --time=2-0:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32000
#SBATCH --ntasks=1
#SBATCH --gres=gpu:mem11g:1

# JOB STEPS
srun -u python train.py configs/relationships.yaml

