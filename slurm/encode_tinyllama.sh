#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --job-name=encode
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=1
#SBATCH --constraint="gpu_40g+"
#SBATCH --constraint="cuda12"
#SBATCH --output=slurm_output/slurm-%j.out
#SBATCH --error=slurm_output/slurm-%j.err
cd ..
nvidia-smi
source ~/.bashrc

CONFIG=retrieve python3 main.py  retriever=tinyllama-chat dataset='kilt_nq'

 
