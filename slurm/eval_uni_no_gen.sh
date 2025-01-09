#!/bin/bash
#SBATCH -n 1
#SBATCH -p calmar
#SBATCH -A calmar
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=1
##SBATCH --constraint="gpu_40g+"
##SBATCH --constraint="cuda12"
#SBATCH --output=slurm_output/%j.out
#SBATCH --error=slurm_output/%j.err
cd ..
source ~/.bashrc
python3 main.py generator='llama-2-7b-chat-uni' generator.batch_size=256 dataset='kilt_nq' # generator.init_args.model_name="experiments_emb/75528dd5aebdd82c/train/checkpoint-170/"
 

