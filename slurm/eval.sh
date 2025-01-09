#!/bin/bash
#SBATCH -n 1
#SBATCH -p calmar
#SBATCH -A calmar
#SBATCH --job-name=encode
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=1
##SBATCH --constraint="gpu_40g+"
##SBATCH --constraint="cuda12"
#SBATCH --output=slurm_output/%j.out
#SBATCH --error=slurm_output/%j.err
cd ..
source ~/.bashrc
python3 main.py retriever='repllama-7b' generator='llama-2-7b-chat' generator.batch_size=16 dataset='kilt_nq'
#python3 main.py retriever='repllama-7b' reranker='debertav3' generator='llama-2-7b-chat' generator.batch_size=16 dataset='kilt_nq'

 

