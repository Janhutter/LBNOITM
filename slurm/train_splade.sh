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
steps=1
python3 main.py retriever='splade' generator='llama-2-7b-chat' dataset='kilt_nq' train='lora' train.trainer.per_device_train_batch_size=8 train.trainer.per_device_eval_batch_size=16 train.trainer.eval_steps=$steps train.trainer.save_steps=$steps train.trainer.logging_steps=1
