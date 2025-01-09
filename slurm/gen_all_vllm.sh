#!/bin/bash

submit_sbatch_job() {
    local args="$1"
    local ngpus="$2"
    local partition="$3"

    echo "$args" 
    sbatch <<EOL
#!/bin/bash
#SBATCH -n 1
#SBATCH -p "$partition"
#SBATCH --job-name=encode
#SBATCH --gres=gpu:$ngpus
#SBATCH --cpus-per-task=4
#SBATCH --mem=1
#SBATCH --constraint="gpu_40g+"
#SBATCH --constraint="cuda12"
#SBATCH --output=slurm_output/%j.out
#SBATCH --error=slurm_output/%j.err
cd ..
source ~/.bashrc

IFS=" " read -ra args <<< "$arguments"

python3 main.py ${args[@]}
EOL
}


batch_sizes=("1024" "32" "32" "8")
generators=("tinyllama-chat" "llama-2-7b-chat" "mixtral-moe-7b-chat" "llama-2-70b-chat")
num_gpus=("1" "1" "1" "2")
partition='gpu'
retrievers=('repllama-7b' "splade" "bm25")


for retriever in ${retrievers[@]};do
    for index in "${!batch_sizes[@]}"; do
        batch_size="${batch_sizes[$index]}"
        generator="${generators[$index]}"
        n_gpu="${num_gpus[$index]}"
        for dataset in kilt_nq kilt_hotpotqa kilt_triviaqa kilt_eli5 kilt_wow; do
            submit_sbatch_job "retriever=${retriever} generator=${generator} generator.batch_size=${batch_size} dataset=${dataset}" $n_gpu $partition
        done
    done
done


