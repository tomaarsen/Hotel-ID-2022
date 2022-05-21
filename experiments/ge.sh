#!/usr/bin/env bash
#SBATCH --partition=csedu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=6
#SBATCH --time=24:00:00
#SBATCH --output=./logs/slurm/%J.out
#SBATCH --error=./logs/slurm/%J.err
#only use this if you want to send the mail to another team member #SBATCH --mail-user=teammember
#only use this if you want to receive mails on you job status
#SBATCH --mail-type=BEGIN,END,FAIL

if [ -z "$*" ]; then echo "Please supply a checkpoint path"; exit; fi

# location of repository and data
project_dir=.
data_folder="$project_dir"/data/data_task2/hotel-id-to-combat-human-trafficking-2022-fgvc9

python "$project_dir"/generate_embeddings.py \
    --checkpoint_path $1 \
    --data_folder $data_folder