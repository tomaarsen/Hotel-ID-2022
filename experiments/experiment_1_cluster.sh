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

# location of repository and data
project_dir=.
data_folder="$project_dir"/data/data_task2/hotel-id-to-combat-human-trafficking-2022-fgvc9


# hyperparameters for optimization
batch_size=16
width=512
height=512
learning_rate=0.001
num_epochs=20
momentum=0.9
weight_decay=0.0005
min_lr=0.0
backbone="eca_nfnet_l2"

# hyperparameters related to data pre-processing and network architecture
embedding_size=4096
num_workers=4

# execute train CLI
source "$project_dir"/venv/bin/activate
python "$project_dir"/cli_train.py \
  --data_folder $data_folder \
  --backbone $backbone \
  --batch_size $batch_size \
  --width $width \
  --height $height \
  --num_workers $num_workers \
  --embedding_size $embedding_size \
  --learning_rate $learning_rate \
  --epochs $num_epochs \
  --momentum $momentum \
  --weight_decay $weight_decay \
  --min_lr $min_lr \
  --gpus 1