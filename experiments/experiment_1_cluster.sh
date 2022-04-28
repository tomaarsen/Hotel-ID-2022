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
data_folder=./scratch/tberns/tiny-voxceleb-shards/
val_trials_path=./data/tiny-voxceleb/val_trials.txt
dev_trials_path=./data/tiny-voxceleb/dev_trials.txt

# hyperparameters for optimization
batch_size=4
width=512
height=512
learning_rate=0.1
num_epochs=120
momentum=0.9
weight_decay=0.0005
min_lr=0.0

# hyperparameters related to data pre-processing and network architecture
audio_length_seconds=3
normalize_channel_wise=true
n_mels=81
embedding_size=512
num_workers=6

# execute train CLI
source "$project_dir"/venv/bin/activate
python "$project_dir"/cli_train.py \
  --shard_folder $shard_folder \
  --val_trials_path $val_trials_path \
  --dev_trials_path $dev_trials_path \
  --batch_size $batch_size \
  --num_workers $num_workers \
  --audio_length_seconds $audio_length_seconds \
  --normalize_channel_wise $normalize_channel_wise \
  --n_mels $n_mels \
  --embedding_size $embedding_size \
  --learning_rate $learning_rate \
  --epochs $num_epochs \
  --momentum $momentum \
  --weight_decay $weight_decay \
  --min_lr $min_lr