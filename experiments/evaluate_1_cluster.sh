#!/usr/bin/env bash
#SBATCH --partition=csedu
#SBATCH --gres=gpu:1
#SBATCH --mem=5G
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --output=./logs/slurm/%J.out
#SBATCH --error=./logs/slurm/%J.err
#only use this if you want to send the mail to another team member #SBATCH --mail-user=teammember
#only use this if you want to receive mails on you job status SBATCH --mail-type=BEGIN,END,FAIL

if [ -z "$*" ]; then echo "Please supply a job id, e.g. 1112891"; exit; fi

# location of repository and data
project_dir=.
shard_folder=data/tiny-voxceleb-shards/
shards_dirs_to_evaluate=data/tiny-voxceleb-shards/dev/,data/tiny-voxceleb-shards/eval
trial_lists=data/tiny-voxceleb/dev_trials.txt,data/tiny-voxceleb/eval_trials_no_gt.txt

# hyperparameters related to data pre-processing and network architecture
normalize_channel_wise=true
n_mels=81
embedding_size=512

version=$1
echo "Evaluating for version: $version"

# execute train CLI
source "$project_dir"/venv/bin/activate
i=0;
for checkpoint_path in logs/lightning_logs/version_"$version"/checkpoints/*; do
  [ -e "$checkpoint_path" ] || continue
  name=${checkpoint_path##*/}
  epoch=${name:0:10}
  val_eer=${name:26:14}
  score_file="scores/scores_${version}_${epoch}_${val_eer}_$i"
  echo "Input checkpoint path: $checkpoint_path";
  echo "Output score file: $score_file";
  python "$project_dir"/cli_evaluate.py \
    $checkpoint_path \
    $shard_folder \
    $shards_dirs_to_evaluate \
    $trial_lists \
    $score_file \
    --normalize_channel_wise $normalize_channel_wise \
    --n_mels $n_mels \
    --use-gpu "1"
  ((i++))
done;