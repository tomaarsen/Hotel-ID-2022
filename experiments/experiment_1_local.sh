#!/usr/bin/env bash

# location of repository and data
project_dir=.
data_folder=F:/MLiPHotel-IDData/Hotel-ID-2022

# hyperparameters for optimization
batch_size=4
learning_rate=0.1
num_epochs=120
momentum=0.9
weight_decay=0.0005
min_lr=0.0

# hyperparameters related to data pre-processing and network architecture
embedding_size=512
num_workers=0

# execute train CLI
python "$project_dir"/cli_train.py \
  --data_folder $data_folder \
  --batch_size $batch_size \
  --num_workers $num_workers \
  --embedding_size $embedding_size \
  --learning_rate $learning_rate \
  --epochs $num_epochs \
  --momentum $momentum \
  --weight_decay $weight_decay \
  --min_lr $min_lr
  # --gpus 0