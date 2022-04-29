#!/usr/bin/env bash

# location of repository and data
project_dir=.
data_folder=/kaggle/input/

# execute train CLI
i=0;
for checkpoint_path in /kaggle/temp/models/*.ckpt; do
  [ -e "$checkpoint_path" ] || continue
  python "$project_dir"/cli_evaluate.py \
    $checkpoint_path \
    $data_folder \
    --gpus 1
  ((i++))
done;
