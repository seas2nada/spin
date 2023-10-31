#!/bin/bash

exp_name=$1
exp_dir=$2
config=$3
# config=config/spin_fz_kbias.yaml

mkdir -p $exp_dir

echo "Name: $exp_name"
echo "Config: $config"

python3 run_task.py \
    SpinPretrainTask \
    --config $config \
    --save-path $exp_dir/$exp_name \
    --gpus 1 \
    --njobs 16
