#!/bin/sh
PARTITION=gpu
PYTHON=python

dataset=$1
exp_name=$2
exp_dir=exp/${dataset}/${exp_name}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}.yaml
now=$(date +"%Y%m%d_%H%M%S")

mkdir -p ${model_dir} ${result_dir}
cp seg_tool/train_banet.sh seg_tool/train_banet.py ${config} ${exp_dir}

export PYTHONPATH=./
#sbatch -p $PARTITION --gres=gpu:8 -c16 --job-name=train \
$PYTHON -u seg_tool/train_banet.py \
  --config=${config} \
  2>&1 | tee ${model_dir}/train_banet-$now.log

$PYTHON -u seg_tool/evaluate_banet.py \
  --config=${config} \
  2>&1 | tee ${result_dir}/evaluate_banet-$now.log
