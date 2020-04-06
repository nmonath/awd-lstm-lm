#!/usr/bin/env bash

set -exu

partition=${1:-1080ti-short}
mem=${2:-12000}
threads=${3:-4}
gpus=${4:-1}

num_features=${5:-500}
feature_dim=${6:-32}
sparsity_every=${7:-100}
sparsity_num_steps=${8:-100}
sparsity_lr=${9:-0.001}

TIME=`(date +%Y-%m-%d-%H-%M-%S)`

export MKL_NUM_THREADS=$threads
export OPENBLAS_NUM_THREADS=$threads
export OMP_NUM_THREADS=$threads



dataset="ptb"
model_name="awd-lstm-feat"
job_name="$model_name-$dataset-$TIME"
log_dir=logs/$model_name/$dataset/$TIME
log_base=$log_dir/$job_name
mkdir -p $log_dir

sbatch -J $job_name \
            -e $log_base.err \
            -o $log_base.log \
            --cpus-per-task $threads \
            --partition=$partition \
            --gres=gpu:$gpus \
            --ntasks=1 \
            --nodes=1 \
            --mem=$mem \
            --time=0-04:00 \
            bin/run_ptb_feat.sh $TIME $num_features $feature_dim $sparsity_every $sparsity_num_steps $sparsity_lr