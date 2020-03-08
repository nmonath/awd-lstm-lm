#!/usr/bin/env bash

set -exu

mem=${1:-12000}
threads=${2:-4}
gpus=${3:-1}
partition=${4:-1080ti-short}

TIME=`(date +%Y-%m-%d-%H-%M-%S)`

export MKL_NUM_THREADS=$threads
export OPENBLAS_NUM_THREADS=$threads
export OMP_NUM_THREADS=$threads

log_dir=logs/inf/$TIME

mkdir -p $log_dir

dataset="ptb"
job_name="awd-lstm-$dataset-$TIME"

sbatch -J $job_name \
            -e $log_dir/inf.err \
            -o $log_dir/inf.log \
            --cpus-per-task $threads \
            --partition=$partition \
            --gres=gpu:$gpus \
            --ntasks=1 \
            --nodes=1 \
            --mem=$mem \
            --time=0-08:00 \
            bin/run_ptb.sh