#!/usr/bin/env bash

set -exu

ts=$1

num_features=${2:-500}
feature_dim=${3:-32}
sparsity_every=${4:-100}
sparsity_num_steps=${5:-100}
sparsity_lr=${6:-0.001}
dropouti=${7:-0.4}
dropouth=${8:-0.25}

outdir="exp_out/$ts/feat_f${num_features}_d${feature_dim}_se${sparsity_every}_sns${sparsity_num_steps}_slr${sparsity_lr}_di${dropouti}_dh${dropouth}"


python main.py \
--batch_size 20 \
--data data/penn \
--dropouti 0.4 \
--dropouth 0.25 \
--seed 141 \
--epoch 500 \
--save $outdir/model.pt \
--cuda \
--num_features 500 \
--feature_dim 32 \
--sparsity_every 100 \
--sparsity_num_steps 100 \
--sparsity_lr 0.001 \
--sparsity_bias 0.001