#!/usr/bin/env bash

set -exu

python main.py \
--batch_size 20 \
--data data/penn \
--dropouti 0.4 \
--dropouth 0.25 \
--seed 141 \
--epoch 500 \
--save PTB.pt \
--cuda \
--num_features 500 \
--feature_dim 32 \
--sparsity_every 100 \
--sparsity_num_steps 100 \
--sparsity_lr 0.001