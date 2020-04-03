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
--feature_dim 16