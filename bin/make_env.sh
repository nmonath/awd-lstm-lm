#!/usr/bin/env bash

set -exu

conda create -n awd-lstm pip python=3.6

conda activate awd-lstm

conda install pytorch=0.4.0 cuda90 -c pytorch

pip install absl-py