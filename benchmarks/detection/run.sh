#!/bin/bash
DET_CFG=$1
WEIGHTS=$2

python $(dirname "$0")/train_smd.py --config-file $DET_CFG \
    --num-gpus 4 MODEL.WEIGHTS $WEIGHTS
