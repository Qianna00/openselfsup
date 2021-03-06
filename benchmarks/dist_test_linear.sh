
#!/usr/bin/env bash

set -x

CFG=$1 # use cfgs under "configs/benchmarks/linear_classification/"
GPUS=$2
CHECKPOINT=$3
WORK_DIR=$4
PORT=${PORT:-2001}


# WORK_DIR="$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/$(echo $PRETRAIN | rev | cut -d/ -f 1 | rev)"
# WORK_DIR="$(dirname $CHECKPOINT)/"

# test

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    tools/test.py \
    $CFG \
    $CHECKPOINT \
    --work_dir $WORK_DIR --launcher="pytorch"
