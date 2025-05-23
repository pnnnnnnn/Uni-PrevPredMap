#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
# PORT=${PORT:-29503}
PORT=$4

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:5} --eval chamfer
