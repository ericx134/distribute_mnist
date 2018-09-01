#!/bin/bash

if [ $# -eq 2 ]; then
    node_rank=$1
    master_ip=$2
elif [ $# -eq 1 ]; then
    node_rank=$1
    master_ip='$(hostname -I)'
else 
    echo "usage: run_multinode.sh <node_rank> <master_ip>"
    exit 1
fi


echo $node_rank
echo $master_ip


ngc batch run \
  --ace nv-us-west-2 \
  --instance ngcv8 \
  --name "distributed test pytorch multi-node No.$node_rank" \
  --image "nvidia/pytorch:18.08-py3" \
  --datasetid 12183:/data/distribute_mnist \
  --result /result \
  --port 1234 \
  --command "hostname -I && cd /data/distribute_mnist && ./run_model.sh $node_rank $master_ip && hostname -I"


# ngc batch run --name "Job-nv-us-west-2-162802" --image "nvidia/pytorch:17.12" --ace nv-us-west-2 --instance ngcv8 --commandline "python -m torch.distributed.launch --nproc_per_node=8 /distribute_mnist/main.py" --result /results --datasetid 12121:/distribute_mnist