#!/bin/bash

if [ $# -ne 1 ]
then
    echo "usage: run_ngc_job.sh <batch_size>"
    exit 1
fi

batch_size=$1

ngc batch run \
  --ace nv-us-west-2 \
  --instance ngcv8 \
  --name "distributed test pytorch multi-node" \
  --image "nvidia/pytorch:18.08-py3" \
  --datasetid 12180:/data/distribute_mnist \
  --result /result \
  --command "cd /data/distribute_mnist && ./run_model.sh $batch_size && hostname -I"


#ngc batch run --name "Job-nv-us-west-2-162802" --image "nvidia/pytorch:17.12" --ace nv-us-west-2 --instance ngcv8 --commandline "python -m torch.distributed.launch --nproc_per_node=8 /distribute_mnist/main.py" --result /results --datasetid 12121:/distribute_mnist