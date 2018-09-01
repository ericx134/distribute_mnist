#!/bin/bash

if [ $# -ne 2 ] 
then
    echo "usage: run_model.sh <node_rank> <master_ip>"
    exit 1
fi

node_rank=$1
master_ip=$2

echo $node_rank
echo $master_ip

time python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=$node_rank \
    --master_addr=$master_ip \
    --master_port=1234 \
    main.py --batch_size=512


sudo mount -t cifs -o username=ericx,sec=ntlm,domain=NVIDIA.COM,dir_mode=0777,file_mode=0777 //dcg-zfs-04.nvidia.com:/export/ericx.cosmos464 /mnt/ericx_nfs_mount