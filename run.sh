#!/bin/bash

mkdir /home/cc/output/
echo "server begin"

python3 server.py &
#python llmserver.py
sleep 5

echo "client begin" 

CUDA_VISIBLE_DEVICES="0" python client.py --partition-id=0 --task_id=0 &
#CUDA_VISIBLE_DEVICES="0" python llmclient.py --partition-id=0 --task_id=0 
CUDA_VISIBLE_DEVICES="1" python client.py --partition-id=1 --task_id=1 &
#CUDA_VISIBLE_DEVICES="1" python llmclient.py --partition-id=1 --task_id=1

#sudo CUDA_VISIBLE_DEVICES="0" nvprof python3 client.py --partition-id=0 --task_id=0
