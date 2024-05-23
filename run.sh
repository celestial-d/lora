#!/bin/bash

mkdir /home/cc/output/
echo "server begin"

python3 server.py &
sleep 5

echo "client begin" 

CUDA_VISIBLE_DEVICES="0" python3 client.py --partition-id=0 --task_id=0 &
CUDA_VISIBLE_DEVICES="1" python3 client.py --partition-id=1 --task_id=1 &


