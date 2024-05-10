#!/bin/bash

echo "server begin"

python3 server.py &
sleep 5

echo "client begin" 

python3 client.py --partition-id=0 --task_id=0 &
python3 client.py --partition-id=1 --task_id=1 &


