#!/bin/bash

MASTER_ADDR=$1
MASTER_PORT=$2
Nodes=$3
CID=$4
export CUDA_VISIBLE_DEVICES=0,1,2,3
CLIENT_ADDR=$(hostname -I | awk '{print $1}')
echo "Client IP: $CLIENT_ADDR"
LOCAL_PORT=$((9093 + PARTITION_ID))
PARTITION_ID=$((CID - 1))

flower-supernode \
  --insecure \
  --superlink "${MASTER_ADDR}:${MASTER_PORT}" \
  --clientappio-api-address "127.0.0.1:${LOCAL_PORT}" \
  --node-config "partition-id=${PARTITION_ID} num-partitions=${Nodes}" \
