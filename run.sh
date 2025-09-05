#installation
pip install -e .
pip install pydantic==2.11.7
copy ds_zero3_offload.json from the repo

#host
flower-superlink --insecure

#client1, pleae modify the ip address
export CUDA_VISIBLE_DEVICES=0,1
export DEEPSPEED_CONFIG_FILE=ds_zero3_offload.json
flower-supernode \
     --insecure \
     --superlink 10.1.0.4:9092 \
     --clientappio-api-address 127.0.0.1:9094 \
     --node-config "partition-id=0 num-partitions=2"

#client2, pleae modify the ip address
export CUDA_VISIBLE_DEVICES=0,1
export DEEPSPEED_CONFIG_FILE=ds_zero3_offload.json
flower-supernode \
     --insecure \
     --superlink 10.1.0.4:9092 \
     --clientappio-api-address 127.0.0.1:9095 \
     --node-config "partition-id=1 num-partitions=2"

#go to the folder and then run
flwr run . local-deployment --stream

deepspeed="ds_zero3_offload.json",