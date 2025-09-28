MASTER_ADDR="$1"
NNODES="$2"
NODE_RANK=$PMI_RANK

# # Libfabric/CXI
# unset NCCL_NET_GDR_LEVEL
# export FI_PROVIDER=cxi
# export FI_CXI_DISABLE_HOST_REGISTER=1
# export FI_MR_CACHE_MONITOR=userfaultfd

# # NCCL over OFI/CXI
# export NCCL_NET="AWS Libfabric"
# export NCCL_IB_DISABLE=1
# export NCCL_CROSS_NIC=1
# export NCCL_COLLNET_ENABLE=0
# export NCCL_SHM_DISABLE=1

# # Force the same non-GDR path NCCL is already using (be explicit)
# export NCCL_NET_GDR_LEVEL=0
# export NCCL_P2P_DISABLE=0
# export NCCL_NVLS_ENABLE=0

# # Usual runtime hygiene
# export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# export CUDA_DEVICE_MAX_CONNECTIONS=1


# # Optional: get verbose proof of what’s being used
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=INIT,NET


# Libfabric / CXI
export FI_PROVIDER=cxi
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_MR_CACHE_MONITOR=userfaultfd

# NCCL over OFI/CXI
export NCCL_NET="AWS Libfabric"
export NCCL_IB_DISABLE=1
export NCCL_CROSS_NIC=1
export NCCL_COLLNET_ENABLE=0
export NCCL_SHM_DISABLE=0         # keep SHM IPC enabled
export NCCL_NET_GDR_LEVEL=0       # be explicit: no GDR on this topo
export NCCL_P2P_DISABLE=0
export NCCL_NVLS_ENABLE=0         # (or omit; default is fine)

# Runtime hygiene
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# export CUDA_DEVICE_MAX_CONNECTIONS=8  # or omit (default)


export MASTER_PORT=10201
export CUDA_VISIBLE_DEVICES=0,1,2,3


APP_DIR="/lus/eagle/projects/SR-APPFL/duo/lora/dp2/dp"

cd "$APP_DIR"


torchrun \
  --nnodes=$NNODES --node_rank=$NODE_RANK \
  --nproc_per_node=4 \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  ds_ds.py \
  --partition-id 0 --num-partitions 1 --num-rounds 1 \
  --model-name /lus/eagle/projects/SR-APPFL/duo/models/llama-3p3-70b-instruct \
  --dataset-name /home/zhangduo4610/CodeAlpaca-20k
