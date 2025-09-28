#/bin/bash
# ds_ds_2nodes_4gpp.pbs
# ===== Polaris PBS job: 2 nodes, 4 GPUs per node (8 ranks total) =====
#PBS -A SR-APPFL
#PBS -q prod
#PBS -l select=32:system=polaris
#PBS -l place=scatter
#PBS -l walltime=02:00:00
#PBS -l filesystems=home:eagle
#PBS -N cds_ds_64
#PBS -j oe
#PBS -V

set -euo pipefail

echo "Nodefile:"
cat "$PBS_NODEFILE"

# --- Activate your environment ---
ENV_NAME_PATH="/lus/eagle/projects/SR-APPFL/duo/env/new_flwr"
APP_DIR="/lus/eagle/projects/SR-APPFL/duo/lora/dp2"

# -------------------------------
# Polaris environment setup
# -------------------------------
# Make sure modules work in batch
#source /etc/profile

module use /soft/modulefiles
module load gcc-native/12.3
#module load conda/2024-04-29
module load conda/2024-04-29-aws-nccl
module load cudatoolkit-standalone/12.4.0
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

conda activate "$ENV_NAME_PATH"

# --- Job parameters (edit as needed) ---
MODEL_PATH="/home/zhangduo4610/opt-125m"
DATASET_PATH="/home/zhangduo4610/CodeAlpaca-20k"
SCRIPT_PATH="/lus/eagle/projects/SR-APPFL/duo/LLM-trl/ds_ds.py"

# Use the first node as master
HEADNODE=$(head -n1 "$PBS_NODEFILE")
export MASTER_ADDR="$HEADNODE"
export MASTER_PORT=10201
export NNODES=$(wc -l < "$PBS_NODEFILE")
export NPROC_PER_NODE=4
export CUDA_VISIBLE_DEVICES=0,1,2,3

# (Optional) helpful NCCL debugging flags if comms act up
# export NCCL_DEBUG=INFO
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_SOCKET_IFNAME=^lo,docker0
# export NCCL_IB_DISABLE=1   # last resort: force TCP if IB is problematic

mpiexec -n $NNODES bash /lus/eagle/projects/SR-APPFL/duo/lora/ds.sh $MASTER_ADDR $NNODES
