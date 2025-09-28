#!/bin/bash
# ===== Polaris PBS job for Flower Deployment Engine =====
#PBS -A SR-APPFL                  
#PBS -q debug-scaling                 
#PBS -l select=3:system=polaris
#PBS -l place=scatter
#PBS -l walltime=00:30:00
#PBS -l filesystems=home:eagle
#PBS -j oe
#PBS -V
#PBS -N flwr-deploy-engine
echo "begin"

## Generate a list of hostnames allocated for the job
cat $PBS_NODEFILE |cut -d "." -f 1 > /home/zhangduo4610/myhostnames
cat /home/zhangduo4610/myhostnames
export HOST_NUM=$(wc -l < $PBS_NODEFILE)
echo "Number of Nodes: $HOST_NUM"
export MASTER_ADDR=$(hostname -I | awk '{print $1}')
export MASTER_PORT=9092
# -------------------------------
# User-configurable parameters
# -------------------------------
NCLIENTS=$((HOST_NUM - 1))      # Number of client nodes (export before qsub or edit here)



# Your environment (as provided)
ENV_NAME_PATH="/lus/eagle/projects/SR-APPFL/duo/env/new_flwr"
APP_DIR="/lus/eagle/projects/SR-APPFL/duo/lora/dp2"

# -------------------------------
# Polaris environment setup
# -------------------------------
# Make sure modules work in batch
#source /etc/profile

module use /soft/modulefiles
module load gcc-native/12.3
module load conda/2024-04-29
module load cudatoolkit-standalone/12.4.0
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

conda activate "$ENV_NAME_PATH"

cd "$APP_DIR"

LOGDIR="$APP_DIR/logs_${PBS_JOBID}"
mkdir -p "$LOGDIR"
## Generate a list of hostnames allocated for the job
cat $PBS_NODEFILE |cut -d "." -f 1 > /home/zhangduo4610/myhostnames
cat /home/zhangduo4610/myhostnames
export HOST_NUM=$(wc -l < $PBS_NODEFILE)
echo "Number of Nodes: $HOST_NUM"
#

mapfile -t HOSTS < /home/zhangduo4610/myhostnames
SERVER_HOST="${HOSTS[0]}"
CLIENT_HOSTS=("${HOSTS[@]:1}")
NCLIENTS="${#CLIENT_HOSTS[@]}"

echo "[INFO] Server host:  ${SERVER_HOST}"
echo "[INFO] Client hosts: ${CLIENT_HOSTS[*]}"

## Launch distributed jobs
echo "[INFO] Launching SuperLink on Server ..."
#mpiexec -n 1 bash /lus/eagle/projects/SR-APPFL/duo/lora/server.sh >"$LOGDIR/superlink.out" 2>"$LOGDIR/superlink.err" &
mpiexec -host "${SERVER_HOST}" -n 1 \
  bash /lus/eagle/projects/SR-APPFL/duo/lora/server.sh "${MASTER_PORT}" \
  > "${LOGDIR}/superlink.out" 2> "${LOGDIR}/superlink.err" &

sleep 10

# for i in $(seq 1 $NCLIENTS); do
#   echo "[INFO] Launching Client $i ..."
#   mpiexec -n 1 bash /lus/eagle/projects/SR-APPFL/duo/lora/client.sh $MASTER_ADDR $MASTER_PORT $NCLIENTS $i > "$LOGDIR/supernode_${i}.out" 2>"$LOGDIR/supernode_${i}.err" &
# done
# --- Launch one client per remaining host ---
cid=1
for H in "${CLIENT_HOSTS[@]}"; do
  echo "[INFO] Launching Client ${cid} on ${H}"
  mpiexec -host "${H}" -n 1 \
    bash /lus/eagle/projects/SR-APPFL/duo/lora/client.sh \
      "${MASTER_ADDR}" "${MASTER_PORT}" "${NCLIENTS}" "${cid}" \
    > "${LOGDIR}/supernode_${cid}.out" 2>&1 &
  cid=$((cid+1))
done

wait

echo "Jobs Done"