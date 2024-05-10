#!/bin/bash
echo "Hello World !"

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/scratch/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/scratch/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/scratch/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/scratch/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate FL-PyT-LLMFT
screen -dmS master 
screen -x -S master -p 0 -X stuff "conda activate FL-PyT-LLMFT\n"
#screen -x -S master -p 0 -X stuff "export NCCL_P2P_DISABLE=1\n"
screen -x -S master -p 0 -X stuff "CUDA_VISIBLE_DEVICES="0" python3 /home/zhazhang/Code/FL_LLMFT/sent/server.py\n"

#conda activate FL-PyT-LLMFT
#screen -dmS worker_1 
#screen -x -S worker_1 -p 0 -X stuff "conda activate FL-PyT-LLMFT\n"
#screen -x -S worker_1 -p 0 -X stuff "export NCCL_P2P_DISABLE=1\n"
#screen -x -S worker_1 -p 0 -X stuff "CUDA_VISIBLE_DEVICES="0" python3 /home/hpcpuzrzhang/xgpu-scratch/Code/FL_LLMFT/Fed_LoRA/client.py --partition-id=0\n"

#conda activate FL-PyT-LLMFT
#screen -dmS worker_2 
#screen -x -S worker_2 -p 0 -X stuff "conda activate FL-PyT-LLMFT\n"
#screen -x -S worker_2 -p 0 -X stuff "export NCCL_P2P_DISABLE=1\n"
#screen -x -S worker_2 -p 0 -X stuff "CUDA_VISIBLE_DEVICES="1" python3 /home/hpcpuzrzhang/xgpu-scratch/Code/FL_LLMFT/Fed_LoRA/client.py --partition-id=1\n"

#CUDA_VISIBLE_DEVICES=""CUDA_LAUNCH_BLOCKING=1