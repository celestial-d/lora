#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
flower-superlink --insecure &

sleep 30

APP_DIR="/lus/eagle/projects/SR-APPFL/duo/lora/dp2"
cd "$APP_DIR"
flwr run . local-deployment --stream