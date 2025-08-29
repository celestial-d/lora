#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import json
from collections import OrderedDict

# Force single GPU if not set (avoid DataParallel surprises)
import os as _os
if "CUDA_VISIBLE_DEVICES" not in _os.environ:
    _os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch
from torch import nn
from multiprocessing import shared_memory

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from trl import SFTTrainer

# -----------------------
# Config
# -----------------------
MODEL_NAME  = os.getenv("MODEL_NAME", "facebook/opt-1.3b")
SHM_NAME    = os.getenv("SHM_NAME", "opt125m_shm")
SYNC_DIR    = os.getenv("SYNC_DIR", "./sync")

MAX_SEQ_LEN = int(os.getenv("MAX_SEQ_LEN", "512"))
TRAIN_SIZE  = int(os.getenv("TRAIN_SIZE", "4000"))
BATCH_SIZE  = int(os.getenv("BATCH_SIZE", "2"))
GR_ACCUM    = int(os.getenv("GR_ACCUM", "4"))
LR          = float(os.getenv("LR", "2e-5"))
NUM_STEPS   = int(os.getenv("NUM_STEPS", "200"))
SEED        = int(os.getenv("SEED", "42"))

os.makedirs(SYNC_DIR, exist_ok=True)

INIT_READY_FLAG  = os.path.join(SYNC_DIR, "init_ready.flag")
START_FLAG       = lambda i: os.path.join(SYNC_DIR, f"round{i}.start")
WRITER_DONE_FLAG = lambda i: os.path.join(SYNC_DIR, f"round{i}.writer_done")
READER_DONE_FLAG = lambda i: os.path.join(SYNC_DIR, f"round{i}.reader_done")
STOP_FLAG        = os.path.join(SYNC_DIR, "stop.flag")
META_JSON        = os.path.join(SYNC_DIR, "meta.json")

def touch(path: str):
    with open(path, "w") as f:
        f.write("ok")

def wait_for(path: str):
    while not os.path.exists(path):
        time.sleep(0.2)

# -----------------------
# Dataset (raw "text"; TRL will tokenize)
# -----------------------
def build_text(ex):
    instr = ex.get("instruction", "").strip()
    inp   = ex.get("input", "").strip()
    out   = ex.get("output", "").strip()
    if inp:
        return f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Answer:\n{out}"
    else:
        return f"### Instruction:\n{instr}\n\n### Answer:\n{out}"

def load_train_dataset(size=4000):
    ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train", trust_remote_code=True).shuffle(seed=SEED)
    if size and size < len(ds):
        ds = ds.select(range(size))
    ds = ds.map(lambda ex: {"text": build_text(ex)}, remove_columns=ds.column_names)
    return ds

# -----------------------
# Model <-> FP16 SHM
# -----------------------
def ordered_float_tensors(model: nn.Module):
    for k, t in model.state_dict().items():
        if t.dtype.is_floating_point:
            yield k, t

def model_numel(model: nn.Module) -> int:
    return sum(t.numel() for _, t in ordered_float_tensors(model))

def write_model_to_shm_fp16(model: nn.Module, shm_buf: np.ndarray):
    offset = 0
    for _, t in ordered_float_tensors(model):
        arr = t.detach().cpu().to(torch.float16).numpy().ravel()
        n = arr.size
        shm_buf[offset:offset+n] = arr
        offset += n

def load_model_from_shm_into_model_dtype(model: nn.Module, shm_buf: np.ndarray):
    device = next(model.parameters()).device
    m_dtype = next((p.dtype for p in model.parameters() if p.dtype.is_floating_point), torch.bfloat16)
    offset = 0
    new_sd = OrderedDict(model.state_dict())
    for k, t in ordered_float_tensors(model):
        n = t.numel()
        arr = shm_buf[offset:offset+n].reshape(t.shape)
        tt = torch.from_numpy(arr).to(device=device, dtype=torch.float16).to(dtype=m_dtype)
        new_sd[k] = tt
        offset += n
    model.load_state_dict(new_sd, strict=False)

def make_trainer(model, tokenizer, train_ds, output_dir):
    args = TrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GR_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_steps=NUM_STEPS,
        logging_steps=50,
        save_steps=0,
        evaluation_strategy="no",  # deprecates to eval_strategy in 4.46+
        bf16=True,                 # BF16 training as requested
        fp16=False,
        gradient_checkpointing=True,
        report_to=[],
        seed=SEED,
        remove_unused_columns=True,
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,      # new arg path (tokenizer)
        train_dataset=train_ds,
        args=args,
        data_collator=collator,          # LM collator
        dataset_text_field="text",       # TRL will tokenize this column
        max_seq_length=MAX_SEQ_LEN,
    )

def main():
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[writer] Waiting for reader init…")
    wait_for(INIT_READY_FLAG)
    wait_for(META_JSON)

    with open(META_JSON, "r") as f:
        meta = json.load(f)

    print("[writer] Loading tokenizer & BF16 model…")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=None,
    ).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id

    total_numel = model_numel(model)
    shm = shared_memory.SharedMemory(name=SHM_NAME, create=False)
    shm_arr = np.ndarray((total_numel,), dtype=np.float16, buffer=shm.buf)

    train_ds = load_train_dataset(TRAIN_SIZE)

    round_idx = 1
    print("[writer] Entering training loop; waiting for round starts or stop.flag")
    while True:
        if os.path.exists(STOP_FLAG):
            print("[writer] stop.flag detected. Exiting.")
            break

        start_file = START_FLAG(round_idx)
        if not os.path.exists(start_file):
            time.sleep(0.2)
            continue

        print(f"[writer] ===== Round {round_idx} =====")
        load_model_from_shm_into_model_dtype(model, shm_arr)

        t1 = make_trainer(model, tokenizer, train_ds, os.path.join(SYNC_DIR, f"train_tmp_r{round_idx}_run1"))
        t1.train()

        t2 = make_trainer(model, tokenizer, train_ds, os.path.join(SYNC_DIR, f"train_tmp_r{round_idx}_run2"))
        t2.train()

        write_model_to_shm_fp16(model, shm_arr)

        touch(WRITER_DONE_FLAG(round_idx))
        print(f"[writer] Round {round_idx}: signaled reader; waiting for reader_done…")
        wait_for(READER_DONE_FLAG(round_idx))

        round_idx += 1

    shm.close()
    print("[writer] Clean exit.")

if __name__ == "__main__":
    main()
