#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import json
from collections import OrderedDict
from typing import List, Tuple

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
ROUNDS      = int(os.getenv("ROUNDS", "5"))
MAX_SEQ_LEN = int(os.getenv("MAX_SEQ_LEN", "512"))
EVAL_SIZE   = int(os.getenv("EVAL_SIZE", "1000"))
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

def load_eval_dataset(size=1000):
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

def write_model_to_shm(model: nn.Module, shm_buf: np.ndarray):
    offset = 0
    for _, t in ordered_float_tensors(model):
        arr = t.detach().cpu().to(torch.float16).numpy().ravel()
        n = arr.size
        shm_buf[offset:offset+n] = arr
        offset += n

def load_model_from_shm(model: nn.Module, shm_buf: np.ndarray):
    device = next(model.parameters()).device
    m_dtype = next((p.dtype for p in model.parameters() if p.dtype.is_floating_point), torch.float16)
    offset = 0
    new_sd = OrderedDict(model.state_dict())
    for k, t in ordered_float_tensors(model):
        n = t.numel()
        arr = shm_buf[offset:offset+n].reshape(t.shape)
        tt = torch.from_numpy(arr).to(device=device, dtype=torch.float16).to(dtype=m_dtype)
        new_sd[k] = tt
        offset += n
    model.load_state_dict(new_sd, strict=False)

# -----------------------
# Eval (let TRL tokenize from "text")
# -----------------------
def evaluate_model(model, tokenizer, eval_ds):
    model.eval()
    args = TrainingArguments(
        per_device_eval_batch_size=4,
        do_eval=True,
        logging_steps=50,
        report_to=[],
        remove_unused_columns=True,
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,       # new arg path (tokenizer)
        train_dataset=None,
        eval_dataset=eval_ds,
        args=args,
        data_collator=collator,           # LM collator for causal LM
        dataset_text_field="text",        # TRL will tokenize this column
        max_seq_length=MAX_SEQ_LEN,
    )
    metrics = trainer.evaluate()
    print(f"[reader] evaluation metrics: {metrics}")
    return metrics

def main():
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[reader] Loading tokenizer & model (FP16)…")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=None,
    ).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id

    total_numel = model_numel(model)
    shm_bytes = total_numel * np.dtype(np.float16).itemsize
    print(f"[reader] Creating SHM '{SHM_NAME}' with {total_numel} fp16 elements (~{shm_bytes/1e6:.1f} MB)…")
    shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=shm_bytes)
    shm_arr = np.ndarray((total_numel,), dtype=np.float16, buffer=shm.buf)

    write_model_to_shm(model, shm_arr)

    with open(META_JSON, "w") as f:
        json.dump({"model": MODEL_NAME, "numel": total_numel, "dtype": "float16"}, f)

    touch(INIT_READY_FLAG)
    print("[reader] Init ready. Driving rounds…")

    eval_ds = load_eval_dataset(EVAL_SIZE)

    for r in range(1, ROUNDS + 1):
        touch(START_FLAG(r))
        print(f"[reader] Round {r} started; waiting for writer_done…")
        wait_for(WRITER_DONE_FLAG(r))

        load_model_from_shm(model, shm_arr)
        with torch.cuda.amp.autocast(enabled=(device=="cuda"), dtype=torch.float16):
            evaluate_model(model, tokenizer, eval_ds)

        write_model_to_shm(model, shm_arr)
        touch(READER_DONE_FLAG(r))
        print(f"[reader] Round {r} done.")

    touch(STOP_FLAG)
    print("[reader] All rounds complete. Stop signaled. Cleaning up SHM.")
    shm.close()
    shm.unlink()

if __name__ == "__main__":
    main()
