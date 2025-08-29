#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Launch with:
  accelerate launch --num_processes 2 writer.py
"""

import os

# Stable NCCL & attention behavior
os.environ.setdefault("PYTORCH_SDP_DISABLE_FAST_PATH", "1")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")

import time
import json
import shutil
import gc
from collections import OrderedDict

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from trl import SFTTrainer

# FSDP consolidated state-dict utilities
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
)

# -----------------------
# Rank / device helpers
# -----------------------
def get_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.environ.get("RANK", "0"))

def barrier():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

# Pin each rank to its local GPU (Accelerate sets LOCAL_RANK)
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
if torch.cuda.is_available():
    torch.cuda.set_device(LOCAL_RANK)

# -----------------------
# Config (env overrides)
# -----------------------
MODEL_NAME  = os.getenv("MODEL_NAME", "facebook/opt-1.3b")
SYNC_DIR    = os.getenv("SYNC_DIR", "./sync")
CKPT_DIR    = os.getenv("CKPT_DIR", "/dev/shm/ckpt_shared")  # shared RAM-backed dir
MAX_SHARD_SIZE = os.getenv("MAX_SHARD_SIZE", "1GB")

MAX_SEQ_LEN = int(os.getenv("MAX_SEQ_LEN", "512"))
TRAIN_SIZE  = int(os.getenv("TRAIN_SIZE", "4000"))
BATCH_SIZE  = int(os.getenv("BATCH_SIZE", "1"))
GR_ACCUM    = int(os.getenv("GR_ACCUM", "4"))
LR          = float(os.getenv("LR", "2e-5"))
NUM_STEPS   = int(os.getenv("NUM_STEPS", "200"))
SEED        = int(os.getenv("SEED", "42"))

SYNC_MODULE_STATES = True  # broadcast init params from rank0 on wrap

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

def wait_for(path: str, sleep: float = 0.2):
    while not os.path.exists(path):
        time.sleep(sleep)

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
# FP16 sharded safetensors save/load helpers
# -----------------------
def cast_state_dict_dtype(sd: "OrderedDict[str, torch.Tensor]", target_dtype: torch.dtype):
    out = OrderedDict()
    for k, v in sd.items():
        if torch.is_tensor(v) and v.dtype.is_floating_point:
            out[k] = v.detach().to(target_dtype)
        else:
            out[k] = v
    return out

def atomic_save_pretrained_fp16_from_sd(model_for_config, state_dict, out_dir: str, max_shard_size: str = "1GB"):
    """
    Save a given state_dict as sharded safetensors in FP16 to out_dir atomically.
    We use `model_for_config` only for its config & save_pretrained logic (never wrapped by FSDP).
    """
    tmp = out_dir + ".tmp"
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp, exist_ok=True)

    sd_fp16 = cast_state_dict_dtype(state_dict, torch.float16)
    model_for_config.save_pretrained(
        tmp,
        state_dict=sd_fp16,
        safe_serialization=True,
        max_shard_size=max_shard_size
    )

    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.rename(tmp, out_dir)

# -----------------------
# Trainer factory (TRL + HF Trainer FSDP)
# -----------------------
def make_trainer(model, tokenizer, train_ds, output_dir):
    # FSDP via fsdp_config; no activation checkpointing for stability on OPT-125M
    fsdp_config = {
        "fsdp_min_num_params": 0,
        "sync_module_states": SYNC_MODULE_STATES,    # broadcast rank0 weights
        "use_orig_params": True,
        "limit_all_gathers": True,
        "forward_prefetch": True,
        "activation_checkpointing": False,           # disabled for stability
        "activation_checkpointing_reentrant": False,
        "transformer_layer_cls_to_wrap": ["OPTDecoderLayer"],
    }
    #output_dir=output_dir,
    args = TrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GR_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_steps=NUM_STEPS,
        logging_steps=50,
        save_steps=0,
        evaluation_strategy="no",
        bf16=True,                     # compute in BF16
        fp16=False,
        gradient_checkpointing=False,  # no activation ckpt here
        report_to=[],
        seed=SEED,
        remove_unused_columns=True,    # dataset has only 'text'
        ddp_find_unused_parameters=False,
        dataloader_num_workers=0,
        fsdp="full_shard",
        fsdp_config=fsdp_config,
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,              # TRL 0.8.1
        train_dataset=train_ds,
        args=args,
        data_collator=collator,
        dataset_text_field="text",        # TRL tokenizes from raw "text"
        max_seq_length=MAX_SEQ_LEN,
    )

# -----------------------
# Main
# -----------------------
def main():
    torch.manual_seed(SEED)

    # 1) Wait for reader to seed a checkpoint
    if get_rank() == 0:
        print("[writer] Waiting for reader init…")
    wait_for(INIT_READY_FLAG)
    wait_for(META_JSON)
    with open(META_JSON, "r") as f:
        meta = json.load(f)

    # 2) Tokenizer + STATIC skeleton model (used ONLY for saving)
    if get_rank() == 0:
        print("[writer] Loading tokenizer & BF16 model (FSDP)…")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Static, never-wrapped skeleton (CPU) so save_pretrained is always clean
    config_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=None,   # keep on CPU
    )
    config_model.config.pad_token_id = tokenizer.pad_token_id
    config_model.config.use_cache = False
    try:
        config_model.set_attn_implementation("eager")
    except Exception:
        pass

    # 3) Data (same across rounds)
    train_ds = load_train_dataset(TRAIN_SIZE)

    round_idx = 1
    if get_rank() == 0:
        print("[writer] Entering FSDP training loop; waiting for round starts or stop.flag")

    while True:
        # Clean exit if reader already signaled stop
        if os.path.exists(STOP_FLAG):
            if get_rank() == 0:
                print("[writer] stop.flag detected. Exiting.")
            break

        start_file = START_FLAG(round_idx)
        if not os.path.exists(start_file):
            time.sleep(0.2)
            continue

        if get_rank() == 0:
            print(f"[writer] ===== Round {round_idx} =====")

        # --- Fresh model for this round, loaded from the shared sharded checkpoint ---
        # Keep it BF16 on CPU; FSDP will place/broadcast to GPUs.
        train_model = AutoModelForCausalLM.from_pretrained(
            CKPT_DIR,                      # <— current shared checkpoint from reader/writer
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map=None,
        )
        train_model.config.pad_token_id = tokenizer.pad_token_id
        train_model.config.use_cache = False
        try:
            train_model.set_attn_implementation("eager")
        except Exception:
            pass

        # 4) Train (FSDP wraps & broadcasts rank0 params)
        trainer = make_trainer(
            train_model, tokenizer, train_ds,
            os.path.join(SYNC_DIR, f"train_tmp_r{round_idx}")
        )
        trainer.train()

        # 5) Consolidate FULL state dict (all ranks enter), then rank0 saves sharded fp16
        barrier()
        full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(trainer.model, StateDictType.FULL_STATE_DICT, full_cfg):
            full_sd = trainer.model.state_dict()  # {} on non-rank0

        if get_rank() == 0:
            atomic_save_pretrained_fp16_from_sd(
                config_model, full_sd, CKPT_DIR, MAX_SHARD_SIZE
            )
            touch(WRITER_DONE_FLAG(round_idx))
            print(f"[writer] Round {round_idx}: signaled reader; waiting for reader_done…")

        # 6) Sync with reader, then next round
        if get_rank() == 0:
            wait_for(READER_DONE_FLAG(round_idx))
        barrier()

        # Clean up the training model for this round
        del trainer
        del train_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        round_idx += 1

    if get_rank() == 0:
        print("[writer] Clean exit.")

if __name__ == "__main__":
    # `accelerate launch` initializes distributed; do not init manually.
    main()
