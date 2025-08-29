#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os, time, json, shutil, gc
from collections import OrderedDict

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, DataCollatorForLanguageModeling,
)
from trl import SFTTrainer

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

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
if torch.cuda.is_available():
    torch.cuda.set_device(LOCAL_RANK)

# -----------------------
# Config (env overrides)
# -----------------------
#meta-llama/Llama-2-7b-hf
MODEL_NAME  = os.getenv("MODEL_NAME", "meta-llama/Llama-2-7b-hf")
SYNC_DIR    = os.getenv("SYNC_DIR", "./sync")
CKPT_DIR    = os.getenv("CKPT_DIR", "/dev/shm/ckpt_shared")   # shared RAM-backed dir
MAX_SHARD_SIZE = os.getenv("MAX_SHARD_SIZE", "1GB")

MAX_SEQ_LEN = int(os.getenv("MAX_SEQ_LEN", "512"))
TRAIN_SIZE  = int(os.getenv("TRAIN_SIZE", "4000"))
BATCH_SIZE  = int(os.getenv("BATCH_SIZE", "1"))               # keep in sync with DS JSON
GR_ACCUM    = int(os.getenv("GR_ACCUM", "16"))                # keep in sync with DS JSON
LR          = float(os.getenv("LR", "2e-5"))
NUM_STEPS   = int(os.getenv("NUM_STEPS", "2"))
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
# Atomic dir replace helper
# -----------------------
def atomic_replace_dir(src_tmp: str, dst: str):
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.rename(src_tmp, dst)

# -----------------------
# Trainer factory (TRL + DeepSpeed ZeRO-3 Offload)
# -----------------------
def make_trainer(model, tokenizer, train_ds, output_dir):
    args = TrainingArguments(
        output_dir=output_dir,
        save_strategy="no",
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GR_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_steps=NUM_STEPS,
        logging_steps=50,
        save_steps=10**9,                    # no periodic saves
        save_total_limit=0,
        evaluation_strategy="no",
        bf16=True,                      # compute in BF16 on A30
        fp16=False,
        optim="adamw_torch",            # DeepSpeed offloads optimizer states to CPU
        report_to=[],
        seed=SEED,
        remove_unused_columns=True,
        ddp_find_unused_parameters=False,
        deepspeed="ds_zero3_offload.json",  # <— DS config below
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        args=args,
        data_collator=collator,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
    )

def teardown_after_round(trainer, model=None):
    import ctypes, gc
    # DS engine is in trainer.model_wrapped (HF) or trainer.model (your case)
    engine = getattr(trainer, "model_wrapped", None) or getattr(trainer, "model", None)
    try:
        if engine is not None and hasattr(engine, "destroy"):
            engine.destroy()  # frees ZeRO partitions/optimizer buffers
        # Drop common refs that pin memory
        for obj in (getattr(engine, "optimizer", None),
                    getattr(engine, "module", None)):
            try:
                del obj
            except Exception:
                pass
        if engine is not None:
            del engine
    except Exception as e:
        print(f"[teardown] warn: {e}")

    try: del trainer
    except: pass
    try: del model
    except: pass

    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)  # return freed heap to OS
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


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

    # 2) Tokenizer (pad/right) — used by trainer
    if get_rank() == 0:
        print("[writer] Loading tokenizer & building dataset…")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 3) Data (same across rounds)
    train_ds = load_train_dataset(TRAIN_SIZE)

    round_idx = 1
    if get_rank() == 0:
        print("[writer] Entering DeepSpeed (ZeRO-3 Offload) loop; waiting for round starts or stop.flag")

    while True:
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

        # Fresh model for this round loaded from shared CKPT_DIR
        train_model = AutoModelForCausalLM.from_pretrained(
            CKPT_DIR,                      # <- reader/writer shared checkpoint
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
            device_map=None,               # let DeepSpeed handle distribution
        )
        train_model.config.pad_token_id = tokenizer.pad_token_id
        train_model.config.use_cache = False
        try:
            train_model.set_attn_implementation("sdpa")
        except Exception:
            pass

        # 4) Train under DeepSpeed
        run_outdir = os.path.join(SYNC_DIR, f"train_tmp_r{round_idx}")
        trainer = make_trainer(train_model, tokenizer, train_ds, run_outdir)
        trainer.train()

        # 5) Rank0 gathers 16-bit weights via DS and saves sharded safetensors atomically
        barrier()
        #if trainer.is_world_process_zero():
        tmp = CKPT_DIR + ".tmp"
        if os.path.isdir(tmp):
            shutil.rmtree(tmp)
            os.makedirs(tmp, exist_ok=True)
        barrier()
            # This triggers DS to gather 16-bit weights (because of stage3_gather_16bit_weights_on_model_save=true)
        #trainer.save_model(tmp)
        barrier()
        if trainer.is_world_process_zero():
            # Replace shared checkpoint dir atomically
            atomic_replace_dir(tmp, CKPT_DIR)
            touch(WRITER_DONE_FLAG(round_idx))
            print(f"[writer] Round {round_idx}: signaled reader; waiting for reader_done…")

        # 6) Sync with reader then next round
        if trainer.is_world_process_zero():
            wait_for(READER_DONE_FLAG(round_idx))
        barrier()
        # 7) Tear down DS engine aggressively to avoid OOM in Round-2
        try:
            engine = trainer.model  # DeepSpeed engine
            if hasattr(engine, "optimizer") and engine.optimizer is not None:
                engine.optimizer.zero_grad(set_to_none=True)
        except Exception:
            pass
        # Cleanup
        # del trainer, train_model
        # gc.collect()
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
        # time.sleep(0.5)  # let reader run first
        # barrier()
        teardown_after_round(trainer, train_model)
        time.sleep(0.2)
        barrier()
        round_idx += 1

    if get_rank() == 0:
        print("[writer] Clean exit.")

if __name__ == "__main__":
    # launch with: torchrun --nproc_per_node=4 writer_ds.py
    main()
