#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import json
import shutil
import gc
from collections import OrderedDict

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForLanguageModeling

# -----------------------
# Config (env overrides)
# -----------------------
#meta-llama/Llama-2-7b-hf
MODEL_NAME  = os.getenv("MODEL_NAME", "meta-llama/Llama-2-7b-hf")
SYNC_DIR    = os.getenv("SYNC_DIR", "./sync")
CKPT_DIR    = os.getenv("CKPT_DIR", "/dev/shm/ckpt_shared")  # shared RAM-backed dir
TMP_DIR     = CKPT_DIR + ".tmp"
MAX_SHARD_SIZE = os.getenv("MAX_SHARD_SIZE", "1GB")

ROUNDS      = int(os.getenv("ROUNDS", "2"))
MAX_SEQ_LEN = int(os.getenv("MAX_SEQ_LEN", "512"))
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

def build_text(ex):
    instr = ex.get("instruction", "").strip()
    inp   = ex.get("input", "").strip()
    out   = ex.get("output", "").strip()
    if inp:
        return f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Answer:\n{out}"
    else:
        return f"### Instruction:\n{instr}\n\n### Answer:\n{out}"

def load_eval_dataset(size=512):
    ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train", trust_remote_code=True).shuffle(seed=SEED)
    size = min(size, len(ds))
    ds = ds.select(range(size))
    ds = ds.map(lambda ex: {"text": build_text(ex)}, remove_columns=ds.column_names)
    return ds

def cast_state_dict_dtype(sd: "OrderedDict[str, torch.Tensor]", target_dtype: torch.dtype):
    out = OrderedDict()
    for k, v in sd.items():
        if torch.is_tensor(v) and v.dtype.is_floating_point:
            out[k] = v.detach().to(target_dtype)
        else:
            out[k] = v
    return out

def atomic_save_pretrained_fp16(model, out_dir: str, max_shard_size: str = "1GB"):
    """
    Save sharded safetensors in FP16 to out_dir atomically (via tmp dir rename).
    """
    tmp = out_dir + ".tmp"
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp, exist_ok=True)

    sd_fp16 = cast_state_dict_dtype(model.state_dict(), torch.float16)
    # Use HF save_pretrained to produce (possibly) sharded safetensors + index
    model.save_pretrained(tmp, state_dict=sd_fp16, safe_serialization=True, max_shard_size=max_shard_size)
    # Atomic replace of the directory
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.rename(tmp, out_dir)

def load_sharded_into_existing_model(model, ckpt_dir: str, dtype: torch.dtype):
    """
    Load a sharded safetensors checkpoint directory into an existing model.
    We instantiate a temporary model from that directory (CPU, low mem), then copy its state dict.
    """
    tmp_model = AutoModelForCausalLM.from_pretrained(
        ckpt_dir,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=None,
    )
    model.load_state_dict(tmp_model.state_dict(), strict=False)
    del tmp_model
    gc.collect()

def main():
    torch.manual_seed(SEED)

    print("[reader] Loading tokenizer & FP16 model…")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,   # reader keeps fp16
        low_cpu_mem_usage=True,
        device_map=None,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    # Seed the shared checkpoint
    os.makedirs(os.path.dirname(CKPT_DIR), exist_ok=True)
    atomic_save_pretrained_fp16(model, CKPT_DIR, MAX_SHARD_SIZE)

    # Write meta & init flag
    meta = {"format": "safetensors", "dtype": "fp16", "ckpt_dir": CKPT_DIR, "max_shard_size": MAX_SHARD_SIZE}
    with open(META_JSON, "w") as f:
        json.dump(meta, f)
    touch(INIT_READY_FLAG)
    print("[reader] Init ready. Waiting for writer…")

    eval_ds = load_eval_dataset()

    # 5 rounds
    for r in range(1, ROUNDS + 1):
        # Tell writer to start this round
        touch(START_FLAG(r))
        print(f"[reader] ===== Round {r} started; waiting for writer_done…")
        wait_for(WRITER_DONE_FLAG(r))

        # Load writer's new checkpoint and do "evolution" (here: quick eval)
        load_sharded_into_existing_model(model, CKPT_DIR, dtype=torch.float16)
        #output_dir=os.path.join(SYNC_DIR, f"eval_tmp_r{r}"),
        # args = TrainingArguments(
        #     per_device_eval_batch_size=4,
        #     dataloader_num_workers=0,
        #     report_to=[],
        # )
        # collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        # trainer = SFTTrainer(
        #     model=model,
        #     tokenizer=tokenizer,
        #     train_dataset=None,
        #     eval_dataset=eval_ds,
        #     args=args,
        #     data_collator=collator,
        #     dataset_text_field="text",
        #     max_seq_length=MAX_SEQ_LEN,
        # )
        # metrics = trainer.evaluate()
        # print(f"[reader] Round {r} eval: {metrics}")

        # Optionally: evolve model (here we just re-save; plug your evolution step here)
        atomic_save_pretrained_fp16(model, CKPT_DIR, MAX_SHARD_SIZE)

        # Signal done to writer
        touch(READER_DONE_FLAG(r))
        print(f"[reader] Round {r} done.")

    # Stop writer
    touch(STOP_FLAG)
    print("[reader] All rounds complete. Stop flag set.")

if __name__ == "__main__":
    main()
