#!/usr/bin/env python3
# fsdp_trainer.py
# Multi-GPU reader/trainer driven via SHM + flag files.
# - SHM/flags from env (per-client isolation)
# - Load initial weights on PLAIN model before FSDP wrapping
# - Use FULL_STATE_DICT for subsequent load/save when FSDP-wrapped
# - Write metrics.json after train/eval for the client to return to server

import os
import time
import json
import numpy as np
import torch
import torch.distributed as dist
from multiprocessing import shared_memory
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM
from trl import SFTTrainer

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
try:
    from torch.distributed.fsdp import FullStateDictConfig
except ImportError:
    from torch.distributed.fsdp.api import FullStateDictConfig  # compat for older torch

# --------- Per-client SHM & flag paths via env ---------
SHM_NAME = os.getenv("SHM_NAME", "opt125m_shared")
FLAG_DIR = os.getenv("FLAG_DIR", ".")
FLAG_READY = os.path.join(FLAG_DIR, "ready.flag")
FLAG_DONE  = os.path.join(FLAG_DIR, "done.flag")
FLAG_MODE  = os.path.join(FLAG_DIR, "mode.flag")
os.makedirs(FLAG_DIR, exist_ok=True)

# --------- Small utils ----------
def read_mode_on_rank0():
    with open(FLAG_MODE, "r") as f:
        return f.read().strip().lower()

def broadcast_mode(local_rank, mode: str | None):
    dev = torch.device("cuda", local_rank)
    if dist.get_rank() == 0:
        mb = (mode or "").encode("utf-8")
        size = torch.tensor([len(mb)], dtype=torch.int64, device=dev)
    else:
        mb = None
        size = torch.zeros(1, dtype=torch.int64, device=dev)
    dist.broadcast(size, src=0)
    sz = int(size.item())
    if sz == 0:
        return ""
    buf = torch.empty(sz, dtype=torch.uint8, device=dev)
    if dist.get_rank() == 0:
        buf[:] = torch.tensor(list(mb), dtype=torch.uint8, device=dev)
    dist.broadcast(buf, src=0)
    return bytes(buf.tolist()).decode("utf-8")

def wait_for_ready_on_rank0():
    print("\n[Rank 0] Waiting for ready.flag...", flush=True)
    while not os.path.exists(FLAG_READY):
        time.sleep(0.1)
    print("[Rank 0] ready.flag detected.", flush=True)

def signal_done_on_rank0():
    with open(FLAG_DONE, "w") as f:
        f.write("done"); f.flush(); os.fsync(f.fileno())
    # Clean flags written by client for the round
    for p in (FLAG_READY, FLAG_MODE):
        try: os.remove(p)
        except FileNotFoundError: pass
    print("✅ [Rank 0] Weights written & done.flag created.", flush=True)

# --------- SHM IO helpers ----------
def _ensure_size(total_elems_fp16):
    shm = shared_memory.SharedMemory(name=SHM_NAME)
    have = shm.size
    need = int(total_elems_fp16) * np.dtype(np.float16).itemsize
    if have < need:
        shm.close()
        raise RuntimeError(
            f"SHM too small: have {have} bytes, need {need} bytes "
            f"(total_elems={total_elems_fp16}, dtype=fp16). "
            "Likely causes: stale SHM from older run, or FSDP wrapped before first load."
        )
    return shm

def load_weights_from_shm_plain(model: torch.nn.Module) -> torch.nn.Module:
    sd = model.state_dict()
    total = sum(p.numel() for p in sd.values())
    shm = _ensure_size(total)
    flat = np.ndarray((total,), dtype=np.float16, buffer=shm.buf)

    ptr = 0
    new_sd = {}
    for k, p in sd.items():
        n = p.numel()
        new_sd[k] = torch.from_numpy(flat[ptr:ptr+n].copy()).view(p.shape).to(p.dtype).contiguous()
        ptr += n

    model.load_state_dict(new_sd, strict=True)
    shm.close()
    print(f"[Rank {dist.get_rank()}] ✅ Loaded (PLAIN) weights from SHM.", flush=True)
    return model

def load_weights_from_shm_fsdp(fsdp_model: torch.nn.Module) -> None:
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
    with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT, cfg):
        ref_sd = fsdp_model.state_dict()
    total = sum(p.numel() for p in ref_sd.values())

    shm = _ensure_size(total)
    flat = np.ndarray((total,), dtype=np.float16, buffer=shm.buf)

    ptr = 0
    new_sd = {}
    for k, p in ref_sd.items():
        n = p.numel()
        new_sd[k] = torch.from_numpy(flat[ptr:ptr+n].copy()).view(p.shape).to(p.dtype).contiguous()
        ptr += n
    shm.close()

    with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT, cfg):
        fsdp_model.load_state_dict(new_sd, strict=True)
    print(f"[Rank {dist.get_rank()}] ✅ Loaded (FSDP) weights from SHM.", flush=True)

def write_weights_to_shm_fsdp(fsdp_model: torch.nn.Module) -> None:
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT, cfg):
        full_state = fsdp_model.state_dict()
    dist.barrier()

    if dist.get_rank() == 0:
        flat = torch.cat([p.detach().to("cpu").to(torch.float16).flatten() for p in full_state.values()])
        shm = shared_memory.SharedMemory(name=SHM_NAME)
        np_out = np.ndarray(flat.shape, dtype=np.float16, buffer=shm.buf)
        np_out[:] = flat[:]
        shm.close()
        signal_done_on_rank0()
    dist.barrier()

# --------- Main ---------
def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    # Tokenizer + base model (PLAIN)
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m", padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", torch_dtype=torch.float16)
    if hasattr(model, "config"):
        model.config.use_cache = False

    # Dataset
    raw = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    def format_prompt(example):
        instruction = example["instruction"].strip()
        input_text  = example["input"].strip()
        q = f"{instruction} {input_text}" if input_text else instruction
        return {"text": f"### Question: {q}\n### Answer: {example['output'].strip()}"}
    ds  = raw.map(format_prompt)
    train_ds = ds.select(range(500))
    eval_ds  = ds.select(range(500, 600))

    # Training args (deprecation warnings are harmless)
    args = TrainingArguments(
        output_dir="./fsdp_output_opt125m",
        per_device_train_batch_size=1,
        learning_rate=2e-5,
        max_steps=10,               # one burst per handshake
        logging_steps=1,
        save_steps=10_000_000,
        report_to="none",
        fp16=True,
        fsdp="full_shard auto_wrap",
        fsdp_transformer_layer_cls_to_wrap="OPTDecoderLayer",
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
    )

    # ---------- First handshake BEFORE FSDP wrapping ----------
    if rank == 0:
        wait_for_ready_on_rank0()
    dist.barrier()

    mode = read_mode_on_rank0() if rank == 0 else None
    mode = broadcast_mode(local_rank, mode)
    if rank == 0:
        print(f"[Rank 0] Mode={mode}", flush=True)

    # Stop?
    if mode == "stop":
        if rank == 0:
            signal_done_on_rank0()
            print("[Rank 0] Stop acknowledged. Exiting.", flush=True)
        dist.barrier()
        dist.destroy_process_group()
        return

    # Load incoming weights into PLAIN model (sizes match client's SHM)
    model = load_weights_from_shm_plain(model)

    # Now build the trainer (this applies FSDP wrapping)
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        dataset_text_field="text",
    )

    # ---------- Per-round loop ----------
    while True:
        # Do work per mode
        if mode == "train":
            if rank == 0: print("[Rank 0] Training...", flush=True)
            out = trainer.train()  # capture TrainOutput for metrics
            if rank == 0:
                with open(os.path.join(FLAG_DIR, "metrics.json"), "w") as f:
                    json.dump({"train_loss": float(getattr(out, "training_loss", 0.0))}, f)
        elif mode == "eval":
            if rank == 0: print("[Rank 0] Evaluating...", flush=True)
            metrics = trainer.evaluate()
            if rank == 0:
                with open(os.path.join(FLAG_DIR, "metrics.json"), "w") as f:
                    json.dump({"eval_loss": float(metrics.get("eval_loss", 0.0))}, f)
        else:
            if rank == 0: print(f"[Rank 0] Unknown mode '{mode}', skipping work.", flush=True)

        dist.barrier()

        # Write back updated weights (FULL state dict) and signal done
        write_weights_to_shm_fsdp(trainer.model)

        # ----- Next handshake -----
        if rank == 0:
            wait_for_ready_on_rank0()
        dist.barrier()
        mode = read_mode_on_rank0() if rank == 0 else None
        mode = broadcast_mode(local_rank, mode)
        if rank == 0:
            print(f"[Rank 0] Mode={mode}", flush=True)

        if mode == "stop":
            if rank == 0:
                signal_done_on_rank0()
                print("[Rank 0] Stop acknowledged. Exiting.", flush=True)
            dist.barrier()
            break

        # Load next incoming weights into FSDP-wrapped model
        load_weights_from_shm_fsdp(trainer.model)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
