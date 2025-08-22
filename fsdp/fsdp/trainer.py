#!/usr/bin/env python3
# export SHM_NAME=opt_client_0
# export FLAG_DIR=$(pwd)/flags_client_0
# mkdir -p "$FLAG_DIR"
# torchrun --nproc_per_node=<NUM_GPUS> fsdp/trainer.py
"""
fsdp/trainer.py
Independent multi-GPU FSDP trainer driven by SHM + flags.

- Uses your fsdp.models / fsdp.dataset APIs.
- Reads trainer_config.json + keys_order.json written by the client.
- First handshake: loads incoming weights into PLAIN model (client order).
- Then FSDP wrap via HF TrainingArguments(fsdp=...)
- Per round:
    * optional knobs.json overrides (LR/steps)
    * run train/eval burst
    * write FULL_STATE_DICT back to SHM in the SAME order as keys_order.json
"""

from __future__ import annotations
import os
import json
import time
from typing import Dict, Any, List

import numpy as np
import torch
import torch.distributed as dist
from multiprocessing import shared_memory

from transformers import TrainingArguments
from trl import SFTTrainer

from fsdp import models as mdl
from fsdp import dataset as ds

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
try:
    from torch.distributed.fsdp import FullStateDictConfig
except ImportError:
    from torch.distributed.fsdp.api import FullStateDictConfig  # torch<2.1 compat


# -----------------------
# Paths and flags
# -----------------------
FLAG_DIR    = os.getenv("FLAG_DIR", os.path.abspath("./flags_client_0"))
SHM_NAME    = os.getenv("SHM_NAME", "opt_client_0")
CFG_JSON    = os.path.join(FLAG_DIR, "trainer_config.json")
KEYS_JSON   = os.path.join(FLAG_DIR, "keys_order.json")
KNOBS_JSON  = os.path.join(FLAG_DIR, "knobs.json")
FLAG_READY  = os.path.join(FLAG_DIR, "ready.flag")
FLAG_DONE   = os.path.join(FLAG_DIR, "done.flag")
FLAG_MODE   = os.path.join(FLAG_DIR, "mode.flag")
METRICS_JSON= os.path.join(FLAG_DIR, "metrics.json")
os.makedirs(FLAG_DIR, exist_ok=True)


# -----------------------
# Config / keys
# -----------------------
DEFAULTS = {
    "model_name": "facebook/opt-125m",
    "model_dtype": "bf16",
    "dataset_name": "sahil2801/CodeAlpaca-20k",
    "partition_id": 0,
    "num_partitions": 1,
    "seq_length": 512,
    "per_device_train_bs": 1,
    "ga_steps": 1,
    "learning_rate": 2e-5,
    "max_steps": 10,
    "logging_steps": 10,
    "lr_scheduler_type": "constant",
    "gradient_checkpointing": True,
    "targs_bf16": True,
    "targs_fp16": False,
    "fsdp": "full_shard auto_wrap",
    "fsdp_transformer_layer_cls_to_wrap": "OPTDecoderLayer",
    "attn_implementation": "sdpa",
    "output_dir": os.path.abspath("./fsdp_output"),
}

def resolve_trainer_cfg() -> Dict[str, Any]:
    cfg = DEFAULTS.copy()
    if not os.path.exists(CFG_JSON):
        print(f"[trainer] Waiting for trainer_config.json in {FLAG_DIR} ...", flush=True)
        while not os.path.exists(CFG_JSON):
            time.sleep(0.2)
    try:
        with open(CFG_JSON, "r") as f:
            cfg.update(json.load(f))
    except Exception as e:
        print(f"[trainer] WARN: failed to read trainer_config.json: {e}")
    return cfg

def read_keys_order() -> List[str]:
    if not os.path.exists(KEYS_JSON):
        print(f"[trainer] WARN: keys_order.json missing; falling back to state_dict order.", flush=True)
        return []
    try:
        with open(KEYS_JSON, "r") as f:
            return list(json.load(f))
    except Exception as e:
        print(f"[trainer] WARN: failed to read keys_order.json: {e}")
        return []


# -----------------------
# Flags
# -----------------------
def read_mode_on_rank0() -> str:
    with open(FLAG_MODE, "r") as f:
        return f.read().strip().lower()

def broadcast_str(local_rank: int, s: str) -> str:
    dev = torch.device("cuda", local_rank)
    rank = dist.get_rank()
    if rank == 0:
        payload = s.encode("utf-8")
        sz = torch.tensor([len(payload)], dtype=torch.int64, device=dev)
    else:
        payload = b""
        sz = torch.zeros(1, dtype=torch.int64, device=dev)
    dist.broadcast(sz, src=0)
    n = int(sz.item())
    if n == 0:
        return ""
    buf = torch.empty(n, dtype=torch.uint8, device=dev)
    if rank == 0:
        buf.copy_(torch.tensor(list(payload), dtype=torch.uint8, device=dev))
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
    for p in (FLAG_READY, FLAG_MODE):
        try: os.remove(p)
        except FileNotFoundError: pass
    print("✅ [Rank 0] done.flag created", flush=True)

def read_knobs() -> Dict[str, Any]:
    if not os.path.exists(KNOBS_JSON):
        return {}
    try:
        with open(KNOBS_JSON, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[trainer] WARN: failed to read knobs.json: {e}")
        return {}


# -----------------------
# SHM I/O (fp16 flat) honoring key order
# -----------------------
def _ensure_size(total_elems_fp16: int):
    shm = shared_memory.SharedMemory(name=SHM_NAME)
    have = shm.size
    need = int(total_elems_fp16) * np.dtype(np.float16).itemsize
    if have < need:
        shm.close()
        raise RuntimeError(
            f"SHM too small: have {have} bytes, need {need} bytes "
            f"(total_elems={total_elems_fp16}, dtype=fp16)."
        )
    return shm

def load_plain_from_shm_with_order(model: torch.nn.Module, keys_order: List[str]) -> None:
    sd = model.state_dict()
    key_list = [k for k in keys_order if k in sd] if keys_order else list(sd.keys())
    total = sum(sd[k].numel() for k in key_list)
    shm = _ensure_size(total)
    flat = np.ndarray((total,), dtype=np.float16, buffer=shm.buf)

    new_sd = {}
    ptr = 0
    for k in key_list:
        p = sd[k]
        n = p.numel()
        new_sd[k] = torch.from_numpy(flat[ptr:ptr+n].copy()).view(p.shape).to(p.dtype).contiguous()
        ptr += n

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if missing or unexpected:
        print(f"[trainer] WARN: load_plain missing={missing}, unexpected={unexpected}")
    shm.close()
    print(f"[Rank {dist.get_rank()}] ✅ Loaded PLAIN weights from SHM", flush=True)

def load_fsdp_from_shm_with_order(fsdp_model: torch.nn.Module, keys_order: List[str]) -> None:
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
    with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT, cfg):
        ref_sd = fsdp_model.state_dict()
    key_list = [k for k in keys_order if k in ref_sd] if keys_order else list(ref_sd.keys())
    total = sum(ref_sd[k].numel() for k in key_list)

    shm = _ensure_size(total)
    flat = np.ndarray((total,), dtype=np.float16, buffer=shm.buf)

    new_sd = {}
    ptr = 0
    for k in key_list:
        p = ref_sd[k]
        n = p.numel()
        new_sd[k] = torch.from_numpy(flat[ptr:ptr+n].copy()).view(p.shape).to(p.dtype).contiguous()
        ptr += n
    shm.close()

    with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT, cfg):
        missing, unexpected = fsdp_model.load_state_dict(new_sd, strict=False)
    if missing or unexpected:
        print(f"[trainer] WARN: load_fsdp missing={missing}, unexpected={unexpected}")
    print(f"[Rank {dist.get_rank()}] ✅ Loaded FSDP FULL_STATE_DICT from SHM", flush=True)

def write_fsdp_to_shm_with_order(fsdp_model: torch.nn.Module, keys_order: List[str]) -> None:
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT, cfg):
        full_state = fsdp_model.state_dict()
    dist.barrier()

    key_list = [k for k in keys_order if k in full_state] if keys_order else list(full_state.keys())

    if dist.get_rank() == 0:
        flats = [full_state[k].detach().to("cpu").to(torch.float16).flatten() for k in key_list]
        flat = torch.cat(flats) if flats else torch.tensor([], dtype=torch.float16)
        shm = shared_memory.SharedMemory(name=SHM_NAME)
        np_out = np.ndarray(flat.shape, dtype=np.float16, buffer=shm.buf)
        if flat.numel() != np_out.size:
            raise RuntimeError(f"[trainer] Flat size mismatch: {flat.numel()} vs SHM {np_out.size}")
        np_out[:] = flat[:]
        shm.close()
        signal_done_on_rank0()
    dist.barrier()


# -----------------------
# Precision policy (respects config but safe-falls back)
# -----------------------
def apply_precision_policy(model: torch.nn.Module, targs: TrainingArguments, want_bf16: bool, want_fp16: bool):
    # Try to honor bf16/fp16 requested by config; fall back if unsupported.
    if want_bf16:
        bf16_ok = False
        try:
            bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        except AttributeError:
            bf16_ok = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
        if bf16_ok:
            model.to(torch.bfloat16)
            targs.bf16 = True
            targs.fp16 = False
            print("[trainer] Using BF16 (per config)", flush=True)
            return
        else:
            print("[trainer] BF16 requested but unsupported; falling back to FP16 AMP.", flush=True)

    # If we get here and FP16 is requested or BF16 was unsupported:
    if want_fp16 or True:
        model.to(torch.float32)  # AMP wants FP32 master; GradScaler will handle casts
        targs.fp16 = True
        targs.bf16 = False
        print("[trainer] Using AMP-FP16", flush=True)


# -----------------------
# Main
# -----------------------
def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    CFG = resolve_trainer_cfg()
    keys_order = read_keys_order()
    if rank == 0:
        print("[trainer] Resolved CFG:", {k: v for k, v in CFG.items() if k != "output_dir"})
        print(f"[trainer] keys_order size: {len(keys_order)}")

    # Tokenizer/collator/formatting via your dataset.py
    tokenizer, data_collator, formatting_fn = ds.get_tokenizer_and_data_collator_and_propt_formatting(CFG["model_name"])

    # Partitioned dataset
    trainset = ds.load_data(int(CFG["partition_id"]), int(CFG["num_partitions"]), CFG["dataset_name"])

    # Plain model (no FSDP yet), using dtype from config
    model_cfg = type("Cfg", (), {
        "name": CFG["model_name"],
        "dtype": CFG.get("model_dtype", "bf16"),
        "gradient_checkpointing": bool(CFG["gradient_checkpointing"]),
        "attn_implementation": CFG.get("attn_implementation", "sdpa"),
    })
    model = mdl.get_model(model_cfg)
    if hasattr(model, "config"):
        model.config.use_cache = False

    # TrainingArguments with FSDP
    args = TrainingArguments(
        output_dir=CFG["output_dir"],
        per_device_train_batch_size=int(CFG["per_device_train_bs"]),
        gradient_accumulation_steps=int(CFG["ga_steps"]),
        learning_rate=float(CFG["learning_rate"]),
        max_steps=int(CFG["max_steps"]),   # burst per handshake
        num_train_epochs=float(3),
        logging_steps=int(CFG["logging_steps"]),
        report_to="none",
        gradient_checkpointing=bool(CFG["gradient_checkpointing"]),
        lr_scheduler_type=str(CFG["lr_scheduler_type"]),
        warmup_ratio=0.03,
        logging_nan_inf_filter=False,
        max_grad_norm=1.0,
        disable_tqdm=True,
        fsdp=str(CFG["fsdp"]),
        fsdp_transformer_layer_cls_to_wrap=str(CFG["fsdp_transformer_layer_cls_to_wrap"]),
        ddp_find_unused_parameters=False,
        # We will set bf16/fp16 via apply_precision_policy() to respect config & hardware
    )

    # ---------- First handshake BEFORE FSDP wrapping ----------
    if rank == 0:
        print("\n[Rank 0] Waiting for ready.flag...", flush=True)
        while not os.path.exists(FLAG_READY):
            time.sleep(0.1)
        print("[Rank 0] ready.flag detected.", flush=True)
    dist.barrier()

    mode = read_mode_on_rank0() if rank == 0 else ""
    mode = broadcast_str(local_rank, mode)
    if rank == 0:
        print(f"[Rank 0] Mode={mode}", flush=True)

    if mode == "stop":
        if rank == 0:
            signal_done_on_rank0()
            print("[Rank 0] Stop acknowledged. Exiting.", flush=True)
        dist.barrier()
        dist.destroy_process_group()
        return

    # Load incoming weights into PLAIN model using client's key order
    load_plain_from_shm_with_order(model, keys_order)

    # Apply precision policy (respect config, fall back safely)
    apply_precision_policy(model, args, want_bf16=bool(CFG["targs_bf16"]), want_fp16=bool(CFG["targs_fp16"]))

    # Build SFTTrainer (wraps with FSDP)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        max_seq_length=int(CFG["seq_length"]),
        train_dataset=trainset,
        formatting_func=formatting_fn,
        data_collator=data_collator,
    )

    # ---------- Per-round loop ----------
    while True:
        # Per-round overrides
        knobs = read_knobs()
        if "learning_rate" in knobs:
            trainer.args.learning_rate = float(knobs["learning_rate"])
        if "max_steps" in knobs:
            trainer.args.max_steps = int(knobs["max_steps"])

        if mode == "train":
            if rank == 0: print("[Rank 0] Training...", flush=True)
            out = trainer.train(resume_from_checkpoint=False)
            if rank == 0:
                loss_attr = getattr(out, "training_loss", None)
                train_loss = float(loss_attr if loss_attr is not None else out.metrics.get("train_loss", 0.0))
                with open(METRICS_JSON, "w") as f:
                    json.dump({"train_loss": train_loss}, f)
        elif mode == "eval":
            if rank == 0: print("[Rank 0] Evaluating...", flush=True)
            metrics = trainer.evaluate()
            if rank == 0:
                with open(METRICS_JSON, "w") as f:
                    json.dump({"eval_loss": float(metrics.get("eval_loss", 0.0))}, f)
        else:
            if rank == 0: print(f"[Rank 0] Unknown mode '{mode}', skipping.", flush=True)

        # Write back FULL weights in the SAME key order and signal done
        dist.barrier()
        write_fsdp_to_shm_with_order(trainer.model, keys_order)

        # ----- Next handshake -----
        if rank == 0:
            print("\n[Rank 0] Waiting for ready.flag...", flush=True)
            while not os.path.exists(FLAG_READY):
                time.sleep(0.1)
            print("[Rank 0] ready.flag detected.", flush=True)
        dist.barrier()
        mode = read_mode_on_rank0() if rank == 0 else ""
        mode = broadcast_str(local_rank, mode)
        if rank == 0:
            print(f"[Rank 0] Mode={mode}", flush=True)

        if mode == "stop":
            if rank == 0:
                signal_done_on_rank0()
                print("[Rank 0] Stop acknowledged. Exiting.", flush=True)
            dist.barrier()
            break

        # Load next incoming weights into the FSDP-wrapped model using the SAME key order
        load_fsdp_from_shm_with_order(trainer.model, keys_order)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
