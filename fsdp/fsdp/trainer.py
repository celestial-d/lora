#!/usr/bin/env python3
# fsdp_trainer.py: Multi-GPU FSDP reader/trainer driven via SHM + flags.
# - First handshake loads plain weights BEFORE FSDP wrapping
# - Per round: train/eval burst -> write FULL_STATE_DICT to SHM -> wait next signal
# - Precision policy: BF16 if supported, else AMP-FP16

# export SHM_NAME=opt_client_0
# export FLAG_DIR=$(pwd)/flags_client_0
# mkdir -p "$FLAG_DIR"

# # Optional knobs the trainer reads (env), unless you add knob JSON reading:
# export MODEL_NAME=facebook/opt-125m
# export DATASET_NAME=sahil2801/CodeAlpaca-20k
# export MAX_STEPS=10
# export LEARNING_RATE=2e-5
# export PER_DEVICE_TRAIN_BS=1
# export LOGGING_STEPS=1

# torchrun --nproc_per_node=<NUM_GPUS> fsdp_trainer.py


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
    from torch.distributed.fsdp.api import FullStateDictConfig  # torch<2.1 compat

# --------- Env/flags ----------
SHM_NAME = os.getenv("SHM_NAME", "opt125m_shared")
FLAG_DIR = os.getenv("FLAG_DIR", ".")
FLAG_READY = os.path.join(FLAG_DIR, "ready.flag")
FLAG_DONE  = os.path.join(FLAG_DIR, "done.flag")
FLAG_MODE  = os.path.join(FLAG_DIR, "mode.flag")
os.makedirs(FLAG_DIR, exist_ok=True)

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
    for p in (FLAG_READY, FLAG_MODE):
        try: os.remove(p)
        except FileNotFoundError: pass
    print("✅ [Rank 0] Weights written & done.flag created.", flush=True)

# --------- SHM IO ----------
def _ensure_size(total_elems_fp16):
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

def load_weights_from_shm_plain(model: torch.nn.Module) -> None:
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

# --------- Precision policy ----------
def prefer_bf16_else_amp_fp16(model: torch.nn.Module, targs: TrainingArguments):
    bf16_ok = False
    try:
        bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    except AttributeError:
        bf16_ok = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8  # SM80+
    if bf16_ok:
        model.to(torch.bfloat16)
        targs.bf16 = True
        targs.fp16 = False
        print("[trainer] Using BF16", flush=True)
    else:
        model.to(torch.float32)
        targs.fp16 = True
        targs.bf16 = False
        print("[trainer] BF16 not supported; using AMP-FP16", flush=True)

# --------- Main ----------
def main():
    # perf niceties
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    model_name   = os.getenv("MODEL_NAME", "facebook/opt-125m")
    dataset_name = os.getenv("DATASET_NAME", "sahil2801/CodeAlpaca-20k")

    # Tokenizer + base model (PLAIN)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    if hasattr(model, "config"):
        model.config.use_cache = False

    # Dataset
    raw = load_dataset(dataset_name, split="train")
    def format_prompt(example):
        instruction = (example.get("instruction") or "").strip()
        input_text  = (example.get("input") or "").strip()
        output      = (example.get("output") or example.get("response") or "").strip()
        q = f"{instruction} {input_text}".strip() if input_text else instruction
        return {"text": f"### Question: {q}\n### Answer: {output}"}
    ds  = raw.map(format_prompt, remove_columns=raw.column_names)
    train_ds = ds.select(range(min(10000, len(ds))))
    eval_ds  = ds.select(range(min(10000, len(ds)), min(10100, len(ds))))

    # Training args (Trainer will wrap with FSDP via accelerate)
    args = TrainingArguments(
        output_dir=os.getenv("OUTPUT_DIR", "./fsdp_output_opt125m"),
        per_device_train_batch_size=int(os.getenv("PER_DEVICE_TRAIN_BS", "1")),
        learning_rate=float(os.getenv("LEARNING_RATE", "2e-5")),
        max_steps=int(os.getenv("MAX_STEPS", "10")),   # burst per handshake
        logging_steps=int(os.getenv("LOGGING_STEPS", "1")),
        save_steps=10_000_000,
        report_to="none",
        fp16=False,  # set by policy below
        bf16=False,  # set by policy below
        fsdp="full_shard auto_wrap",
        fsdp_transformer_layer_cls_to_wrap=os.getenv("FSDP_WRAP_CLS", "OPTDecoderLayer"),
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        lr_scheduler_type=os.getenv("LR_SCHEDULER_TYPE", "constant"),
        warmup_ratio=0.03,
        logging_nan_inf_filter=False,
        max_grad_norm=1.0,
        disable_tqdm=True,
    )
    prefer_bf16_else_amp_fp16(model, args)

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

    # Load incoming weights into PLAIN model
    load_weights_from_shm_plain(model)

    # Build the trainer (FSDP wrapping happens here)
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds if len(eval_ds) > 0 else None,
        tokenizer=tokenizer,
        dataset_text_field="text",
    )

    # ---------- Per-round loop ----------
    while True:
        if mode == "train":
            if rank == 0: print("[Rank 0] Training...", flush=True)
            out = trainer.train(resume_from_checkpoint=False)
            if rank == 0:
                # Prefer attr; fallback to metrics
                loss_attr = getattr(out, "training_loss", None)
                train_loss = float(loss_attr if loss_attr is not None else out.metrics.get("train_loss", 0.0))
                with open(os.path.join(FLAG_DIR, "metrics.json"), "w") as f:
                    json.dump({"train_loss": train_loss}, f)
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
