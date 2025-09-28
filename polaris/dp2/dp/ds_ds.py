# ds_trl.py
# DeepSpeed + TRL SFT trainer with bidirectional streaming only.
# Ingests initial weights from SHM_DIR/in_stream/, trains, then streams
# updated weights to SHM_DIR/out_stream/ (double-buffered safetensors).
#
# All ranks enter ZeRO gather collectives; only writer rank (0) writes.

# ===== COMPILER & DS OPTIMIZER HARDENING (add near top, before any DS/HF imports) =====

import os, torch
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
# If you call init_process_group yourself, pass device_id=local_rank on torch>=2.4


import os, sys, json
from pathlib import Path
import shutil

def _which_many(cands):
    for c in cands:
        p = shutil.which(c)
        if p:
            return p
        if os.path.isabs(c) and os.path.exists(c):
            return c
    return None

# 1) Force modern compilers for torch cpp_extension / DS JIT
gcc_candidates = [
    "g++-12", "gcc-12", "/usr/bin/g++-12", "/usr/local/bin/g++-12",
    "/soft/compilers/gcc/12.3.0/bin/g++", "/soft/compilers/gcc/12.3.0/bin/gcc",
    "/opt/gcc/12.3.0/bin/g++", "/opt/gcc/12.3.0/bin/gcc",
]
gxx = _which_many(gcc_candidates)
gcc = gxx  # ok to set both to the same path; torch checks CXX
if gxx:
    os.environ["CXX"] = gxx
    os.environ["CC"]  = gcc
    print(f"[ds_trl] Using CXX={gxx}")
else:
    # Fall back to whatever is present; we’ll also prepare a DS config fallback below
    print("[ds_trl] WARNING: Could not find GCC/G++ 12; will enable DS fallback if needed")

# 2) Speed up builds and avoid over-broad arch compile
# Polaris = A100 → sm_80; adjust if you’re on H100 (set '90') or V100 ('70')
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "80")

# 3) Optional: avoid building extra DS ops you don’t use
os.environ.setdefault("DS_BUILD_AIO", "0")
os.environ.setdefault("DS_BUILD_SPARSE_ATTN", "0")
os.environ.setdefault("DS_BUILD_QUANTIZATION", "0")

# 4) If we fail to provide modern compiler, plan to avoid CPUAdam op build:
def _rewrite_deepspeed_config_if_needed(ds_cfg_path: str) -> str:
    """
    If no GCC>=9 was found AND the config requests CPUAdam, rewrite to avoid CPUAdam.
    Returns path to the (possibly) rewritten config file.
    """
    need_fallback = ("CXX" not in os.environ) or ("-12" not in os.environ.get("CXX","") and "12." not in os.environ.get("CXX",""))
    if not Path(ds_cfg_path).exists():
        return ds_cfg_path
    try:
        cfg = json.loads(Path(ds_cfg_path).read_text())
    except Exception:
        return ds_cfg_path

    opt = (cfg.get("optimizer") or {})
    opt_type = (opt.get("type") or "").lower()
    if need_fallback and "cpuadam" in opt_type:
        print("[ds_trl] Rewriting DeepSpeed config to avoid CPUAdam (no modern GCC).")
        # Switch to AdamW (PyTorch) or FusedAdam (if available) to avoid DS CPU extension.
        cfg["optimizer"] = {
            "type": "AdamW",
            "params": {
                "lr": opt.get("params", {}).get("lr", 2e-5),
                "betas": opt.get("params", {}).get("betas", [0.9, 0.999]),
                "eps": opt.get("params", {}).get("eps", 1e-8),
                "weight_decay": opt.get("params", {}).get("weight_decay", 0.0)
            }
        }
        # If original config had ZeRO-3 offload to CPU, keep it; optimizer now runs in PyTorch.
        out_dir = os.environ.get("DS_RUNTIME_DIR", str(Path.home() / ".ds_runtime"))
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        new_path = str(Path(out_dir) / "ds_cfg_no_cpuadam.json")
        Path(new_path).write_text(json.dumps(cfg, indent=2))
        return new_path

    return ds_cfg_path

# Expose helper to main() by putting it in globals
globals()["_rewrite_deepspeed_config_if_needed"] = _rewrite_deepspeed_config_if_needed
# =======================================================================================

# ===== MUST BE FIRST LINES OF FILE =====
import os, sys
from pathlib import Path

def _mk(p: str) -> str:
    Path(p).mkdir(parents=True, exist_ok=True); return p

# Choose a safe, exec-able filesystem (home/scratch). Never /dev/shm.
_CACHE_ROOT = os.environ.get("CACHE_ROOT_OVERRIDE", str(Path.home() / ".cache_ds"))
XDG      = _mk(os.path.join(_CACHE_ROOT, "xdg"))
TRITON   = _mk(os.path.join(_CACHE_ROOT, "triton"))
INDUCT   = _mk(os.path.join(_CACHE_ROOT, "torch_inductor"))
NVCCACHE = _mk(str(Path.home() / ".nv" / "ComputeCache"))

# Hard override envs so downstream imports see them
os.environ["XDG_CACHE_HOME"] = XDG
os.environ["TRITON_CACHE_DIR"] = TRITON
os.environ["TORCHINDUCTOR_CACHE_DIR"] = INDUCT
os.environ["CUDA_CACHE_PATH"] = NVCCACHE  # NVRTC temp cache (some stacks look here)
os.environ.pop("TRITON_CACHE_PATH", None)    # nuke alias pointing to /dev/shm if present
os.environ.pop("TRITON_HOME", None)

# If anything still points to /dev/shm, fix it and re-exec before importing heavy libs
def _poisoned_env() -> bool:
    for k in ("TRITON_CACHE_DIR","TRITON_CACHE_PATH","TRITON_HOME"):
        v = os.environ.get(k, "")
        if v.startswith("/dev/shm"):
            return True
    return False

# If Triton was somehow pre-imported (via sitecustomize or a wrapper), re-exec now
if "triton" in sys.modules or _poisoned_env():
    os.environ["TRITON_CACHE_DIR"] = TRITON
    # Clean any stale shm cache that could be preferred
    try:
        import shutil
        shutil.rmtree("/dev/shm/triton_cache", ignore_errors=True)
    except Exception:
        pass
    os.execvpe(sys.executable, [sys.executable] + sys.argv, os.environ)

# Import Triton early so it binds to our dirs; also patch cache dir defensively
try:
    import triton  # noqa: F401
    try:
        # Triton 2.x style
        from triton.runtime.cache import get_cache_manager
        cm = get_cache_manager()
        try:
            # Force cache dir in case defaults were computed already
            cm._cache_dir = TRITON  # private attr but effective
        except Exception:
            pass
        print("[ds_trl] Triton cache dir =", cm._cache_dir)
    except Exception:
        # Older/newer Triton: at least show env value
        print("[ds_trl] Triton cache env TRITON_CACHE_DIR =", os.environ.get("TRITON_CACHE_DIR"))
except Exception as e:
    print("[ds_trl] Triton import not available yet:", e)

# Also steer TorchInductor explicitly (in addition to env)
try:
    import torch._inductor.config as inductor_cfg  # lightweight import
    inductor_cfg.cache_dir = INDUCT
except Exception:
    pass
# ===== END HARDENED CACHE HEADER =====



import json
import time
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from safetensors.torch import save_file, load_file


# DeepSpeed (optional)
try:
    # import deepspeed
    from deepspeed import zero as ds_zero
    HAS_DEEPSPEED = True
except Exception:
    HAS_DEEPSPEED = False

# your modules
from dp.dataset import (
    get_tokenizer_and_data_collator_and_propt_formatting,
    load_data,
)

print("\n========== ds_trl.py STREAM_PROTOCOL = ON (double-buffered safetensors) ==========\n")

# -------------------------- Distributed setup --------------------------
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.backends.cuda.matmul.allow_tf32 = True

SHM_DIR = os.environ.get("SHM_DIR", "/dev/shm/llama7b_cycle")
STREAM_CHUNK_BYTES = int(os.environ.get("STREAM_CHUNK_BYTES", str(512 * 1024**2)))
STREAM_WINDOW_SIZE = int(os.environ.get("STREAM_WINDOW_SIZE", "2"))

def rank() -> int:
    try:
        return int(os.environ.get("RANK", "0"))
    except Exception:
        return 0

def world_size() -> int:
    try:
        return int(os.environ.get("WORLD_SIZE", "128"))
    except Exception:
        return 1

def is_rank0() -> bool:
    return rank() == 0

def barrier():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

# ----------------------- Arg/env/json resolution -----------------------
def parse_cli():
    p = argparse.ArgumentParser()
    p.add_argument("--partition-id", type=int, default=None)
    p.add_argument("--num-partitions", type=int, default=None)
    p.add_argument("--num-rounds", type=int, default=None)
    p.add_argument("--model-name", type=str, default=None)
    p.add_argument("--dataset-name", type=str, default=None)
    return p.parse_args()

def read_json_handoff(shm_path: Path):
    f = shm_path / "context.json"
    if f.exists():
        try:
            return json.loads(f.read_text())
        except Exception:
            pass
    return {}

def resolve_ctx():
    ctx = {
        "partition_id": 0,
        "num_partitions": 1,
        "num_rounds": 1,
        "model_name": os.getenv("MODEL_NAME", "/home/zhangduo4610/opt-125m"),
        "dataset_name": os.getenv("DATASET_NAME", "/home/zhangduo4610/CodeAlpaca-20k "),
    }
    hand = read_json_handoff(Path(SHM_DIR))
    for k in ("partition_id", "num_partitions", "num_rounds", "model_name", "dataset_name"):
        if k in hand and k is not None:
            ctx[k] = hand[k]
    ctx["partition_id"]   = int(os.getenv("FL_PARTITION_ID",   ctx["partition_id"]))
    ctx["num_partitions"] = int(os.getenv("FL_NUM_PARTITIONS", ctx["num_partitions"]))
    ctx["num_rounds"]     = int(os.getenv("FL_NUM_ROUNDS",     ctx["num_rounds"]))
    args = parse_cli()
    if args.partition_id  is not None: ctx["partition_id"] = int(args.partition_id)
    if args.num_partitions is not None: ctx["num_partitions"] = int(args.num_partitions)
    if args.num_rounds    is not None: ctx["num_rounds"] = int(args.num_rounds)
    if args.model_name    is not None: ctx["model_name"] = args.model_name
    if args.dataset_name  is not None: ctx["dataset_name"] = args.dataset_name
    return ctx

# --------------------------- Streaming helpers -------------------------
def apply_safetensors_chunk_inplace(model: torch.nn.Module, path: Path):
    tensors = load_file(str(path))
    name_to_param = dict(model.named_parameters())
    name_to_buffer = dict(model.named_buffers())

    def _lookup(name: str):
        t = name_to_param.get(name)
        if t is None:
            t = name_to_buffer.get(name)
        if t is None and name.startswith("module."):
            base = name[len("module."):]
            t = name_to_param.get(base) or name_to_buffer.get(base)
        if t is None:
            mod = "module." + name
            t = name_to_param.get(mod) or name_to_buffer.get(mod)
        return t

    with torch.no_grad():
        for name, v in tensors.items():
            if v.numel() == 0:
                continue
            target = _lookup(name)
            if target is None:
                continue
            if target.shape != v.shape:
                raise RuntimeError(f"Shape mismatch for {name}: {target.shape} vs {v.shape}")
            target.data.copy_(v.to(dtype=target.dtype))

def read_stream_into_model(model: torch.nn.Module, stream_dir: Path):
    i, total = 0, None
    end_path = stream_dir / "end.json"
    print(f"[ds_trl] Waiting for input stream at {stream_dir}")
    while True:
        if total is None and end_path.exists():
            try:
                total = json.loads(end_path.read_text()).get("total", None)
            except Exception:
                total = None

        ready = stream_dir / f"chunk_{i:05d}.ready"
        if ready.exists():
            chunk = stream_dir / f"chunk_{i:05d}.safetensors"
            while not chunk.exists():
                time.sleep(0.02)
            apply_safetensors_chunk_inplace(model, chunk)
            (stream_dir / f"chunk_{i:05d}.done").touch()
            try:
                chunk.unlink(missing_ok=True)
                ready.unlink(missing_ok=True)
            except Exception:
                pass
            i += 1
            continue

        if total is not None and i >= total:
            print(f"[ds_trl] Input stream applied ({i} chunks).")
            break
        time.sleep(0.05)

def _to_cpu_full_tensor(param: torch.nn.Parameter, writer_rank: int) -> torch.Tensor:
    """Return full (gathered) CPU tensor for a possibly ZeRO-3 partitioned param."""
    if HAS_DEEPSPEED and hasattr(param, "ds_id"):
        # All ranks must enter the context in same order.
        with ds_zero.GatheredParameters([param], modifier_rank=writer_rank):
            if rank() == writer_rank:
                if param.numel() == 0:
                    return torch.empty(0, dtype=param.dtype)
                return param.data.detach().cpu().contiguous().clone()
            else:
                # Non-writer: participate in collective, discard
                return torch.empty(0, dtype=param.dtype)
    # Regular param
    return param.detach().cpu().contiguous() if rank() == writer_rank else torch.empty(0, dtype=param.dtype)

def write_streamed_safetensors(model: torch.nn.Module, stream_dir: Path,
                               writer_rank: int = 0,
                               max_chunk_bytes: int = STREAM_CHUNK_BYTES,
                               window_size: int = STREAM_WINDOW_SIZE):
    """All ranks iterate & participate in ZeRO gathers. Only writer_rank writes."""
    is_writer = rank() == writer_rank
    if is_writer:
        stream_dir.mkdir(parents=True, exist_ok=True)

    def wait_for_done(idx_needed: int):
        if not is_writer:
            return  # only writer does backpressure waiting
        while not (stream_dir / f"chunk_{idx_needed:05d}.done").exists():
            time.sleep(0.05)

    def flush_chunk(tensors_dict: dict, idx: int):
        if not is_writer:
            return
        tmp = stream_dir / f"chunk_{idx:05d}.safetensors.tmp"
        final = stream_dir / f"chunk_{idx:05d}.safetensors"
        save_file(tensors_dict, str(tmp))
        os.replace(tmp, final)
        (stream_dir / f"chunk_{idx:05d}.ready").touch()

    idx = 0
    cur = {}
    cur_bytes = 0

    with torch.no_grad():
        # Parameters (handle ZeRO gathering) — all ranks enter contexts in same order
        for name, param in model.named_parameters():
            t = _to_cpu_full_tensor(param, writer_rank)
            if is_writer and t.numel() > 0:
                nbytes = t.numel() * t.element_size()
                if cur and cur_bytes + nbytes > max_chunk_bytes:
                    if idx >= window_size:
                        wait_for_done(idx - window_size)
                    flush_chunk(cur, idx)
                    idx += 1
                    cur, cur_bytes = {}, 0
                cur[name] = t
                cur_bytes += nbytes

        # Buffers (not ZeRO-partitioned typically) — writer only needs to handle
        for name, buf in model.named_buffers():
            if not is_writer:
                continue
            t = buf.detach().cpu().contiguous()
            if t.numel() == 0:
                continue
            nbytes = t.numel() * t.element_size()
            if cur and cur_bytes + nbytes > max_chunk_bytes:
                if idx >= window_size:
                    wait_for_done(idx - window_size)
                flush_chunk(cur, idx)
                idx += 1
                cur, cur_bytes = {}, 0
            cur[name] = t
            cur_bytes += nbytes

        if is_writer and cur:
            if idx >= window_size:
                wait_for_done(idx - window_size)
            flush_chunk(cur, idx)
            idx += 1

    # Signal completion
    if is_writer:
        (stream_dir / "end.json").write_text(json.dumps({"total": idx}))
        print(f"[ds_trl][writer rank={writer_rank}] Output stream written ({idx} chunks) to {stream_dir}")

import deepspeed
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ------------------------------- Main ----------------------------------
def main():
    # Training defaults
    max_seq_length = 512
    per_device_train_batch_size = 1
    gradient_accumulation_steps = 128
    num_train_epochs = 1
    learning_rate = 2e-5
    weight_decay = 0.0
    warmup_ratio = 0.03
    lr_scheduler_type = "cosine"
    logging_steps = 10
    seed = 42
    bf16 = True
    fp16 = False
    gradient_checkpointing = True
    attn_impl = "sdpa"
    packing = False
    optim_name = "adamw_torch"
    deepspeed_cfg = "/lus/eagle/projects/SR-APPFL/duo/lora/dp2/dp/ds_zero3_offload.json"

    print("[ds_trl] Using DeepSpeed config:", deepspeed_cfg)

    ctx = resolve_ctx()
    partition_id   = 0
    num_partitions = 2
    model_name     = ctx["model_name"]
    dataset_name   = ctx["dataset_name"]

    # Data/Tokenizer
    trainset = load_data(partition_id, num_partitions, dataset_name)
    
    #print to check 
    print(f"[ds_trl] len(trainset) = {len(trainset)}")
    G = int(os.environ.get("WORLD_SIZE", "1"))
    micro = per_device_train_batch_size
    gas = gradient_accumulation_steps
    global_bs = G * micro * gas
    import math
    print(f"[ds_trl] global_batch={global_bs}, expected_steps/epoch={math.ceil(len(trainset)/global_bs)}")

    
    tokenizer, data_collator, formatting_prompts_func = (
        get_tokenizer_and_data_collator_and_propt_formatting(model_name)
    )

    # Model skeleton, then ingest initial stream
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     torch_dtype=(torch.bfloat16 if bf16 else torch.float16 if fp16 else None),
    #     low_cpu_mem_usage=True,
    #     attn_implementation=attn_impl,
    #     trust_remote_code=True,
    # )
    # if gradient_checkpointing:
    #     model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    #     model.config.use_cache = False
    def model_init():
    # Build under ZeRO partitioning so no rank holds the full 70B
        dtype = torch.bfloat16 if bf16 else (torch.float16 if fp16 else None)
        with deepspeed.zero.Init(remote_device="cpu", pin_memory=True, dtype=dtype):
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                attn_implementation=attn_impl,
                trust_remote_code=True,
            )
        # activation savings
        if gradient_checkpointing:
            model.gradient_checkpointing_enable(use_reentrant=False)
        model.config.use_cache = False
        return model


    # in_dir = Path(SHM_DIR) / "in_stream"
    # if is_rank0():
    #     t_in_start = time.time()
    #     read_stream_into_model(model, in_dir)
    #     t_in_end = time.time()
    #     # bytes_in = 0
    #     # for p in in_dir.glob("chunk_*.safetensors"):
    #     #     try:
    #     #         bytes_in += p.stat().st_size
    #     #     except Exception:
    #     #         pass
    #     # print(f"[ds_trl][rank0] ✅ input stream written in {t_in_end - t_in_start:.3f}s "
    #     #       f"(~{bytes_in} bytes)")
    #     print(f"[ds_trl][rank0] ✅ input stream written in {t_in_end - t_in_start:.3f}s ")
    # barrier()

    # Trainer args (no full HF saves)
    out_tmp = str(Path(SHM_DIR) / "hf_trainer_tmp")  # scratch only
    Path(out_tmp).mkdir(parents=True, exist_ok=True)
    targs = TrainingArguments(
        output_dir=out_tmp,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        logging_steps=logging_steps,
        eval_strategy="no",
        bf16=bf16,
        fp16=fp16,
        optim=optim_name,
        report_to=["none"],
        seed=seed,
        ddp_find_unused_parameters=False,
        deepspeed=deepspeed_cfg,
        save_strategy="no",
        save_total_limit=0,
        max_steps=12, 
    )

    trainer = SFTTrainer(
        model=model_init,
        tokenizer=tokenizer,
        train_dataset=trainset,
        args=targs,
        max_seq_length=max_seq_length,
        packing=packing,
        data_collator=data_collator,
        formatting_func=formatting_prompts_func,
    )

    results = trainer.train()

    # IMPORTANT: all ranks participate in ZeRO gathers; only rank0 writes.
    # out_dir = Path(SHM_DIR) / "out_stream"
    # print(f"[ds_trl] Starting streamed write on all ranks (writer=0).")
    # if is_rank0():
    #     t_out_start = time.time()
    # write_streamed_safetensors(model, out_dir, writer_rank=0,
    #                            max_chunk_bytes=STREAM_CHUNK_BYTES,
    #                            window_size=STREAM_WINDOW_SIZE)
    # if is_rank0():
    #     t_out_end = time.time()
    if is_rank0():
        # metrics for client
        # bytes_out = 0
        # for p in out_dir.glob("chunk_*.safetensors"):
        #     try:
        #         bytes_out += p.stat().st_size
        #     except Exception:
        #         pass
        # print(f"[ds_trl][rank0] ✅ Output stream written in {t_out_end - t_out_start:.3f}s "
        #       f"(~{bytes_out} bytes)")
        # print(f"[ds_trl][rank0] ✅ Output stream written in {t_out_end - t_out_start:.3f}s ")
        # Path("/dev/shm/loss.txt").write_text(f"{results.training_loss}\n")
        # (Path(SHM_DIR) / "metrics.json").write_text(
        #     json.dumps({"train_loss": float(results.training_loss)}, indent=2)
        # )
        #print(f"[rank0] Train done. train_loss={results.training_loss}. Streamed to {out_dir}")
        print(f"[rank0] Train done. train_loss={results.training_loss}.")

    barrier()

if __name__ == "__main__":
    main()
