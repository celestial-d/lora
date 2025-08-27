# reader.py
# launch with: accelerate launch --multi_gpu reader.py
import os
import time
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
    from torch.distributed.fsdp.api import FullStateDictConfig  # compat

SHM_NAME   = "opt125m_shared"
FLAG_READY = "ready.flag"
FLAG_DONE  = "done.flag"
FLAG_MODE  = "mode.flag"

def format_prompt(example):
    instruction = example["instruction"].strip()
    input_text  = example["input"].strip()
    q = f"{instruction} {input_text}" if input_text else instruction
    return {"text": f"### Question: {q}\n### Answer: {example['output'].strip()}"}

def load_weights_from_shm(model):
    sd = model.state_dict()
    total = sum(p.numel() for p in sd.values())
    shm = shared_memory.SharedMemory(name=SHM_NAME)
    flat = np.ndarray((total,), dtype=np.float16, buffer=shm.buf)
    ptr = 0
    new_sd = {}
    for name, param in sd.items():
        n = param.numel()
        new_sd[name] = torch.from_numpy(flat[ptr:ptr+n].copy()).view(param.shape).to(param.dtype)
        ptr += n
    model.load_state_dict(new_sd)
    shm.close()
    print(f"[Rank {dist.get_rank()}] ✅ Loaded weights from shared memory.", flush=True)
    return model

def read_mode_on_rank0():
    with open(FLAG_MODE, "r") as f:
        return f.read().strip().lower()

def broadcast_mode(local_rank, mode: str | None):
    # simple robust string broadcast
    if dist.get_rank() == 0:
        mb = mode.encode("utf-8")
        size = torch.tensor([len(mb)], dtype=torch.int64, device=local_rank)
    else:
        mb = None
        size = torch.zeros(1, dtype=torch.int64, device=local_rank)
    dist.broadcast(size, src=0)
    buf = torch.empty(int(size.item()), dtype=torch.uint8, device=local_rank)
    if dist.get_rank() == 0:
        buf[:] = torch.tensor(list(mb), dtype=torch.uint8, device=local_rank)
    dist.broadcast(buf, src=0)
    return bytes(buf.tolist()).decode("utf-8")

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m", padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", torch_dtype=torch.float16)
    if hasattr(model, "config"):
        model.config.use_cache = False  # silence ckpt warning

    raw = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    ds  = raw.map(format_prompt)
    train_ds = ds.select(range(500))
    eval_ds  = ds.select(range(500, 600))

    args = TrainingArguments(
        output_dir="./fsdp_output_opt27b",
        per_device_train_batch_size=1,
        learning_rate=2e-5,
        max_steps=10,               # single burst per handshake
        logging_steps=1,
        save_steps=10_000_000,
        report_to="none",
        fp16=True,
        fsdp="full_shard auto_wrap",
        fsdp_transformer_layer_cls_to_wrap="OPTDecoderLayer",
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        dataset_text_field="text",
    )

    while True:
        if rank == 0:
            print("\n[Rank 0] Waiting for ready.flag...", flush=True)
            while not os.path.exists(FLAG_READY):
                time.sleep(0.1)
            print("[Rank 0] ready.flag detected.", flush=True)

        dist.barrier()

        mode = read_mode_on_rank0() if rank == 0 else None
        mode = broadcast_mode(local_rank, mode)
        if rank == 0:
            print(f"[Rank 0] Mode={mode}", flush=True)

        # Stop signal from writer: acknowledge and exit
        if mode == "stop":
            if rank == 0:
                with open(FLAG_DONE, "w") as f:
                    f.write("done"); f.flush(); os.fsync(f.fileno())
                # clean up flags written by writer for the final round
                for p in (FLAG_READY, FLAG_MODE):
                    try: os.remove(p)
                    except FileNotFoundError: pass
                print("[Rank 0] Stop acknowledged. Exiting.", flush=True)
            dist.barrier()
            break

        # Load incoming weights
        trainer.model = load_weights_from_shm(trainer.model)

        # Do work per mode
        if mode == "train":
            if rank == 0: print("[Rank 0] Training...", flush=True)
            trainer.train()
        elif mode == "eval":
            if rank == 0: print("[Rank 0] Evaluating...", flush=True)
            metrics = trainer.evaluate()
            if rank == 0: print(f"[Rank 0] Eval metrics: {metrics}", flush=True)
        else:
            if rank == 0: print(f"[Rank 0] Unknown mode '{mode}', skipping.", flush=True)

        dist.barrier()

        # Gather full model (all ranks participate; only rank0 receives)
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(trainer.model, StateDictType.FULL_STATE_DICT, cfg):
            full_state = trainer.model.state_dict()
        dist.barrier()

        # Write back + signal done (even after eval to keep protocol uniform)
        if rank == 0:
            flat = torch.cat([p.detach().flatten().cpu().to(torch.float16) for p in full_state.values()])
            shm_out = shared_memory.SharedMemory(name=SHM_NAME)
            np_out = np.ndarray(flat.shape, dtype=np.float16, buffer=shm_out.buf)
            np_out[:] = flat[:]
            shm_out.close()

            with open(FLAG_DONE, "w") as f:
                f.write("done"); f.flush(); os.fsync(f.fileno())

            # clean flags for the next handshake
            for path in (FLAG_READY, FLAG_MODE):
                try: os.remove(path)
                except FileNotFoundError: pass
            print("✅ [Rank 0] Weights written & done.flag created.", flush=True)

        dist.barrier()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()