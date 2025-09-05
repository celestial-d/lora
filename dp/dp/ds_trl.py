# ds_trl.py
import os, json, math, time, shutil, argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer

# your modules
from dp.dataset import (
    get_tokenizer_and_data_collator_and_propt_formatting,
    load_data,
)

# ---------- distributed basics ----------
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.backends.cuda.matmul.allow_tf32 = True

SHM_DIR = os.environ.get("SHM_DIR", "/dev/shm/llama7b_cycle")

def is_rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0

def barrier():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

# ---------- arg/env/json resolution ----------
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
    # defaults
    ctx = {
        "partition_id": 0,
        "num_partitions": 1,
        "num_rounds": 1,
        "model_name": os.getenv("MODEL_NAME", "facebook/opt-125m"),
        "dataset_name": os.getenv("DATASET_NAME", "unknown"),
    }
    # JSON in SHM (optional)
    hand = read_json_handoff(Path(SHM_DIR))
    for k in ("partition_id", "num_partitions", "num_rounds", "model_name", "dataset_name"):
        if k in hand and hand[k] is not None:
            ctx[k] = hand[k]
    # env fallbacks
    ctx["partition_id"]   = int(os.getenv("FL_PARTITION_ID",   ctx["partition_id"]))
    ctx["num_partitions"] = int(os.getenv("FL_NUM_PARTITIONS", ctx["num_partitions"]))
    ctx["num_rounds"]     = int(os.getenv("FL_NUM_ROUNDS",     ctx["num_rounds"]))
    # CLI has highest precedence
    args = parse_cli()
    if args.partition_id  is not None: ctx["partition_id"] = int(args.partition_id)
    if args.num_partitions is not None: ctx["num_partitions"] = int(args.num_partitions)
    if args.num_rounds    is not None: ctx["num_rounds"] = int(args.num_rounds)
    if args.model_name    is not None: ctx["model_name"] = args.model_name
    if args.dataset_name  is not None: ctx["dataset_name"] = args.dataset_name
    return ctx

def main():
    # -------- training config (static defaults; override via run_config if you want) --------
    max_seq_length = 512
    per_device_train_batch_size = 1
    gradient_accumulation_steps = 16
    num_train_epochs = 3
    learning_rate = 2e-5
    weight_decay = 0.0
    warmup_ratio = 0.03
    lr_scheduler_type = "cosine"
    logging_steps = 10
    eval_steps = 250
    seed = 42
    bf16 = True
    fp16 = False
    gradient_checkpointing = True
    attn_impl = "sdpa"
    packing = False
    optim_name = "adamw_torch"
    deepspeed_cfg = "ds_zero3_offload.json"  # ensure this path exists

    ctx = resolve_ctx()
    partition_id   = ctx["partition_id"]
    num_partitions = ctx["num_partitions"]
    model_name     = ctx["model_name"]
    dataset_name   = ctx["dataset_name"]

    # -------- dataset/tokenizer via your existing API --------
    client_trainset = load_data(partition_id, num_partitions, dataset_name)
    tokenizer, data_collator, formatting_prompts_func = (
        get_tokenizer_and_data_collator_and_propt_formatting(model_name)
    )

    # -------- model: load from SHM checkpoint if present, else base model --------
    load_path = Path(SHM_DIR) if (Path(SHM_DIR) / "config.json").exists() else model_name
    model = AutoModelForCausalLM.from_pretrained(
        load_path,
        torch_dtype=(torch.bfloat16 if bf16 else torch.float16 if fp16 else None),
        low_cpu_mem_usage=True,
        attn_implementation=attn_impl,
        trust_remote_code=True,
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False

    # -------- training args --------
    output_dir = "./opt67b_codealpaca_zero3"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    targs = TrainingArguments(
        output_dir=output_dir,
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
    )

    # -------- SFTTrainer: use your formatting func (so DO NOT set dataset_text_field) --------
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=client_trainset,
        args=targs,
        max_seq_length=max_seq_length,
        packing=packing,
        data_collator=data_collator,
        formatting_func=formatting_prompts_func,
    )

    results = trainer.train()

    # -------- atomic save to SHM + write loss (rank0 only) --------
    tmp_dir = Path(SHM_DIR).with_name(Path(SHM_DIR).name + f"_tmp_{int(time.time())}")
    if is_rank0():
        tmp_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(tmp_dir)  # save model + config
        # tokenizer.save_pretrained(tmp_dir)  # optional, if not already in tmp_dir
    if is_rank0():
        # swap in
        if Path(SHM_DIR).exists():
            shutil.rmtree(SHM_DIR)
        shutil.move(str(tmp_dir), SHM_DIR)
        print(f"[rank0] Saved updated checkpoint to {SHM_DIR}")

        # write loss next to checkpoint to avoid cross-client collisions
        loss_file = Path("/dev/shm/loss.txt")
        with open(loss_file, "w") as f:
            f.write(f"{results.training_loss}\n")
        print(f"[rank0] Wrote train_loss={results.training_loss} to {loss_file}")

        # optional: write metrics json
        (Path(SHM_DIR) / "metrics.json").write_text(json.dumps(
            {"train_loss": float(results.training_loss)}, indent=2
        ))
    barrier()

if __name__ == "__main__":
    main()
