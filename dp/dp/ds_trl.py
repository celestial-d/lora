# --- add near imports ---
import os, json, math, time, shutil
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer

# rank/device (keep your originals)
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.backends.cuda.matmul.allow_tf32 = True

# NEW: shared dir for HF-style checkpoint in /dev/shm
SHM_DIR = os.environ.get("SHM_DIR", "/dev/shm/llama7b_cycle")

def is_rank0():
    return int(os.environ.get("RANK", "0")) == 0

def barrier():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

def format_row(ex):
    instr = (ex.get("instruction") or "").strip()
    inp   = (ex.get("input") or "").strip()
    out   = (ex.get("output") or "").strip()
    if inp:
        return f"### Question:\n{instr}\n\n### Input:\n{inp}\n\n### Answer:\n{out}"
    return f"### Question:\n{instr}\n\n### Answer:\n{out}"

def main():
    # ===== Config (yours, unchanged) =====
    model_name = "meta-llama/Llama-2-7b-hf"
    dataset_name = "sahil2801/CodeAlpaca-20k"
    output_dir = "./opt67b_codealpaca_zero3"

    max_seq_length = 512
    per_device_train_batch_size = 1
    gradient_accumulation_steps = 16
    num_train_epochs = 3
    learning_rate = 2e-5
    weight_decay = 0.0
    warmup_ratio = 0.03
    lr_scheduler_type = "cosine"
    logging_steps = 10
    save_steps = 500
    save_total_limit = 2
    eval_steps = 250
    seed = 42

    bf16 = True
    fp16 = False
    gradient_checkpointing = True
    attn_impl = "sdpa"
    packing = False

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ===== Dataset (yours) =====
    raw = load_dataset(dataset_name, split="train[:5]").shuffle(seed=seed)
    train = raw.map(lambda ex: {"text": format_row(ex)}, remove_columns=raw.column_names)
    eval_ds = train

    # ===== Tokenizer =====
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # ===== Model: use HF API to load either from SHM_DIR or base model =====
    load_path = SHM_DIR if Path(SHM_DIR, "config.json").exists() else model_name
    model = AutoModelForCausalLM.from_pretrained(
        load_path,
        torch_dtype=torch.bfloat16 if bf16 else torch.float16 if fp16 else None,
        low_cpu_mem_usage=True,
        attn_implementation=attn_impl,
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False

    # ===== TrainingArguments (yours) =====
    optim_name = "adamw_torch"
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
        eval_strategy="steps",
        eval_steps=eval_steps,
        eval_accumulation_steps=1,
        bf16=bf16,
        fp16=fp16,
        optim=optim_name,
        report_to=["none"],
        seed=seed,
        ddp_find_unused_parameters=False,
        deepspeed="ds_zero3_offload.json",
        save_strategy="no",
        save_total_limit=0,
    )

    # ===== Trainer =====
    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=train,
        eval_dataset=eval_ds,
        args=targs,
        max_seq_length=max_seq_length,
        packing=packing,
        dataset_text_field="text",
    )

    results = trainer.train()
    #print("Training completed. Results:", results.training_loss)
    # metrics = trainer.evaluate()
    # if "eval_loss" in metrics and metrics["eval_loss"] is not None:
    #     try:
    #         metrics["perplexity"] = math.exp(metrics["eval_loss"])
    #     except OverflowError:
    #         metrics["perplexity"] = float("inf")
    # with open(Path(output_dir) / "eval_metrics.json", "w") as f:
    #     json.dump(metrics, f, indent=2)
    # print("Final eval (train-as-eval):", metrics)

    # ===== Save back to SHM via HF API (rank-0 only) =====
    
    tmp_dir = Path(SHM_DIR).with_name(Path(SHM_DIR).name + f"_tmp_{int(time.time())}")
    if is_rank0():
        tmp_dir.mkdir(parents=True, exist_ok=True)
        # save model + tokenizer
    trainer.save_model(tmp_dir)      # saves model + config
        #tok.save_pretrained(tmp_dir)     # ensure tokenizer artifacts are there too
        # swap-in atomically (best effort)
    if is_rank0():
        if Path(SHM_DIR).exists():
            shutil.rmtree(SHM_DIR)
        shutil.move(str(tmp_dir), SHM_DIR)
        print(f"[rank0] Saved updated checkpoint to {SHM_DIR}")
        loss_file = Path("/dev/shm/loss.txt")
        if not loss_file.exists():
            with open(loss_file, "w") as f:
                f.write("0.0\n")   # initialize with a default loss value
        with open(loss_file, "w") as f:
            f.write(f"{results.training_loss}\n")
        print(f"[rank0] Wrote train_loss={results.training_loss} to {loss_file}")
    barrier()  # let rank-0 finish saving first

if __name__ == "__main__":
    main()
