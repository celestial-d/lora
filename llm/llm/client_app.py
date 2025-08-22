"""flowertune-llm: A Flower / FlowerTune client (dense FT, no LoRA/quant).
Adds stability fixes:
- Switch to AMP-FP16 (GradScaler) to avoid FP16 NaNs
- Disable logging_nan_inf_filter to reveal real NaNs during debug
- NaN guard before sending weights
- Detailed diagnostics
"""

import os
import math
import warnings
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flwr.common.config import unflatten_dict
from flwr.common.typing import NDArrays, Scalar
from omegaconf import DictConfig

from transformers import TrainingArguments
from trl import SFTTrainer

from llm.dataset import (
    get_tokenizer_and_data_collator_and_propt_formatting,
    load_data,
    replace_keys,
)
from llm.models import (
    cosine_annealing,
    get_model,
    set_parameters,
    get_parameters,
)

# Avoid noisy warnings
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["RAY_DISABLE_DOCKER_CPU_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)


def _human_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024 or unit == "TB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{n} {unit}"
        n /= 1024
    return f"{n:.2f} TB"


def _summarize_model_size(model: torch.nn.Module) -> Dict[str, object]:
    sd = model.state_dict()
    total_params = 0
    total_bytes = 0
    per_dtype_bytes: Dict[str, int] = {}
    largest_name = None
    largest_bytes = 0
    for name, t in sd.items():
        numel = t.numel()
        total_params += numel
        b = numel * t.element_size()
        total_bytes += b
        dt = str(t.dtype).replace("torch.", "")
        per_dtype_bytes[dt] = per_dtype_bytes.get(dt, 0) + b
        if b > largest_bytes:
            largest_bytes = b
            largest_name = name
    transport_bytes_fp16 = sum(t.numel() * 2 for t in sd.values())
    return {
        "total_params": total_params,
        "total_bytes": total_bytes,
        "per_dtype_bytes": per_dtype_bytes,
        "largest_name": largest_name,
        "largest_bytes": largest_bytes,
        "transport_bytes_fp16": transport_bytes_fp16,
    }


def _count_supervised_tokens(tokenizer, data_collator, formatting_fn, dataset, seq_len: int) -> int:
    """Build one real batch using the exact collator/formatter and count non-masked labels."""
    if len(dataset) == 0:
        return 0
    take = min(4, len(dataset))
    subset = dataset.select(range(take))

    def _as_dict_of_lists(batch_list):
        keys = batch_list[0].keys()
        return {k: [row[k] for row in batch_list] for k in keys}

    def _cf(batch_list):
        batch_dict = _as_dict_of_lists(batch_list)
        texts = formatting_fn(batch_dict)
        enc = tokenizer(texts, padding=True, truncation=True, max_length=seq_len, return_tensors="pt")
        examples = [{k: enc[k][i] for k in enc} for i in range(enc["input_ids"].size(0))]
        return data_collator(examples)

    dl = DataLoader(subset, batch_size=min(2, take), shuffle=False, collate_fn=_cf)
    batch = next(iter(dl))
    labels = batch.get("labels", None)
    if labels is None:
        return 0
    return int((labels != -100).sum().item())


def _has_nan_or_inf(arrs: NDArrays) -> bool:
    for a in arrs:
        if not np.isfinite(a).all():
            return True
    return False


class FlowerClient(NumPyClient):
    """Flower client for dense full-model fine-tuning (no LoRA, no quantization)."""

    def __init__(
        self,
        model_cfg: DictConfig,
        train_cfg: DictConfig,
        trainset,
        tokenizer,
        formatting_prompts_func,
        data_collator,
        num_rounds: int,
        partition_id: int,
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.training_arguments = TrainingArguments(**train_cfg.training_arguments)
        self.tokenizer = tokenizer
        self.formatting_prompts_func = formatting_prompts_func
        self.data_collator = data_collator
        self.num_rounds = num_rounds
        self.trainset = trainset
        self.partition_id = partition_id

        # --- Stability defaults
        # reveal NaNs in logs (HF otherwise prints 0.0)
        self.training_arguments.logging_nan_inf_filter = False
        # ensure there is some grad clipping (HF default is 1.0, keep/override here if needed)
        if getattr(self.training_arguments, "max_grad_norm", None) is None:
            self.training_arguments.max_grad_norm = 1.0

        # Instantiate dense model
        self.model = get_model(model_cfg)

        # Prefer AMP-FP16 over pure FP16 to avoid NaNs:
        self._prefer_amp_fp16()

    def _prefer_amp_fp16(self) -> None:
        """
        If the model was loaded in FP16, switch to AMP-FP16:
          - cast model to FP32 master params
          - enable fp16=True in Trainer (GradScaler on)
        BF16 stays BF16 (no scaler). FP32 stays FP32 (can still set fp16=True if desired).
        """
        param_dtype = next(self.model.parameters()).dtype
        if param_dtype == torch.float16:
            print(f"[client {self.partition_id}] Switching to AMP-FP16 (cast model->fp32, Trainer.fp16=True)")
            # Cast model to fp32 master weights to work with GradScaler
            self.model = self.model.to(torch.float32)
            self.training_arguments.fp16 = True
            self.training_arguments.bf16 = False
        elif param_dtype == torch.bfloat16:
            self.training_arguments.bf16 = True
            self.training_arguments.fp16 = False
        else:
            # model already fp32; if user didn't force bf16, enable fp16 for speed
            if not getattr(self.training_arguments, "bf16", False) and getattr(self.training_arguments, "fp16", None) is None:
                self.training_arguments.fp16 = True

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict]:
        """One FL local training round."""
        # Load global weights
        set_parameters(self.model, parameters)

        # Per-round cosine LR
        round_id = int(config["current_round"])
        new_lr = cosine_annealing(
            round_id,
            self.num_rounds,
            self.train_cfg.learning_rate_max,
            self.train_cfg.learning_rate_min,
        )

        # Unique output dir per client & round to avoid HF auto-resume
        outdir = os.path.join(config["save_path"], f"client_{self.partition_id}", f"round_{round_id}")
        os.makedirs(outdir, exist_ok=True)

        # ---- Print model size BEFORE training ----
        stats = _summarize_model_size(self.model)
        per_dtype_str = ", ".join(
            f"{dt}:{_human_bytes(b)}" for dt, b in sorted(stats["per_dtype_bytes"].items())
        )
        print(
            f"[client {self.partition_id}] round {round_id} | "
            f"model params: {stats['total_params']:,} | "
            f"on-device: {_human_bytes(stats['total_bytes'])} "
            f"({per_dtype_str}) | "
            f"largest tensor: {stats['largest_name']}={_human_bytes(stats['largest_bytes'])} | "
            f"est. transport(fp16): {_human_bytes(stats['transport_bytes_fp16'])}"
        )

        # Dataset / masking diagnostics
        sup_tokens = _count_supervised_tokens(
            self.tokenizer, self.data_collator, self.formatting_prompts_func, self.trainset, self.train_cfg.seq_length
        )
        print(
            f"[client {self.partition_id}] round {round_id} | "
            f"dataset_size={len(self.trainset)} | supervised_tokens_in_probe_batch={sup_tokens}"
        )

        # Dataloader / effective steps diagnostics (pre-train)
        tmp_args = self.training_arguments
        tmp_args.output_dir = outdir
        tmp_trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=tmp_args,
            max_seq_length=self.train_cfg.seq_length,
            train_dataset=self.trainset,
            formatting_func=self.formatting_prompts_func,
            data_collator=self.data_collator,
        )
        dl_len = len(tmp_trainer.get_train_dataloader())
        ga = int(self.training_arguments.gradient_accumulation_steps or 1)
        epochs = float(self.training_arguments.num_train_epochs or 1.0)
        eff_steps = math.ceil(dl_len / ga) * max(1, int(epochs))
        max_steps = int(self.training_arguments.max_steps or 0)
        if max_steps > 0:
            eff_steps = min(eff_steps, max_steps)
        print(
            f"[client {self.partition_id}] round {round_id} | "
            f"len(train_dataloader)={dl_len} | GA={ga} | epochs={epochs} | "
            f"max_steps={max_steps} | expected_steps_this_round≈{eff_steps}"
        )

        # ---- Apply training args updates
        self.training_arguments.learning_rate = float(new_lr)
        self.training_arguments.output_dir = outdir
        self.training_arguments.overwrite_output_dir = True  # fresh run, no resume

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.training_arguments,
            max_seq_length=self.train_cfg.seq_length,
            train_dataset=self.trainset,
            formatting_func=self.formatting_prompts_func,
            data_collator=self.data_collator,
        )

        # Train (fresh each round)
        results = trainer.train(resume_from_checkpoint=False)

        # Loss + steps diagnostics
        raw_attr = getattr(results, "training_loss", None)
        raw_metric = results.metrics.get("train_loss", None) if hasattr(results, "metrics") else None
        gstep = getattr(results, "global_step", None)
        print(
            f"[client {self.partition_id}] round {round_id} | "
            f"global_step={gstep} | "
            f"loss_attr={raw_attr} (is_none={raw_attr is None}, is_zero={(raw_attr == 0.0) if raw_attr is not None else False}) | "
            f"metrics.train_loss={raw_metric} (is_none={raw_metric is None}, is_zero={(raw_metric == 0.0) if raw_metric is not None else False})"
        )

        # Choose loss to report upstream
        train_loss = (
            float(raw_attr)
            if raw_attr is not None
            else float(raw_metric if raw_metric is not None else 0.0)
        )

        # ---- NaN guard on outgoing weights (prevents poisoning the aggregate)
        new_params = get_parameters(self.model)
        if _has_nan_or_inf(new_params):
            print(f"[client {self.partition_id}] round {round_id} | DETECTED NaN/Inf in weights — reverting update")
            new_params = parameters  # send back received (clean) weights
            # Optional: surface the issue in metrics
            return new_params, len(self.trainset), {"train_loss": float("nan"), "nan_guard": 1}

        return new_params, len(self.trainset), {"train_loss": train_loss, "nan_guard": 0}


def client_fn(context: Context) -> FlowerClient:
    """Create a Flower client representing a single organization."""
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    num_rounds = context.run_config["num-server-rounds"]
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))

    # Client partition
    client_trainset = load_data(partition_id, num_partitions, cfg.dataset.name)
    tokenizer, data_collator, formatting_prompts_func = (
        get_tokenizer_and_data_collator_and_propt_formatting(cfg.model.name)
    )

    return FlowerClient(
        cfg.model,
        cfg.train,
        client_trainset,
        tokenizer,
        formatting_prompts_func,
        data_collator,
        num_rounds,
        partition_id,
    ).to_client()


# Flower ClientApp
app = ClientApp(client_fn)
