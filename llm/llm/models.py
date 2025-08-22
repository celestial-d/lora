# models.py  (dense full fine-tuning; no LoRA, no quantization)

import math
from collections import OrderedDict
from typing import List

import torch
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM

from flwr.common.typing import NDArrays


# -----------------------
# LR schedule (unchanged)
# -----------------------
def cosine_annealing(
    current_round: int,
    total_round: int,
    lrate_max: float = 0.001,
    lrate_min: float = 0.0,
) -> float:
    cos_inner = math.pi * current_round / total_round
    return lrate_min + 0.5 * (lrate_max - lrate_min) * (1 + math.cos(cos_inner))


# -----------------------
# Model loader (dense)
# -----------------------
def _str_to_dtype(name: str) -> torch.dtype:
    name = (name or "fp16").lower()
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp32", "float32"):
        return torch.float32
    # default
    return torch.float16


def get_model(model_cfg: DictConfig):
    """
    Load a dense CausalLM model for full fine-tuning (no LoRA, no quantization).

    Expected (non-strict) fields in model_cfg:
      - name: str (HF repo or path)
      - dtype: "fp16" | "bf16" | "fp32"   (optional; default "fp16")
      - gradient_checkpointing: bool      (optional; default True)
      - attn_implementation: "sdpa"|"eager"|"flash_attention_2" (optional; default "sdpa")
    """
    name = getattr(model_cfg, "name", None)
    if not name:
        raise ValueError("model_cfg.name must be set (HF model id or local path)")

    dtype = _str_to_dtype(getattr(model_cfg, "dtype", "fp16"))
    grad_ckpt = bool(getattr(model_cfg, "gradient_checkpointing", True))
    attn_impl = getattr(model_cfg, "attn_implementation", "sdpa")

    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )

    # Good practice for training
    model.config.use_cache = False
    if grad_ckpt:
        model.gradient_checkpointing_enable()

    return model


# -----------------------
# Param <-> ndarray utils
# -----------------------
def get_parameters(model) -> NDArrays:
    """
    Return full model parameters as a list of numpy arrays.
    Transport in fp16 by default to reduce size (not quantization, just dtype cast).
    """
    arrays: List = []
    with torch.no_grad():
        for _, t in model.state_dict().items():
            arrays.append(t.detach().cpu().contiguous().to(torch.float16).numpy())
    return arrays


def set_parameters(model, parameters: NDArrays) -> None:
    """
    Load full model parameters from a list of numpy arrays.
    We cast back to each tensor's original dtype from the current model.
    """
    sd = model.state_dict()
    if len(sd) != len(parameters):
        raise ValueError(f"Mismatched tensor count: model has {len(sd)}, got {len(parameters)}")

    new_sd = OrderedDict()
    for (k, t), arr in zip(sd.items(), parameters):
        ten = torch.from_numpy(arr).to(dtype=t.dtype)
        new_sd[k] = ten

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if missing or unexpected:
        # Usually should be empty; keep a gentle check
        raise RuntimeError(f"State dict load mismatch. Missing: {missing}, Unexpected: {unexpected}")
