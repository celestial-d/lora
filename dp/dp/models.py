# models.py  (dense full fine-tuning; no LoRA, no quantization)

import math
from collections import OrderedDict
from typing import List

import torch
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM

from flwr.common.typing import NDArrays
from transformers import BitsAndBytesConfig

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

    # if model_cfg.quantization == 4:
    #     quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    # elif model_cfg.quantization == 8:
    #     quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    # else:
    #     raise ValueError(
    #         f"Only 4-bit or 8-bit quantization supported. Got: {model_cfg.quantization}"
    #     )

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
def get_parameters(model: torch.nn.Module) -> list:
    return [
        val.detach().cpu().to(torch.float16).numpy()
        if val.dtype == torch.bfloat16 else val.detach().cpu().numpy()
        for val in model.state_dict().values()
    ]


def set_parameters(model: torch.nn.Module, parameters: list) -> None:
    state_dict = model.state_dict()
    for key, val in zip(state_dict.keys(), parameters):
        tensor_val = torch.tensor(val)

        # Cast back to original dtype if needed
        if state_dict[key].dtype != tensor_val.dtype:
            tensor_val = tensor_val.to(state_dict[key].dtype)

        state_dict[key] = tensor_val
    model.load_state_dict(state_dict, strict=True)