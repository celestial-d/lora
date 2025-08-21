# Full-model fine-tuning (no LoRA, no quantization)
import math
from collections import OrderedDict

import torch
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM

from flwr.common.typing import NDArrays


def cosine_annealing(
    current_round: int,
    total_round: int,
    lrate_max: float = 0.001,
    lrate_min: float = 0.0,
) -> float:
    """Cosine annealing LR schedule."""
    cos_inner = math.pi * current_round / total_round
    return lrate_min + 0.5 * (lrate_max - lrate_min) * (1 + math.cos(cos_inner))


def get_model(model_cfg: DictConfig):
    """Load the full base model (no PEFT/BitsAndBytes)."""
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.name,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if getattr(model_cfg, "gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    return model


def get_parameters(model) -> NDArrays:
    """Serialize the full model state_dict to a list of NumPy arrays (stable order)."""
    state = model.state_dict()
    return [t.detach().cpu().contiguous().numpy() for t in state.values()]


def set_parameters(model, parameters: NDArrays) -> None:
    """Load the full model weights from a list of NumPy arrays (dtype/device-safe)."""
    current = model.state_dict()
    keys = list(current.keys())
    if len(keys) != len(parameters):
        raise ValueError(f"Parameter length mismatch: {len(keys)} vs {len(parameters)}")

    new_state = OrderedDict()
    for k, arr in zip(keys, parameters):
        ref = current[k]
        new_state[k] = torch.from_numpy(arr).to(dtype=ref.dtype, device=ref.device)

    model.load_state_dict(new_state, strict=True)
