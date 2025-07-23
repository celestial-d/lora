import torch
import math
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig


def cosine_annealing(
    current_round: int,
    total_round: int,
    lrate_max: float = 0.001,
    lrate_min: float = 0.0,
) -> float:
    """Implement cosine annealing learning rate schedule."""
    cos_inner = math.pi * current_round / total_round
    return lrate_min + 0.5 * (lrate_max - lrate_min) * (1 + math.cos(cos_inner))


def get_model(model_cfg: DictConfig):
    """Load full LLM model with optional quantization and gradient checkpointing."""

    if model_cfg.quantization == 4:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    elif model_cfg.quantization == 8:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        raise ValueError(
            f"Only 4-bit or 8-bit quantization supported. Got: {model_cfg.quantization}"
        )
    # Load the model with the specified quantization
    #
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.name,
        torch_dtype=torch.bfloat16,
    )

    # Optionally enable gradient checkpointing
    if model_cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    return model


def get_parameters(model: torch.nn.Module) -> list:
    return [
        val.detach().cpu().to(torch.float32).numpy()
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

