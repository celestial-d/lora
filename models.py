import torch
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig
from peft.utils import prepare_model_for_kbit_training

import math


def cosine_annealing(
    current_round: int,
    total_round: int,
    lrate_max: float = 0.001,
    lrate_min: float = 0.0,
) -> float:
    """Implement cosine annealing learning rate schedule."""

    cos_inner = math.pi * current_round / total_round
    return lrate_min + 0.5 * (lrate_max - lrate_min) * (1 + math.cos(cos_inner))

def get_module(start_module: torch.nn.Module, module_names):
    """
    Recursively get a PyTorch module starting from the start module with
    a given list of module names.
    """
    module = start_module
    for module_name in module_names:
        module = getattr(module, module_name)
    return module

def get_model(model_cfg: DictConfig):
    """Load model with appropriate quantization config and other optimizations.

    Please refer to this example for `peft + BitsAndBytes`:
    https://github.com/huggingface/peft/blob/main/examples/fp4_finetuning/finetune_fp4_opt_bnb_peft.py
    """

    if model_cfg.quantization == 4:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    elif model_cfg.quantization == 8:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        raise ValueError(
            f"Use 4-bit or 8-bit quantization. You passed: {model_cfg.quantization}/"
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.name,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
    )

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=model_cfg.gradient_checkpointing
    )

    peft_config = LoraConfig(
        r=model_cfg.lora.peft_lora_r,
        lora_alpha=model_cfg.lora.peft_lora_alpha,
        lora_dropout=0.075,
        task_type="CAUSAL_LM",
    )
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print(model)
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    return get_peft_model(model, peft_config)

def get_model_client(model_cfg: DictConfig):
    """Load model with appropriate quantization config and other optimizations.

    Please refer to this example for `peft + BitsAndBytes`:
    https://github.com/huggingface/peft/blob/main/examples/fp4_finetuning/finetune_fp4_opt_bnb_peft.py
    """

    if model_cfg.quantization == 4:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    elif model_cfg.quantization == 8:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        raise ValueError(
            f"Use 4-bit or 8-bit quantization. You passed: {model_cfg.quantization}/"
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.name,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
    )
    transformer_module = model
    cut_layer = model_cfg.cut_layer
    print("#################")
    print(model_cfg)
    print("#################")
    for module_name in model_cfg.transformer_module_name.split("."):
        transformer_module = getattr(transformer_module,module_name)
    client_layers = transformer_module[: cut_layer]
    client_module_names = model_cfg.transformer_module_name.split(".")
    client_module = get_module(model, client_module_names[:-1])
    setattr(client_module, client_module_names[-1], client_layers)

    #for layer in Config().parameters.model.layers_after_transformer:
    #    layer = layer.split(".")
    #    if len(layer) > 1:
    #        module = get_module(model, layer[:-1])
    #        setattr(module, layer[-1], torch.nn.Identity())
    #    else:
    #        setattr(model, layer[0], torch.nn.Identity())

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=model_cfg.gradient_checkpointing
    )

    peft_config = LoraConfig(
        r=model_cfg.lora.peft_lora_r,
        lora_alpha=model_cfg.lora.peft_lora_alpha,
        lora_dropout=0.075,
        task_type="CAUSAL_LM",
    )
    print(model)
    return get_peft_model(model, peft_config)

def get_model_server(model_cfg: DictConfig):
    """Load model with appropriate quantization config and other optimizations.

    Please refer to this example for `peft + BitsAndBytes`:
    https://github.com/huggingface/peft/blob/main/examples/fp4_finetuning/finetune_fp4_opt_bnb_peft.py
    """

    if model_cfg.quantization == 4:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    elif model_cfg.quantization == 8:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        raise ValueError(
            f"Use 4-bit or 8-bit quantization. You passed: {model_cfg.quantization}/"
        )

    model = AutoModelForCausalLM.from_pretrained(
            model_cfg.name,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
        )
    transformer_module = model
    cut_layer = model_cfg.cut_layer

    for module_name in model_cfg.transformer_module_name.split("."):
        transformer_module = getattr(transformer_module,module_name)
    server_layers = transformer_module[cut_layer:]
    server_module_names = model_cfg.transformer_module_name.split(".")
    server_module = get_module(model, server_module_names[:-1])
    setattr(server_module, server_module_names[-1], server_layers)

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=model_cfg.gradient_checkpointing
    )

    peft_config = LoraConfig(
        r=model_cfg.lora.peft_lora_r,
        lora_alpha=model_cfg.lora.peft_lora_alpha,
        lora_dropout=0.075,
        task_type="CAUSAL_LM",
    )
    
    return get_peft_model(model, peft_config)
