import os
import warnings
from typing import Dict, Tuple

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flwr.common.config import unflatten_dict
from flwr.common.typing import NDArrays, Scalar
from omegaconf import DictConfig
from transformers import TrainingArguments
from trl import SFTTrainer

from flowertune_llm.dataset import (
    get_tokenizer_and_data_collator_and_propt_formatting,
    load_data,
    replace_keys,
)
from flowertune_llm.models import (
    cosine_annealing,
    get_model,
    set_parameters,
    get_parameters,
)

# Quieter logs
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["RAY_DISABLE_DOCKER_CPU_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)


class FlowerClient(NumPyClient):
    """Flower NumPyClient for full-model supervised fine-tuning via TRL SFTTrainer."""

    def __init__(
        self,
        model_cfg: DictConfig,
        train_cfg: DictConfig,
        trainset,
        tokenizer,
        formatting_prompts_func,
        data_collator,
        num_rounds: int,
    ):
        self.train_cfg = train_cfg
        self.training_arguments = TrainingArguments(**train_cfg.training_arguments)
        self.tokenizer = tokenizer
        self.formatting_prompts_func = formatting_prompts_func
        self.data_collator = data_collator
        self.num_rounds = num_rounds
        self.trainset = trainset

        self.model = get_model(model_cfg)

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict]:
        set_parameters(self.model, parameters)

        # Cosine LR schedule per round
        new_lr = cosine_annealing(
            int(config["current_round"]),
            self.num_rounds,
            self.train_cfg.learning_rate_max,
            self.train_cfg.learning_rate_min,
        )
        self.training_arguments.learning_rate = new_lr
        self.training_arguments.output_dir = str(config["save_path"])

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.training_arguments,
            max_seq_length=self.train_cfg.seq_length,
            train_dataset=self.trainset,
            formatting_func=self.formatting_prompts_func,
            data_collator=self.data_collator,
            packing=False,
        )

        results = trainer.train()
        train_loss = float(results.training_loss) if results.training_loss is not None else 0.0

        return get_parameters(self.model), len(self.trainset), {"train_loss": train_loss}


def client_fn(context: Context) -> FlowerClient:
    """Construct one client instance."""
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    num_rounds = context.run_config["num-server-rounds"]

    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))

    # Data and tokenizer/collator
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
    ).to_client()


# Flower ClientApp
app = ClientApp(client_fn)
