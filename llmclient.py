from collections import OrderedDict
from typing import Callable, Dict, Tuple
from hydra import compose, initialize
import flwr as fl
import torch
from flwr.common.typing import NDArrays, Scalar
from omegaconf import DictConfig
from trl import SFTTrainer
from transformers import TrainingArguments
from flwr_datasets import FederatedDataset
from llmmodels import get_model, cosine_annealing
import os
import warnings
from llmdataset import get_tokenizer_and_data_collator_and_propt_formatting
import argparse
from llmmodels import get_model, set_parameters, get_parameters


PATH = "./results/"

parser = argparse.ArgumentParser(description="Flower")
parser.add_argument(
    "--partition-id",
    choices=list(range(1_000)),
    required=True,
    type=int,
    help="Partition of the dataset divided into 1,000 iid partitions created artificially.",
)
parser.add_argument(
    "--task_id",
    choices=list(range(1_000)),
    required=True,
    type=int,
    help="Task ID artificially assigned.",
)

args = parser.parse_args()

# Load config
with initialize(config_path="conf"):
    cfg = compose(config_name="config")

# Load federated dataset and tokenizer
fds = FederatedDataset(
    dataset=cfg.dataset.name, partitioners={"train": cfg.num_clients}
)
tokenizer, data_collator, formatting_prompts_func = get_tokenizer_and_data_collator_and_propt_formatting(cfg.model.name)


# Flower client definition
class FlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        model_cfg: DictConfig,
        train_cfg: DictConfig,
        trainset,
        tokenizer,
        formatting_prompts_func,
        data_collator,
        save_path,
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_cfg = train_cfg
        self.training_arguments = TrainingArguments(**train_cfg.training_arguments)
        self.tokenizer = tokenizer
        self.formatting_prompts_func = formatting_prompts_func
        self.data_collator = data_collator
        self.save_path = save_path
        self.model = get_model(model_cfg)
        self.trainset = trainset

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return get_parameters(self.model)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        set_parameters(self.model, parameters)

        new_lr = cosine_annealing(
            int(config["current_round"]),
            self.train_cfg.num_rounds,
            self.train_cfg.learning_rate_max,
            self.train_cfg.learning_rate_min,
        )

        self.training_arguments.learning_rate = new_lr
        self.training_arguments.output_dir = self.save_path

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

        # Debug: parameter shape and total size
        param_list = self.get_parameters({})
        total_param = sum(p.size for p in param_list)
        print(f"Parameter count: {len(param_list)}, total size: {total_param * 32.0 / 1e9:.2f} Gb")

        return param_list, len(self.trainset), {"train_loss": results.training_loss}

# Load dataset partition
client_trainset = fds.load_partition(args.partition_id, "train")

# Start Flower client
fl.client.start_client(
    server_address="0.0.0.0:8000",
    client=FlowerClient(
        model_cfg=cfg.model,
        train_cfg=cfg.train,
        trainset=client_trainset,
        tokenizer=tokenizer,
        formatting_prompts_func=formatting_prompts_func,
        data_collator=data_collator,
        save_path=PATH,
    ).to_client()
)

torch.cuda.synchronize()
