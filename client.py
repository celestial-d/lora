from collections import OrderedDict
from typing import Callable, Dict, Tuple
from hydra import compose, initialize
import flwr as fl
import torch
from flwr.common.typing import NDArrays, Scalar
from omegaconf import DictConfig
from trl import SFTTrainer
from transformers import TrainingArguments
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from flwr_datasets import FederatedDataset
from models import get_model_client, get_model, cosine_annealing
import os
import warnings
from dataset import get_tokenizer_and_data_collator_and_propt_formatting
import argparse

PATH = '/home/cc/output/'

#profiler # define timer
#start = torch.cuda.Event(enable_timing=True)
#end = torch.cuda.Event(enable_timing=True)
# # start pt
#start.record()

parser = argparse.ArgumentParser(description="Flower")
parser.add_argument(
        "--partition-id",
        choices=list(range(1_000)),
        required=True,
        type=int,
        help="Partition of the dataset divided into 1,000 iid partitions created "
        "artificially.",
    )
parser.add_argument(
        "--task_id",
        choices=list(range(1_000)),
        required=True,
        type=int,
        help="task id "
        "artificially.",
    )

#os.environ["CUDA_VISIBLE_DEVICES"] = str(parser.parse_args().partition_id)
#print(os.environ["CUDA_VISIBLE_DEVICES"])
with initialize(config_path="conf"):
    cfg = compose(config_name="config")


fds = FederatedDataset(
    dataset=cfg.dataset.name, partitioners={"train": cfg.num_clients}
)
(
    tokenizer,
    data_collator,
    formatting_prompts_func,
) = get_tokenizer_and_data_collator_and_propt_formatting(
    cfg.model.name,
)

def set_parameters(model, parameters: NDArrays) -> None:
    """Change the parameters of the model using the given ones."""
    peft_state_dict_keys = get_peft_model_state_dict(model).keys()
    params_dict = zip(peft_state_dict_keys, parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    set_peft_model_state_dict(model, state_dict)


# pylint: disable=too-many-arguments
class FlowerClient(
    fl.client.NumPyClient
):  # pylint: disable=too-many-instance-attributes
    """Standard Flower client for CNN training."""

    def __init__(
        self,
        model_cfg: DictConfig,
        train_cfg: DictConfig,
        trainset,
        tokenizer,
        formatting_prompts_func,
        data_collator,
        save_path,
    ):  # pylint: disable=too-many-arguments
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_cfg = train_cfg
        self.training_argumnets = TrainingArguments(**train_cfg.training_arguments)
        self.tokenizer = tokenizer
        self.formatting_prompts_func = formatting_prompts_func
        self.data_collator = data_collator
        self.save_path = save_path

        # instantiate model
        self.model = get_model(model_cfg)

        self.trainset = trainset

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return the parameters of the current net."""

        state_dict = get_peft_model_state_dict(self.model)
        return [val.cpu().numpy() for _, val in state_dict.items()]

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Implement distributed fit function for a given client."""
        set_parameters(self.model, parameters)

        new_lr = cosine_annealing(
            int(config["current_round"]),
            self.train_cfg.num_rounds,
            self.train_cfg.learning_rate_max,
            self.train_cfg.learning_rate_min,
        )

        self.training_argumnets.learning_rate = new_lr
        self.training_argumnets.output_dir = self.save_path

        # Construct trainer
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.training_argumnets,
            max_seq_length=self.train_cfg.seq_length,
            train_dataset=self.trainset,
            formatting_func=self.formatting_prompts_func,
            data_collator=self.data_collator,
            packing=False,
        )

        # Do local training
        results = trainer.train()


        print("********shape*************", len(self.get_parameters({})), self.get_parameters({})[0].shape, self.get_parameters({})[1].shape, self.get_parameters({})[2].shape)
        print("*********************", self.get_parameters({})[0].size)
        num_parameter = 0
        for i in range(len(self.get_parameters({}))):
            num_parameter = num_parameter + self.get_parameters({})[i].size
        total_size = num_parameter*32.0/1e9
        print("**********total_size_Gb***********", total_size)

        return (
            self.get_parameters({}),
            len(self.trainset),
            {"train_loss": results.training_loss},
        )


client_trainset = (fds.load_partition(parser.parse_args().partition_id, "train"))
#client_trainset = client_trainset.rename_column("output", "Answer")

fl.client.start_client(
        server_address="0.0.0.0:8000", client=FlowerClient(
            model_cfg=cfg.model,
            train_cfg=cfg.train,
            trainset=client_trainset,
            tokenizer=tokenizer,
            formatting_prompts_func=formatting_prompts_func,
            data_collator=data_collator,           
            save_path=PATH
            ).to_client()
    )

# # Waits for everything to finish running
torch.cuda.synchronize()
#end.record()

# # print time
#print(start.elapsed_time(end))
