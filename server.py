import os
import warnings
from hydra import compose, initialize
import flwr as fl
from models import get_model
warnings.filterwarnings("ignore", category=UserWarning)
#os.environ["CUDA_VISIBLE_DEVICES"]='-1'
#torch.cuda.device_count()

NUM_ROUNDS = 100
save_path = "./results/"

with initialize(config_path="conf"):
    cfg = compose(config_name="config")

# Reset the number of number
cfg.num_rounds = NUM_ROUNDS
cfg.train.num_rounds = NUM_ROUNDS

# Create output directory
if not os.path.exists(save_path):
    os.mkdir(save_path)
###########################################################
def get_evaluate_fn(model_cfg, save_every_round, total_round, save_path):
    """Return an evaluation function for saving global model."""

    def evaluate(server_round: int, parameters, config):
        # Save model
        if server_round != 0 and (
            server_round == total_round or server_round % save_every_round == 0
        ):
            # Init model
            model = get_model(model_cfg)
            set_parameters(model, parameters)

            model.save_pretrained(f"{save_path}/peft_{server_round}")

        return 0.0, {}

    return evaluate


# Get a function that will be used to construct the config that the client's
# fit() method will receive
def get_on_fit_config():
    def fit_config_fn(server_round: int):
        fit_config = {"current_round": server_round}
        return fit_config

    return fit_config_fn


def fit_weighted_average(metrics):
    """Aggregation function for (federated) evaluation metrics."""
    # Multiply accuracy of each client by number of examples used
    losses = [num_examples * m["train_loss"] for num_examples, m in metrics]  
    examples = [num_examples for num_examples, _ in metrics]
    
    # Aggregate and return custom metric (weighted average)
    return {"train_loss": sum(losses) / sum(examples)}
#############################################################

# Instantiate strategy.
strategy = fl.server.strategy.FedAvg(
    min_available_clients=2,  # Simulate a 2-client setting
    fraction_fit=1.0,
    fraction_evaluate=0.0,  # no client evaluation
    on_fit_config_fn=get_on_fit_config(),
    fit_metrics_aggregation_fn=fit_weighted_average,
)

fl.server.start_server(
        server_address="192.168.21.114:8000",
        config=fl.server.ServerConfig(num_rounds=100),
        strategy=strategy,
    )
