import os
import warnings
from hydra import compose, initialize
import flwr as fl
from llmmodels import get_model, set_parameters, get_parameters  # make sure set_parameters is available
warnings.filterwarnings("ignore", category=UserWarning)

NUM_ROUNDS = 100
save_path = "./results/"

# Load configuration
with initialize(config_path="conf"):
    cfg = compose(config_name="config")

# Update config for number of rounds
cfg.num_rounds = NUM_ROUNDS
cfg.train.num_rounds = NUM_ROUNDS

# Create output directory if not exists
os.makedirs(save_path, exist_ok=True)

###########################################################
def get_evaluate_fn(model_cfg, save_every_round, total_round, save_path):
    """Return an evaluation function for saving the global full LLM model."""

    def evaluate(server_round: int, parameters, config):
        # Save model at specific intervals
        if server_round != 0 and (
            server_round == total_round or server_round % save_every_round == 0
        ):
            model = get_model(model_cfg)
            set_parameters(model, parameters)

            model.save_pretrained(f"{save_path}/full_model_{server_round}")

        return 0.0, {}

    return evaluate
###########################################################

def get_on_fit_config():
    def fit_config_fn(server_round: int):
        return {"current_round": server_round}
    return fit_config_fn

def fit_weighted_average(metrics):
    losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"train_loss": sum(losses) / sum(examples)}

#############################################################

# Instantiate FedAvg strategy
strategy = fl.server.strategy.FedAvg(
    min_available_clients=2,
    fraction_fit=1.0,
    fraction_evaluate=0.0,
    on_fit_config_fn=get_on_fit_config(),
    fit_metrics_aggregation_fn=fit_weighted_average,
    evaluate_fn=get_evaluate_fn(cfg.model, save_every_round=5, total_round=NUM_ROUNDS, save_path=save_path)
)

# Start server
fl.server.start_server(
    server_address="0.0.0.0:8000",
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy,
)
