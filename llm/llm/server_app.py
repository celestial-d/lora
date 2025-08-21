import os
from datetime import datetime

from flwr.common import Context, ndarrays_to_parameters
from flwr.common.config import unflatten_dict
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from omegaconf import DictConfig

from flowertune_llm.models import get_model, get_parameters, set_parameters
from flowertune_llm.dataset import replace_keys


def get_evaluate_fn(model_cfg, save_every_round: int, total_round: int, save_path: str):
    """Used by strategy.evaluate() to periodically save the global model."""
    def evaluate(server_round: int, parameters, config):
        if server_round != 0 and (
            server_round == total_round or server_round % save_every_round == 0
        ):
            model = get_model(model_cfg)
            set_parameters(model, parameters)
            model.save_pretrained(f"{save_path}/full_model_{server_round}")
        return 0.0, {}
    return evaluate


def get_on_fit_config(save_path: str):
    """Attach per-round config (round index and output dir) for clients."""
    def fit_config_fn(server_round: int):
        return {"current_round": server_round, "save_path": save_path}
    return fit_config_fn


def fit_weighted_average(metrics):
    """Weighted average of client-reported training loss."""
    losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    denom = max(1, sum(examples))
    return {"train_loss": sum(losses) / denom}


def server_fn(context: Context):
    """Assemble the strategy and server config."""
    # Output dir per run
    folder = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(os.getcwd(), f"results/{folder}")
    os.makedirs(save_path, exist_ok=True)

    # Load config
    num_rounds = int(context.run_config["num-server-rounds"])
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))

    # Seed initial global weights on the server (avoids heavy round-1 upload)
    init_model = get_model(cfg.model)
    init_params = ndarrays_to_parameters(get_parameters(init_model))

    # Strategy with periodic saving
    strategy = FedAvg(
        fraction_fit=cfg.strategy.fraction_fit,
        fraction_evaluate=cfg.strategy.fraction_evaluate,
        on_fit_config_fn=get_on_fit_config(save_path),
        fit_metrics_aggregation_fn=fit_weighted_average,
        initial_parameters=init_params,
        evaluate_fn=get_evaluate_fn(cfg.model, cfg.train.save_every_round, num_rounds, save_path),
        # tune these for your deployment size:
        min_fit_clients=max(1, int(1)),
        min_available_clients=max(1, int(1)),
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


# Flower ServerApp
app = ServerApp(server_fn=server_fn)
