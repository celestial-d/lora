# hf_fulltrain_client_app.py
# Flower ClientApp that orchestrates DeepSpeed+TRL full fine-tuning by
# saving a HF checkpoint into /dev/shm and running: torchrun ds_trl.py

from __future__ import annotations

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flwr.common.config import unflatten_dict
from flwr.common.typing import NDArrays, Scalar
from omegaconf import DictConfig

# your modules
from dp import models as mdl
from dp import dataset as ds


# =============== config / defaults ===============
DEFAULT_MODEL_NAME = os.environ.get("MODEL_NAME", "facebook/opt-125m")
DEFAULT_NPROC      = int(os.environ.get("NPROC", "2"))
DEFAULT_MASTER_PORT= os.environ.get("MASTER_PORT", "29517")
DEFAULT_METRICS_PATH = Path("./opt67b_codealpaca_zero3/eval_metrics.json")  # ds_trl.py writes here


# =============== HF checkpoint helpers ===============
def hf_checkpoint_exists(d: Path) -> bool:
    return d.exists() and (d / "config.json").exists()

def save_hf_ckpt_from_model(model: torch.nn.Module, tok_name: str, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    try:
        model.save_pretrained(dst)
    except Exception:
        base = AutoModelForCausalLM.from_pretrained(tok_name, low_cpu_mem_usage=True)
        base.load_state_dict(model.state_dict(), strict=False)
        base.save_pretrained(dst)
    tok = AutoTokenizer.from_pretrained(tok_name, use_fast=True)
    tok.save_pretrained(dst)

def load_hf_ckpt_into_model(src: Path, model: torch.nn.Module):
    hf_model = AutoModelForCausalLM.from_pretrained(src, low_cpu_mem_usage=True)
    model.load_state_dict(hf_model.state_dict(), strict=False)

def count_layers_from_dir(d: Path) -> int | str:
    mdl_tmp = AutoModelForCausalLM.from_pretrained(d, low_cpu_mem_usage=True)
    if hasattr(mdl_tmp, "model") and hasattr(mdl_tmp.model, "layers"):
        return len(mdl_tmp.model.layers)
    if hasattr(mdl_tmp, "transformer") and hasattr(mdl_tmp.transformer, "h"):
        return len(mdl_tmp.transformer.h)
    if hasattr(mdl_tmp.config, "num_hidden_layers"):
        return mdl_tmp.config.num_hidden_layers
    return "UNKNOWN"


# =============== the Flower NumPyClient ===============
class FullHFClient(NumPyClient):
    def __init__(self, cfg: DictConfig, num_rounds: int, partition_id: int, num_partitions: int):
        self.cfg = cfg
        self.partition_id = partition_id
        self.num_rounds = num_rounds
        self.num_partitions = num_partitions

        # model used only for param (de)serialization with server
        self.model_name = cfg.model.name if hasattr(cfg, "model") else DEFAULT_MODEL_NAME
        self.model = mdl.get_model(cfg.model if hasattr(cfg, "model") else {"name": self.model_name})
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")

        # unique SHM dir per client to avoid collisions
        base_shm = os.environ.get("SHM_BASE", "/dev/shm")
        self.shm_dir = Path(base_shm) / f"llama7b_client_{partition_id}"
        self.nproc = int(os.environ.get("NPROC", str(DEFAULT_NPROC)))
        self.master_port = os.environ.get("MASTER_PORT", DEFAULT_MASTER_PORT)

        # optional: dataset size for weighted FedAvg on server
        try:
            self.num_examples = len(ds.load_data(partition_id, num_partitions, cfg.dataset.name))
        except Exception:
            self.num_examples = 0

    # ---- Flower interface ----
    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return mdl.get_parameters(self.model)

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict]:

        # 1) load global params into local model
        mdl.set_parameters(self.model, parameters)

        # 2) write HF checkpoint to SHM_DIR (fresh)
        if self.shm_dir.exists():
            shutil.rmtree(self.shm_dir)
        save_hf_ckpt_from_model(self.model.cpu(), self.model_name, self.shm_dir)

        # 3) run a single training/eval round via torchrun ds_trl.py
        # build args for ds_trl.py from Context-derived values
        ds_args = [
            "--partition-id", str(self.partition_id),
            "--num-partitions", str(self.num_partitions),
            "--num-rounds", str(self.num_rounds),
            "--model-name", self.model_name,
            "--dataset-name", getattr(self.cfg.dataset, "name", "unknown"),
        ] 

        env = os.environ.copy()
        env.setdefault("TRITON_CACHE_DIR", "/dev/shm/triton_cache")
        env.setdefault("MASTER_PORT", self.master_port)
        env["SHM_DIR"] = str(self.shm_dir)
        env["FL_PARTITION_ID"] = str(self.partition_id)   # env fallback
        env["FL_NUM_PARTITIONS"] = str(self.num_partitions)
        env["FL_NUM_ROUNDS"] = str(self.num_rounds)

        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            f"--nproc_per_node={self.nproc}",
            "lora/dp/dp/ds_trl.py",
            *ds_args,  # <-- args go to ds_trl.py
        ]
        print(f"[client {self.partition_id}] launching: {' '.join(cmd)} (SHM_DIR={self.shm_dir})")
        ret = subprocess.call(cmd, env=env)

        if ret != 0:
            raise RuntimeError(f"ds_trl.py failed with exit {ret}")

        # 4) read metrics if present (optional)
        metrics: Dict[str, Scalar] = {}
        if DEFAULT_METRICS_PATH.exists():
            try:
                metrics = json.loads(DEFAULT_METRICS_PATH.read_text())
            except Exception:
                metrics = {}
        # add a small debug metric
        metrics["num_layers"] = count_layers_from_dir(self.shm_dir)

        # 5) load updated checkpoint back into local model
        load_hf_ckpt_into_model(self.shm_dir, self.model)

        # 6) return updated params + examples + metrics
        out_params = mdl.get_parameters(self.model)

        # 7) load training loss if written by ds_trl.py
        loss_file = Path("/dev/shm/loss.txt")
        if loss_file.exists():
            try:
                with open(loss_file, "r") as f:
                    line = f.readline()
                    metrics["train_loss"] = float(line.strip())
            except Exception:
                pass
        return out_params, (self.num_examples or 0), {"train_loss": metrics["train_loss"]}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        # simple local no-op eval (you can wire an eval-only run in ds_trl.py if desired)
        mdl.set_parameters(self.model, parameters)
        return 0.0, (self.num_examples or 0), {}


# =============== ClientApp factory (new API) ===============
def client_fn(context: Context):
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    num_rounds = int(context.run_config["num-server-rounds"])
    cfg = DictConfig(ds.replace_keys(unflatten_dict(context.run_config)))
    return FullHFClient(cfg, num_rounds, partition_id, num_partitions).to_client()


# register ClientApp
app = ClientApp(client_fn)
