"""
fsdp/client_app.py
Flower ClientApp (controller) that orchestrates an external FSDP trainer
via shared memory (fp16) + flag files.

- Uses your fsdp.models / fsdp.dataset modules (as provided).
- Respects pyproject config: bf16 preference, dtype, etc.
- Writes trainer_config.json + keys_order.json into FLAG_DIR so the trainer
  can exactly match model/dataset/knobs without extra envs.
"""

from __future__ import annotations

import os
import json
import time
import warnings
from typing import Dict, Tuple

import numpy as np
import torch

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flwr.common.config import unflatten_dict
from flwr.common.typing import NDArrays, Scalar
from omegaconf import DictConfig

# Your modules
from dp import models as mdl
from dp import dataset as ds

from multiprocessing import shared_memory


# ---------------------------
# Shared-memory bridge (fp16)
# ---------------------------
class ShmBridge:
    def __init__(self, shm_name: str, flag_dir: str, model: torch.nn.Module, dtype=np.float16):
        self.shm_name = shm_name
        self.flag_dir = flag_dir
        self.dtype = dtype
        os.makedirs(flag_dir, exist_ok=True)

        sd = model.state_dict()
        self._keys = list(sd.keys())
        self._numels = [sd[k].numel() for k in self._keys]
        self._total = int(sum(self._numels))
        self._nbytes = np.dtype(self.dtype).itemsize * self._total

        # Persist the key order so trainer can flatten/unflatten consistently
        self.keys_json = os.path.join(self.flag_dir, "keys_order.json")
        with open(self.keys_json, "w") as f:
            json.dump(self._keys, f)

        try:
            self.shm = shared_memory.SharedMemory(name=self.shm_name, create=True, size=self._nbytes)
            print(f"[client] Created SHM '{self.shm_name}' with {self._nbytes/1e6:.2f} MB")
        except FileExistsError:
            self.shm = shared_memory.SharedMemory(name=self.shm_name, create=False)
            if self.shm.size < self._nbytes:
                raise RuntimeError(
                    f"Existing SHM '{self.shm_name}' too small: {self.shm.size} < {self._nbytes}"
                )
            print(f"[client] Connected to existing SHM '{self.shm_name}' ({self.shm.size/1e6:.2f} MB)")

        self.flag_ready   = os.path.join(self.flag_dir, "ready.flag")
        self.flag_done    = os.path.join(self.flag_dir, "done.flag")
        self.flag_mode    = os.path.join(self.flag_dir, "mode.flag")
        self.metrics_json = os.path.join(self.flag_dir, "metrics.json")
        self.cfg_json     = os.path.join(self.flag_dir, "trainer_config.json")
        self.knobs_json   = os.path.join(self.flag_dir, "knobs.json")

    def _flat_view(self):
        return np.ndarray((self._total,), dtype=self.dtype, buffer=self.shm.buf)

    def write_mode(self, mode: str):
        with open(self.flag_mode, "w") as f:
            f.write(mode); f.flush(); os.fsync(f.fileno())

    def signal_ready(self):
        with open(self.flag_ready, "w") as f:
            f.write("ready"); f.flush(); os.fsync(f.fileno())

    def wait_done(self, poll: float = 0.1, timeout: float | None = None):
        start = time.time()
        while not os.path.exists(self.flag_done):
            time.sleep(poll)
            if timeout is not None and (time.time() - start) > timeout:
                raise TimeoutError("Timed out waiting for done.flag")

    def clear_flags(self):
        for p in (self.flag_done, self.flag_ready, self.flag_mode):
            try: os.remove(p)
            except FileNotFoundError: pass

    def write_model_to_shm(self, model: torch.nn.Module):
        sd = model.state_dict()
        flat = self._flat_view()
        ptr = 0
        for k, n in zip(self._keys, self._numels):
            v = (
                sd[k].detach().to("cpu").to(torch.float16)
                .contiguous().view(-1).numpy()
            )
            flat[ptr:ptr+n] = v
            ptr += n

    def load_model_from_shm(self, model: torch.nn.Module):
        sd = model.state_dict()
        flat = self._flat_view()
        ptr = 0
        new_sd = {}
        for k, n in zip(self._keys, self._numels):
            p = sd[k]
            chunk = torch.from_numpy(flat[ptr:ptr+n].copy()).view(p.shape).to(p.dtype).contiguous()
            new_sd[k] = chunk
            ptr += n
        model.load_state_dict(new_sd, strict=True)

    def read_metrics(self) -> dict:
        try:
            with open(self.metrics_json, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except Exception as e:
            print(f"[client] Warning: failed to read metrics.json: {e}")
            return {}

    def rm_metrics(self):
        try: os.remove(self.metrics_json)
        except FileNotFoundError: pass

    def write_trainer_config(self, cfg: DictConfig, partition_id: int, num_partitions: int):
        # Respect your pyproject config (bf16 preference, dtype, etc.)
        targs = cfg.train.training_arguments
        trainer_cfg = {
            "model_name": cfg.model.name,
            "model_dtype": getattr(cfg.model, "dtype", "bf16"),
            "dataset_name": cfg.dataset.name,
            "partition_id": int(partition_id),
            "num_partitions": int(num_partitions),
            "seq_length": int(cfg.train.seq_length),
            "per_device_train_bs": int(targs.per_device_train_batch_size),
            "ga_steps": int(targs.gradient_accumulation_steps),
            "learning_rate": float(targs.learning_rate or 2e-5),
            "max_steps": int(targs.max_steps or 10),
            "logging_steps": int(targs.logging_steps or 10),
            "lr_scheduler_type": str(targs.lr_scheduler_type or "constant"),
            "gradient_checkpointing": bool(cfg.model.gradient_checkpointing),
            "targs_bf16": bool(getattr(targs, "bf16", False)),
            "targs_fp16": bool(getattr(targs, "fp16", False)),
            "fsdp": "full_shard auto_wrap",
            "fsdp_transformer_layer_cls_to_wrap": "OPTDecoderLayer",
            "attn_implementation": getattr(cfg.model, "attn_implementation", "sdpa"),
            "output_dir": os.path.abspath("./fsdp_output"),
        }
        with open(self.cfg_json, "w") as f:
            json.dump(trainer_cfg, f)
        os.sync()

        # Optional heads-up if quantization is present in config (ignored)
        if hasattr(cfg.model, "quantization"):
            print("[client] NOTE: cfg.model.quantization is set but ignored (dense full FT).")

    def write_knobs(self, *, learning_rate: float, max_steps: int | None):
        data = {"learning_rate": float(learning_rate)}
        if max_steps is not None:
            data["max_steps"] = int(max_steps)
        with open(self.knobs_json, "w") as f:
            json.dump(data, f)
        os.sync()


# ---------------------------
# Flower NumPyClient (writer)
# ---------------------------
class FlowerClient(NumPyClient):
    def __init__(self, cfg: DictConfig, num_rounds: int, partition_id: int, num_partitions: int):
        self.cfg = cfg
        self.partition_id = partition_id
        self.num_rounds = num_rounds
        self.num_partitions = num_partitions

        # Plain model for state_dict (FSDP model lives in external trainer)
        self.model = mdl.get_model(cfg.model)
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")

        # SHM + flags (per client)
        shm_name = f"opt_client_{partition_id}"
        flag_dir = os.path.abspath(f"./flags_client_{partition_id}")
        self.shm = ShmBridge(shm_name, flag_dir, self.model, dtype=np.float16)

        # (optional) expose to trainer via envs (trainer mainly uses trainer_config.json)
        os.environ["SHM_NAME"] = shm_name
        os.environ["FLAG_DIR"] = flag_dir
        print(f"[client] SHM_NAME={shm_name} FLAG_DIR={flag_dir}")

        # Write trainer_config.json once
        self.shm.write_trainer_config(cfg, partition_id, num_partitions)

        # Count examples for reporting to server
        self.num_examples = len(ds.load_data(partition_id, num_partitions, cfg.dataset.name))

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        arrs = mdl.get_parameters(self.model)
        mb = sum(a.nbytes for a in arrs) / 1e6
        print(f"[client {self.partition_id}] get_parameters -> {len(arrs)} arrays, {mb:.2f} MB")
        return arrs

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict]:
        # Load global -> SHM
        mdl.set_parameters(self.model, parameters)
        self.shm.write_model_to_shm(self.model)

        # Per-round cosine LR knobs for trainer
        round_id = int(config.get("current_round", 1))
        new_lr = mdl.cosine_annealing(
            round_id,
            self.num_rounds,
            self.cfg.train.learning_rate_max,
            self.cfg.train.learning_rate_min,
        )
        max_steps = int(self.cfg.train.training_arguments.max_steps or 10)
        self.shm.write_knobs(learning_rate=float(new_lr), max_steps=max_steps)

        # Signal trainer
        mode = str(config.get("mode", "train")).lower()
        print(f"[client {self.partition_id}] signaling trainer mode='{mode}', lr={new_lr:.6g}, max_steps={max_steps}")
        self.shm.write_mode(mode)
        self.shm.signal_ready()

        # Wait, read back
        self.shm.wait_done()
        self.shm.load_model_from_shm(self.model)
        metrics = self.shm.read_metrics()
        self.shm.clear_flags()
        self.shm.rm_metrics()

        # Final round? tell trainer to stop
        total = int(config.get("total_rounds", self.num_rounds))
        if total and round_id == total:
            print(f"[client {self.partition_id}] final round -> sending 'stop'")
            self.shm.write_mode("stop")
            self.shm.signal_ready()
            self.shm.wait_done()
            self.shm.clear_flags()
            self.shm.rm_metrics()

        # Return to server
        out_params = mdl.get_parameters(self.model)
        mb = sum(a.nbytes for a in out_params) / 1e6
        print(f"[client {self.partition_id}] fit -> returning {len(out_params)} arrays, {mb:.2f} MB")
        train_loss = float(metrics.get("train_loss", 0.0))
        return out_params, self.num_examples, {"train_loss": train_loss}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        mdl.set_parameters(self.model, parameters)
        self.shm.write_model_to_shm(self.model)
        print(f"[client {self.partition_id}] signaling trainer mode='eval'")
        self.shm.write_mode("eval")
        self.shm.signal_ready()
        self.shm.wait_done()
        self.shm.load_model_from_shm(self.model)
        metrics = self.shm.read_metrics()
        self.shm.clear_flags()
        self.shm.rm_metrics()
        loss = float(metrics.get("eval_loss", 0.0))
        return loss, self.num_examples, metrics


# ---------------------------
# ClientApp factory (new API)
# ---------------------------
def client_fn(context: Context) -> FlowerClient:
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    num_rounds = int(context.run_config["num-server-rounds"])
    cfg = DictConfig(ds.replace_keys(unflatten_dict(context.run_config)))
    return FlowerClient(cfg, num_rounds, partition_id, num_partitions).to_client()


# Register ClientApp
app = ClientApp(client_fn)
