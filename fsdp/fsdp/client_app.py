"""
flowertune-llm (writer/controller): New-style Flower ClientApp that orchestrates
an external FSDP trainer via shared memory (fp16) + flag files.

- Uses ClientApp/NumPyClient/Context (Flower ≥1.19)
- Client keeps a plain (non-FSDP) model only to size/serialize state_dict
- FSDP lives in a separate torchrun process (fsdp_trainer.py) which:
    * waits for flags,
    * runs a train/eval burst,
    * writes FULL_STATE_DICT back to SHM,
    * writes metrics.json, and
    * idles until the next signal.

Make sure to start fsdp_trainer.py separately with the SAME SHM_NAME/FLAG_DIR.
"""

from __future__ import annotations
import os
import time
import json
import warnings
from typing import Dict, Tuple

import numpy as np
import torch

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flwr.common.config import unflatten_dict
from flwr.common.typing import NDArrays, Scalar
from omegaconf import DictConfig

# Plain (non-FSDP) helpers on the client
from fsdp.models import get_model, set_parameters, get_parameters
from fsdp.dataset import load_data, replace_keys

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

        self.flag_ready = os.path.join(self.flag_dir, "ready.flag")
        self.flag_done  = os.path.join(self.flag_dir, "done.flag")
        self.flag_mode  = os.path.join(self.flag_dir, "mode.flag")
        self.metrics_json = os.path.join(self.flag_dir, "metrics.json")

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
            try:
                os.remove(p)
            except FileNotFoundError:
                pass

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
        try:
            os.remove(self.metrics_json)
        except FileNotFoundError:
            pass

    def close(self, unlink: bool = False):
        try:
            self.shm.close()
        except Exception:
            pass
        if unlink:
            try:
                self.shm.unlink()
            except Exception:
                pass


# ---------------------------
# Flower NumPyClient (writer)
# ---------------------------
class FlowerClient(NumPyClient):
    """Controller client: sends/receives weights via SHM and signals the FSDP trainer."""

    def __init__(
        self,
        model_cfg: DictConfig,
        dataset_name: str,
        num_rounds: int,
        partition_id: int,
        num_partitions: int,
        save_path: str = "./results",
    ):
        self.partition_id = partition_id
        self.num_rounds = num_rounds
        self.dataset_name = dataset_name
        self.save_path = save_path

        # Plain model for state_dict compatibility (FSDP lives in separate proc)
        self.model = get_model(model_cfg)
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")

        # Set up SHM + flags (one segment per client)
        shm_name = f"opt_client_{partition_id}"
        flag_dir = os.path.abspath(f"./flags_client_{partition_id}")
        self.shm = ShmBridge(shm_name, flag_dir, self.model, dtype=np.float16)

        # Tell/confirm env used by fsdp_trainer.py (launched separately)
        os.environ["SHM_NAME"] = shm_name
        os.environ["FLAG_DIR"] = flag_dir
        print(f"[client] SHM_NAME={shm_name} FLAG_DIR={flag_dir}")

        # For reporting sample count to Flower (training itself is remote)
        # If your fsdp_trainer uses the same dataset and partitioning, match it here.
        self.num_examples = len(load_data(partition_id, num_partitions, dataset_name))

        # Optional: keep last metrics read from trainer
        self._last_metrics: Dict[str, float] = {}

    # ---- Flower NumPyClient API ----
    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        arrs = get_parameters(self.model)
        mb = sum(a.nbytes for a in arrs) / 1e6
        print(f"[client {self.partition_id}] get_parameters -> {len(arrs)} arrays, {mb:.2f} MB")
        return arrs

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict]:
        # 1) Load global weights into our plain model
        set_parameters(self.model, parameters)

        # 2) Write to SHM
        self.shm.write_model_to_shm(self.model)

        # 3) Mode (default train). You can pass 'mode' in strategy's on_fit_config_fn.
        mode = str(config.get("mode", "train")).lower()
        if mode not in {"train", "eval"}:
            warnings.warn(f"Unknown mode '{mode}', defaulting to 'train'")
            mode = "train"

        # 4) Optionally pass round-specific knobs to trainer (e.g., LR, steps)
        # Drop a small JSON in FLAG_DIR if you want; trainer can read it.
        # Example:
        # knobs = {"learning_rate": float(config.get("lr", 2e-5)), "max_steps": int(config.get("max_steps", 10))}
        # with open(os.path.join(os.environ["FLAG_DIR"], "knobs.json"), "w") as f:
        #     json.dump(knobs, f)

        # 5) Signal trainer and wait
        print(f"[client {self.partition_id}] signaling trainer mode='{mode}'")
        self.shm.write_mode(mode)
        self.shm.signal_ready()
        self.shm.wait_done()

        # 6) Read back updated weights + metrics
        self.shm.load_model_from_shm(self.model)
        metrics = self.shm.read_metrics()
        self._last_metrics = metrics

        # 7) Cleanup flags
        self.shm.clear_flags()
        self.shm.rm_metrics()

        # 8) Final round? tell trainer to stop
        cur = int(config.get("current_round", 0))
        total = int(config.get("total_rounds", 0))
        if total and cur == total:
            print(f"[client {self.partition_id}] final round -> sending 'stop'")
            self.shm.write_mode("stop")
            self.shm.signal_ready()
            self.shm.wait_done()
            self.shm.clear_flags()
            self.shm.rm_metrics()

        # 9) Return to server
        out_params = get_parameters(self.model)
        mb = sum(a.nbytes for a in out_params) / 1e6
        print(f"[client {self.partition_id}] fit -> returning {len(out_params)} arrays, {mb:.2f} MB")
        # prefer 'train_loss' if present
        train_loss = float(metrics.get("train_loss", 0.0))
        return out_params, self.num_examples, {"train_loss": train_loss}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        set_parameters(self.model, parameters)
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
    """Create a Flower client using run/node config from Context."""
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    num_rounds = int(context.run_config["num-server-rounds"])

    # Hydra-like: flatten -> omegaconf DictConfig
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))

    # Build controller client
    return FlowerClient(
        model_cfg=cfg.model,                 # must match fsdp_trainer model arch
        dataset_name=cfg.dataset.name,       # for local sample count only
        num_rounds=num_rounds,
        partition_id=partition_id,
        num_partitions=num_partitions,
        save_path=os.getcwd(),
    ).to_client()


# Register ClientApp
app = ClientApp(client_fn)
