#!/usr/bin/env python3
# client_writer.py
# Flower NumPyClient that orchestrates training via SHM + flags.
# - Creates per-client SHM sized for fp16
# - Signals a separate fsdp_trainer.py (reader)
# - Reads metrics.json from reader and returns to server
from __future__ import annotations
from typing import Dict, Tuple
import argparse
import os
import time
import json
import warnings

import numpy as np
import torch
import flwr as fl
from flwr.common.typing import NDArrays, Scalar

from hydra import compose, initialize
from omegaconf import DictConfig
from flwr_datasets import FederatedDataset

from fsdp_model import get_model, set_parameters, get_parameters
from fsdp_dataset import get_tokenizer_and_data_collator_and_propt_formatting
from multiprocessing import shared_memory

PATH = "./results/"

# --------- SHM bridge (writer side) ----------
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
            print(f"[Client] Created SHM '{self.shm_name}' with {self._nbytes/1e6:.2f} MB")
        except FileExistsError:
            self.shm = shared_memory.SharedMemory(name=self.shm_name, create=False)
            if self.shm.size < self._nbytes:
                raise RuntimeError(
                    f"Existing SHM '{self.shm_name}' too small: {self.shm.size} < {self._nbytes}"
                )
            print(f"[Client] Connected to existing SHM '{self.shm_name}' ({self.shm.size/1e6:.2f} MB)")

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
            try: os.remove(p)
            except FileNotFoundError: pass

    def write_model_to_shm(self, model: torch.nn.Module):
        sd = model.state_dict()
        flat = self._flat_view()
        ptr = 0
        for k, n in zip(self._keys, self._numels):
            arr = (
                sd[k]
                .detach()
                .to("cpu")
                .to(torch.float16)
                .contiguous()
                .view(-1)
                .numpy()
            )
            flat[ptr:ptr+n] = arr
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
            print(f"[Client] Warning: failed to read metrics.json: {e}")
            return {}

    def rm_metrics(self):
        try: os.remove(self.metrics_json)
        except FileNotFoundError: pass

    def close(self, unlink: bool = False):
        try: self.shm.close()
        except Exception: pass
        if unlink:
            try: self.shm.unlink()
            except Exception: pass

# --------- Flower Client (writer) ----------
class FlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        model_cfg: DictConfig,
        train_cfg: DictConfig,
        trainset,
        tokenizer,
        formatting_prompts_func,
        data_collator,
        save_path: str,
        partition_id: int,
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_cfg = train_cfg
        self.tokenizer = tokenizer
        self.formatting_prompts_func = formatting_prompts_func
        self.data_collator = data_collator
        self.save_path = save_path

        self.model = get_model(model_cfg)
        if torch.cuda.is_available():
            self.model = self.model.to(self.device)

        self.trainset = trainset  # reader performs actual training

        # Per-client SHM & flags
        shm_name = f"opt_client_{partition_id}"
        flag_dir = os.path.abspath(f"./flags_client_{partition_id}")
        self.shm = ShmBridge(shm_name, flag_dir, self.model, dtype=np.float16)

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        params = get_parameters(self.model)  # your helper; can be fp16-on-CPU internally
        total_mb = sum(p.nbytes for p in params) / 1e6
        print(f"[Client] get_parameters -> {len(params)} arrays, {total_mb:.2f} MB")
        return params

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict]:
        # Load server params
        set_parameters(self.model, parameters)

        # Push to SHM
        self.shm.write_model_to_shm(self.model)

        # Decide mode (server may provide; default train)
        mode = str(config.get("mode", "train")).lower()
        if mode not in {"train", "eval"}:
            warnings.warn(f"Unknown mode '{mode}', defaulting to 'train'")
            mode = "train"

        # Signal reader
        print(f"[Client] Signaling reader mode='{mode}'")
        self.shm.write_mode(mode)
        self.shm.signal_ready()

        # Wait completion
        self.shm.wait_done()

        # Pull back weights
        self.shm.load_model_from_shm(self.model)

        # Read metrics.json (if present)
        client_metrics = self.shm.read_metrics()

        # Cleanup flags and metrics file
        self.shm.clear_flags()
        self.shm.rm_metrics()

        # If this was the final round, cleanly stop the reader
        cur   = int(config.get("current_round", 0))
        total = int(config.get("total_rounds", 0))
        if total and cur == total:
            print("[Client] Final round reached, sending 'stop' to reader.")
            self.shm.write_mode("stop")
            self.shm.signal_ready()
            self.shm.wait_done()
            self.shm.clear_flags()
            self.shm.rm_metrics()

        # Return params to server
        out_params = get_parameters(self.model)
        total_mb = sum(p.nbytes for p in out_params) / 1e6
        print(f"[Client] fit -> returning {len(out_params)} arrays, {total_mb:.2f} MB")
        return out_params, len(self.trainset), client_metrics

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        set_parameters(self.model, parameters)
        self.shm.write_model_to_shm(self.model)
        print("[Client] Signaling reader mode='eval'")
        self.shm.write_mode("eval")
        self.shm.signal_ready()
        self.shm.wait_done()
        self.shm.load_model_from_shm(self.model)
        client_metrics = self.shm.read_metrics()
        self.shm.clear_flags()
        self.shm.rm_metrics()
        return float(client_metrics.get("eval_loss", 0.0)), len(self.trainset), client_metrics

# --------- Entrypoint ----------
def main():
    parser = argparse.ArgumentParser(description="Flower Client Orchestrator (Writer)")
    parser.add_argument(
        "--partition-id",
        choices=list(range(1_000)),
        required=True,
        type=int,
        help="Dataset partition ID (of 1,000 iid partitions).",
    )
    parser.add_argument(
        "--task_id",
        choices=list(range(1_000)),
        required=True,
        type=int,
        help="Artificial task ID.",
    )
    parser.add_argument(
        "--server",
        type=str,
        default="10.1.0.4:8000",
        help="Flower server address host:port",
    )
    args = parser.parse_args()

    # Hydra config
    with initialize(config_path="conf"):
        cfg = compose(config_name="config")

    # Dataset + tokenizer/collator (kept for consistency/length reporting)
    fds = FederatedDataset(dataset=cfg.dataset.name, partitioners={"train": cfg.num_clients})
    tokenizer, data_collator, formatting_prompts_func = get_tokenizer_and_data_collator_and_propt_formatting(
        cfg.model.name
    )
    client_trainset = fds.load_partition(args.partition_id, "train")

    # Start client (your preferred style)
    fl.client.start_client(
        server_address=args.server,
        client=FlowerClient(
            model_cfg=cfg.model,
            train_cfg=cfg.train,
            trainset=client_trainset,
            tokenizer=tokenizer,
            formatting_prompts_func=formatting_prompts_func,
            data_collator=data_collator,
            save_path=PATH,
            partition_id=args.partition_id,  # unique SHM/flags per client
        ).to_client(),
        # Optional if your Flower version supports:
        # grpc_max_message_length=1024 * 1024 * 1024,
        # grpc_keepalive_time_ms=30_000,
        # grpc_keepalive_timeout_ms=10_000,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

if __name__ == "__main__":
    main()
