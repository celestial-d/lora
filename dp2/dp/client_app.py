# client_app.py
# Flower ClientApp orchestrating DeepSpeed+TRL fine-tuning with
# bidirectional streaming (safetensors shards) via /dev/shm.
#
# - Streams initial global weights -> ds_trl.py:   SHM_DIR/in_stream/
# - Ingests trained weights <- ds_trl.py:          SHM_DIR/out_stream/
#
# Streaming-only.

from __future__ import annotations

import os
import sys
import json
import time
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Tuple

import torch
from safetensors.torch import save_file as safe_save_file, load_file as safe_load_file

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flwr.common.config import unflatten_dict
from flwr.common.typing import NDArrays, Scalar  # note: "flwr"
from omegaconf import DictConfig

# your modules
from dp import models as mdl
from dp import dataset as ds

# ------------------------------- Config --------------------------------
DEFAULT_MODEL_NAME = os.environ.get("MODEL_NAME", "facebook/opt-125m")
DEFAULT_NPROC = int(os.environ.get("NPROC", "2"))
DEFAULT_MASTER_PORT = os.environ.get("MASTER_PORT", "29517")

STREAM_CHUNK_BYTES = int(os.environ.get("STREAM_CHUNK_BYTES", str(512 * 1024**2)))  # 512MB
STREAM_WINDOW_SIZE = int(os.environ.get("STREAM_WINDOW_SIZE", "2"))                 # double buffer

# ------------------------------- Utils ---------------------------------
def count_layers_from_model(model: torch.nn.Module) -> int | str:
    try:
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return len(model.model.layers)
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return len(model.transformer.h)
        if hasattr(model.config, "num_hidden_layers"):
            return model.config.num_hidden_layers
    except Exception:
        pass
    return "UNKNOWN"

def apply_safetensors_chunk_inplace(model: torch.nn.Module, path: Path):
    tensors = safe_load_file(str(path))
    name_to_param = dict(model.named_parameters())
    name_to_buffer = dict(model.named_buffers())

    def _lookup(name: str):
        t = name_to_param.get(name)
        if t is None:
            t = name_to_buffer.get(name)
        # optional "module." prefix tolerance
        if t is None and name.startswith("module."):
            base = name[len("module."):]
            t = name_to_param.get(base) or name_to_buffer.get(base)
        if t is None:
            mod = "module." + name
            t = name_to_param.get(mod) or name_to_buffer.get(mod)
        return t

    with torch.no_grad():
        for name, v in tensors.items():
            if v.numel() == 0:  # defensive: ignore empty shards
                continue
            target = _lookup(name)
            if target is None:
                continue
            if target.shape != v.shape:
                raise RuntimeError(f"Shape mismatch for {name}: {target.shape} vs {v.shape}")
            target.data.copy_(v.to(dtype=target.dtype))

def write_streamed_safetensors(model: torch.nn.Module, stream_dir: Path,
                               max_chunk_bytes: int = STREAM_CHUNK_BYTES,
                               window_size: int = STREAM_WINDOW_SIZE):
    """Producer: write params+buffers as .safetensors shards with backpressure."""
    stream_dir.mkdir(parents=True, exist_ok=True)

    def wait_for_done(idx_needed: int):
        while not (stream_dir / f"chunk_{idx_needed:05d}.done").exists():
            time.sleep(0.05)

    def flush_chunk(tensors_dict: dict, idx: int):
        tmp = stream_dir / f"chunk_{idx:05d}.safetensors.tmp"
        final = stream_dir / f"chunk_{idx:05d}.safetensors"
        safe_save_file(tensors_dict, str(tmp))
        os.replace(tmp, final)                           # atomic
        (stream_dir / f"chunk_{idx:05d}.ready").touch()  # signal ready

    idx = 0
    cur = {}
    cur_bytes = 0

    with torch.no_grad():
        for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
            t = tensor.detach().cpu().contiguous()
            nbytes = t.numel() * t.element_size()
            if cur and cur_bytes + nbytes > max_chunk_bytes:
                if idx >= window_size:
                    wait_for_done(idx - window_size)
                flush_chunk(cur, idx)
                idx += 1
                cur, cur_bytes = {}, 0
            cur[name] = t
            cur_bytes += nbytes

        if cur:
            if idx >= window_size:
                wait_for_done(idx - window_size)
            flush_chunk(cur, idx)
            idx += 1

    (stream_dir / "end.json").write_text(json.dumps({"total": idx}))

# --------------------------- Flower Client -----------------------------
class FullHFClient(NumPyClient):
    def __init__(self, cfg: DictConfig, num_rounds: int, partition_id: int, num_partitions: int):
        self.cfg = cfg
        self.partition_id = partition_id
        self.num_rounds = num_rounds
        self.num_partitions = num_partitions

        self.model_name = cfg.model.name if hasattr(cfg, "model") else DEFAULT_MODEL_NAME
        self.model = mdl.get_model(cfg.model if hasattr(cfg, "model") else {"name": self.model_name})
        self.model = self.model.to("cpu")  # keep CPU during streaming

        base_shm = os.environ.get("SHM_BASE", "/dev/shm")
        self.shm_dir = Path(base_shm) / f"llama7b_client_{partition_id}"
        self.nproc = int(os.environ.get("NPROC", str(DEFAULT_NPROC)))
        self.master_port = os.environ.get("MASTER_PORT", DEFAULT_MASTER_PORT)

        try:
            self.num_examples = len(ds.load_data(partition_id, num_partitions, cfg.dataset.name))
        except Exception:
            self.num_examples = 0

        # Allow overriding the trainer path via env
        env_trainer = os.environ.get("TRAINER_PATH", "").strip()
        if env_trainer:
            self.trainer_path = Path(env_trainer)
        else:
            self.trainer_path = Path(__file__).parent / "ds_trl.py"

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return mdl.get_parameters(self.model)

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict]:
        # Load global params
        mdl.set_parameters(self.model, parameters)

        # Clean shm
        if self.shm_dir.exists():
            shutil.rmtree(self.shm_dir)
        self.shm_dir.mkdir(parents=True, exist_ok=True)

        # Launch ds_trl.py
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
        env["FL_PARTITION_ID"] = str(self.partition_id)
        env["FL_NUM_PARTITIONS"] = str(self.num_partitions)
        env["FL_NUM_ROUNDS"] = str(self.num_rounds)
        env["STREAM_CHUNK_BYTES"] = str(STREAM_CHUNK_BYTES)
        env["STREAM_WINDOW_SIZE"] = str(STREAM_WINDOW_SIZE)

        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            f"--nproc_per_node={self.nproc}",
            str(self.trainer_path),
            *ds_args,
        ]
        print(f"[client {self.partition_id}] launching: {' '.join(cmd)} (SHM_DIR={self.shm_dir})")
        proc = subprocess.Popen(cmd, env=env)

        # Stream INPUT → trainer
        in_dir = self.shm_dir / "in_stream"
        print(f"[client] STREAM OUT → ds_trl: writing initial weights to {in_dir}")
        write_streamed_safetensors(self.model, in_dir)

        # Receive OUTPUT ← trainer (streaming only)
        out_dir = self.shm_dir / "out_stream"
        i = 0
        total = None
        end_path = out_dir / "end.json"

        print(f"[client] Waiting for streamed updated weights at {out_dir}")
        while True:
            ready = out_dir / f"chunk_{i:05d}.ready"
            if ready.exists():
                chunk = out_dir / f"chunk_{i:05d}.safetensors"
                while not chunk.exists():
                    time.sleep(0.02)
                apply_safetensors_chunk_inplace(self.model, chunk)
                (out_dir / f"chunk_{i:05d}.done").touch()
                try:
                    chunk.unlink(missing_ok=True)
                    ready.unlink(missing_ok=True)
                except Exception:
                    pass
                i += 1
                continue

            if total is None and end_path.exists():
                try:
                    total = json.loads(end_path.read_text()).get("total", None)
                except Exception:
                    total = None

            if total is not None and i >= total:
                print("[client] STREAM mode complete.")
                break

            poll = proc.poll()
            if poll is not None and not end_path.exists():
                raise RuntimeError(
                    f"ds_trl.py exited with code {poll}; no streamed out_stream produced."
                )
            time.sleep(0.05)

        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"ds_trl.py failed with exit {ret}")

        metrics: Dict[str, Scalar] = {}
        loss_file = Path("/dev/shm/loss.txt")
        if loss_file.exists():
            try:
                with open(loss_file, "r") as f:
                    line = f.readline()
                    metrics["train_loss"] = float(line.strip())
            except Exception:
                pass

        metrics["num_layers"] = count_layers_from_model(self.model)
        metrics["handoff_mode"] = "stream"

        out_params = mdl.get_parameters(self.model)
        return out_params, (self.num_examples or 0), metrics

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        mdl.set_parameters(self.model, parameters)
        return 0.0, (self.num_examples or 0), {}


# ------------------------- ClientApp factory ---------------------------
def client_fn(context: Context):
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    num_rounds = int(context.run_config["num-server-rounds"])
    cfg = DictConfig(ds.replace_keys(unflatten_dict(context.run_config)))
    return FullHFClient(cfg, num_rounds, partition_id, num_partitions).to_client()

app = ClientApp(client_fn)
