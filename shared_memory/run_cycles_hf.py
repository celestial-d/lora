# run_cycles_hf.py
import os, sys, subprocess, shutil
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

SHM_DIR   = os.environ.get("SHM_DIR", "/dev/shm/llama7b_cycle")
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-2-7b-hf")
NPROC     = int(os.environ.get("NPROC", "4"))
ROUNDS    = int(os.environ.get("ROUNDS", "4"))

def hf_checkpoint_exists(d: Path) -> bool:
    return d.exists() and (d / "config.json").exists()

def preload_if_needed():
    d = Path(SHM_DIR)
    if hf_checkpoint_exists(d):
        print("[orchestrator] Found existing HF checkpoint in SHM; skipping preload.")
        return
    print("[orchestrator] Preloading base model to SHM via HF save_pretrained ...")
    d.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,   # match ds_trl default
        low_cpu_mem_usage=True,
    )
    model.save_pretrained(d)
    tok.save_pretrained(d)
    print("[orchestrator] Preload complete.")

def print_num_layers_from_shm():
    print("[orchestrator] Loading model from SHM to count layers ...")
    model = AutoModelForCausalLM.from_pretrained(
        SHM_DIR,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    # Try common layer attributes
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        n = len(model.model.layers)
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        n = len(model.transformer.h)
    elif hasattr(model.config, "num_hidden_layers"):
        n = model.config.num_hidden_layers
    else:
        n = "UNKNOWN"
    print(f"[orchestrator] Number of layers: {n}")

    # Save back (unchanged), just to follow your spec
    tmp_dir = Path(SHM_DIR).with_name(Path(SHM_DIR).name + "_touch")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(tmp_dir)
    # keep tokenizer stable; no need to rewrite unless you changed it
    shutil.rmtree(SHM_DIR)
    shutil.move(str(tmp_dir), SHM_DIR)
    print("[orchestrator] Saved checkpoint back to SHM.")

def main():
    preload_if_needed()

    for r in range(1, ROUNDS + 1):
        print(f"\n[orchestrator] === Round {r}/{ROUNDS} ===")
        env = os.environ.copy()
        env["SHM_DIR"] = SHM_DIR  # ds_trl.py will use this

        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            f"--nproc_per_node={NPROC}", "ds_trl.py"
        ]
        print(f"[orchestrator] Launching: {' '.join(cmd)}")
        ret = subprocess.call(cmd, env=env)
        print(f"[orchestrator] ds_trl.py exit code: {ret}")

        # Exit code 0 is our done signal; also assert the HF checkpoint exists
        if ret != 0:
            print("[orchestrator] ERROR: training failed (non-zero exit). Aborting.")
            sys.exit(1)
        if not hf_checkpoint_exists(Path(SHM_DIR)):
            print("[orchestrator] ERROR: checkpoint missing after success. Aborting.")
            sys.exit(1)

        print_num_layers_from_shm()

    print("\n[orchestrator] All rounds completed successfully.")

if __name__ == "__main__":
    main()
