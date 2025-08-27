# writer.py
import torch
import numpy as np
import os
import time
from transformers import AutoModelForCausalLM
from multiprocessing import shared_memory

SHM_NAME   = "opt125m_shared"
FLAG_READY = "ready.flag"
FLAG_DONE  = "done.flag"
FLAG_MODE  = "mode.flag"
NUM_ROUNDS = 10  # only writer knows this

def write_mode(mode: str):
    with open(FLAG_MODE, "w") as f:
        f.write(mode)
        f.flush()
        os.fsync(f.fileno())

def log(msg):
    print(f"[Writer] {msg}", flush=True)

log("Loading OPT-2.7B...")
model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", torch_dtype=torch.float16)
sd = model.state_dict()
flat = torch.cat([p.detach().flatten().cpu().to(torch.float16) for p in sd.values()]).numpy()

# Create or connect shared memory
try:
    shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=flat.nbytes)
    log("Created shared memory segment.")
except FileExistsError:
    shm = shared_memory.SharedMemory(name=SHM_NAME)
    log("Connected to existing shared memory segment.")
np.ndarray(flat.shape, dtype=np.float16, buffer=shm.buf)[:] = flat[:]
shm.close()
log("Initial weights written to shared memory.")

# Rounds: odd=train, even=eval
for r in range(1, NUM_ROUNDS + 1):
    mode = "train" if (r % 2 == 1) else "eval"
    log(f"===== Round {r}/{NUM_ROUNDS} — mode={mode} =====")

    write_mode(mode)
    with open(FLAG_READY, "w") as f:
        f.write("ready"); f.flush(); os.fsync(f.fileno())
    log("ready.flag written.")

    while not os.path.exists(FLAG_DONE):
        time.sleep(0.1)
    log("Detected done.flag. Loading updated weights...")

    shm_in = shared_memory.SharedMemory(name=SHM_NAME)
    flat_updated = np.ndarray(flat.shape, dtype=np.float16, buffer=shm_in.buf)
    ptr = 0
    new_sd = {}
    for name, param in sd.items():
        n = param.numel()
        new_sd[name] = torch.from_numpy(flat_updated[ptr:ptr+n].copy()).view(param.shape)
        ptr += n
    shm_in.close()
    model.load_state_dict(new_sd)
    log("Model updated for next round.")

    for path in (FLAG_DONE, FLAG_READY, FLAG_MODE):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

# Send stop signal
log("Sending stop signal.")
write_mode("stop")
with open(FLAG_READY, "w") as f:
    f.write("ready"); f.flush(); os.fsync(f.fileno())

# Wait for reader ack and clean up
while not os.path.exists(FLAG_DONE):
    time.sleep(0.1)
log("Reader acknowledged stop (done.flag). Cleaning up flags.")
for path in (FLAG_DONE, FLAG_READY, FLAG_MODE):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass

log("Finished. Bye.")