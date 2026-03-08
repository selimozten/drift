"""Integration test: DDP with drift communication hook.

Mocks the shm+stdio layer to verify DDP calls allreduce() during backward
pass and that gradients are averaged correctly for a trivial model.
"""

import os
import sys
import struct
import subprocess
import threading
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


# Child script that runs DDP with drift comm_hook
CHILD_SCRIPT = """
import os, sys, warnings
warnings.filterwarnings("ignore", category=UserWarning)
sys.path.insert(0, os.environ["DRIFT_PYTHON_PATH"])

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import drift

drift.init()

model = nn.Linear(4, 2, bias=False)
with torch.no_grad():
    model.weight.fill_(1.0)

model = DDP(model)
drift.register(model)

x = torch.ones(1, 4)
out = model(x)
loss = out.sum()
loss.backward()

grad = model.module.weight.grad
sys.stderr.write(f"GRAD_SHAPE {list(grad.shape)}\\n")
sys.stderr.write(f"GRAD_SUM {grad.sum().item():.4f}\\n")
sys.stderr.flush()

sys.stdout.write("DRIFT_DONE\\n")
sys.stdout.flush()

import torch.distributed as dist
dist.destroy_process_group()
"""


class TestProcessGroup:
    def test_ddp_calls_allreduce(self):
        """Verify DDP invokes allreduce during backward pass via comm_hook."""
        shm_name = "test-pg-ddp"

        from multiprocessing.shared_memory import SharedMemory
        from drift.shm import HEADER_SIZE, MAGIC, VERSION

        size = HEADER_SIZE + 1024 * 1024
        try:
            stale = SharedMemory(name=shm_name, create=False)
            stale.close()
            stale.unlink()
        except FileNotFoundError:
            pass
        shm = SharedMemory(name=shm_name, create=True, size=size)
        struct.pack_into("<I", shm.buf, 0, MAGIC)
        struct.pack_into("<I", shm.buf, 4, VERSION)
        struct.pack_into("<Q", shm.buf, 8, size)

        try:
            drift_python_path = os.path.join(os.path.dirname(__file__), "..")

            proc = subprocess.Popen(
                [sys.executable, "-c", CHILD_SCRIPT],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={
                    **os.environ,
                    "DRIFT_SHM_NAME": shm_name,
                    "DRIFT_RANK": "0",
                    "DRIFT_WORLD_SIZE": "1",
                    "DRIFT_PYTHON_PATH": drift_python_path,
                    "MASTER_ADDR": "127.0.0.1",
                    "MASTER_PORT": "29502",
                },
            )

            allreduce_count = 0
            done = False
            error = None

            def ipc_loop():
                nonlocal allreduce_count, done, error
                try:
                    while True:
                        line = proc.stdout.readline()
                        if not line:
                            break
                        line = line.decode().strip()
                        if not line:
                            continue

                        if line == "DRIFT_READY":
                            continue

                        if line.startswith("DRIFT_ALLREDUCE"):
                            parts = line.split()
                            op_id = int(parts[1])
                            response = f"DRIFT_ALLREDUCE_DONE {op_id}\n"
                            proc.stdin.write(response.encode())
                            proc.stdin.flush()
                            allreduce_count += 1
                            continue

                        if line == "DRIFT_DONE":
                            done = True
                            break
                except Exception as e:
                    error = e

            t = threading.Thread(target=ipc_loop, daemon=True)
            t.start()
            t.join(timeout=30)

            try:
                proc.stdin.close()
            except Exception:
                pass
            try:
                stderr = proc.stderr.read().decode()
            except Exception:
                stderr = ""
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

            assert error is None, f"IPC loop error: {error}"
            assert done, f"child did not send DRIFT_DONE. stderr={stderr}"
            assert proc.returncode == 0, f"child failed (rc={proc.returncode}): {stderr}"
            assert allreduce_count >= 1, f"expected allreduce calls, got {allreduce_count}"

            # Check gradient values
            stderr_lines = [l for l in stderr.strip().split("\n") if l.startswith("GRAD_")]
            grad_shape = None
            grad_sum = None
            for line in stderr_lines:
                if line.startswith("GRAD_SHAPE"):
                    grad_shape = line.split(" ", 1)[1]
                elif line.startswith("GRAD_SUM"):
                    grad_sum = float(line.split(" ", 1)[1])

            assert grad_shape == "[2, 4]", f"unexpected grad shape: {grad_shape}"
            assert grad_sum is not None
            assert abs(grad_sum - 8.0) < 0.01, f"unexpected grad sum: {grad_sum}"

        finally:
            shm.close()
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
