"""Integration test: Python IPC round-trip without networking.

A parent Python process plays "mock Rust node": creates shm, spawns a child
that calls allreduce_bytes(), reads control messages from child stdout,
manipulates shm data, and writes responses to child stdin.
"""

import os
import sys
import struct
import subprocess

from multiprocessing.shared_memory import SharedMemory
from drift.shm import HEADER_SIZE, MAGIC, VERSION


def _create_shm(name, num_floats):
    size = HEADER_SIZE + num_floats * 4
    shm = SharedMemory(name=name, create=True, size=size)
    struct.pack_into("<I", shm.buf, 0, MAGIC)
    struct.pack_into("<I", shm.buf, 4, VERSION)
    struct.pack_into("<Q", shm.buf, 8, size)
    return shm


def _safe_unlink(shm):
    try:
        shm.unlink()
    except FileNotFoundError:
        pass


# Child script that performs one allreduce via stdin/stdout
CHILD_SCRIPT = """
import os, sys, struct, warnings
warnings.filterwarnings("ignore", category=UserWarning)
sys.path.insert(0, os.environ["DRIFT_PYTHON_PATH"])
from drift.shm import DriftShm
from drift.allreduce import allreduce_bytes

shm = DriftShm(os.environ["DRIFT_SHM_NAME"])
num_floats = int(os.environ["NUM_FLOATS"])

# Write [1.0, 2.0, 3.0, 4.0] to shm
data = struct.pack(f"<{num_floats}f", *[float(i+1) for i in range(num_floats)])
result = allreduce_bytes(data, num_floats, shm)

# Write result to stderr so parent can verify
vals = struct.unpack(f"<{num_floats}f", result)
sys.stderr.write("RESULT " + " ".join(f"{v:.1f}" for v in vals) + "\\n")
sys.stderr.flush()

shm.close()
"""


class TestIPCRoundtrip:
    def test_single_allreduce(self):
        """Verify full data-plane round-trip: write -> signal -> manipulate -> respond -> read."""
        num_floats = 4
        shm_name = "test-ipc-rt"
        owner = _create_shm(shm_name, num_floats)

        try:
            drift_python_path = os.path.join(
                os.path.dirname(__file__), ".."
            )

            proc = subprocess.Popen(
                [sys.executable, "-c", CHILD_SCRIPT],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={
                    **os.environ,
                    "DRIFT_SHM_NAME": shm_name,
                    "NUM_FLOATS": str(num_floats),
                    "DRIFT_PYTHON_PATH": drift_python_path,
                },
            )

            # Read the DRIFT_ALLREDUCE line from child stdout
            line = proc.stdout.readline().decode().strip()
            assert line.startswith("DRIFT_ALLREDUCE"), f"unexpected: {line}"
            parts = line.split()
            op_id = int(parts[1])
            n = int(parts[2])
            assert n == num_floats

            # Read what child wrote to shm
            raw = bytes(owner.buf[HEADER_SIZE : HEADER_SIZE + num_floats * 4])
            vals = struct.unpack(f"<{num_floats}f", raw)
            assert vals == (1.0, 2.0, 3.0, 4.0)

            # "All-reduce": multiply each value by 2 (simulating average of 2 nodes)
            result = struct.pack(
                f"<{num_floats}f", *[v * 2.0 for v in vals]
            )
            owner.buf[HEADER_SIZE : HEADER_SIZE + num_floats * 4] = result

            # Send DRIFT_ALLREDUCE_DONE
            response = f"DRIFT_ALLREDUCE_DONE {op_id}\n"
            proc.stdin.write(response.encode())
            proc.stdin.flush()

            # Wait for child to exit and check result
            stdout, stderr = proc.communicate(timeout=10)
            assert proc.returncode == 0, f"child failed: {stderr.decode()}"

            # Parse result from stderr (first line only, ignore resource_tracker warnings)
            first_line = stderr.decode().strip().split("\n")[0]
            assert first_line == "RESULT 2.0 4.0 6.0 8.0", f"got: {first_line}"

        finally:
            owner.close()
            _safe_unlink(owner)

    def test_multiple_allreduces(self):
        """Verify sequential allreduce calls with incrementing op_ids."""
        num_floats = 2
        shm_name = "test-ipc-multi"
        owner = _create_shm(shm_name, num_floats)

        child_script = """
import os, sys, struct, warnings
warnings.filterwarnings("ignore", category=UserWarning)
sys.path.insert(0, os.environ["DRIFT_PYTHON_PATH"])
from drift.shm import DriftShm
from drift.allreduce import allreduce_bytes

shm = DriftShm(os.environ["DRIFT_SHM_NAME"])

for i in range(3):
    data = struct.pack("<2f", float(i), float(i+1))
    result = allreduce_bytes(data, 2, shm)
    vals = struct.unpack("<2f", result)
    sys.stderr.write(f"ROUND {i}: {vals[0]:.1f} {vals[1]:.1f}\\n")
    sys.stderr.flush()

shm.close()
"""

        try:
            drift_python_path = os.path.join(os.path.dirname(__file__), "..")
            proc = subprocess.Popen(
                [sys.executable, "-c", child_script],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={
                    **os.environ,
                    "DRIFT_SHM_NAME": shm_name,
                    "NUM_FLOATS": str(num_floats),
                    "DRIFT_PYTHON_PATH": drift_python_path,
                },
            )

            for round_idx in range(3):
                line = proc.stdout.readline().decode().strip()
                assert line.startswith("DRIFT_ALLREDUCE")
                parts = line.split()
                op_id = int(parts[1])
                assert op_id == round_idx  # op_ids should increment

                # Echo back same data (identity all-reduce)
                response = f"DRIFT_ALLREDUCE_DONE {op_id}\n"
                proc.stdin.write(response.encode())
                proc.stdin.flush()

            stdout, stderr = proc.communicate(timeout=10)
            assert proc.returncode == 0, f"child failed: {stderr.decode()}"

            # Filter out resource_tracker warnings, keep only ROUND lines
            lines = [l for l in stderr.decode().strip().split("\n")
                     if l.startswith("ROUND")]
            assert len(lines) == 3
            assert "ROUND 0:" in lines[0]
            assert "ROUND 1:" in lines[1]
            assert "ROUND 2:" in lines[2]

        finally:
            owner.close()
            _safe_unlink(owner)
