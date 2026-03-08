#!/usr/bin/env python3
"""Helper script for Rust-Python IPC integration test.

Opens shared memory, writes gradient data, sends DRIFT_ALLREDUCE,
waits for DRIFT_ALLREDUCE_DONE, reads result, prints verification to stderr.
"""

import os
import sys
import struct
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Add drift-python to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "drift-python"))

from drift.shm import DriftShm
from drift.allreduce import allreduce_bytes

shm_name = os.environ["DRIFT_SHM_NAME"]
num_floats = 4

shm = DriftShm(shm_name)

# Write [1.0, 2.0, 3.0, 4.0]
data = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
result = allreduce_bytes(data, num_floats, shm)

# Output result for verification
vals = struct.unpack("<4f", result)
sys.stderr.write(f"RESULT {' '.join(f'{v:.1f}' for v in vals)}\n")
sys.stderr.flush()

shm.close()
