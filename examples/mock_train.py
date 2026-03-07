#!/usr/bin/env python3
"""Mock training script for drift integration testing.

Reads config from DRIFT_* environment variables and prints
DRIFT_PROGRESS lines that the node parses into TrainProgress messages.
"""

import os
import sys
import time
import math

def main():
    epochs = int(os.environ.get("DRIFT_EPOCHS", "3"))
    batch_size = int(os.environ.get("DRIFT_BATCH_SIZE", "32"))
    shard_size = int(os.environ.get("DRIFT_SHARD_SIZE", "10000"))
    shard_index = int(os.environ.get("DRIFT_SHARD_INDEX", "0"))
    lr = float(os.environ.get("DRIFT_LEARNING_RATE", "0.001"))
    node_id = os.environ.get("DRIFT_NODE_ID", "unknown")

    steps_per_epoch = max(shard_size // batch_size, 1)
    loss = 2.5

    print(f"[mock_train] node={node_id[:12]} shard={shard_index} "
          f"epochs={epochs} steps/epoch={steps_per_epoch}", flush=True)

    for epoch in range(epochs):
        for step in range(steps_per_epoch):
            global_step = epoch * steps_per_epoch + step
            loss *= 0.97  # simulate convergence
            throughput = batch_size * (8.0 + shard_index)  # vary by shard

            # This is the magic line the node parses
            print(f"DRIFT_PROGRESS {epoch} {global_step} {loss:.6f} {throughput:.1f}",
                  flush=True)
            time.sleep(0.01)  # simulate compute

    print(f"[mock_train] training complete, final loss={loss:.6f}", flush=True)

if __name__ == "__main__":
    main()
