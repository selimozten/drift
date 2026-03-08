"""PyTorch DDP communication hook for drift.

Instead of subclassing ProcessGroup (which requires C++ backend registration),
we use gloo for DDP initialization and install a comm_hook that redirects
allreduce calls through our shm+stdio IPC to the Rust node.
"""

import torch
import torch.distributed as dist

from drift.allreduce import allreduce


def drift_comm_hook(shm):
    """Create a DDP communication hook that routes allreduce through drift IPC.

    Args:
        shm: A DriftShm instance connected to the Rust node's shared memory.

    Returns:
        A hook function compatible with DDP.register_comm_hook().
    """

    def hook(state, bucket):
        tensor = bucket.buffer()

        # Flatten to contiguous f32 on CPU
        original_dtype = tensor.dtype
        t = tensor.detach().float().cpu().contiguous().view(-1)

        result = allreduce(t, shm)

        # Copy result back
        if original_dtype != torch.float32:
            result = result.to(original_dtype)
        if tensor.device.type != "cpu":
            result = result.to(tensor.device)

        tensor.copy_(result.view(tensor.shape))

        fut = torch.futures.Future()
        fut.set_result(tensor)
        return fut

    return hook
