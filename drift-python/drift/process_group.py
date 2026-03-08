"""PyTorch DDP communication hook for drift.

Instead of subclassing ProcessGroup (which requires C++ backend registration),
we use gloo for DDP initialization and install a comm_hook that redirects
allreduce calls through our shm+stdio IPC to the Rust node.

DDP splits gradients into ~25MB buckets and calls allreduce per bucket.
The comm_hook handles each bucket independently. Dtype conversion
(f16/bf16 -> f32 -> allreduce -> back) is handled transparently.
"""

import torch
import torch.distributed as dist

from drift.allreduce import allreduce


def drift_comm_hook(shm):
    """Create a DDP communication hook that routes allreduce through drift IPC.

    Handles multiple gradient buckets (DDP calls the hook once per bucket)
    and dtype conversion for mixed-precision training.

    Args:
        shm: A DriftShm instance connected to the Rust node's shared memory.

    Returns:
        A hook function compatible with DDP.register_comm_hook().
    """

    def hook(state, bucket):
        tensor = bucket.buffer()

        original_dtype = tensor.dtype
        original_device = tensor.device

        # Convert to f32 on CPU for IPC (handles f16/bf16 mixed precision)
        t = tensor.detach()
        if original_dtype != torch.float32:
            t = t.float()
        if original_device.type != "cpu":
            t = t.cpu()
        t = t.contiguous().view(-1)

        result = allreduce(t, shm)

        # Convert back to original dtype/device
        if original_dtype != torch.float32:
            result = result.to(original_dtype)
        if original_device.type != "cpu":
            result = result.to(original_device)

        tensor.copy_(result.view(tensor.shape))

        fut = torch.futures.Future()
        fut.set_result(tensor)
        return fut

    return hook
