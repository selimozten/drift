"""PyTorch ProcessGroup backend for DDP over drift.

Implements the minimal ProcessGroup interface needed for DDP to route
allreduce calls through our shm+stdio IPC to the Rust node.
"""

from drift.allreduce import allreduce


class DriftWork:
    """Synchronous work handle — allreduce is blocking, so work is always done."""

    def is_completed(self):
        return True

    def is_success(self):
        return True

    def wait(self, timeout=None):
        return True


class DriftProcessGroup:
    """ProcessGroup that routes tensor operations through drift's IPC layer."""

    def __init__(self, rank, world_size, shm):
        self._rank = rank
        self._world_size = world_size
        self._shm = shm

    def rank(self):
        return self._rank

    def size(self):
        return self._world_size

    def getBackendName(self):
        return "drift"

    def allreduce(self, tensors, opts=None):
        """All-reduce a list of tensors (DDP passes one tensor per call)."""
        import torch

        for i, tensor in enumerate(tensors):
            # Move to CPU f32 for IPC
            was_cuda = tensor.device.type == "cuda"
            original_dtype = tensor.dtype

            t = tensor.detach().float().cpu().contiguous()
            result = allreduce(t, self._shm)

            # Copy result back in-place
            if original_dtype != result.dtype:
                result = result.to(original_dtype)
            if was_cuda:
                result = result.to(tensor.device)
            tensor.data.copy_(result)

        return DriftWork()

    def broadcast(self, tensors, opts=None):
        """Broadcast — no-op stub (rank 0 already has the data in single-coordinator setup)."""
        return DriftWork()

    def barrier(self, opts=None):
        """Barrier — implemented as a zero-size allreduce."""
        import torch

        zero = torch.zeros(1)
        self.allreduce([zero])
        return DriftWork()
