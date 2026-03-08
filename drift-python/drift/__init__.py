"""drift — PyTorch DDP backend for P2P distributed training over QUIC."""

import os
import sys

_rank = None
_world_size = None
_shm = None
_process_group = None


def init():
    """Initialize drift distributed training.

    Reads DRIFT_RANK, DRIFT_WORLD_SIZE, DRIFT_SHM_NAME from env,
    opens shared memory, registers the drift ProcessGroup backend,
    and calls dist.init_process_group.
    """
    global _rank, _world_size, _shm, _process_group

    import torch.distributed as dist
    from drift.shm import DriftShm
    from drift.process_group import DriftProcessGroup

    _rank = int(os.environ["DRIFT_RANK"])
    _world_size = int(os.environ["DRIFT_WORLD_SIZE"])
    shm_name = os.environ["DRIFT_SHM_NAME"]

    _shm = DriftShm(shm_name)
    _process_group = DriftProcessGroup(_rank, _world_size, _shm)

    # Register the drift backend with PyTorch
    def _create_drift_pg(prefix_store, rank, world_size, timeout):
        return _process_group

    dist.Backend.register_backend("drift", _create_drift_pg)
    dist.init_process_group(backend="drift", rank=_rank, world_size=_world_size)

    # Signal Rust node that we're ready
    sys.stdout.write("DRIFT_READY\n")
    sys.stdout.flush()


def rank():
    """Return this node's rank."""
    return _rank


def world_size():
    """Return the total number of nodes."""
    return _world_size
