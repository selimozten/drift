"""drift — PyTorch DDP backend for P2P distributed training over QUIC."""

import os
import sys

_rank = None
_world_size = None
_shm = None
_comm_hook = None


def init():
    """Initialize drift distributed training.

    Reads DRIFT_RANK, DRIFT_WORLD_SIZE, DRIFT_SHM_NAME from env,
    opens shared memory, initializes gloo process group, and prepares
    the drift comm_hook for DDP.

    After init(), wrap your model with DDP and call drift.register(ddp_model)
    to install the communication hook.
    """
    global _rank, _world_size, _shm, _comm_hook

    import torch.distributed as dist
    from drift.shm import DriftShm
    from drift.process_group import drift_comm_hook

    _rank = int(os.environ["DRIFT_RANK"])
    _world_size = int(os.environ["DRIFT_WORLD_SIZE"])
    shm_name = os.environ["DRIFT_SHM_NAME"]

    _shm = DriftShm(shm_name)
    _comm_hook = drift_comm_hook(_shm)

    # Use gloo for DDP initialization (param verification, broadcast, etc.)
    # The actual gradient allreduce goes through our comm_hook
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(backend="gloo", rank=_rank, world_size=_world_size)

    # Signal Rust node that we're ready
    sys.stdout.write("DRIFT_READY\n")
    sys.stdout.flush()


def register(ddp_model):
    """Register the drift communication hook on a DDP model.

    Call this after wrapping your model with DistributedDataParallel:
        model = DDP(model)
        drift.register(model)
    """
    if _comm_hook is None:
        raise RuntimeError("drift.init() must be called before drift.register()")
    ddp_model.register_comm_hook(state=None, hook=_comm_hook)


def rank():
    """Return this node's rank."""
    return _rank


def world_size():
    """Return the total number of nodes."""
    return _world_size
