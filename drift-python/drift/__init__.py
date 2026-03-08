"""drift — PyTorch DDP backend for P2P distributed training over QUIC."""

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
    raise NotImplementedError("drift.init() not yet implemented")


def rank():
    """Return this node's rank."""
    return _rank


def world_size():
    """Return the total number of nodes."""
    return _world_size
