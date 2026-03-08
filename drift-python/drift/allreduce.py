"""Low-level allreduce via shared memory and stdio IPC.

The Python process writes gradient data to shared memory, then signals the
Rust parent via stdout. The Rust side performs the actual ring all-reduce
over QUIC and writes the result back to shm. Python blocks on stdin until done.
"""

import sys
import struct
import threading

_op_counter = 0
_op_lock = threading.Lock()


def _next_op_id():
    global _op_counter
    with _op_lock:
        op_id = _op_counter
        _op_counter += 1
        return op_id


def allreduce(tensor, shm):
    """Perform all-reduce on a tensor via shared memory IPC.

    Args:
        tensor: A torch.Tensor (will be flattened to contiguous f32).
        shm: A DriftShm instance connected to the Rust node's shared memory.

    Returns:
        torch.Tensor: The averaged tensor, same shape and device as input.
    """
    import torch

    original_shape = tensor.shape
    original_device = tensor.device
    original_dtype = tensor.dtype

    # Move to CPU f32 if needed
    t = tensor.detach().float().cpu().contiguous().view(-1)
    num_floats = t.numel()

    # Write to shm (raw bytes)
    data = t.numpy().tobytes()
    shm.write_floats(data, num_floats)

    # Signal Rust via stdout
    op_id = _next_op_id()
    sys.stdout.write(f"DRIFT_ALLREDUCE {op_id} {num_floats}\n")
    sys.stdout.flush()

    # Block until Rust responds on stdin
    response = sys.stdin.readline().strip()
    expected = f"DRIFT_ALLREDUCE_DONE {op_id}"
    if response != expected:
        raise RuntimeError(
            f"unexpected IPC response: {response!r}, expected {expected!r}"
        )

    # Read result from shm
    result_bytes = shm.read_floats(num_floats)
    result = torch.frombuffer(bytearray(result_bytes), dtype=torch.float32).clone()
    result = result.view(original_shape)

    # Convert back to original dtype/device
    if original_dtype != torch.float32:
        result = result.to(original_dtype)
    if original_device.type != "cpu":
        result = result.to(original_device)

    return result


def allreduce_bytes(data_bytes, num_floats, shm):
    """Perform all-reduce on raw f32 bytes via shared memory IPC.

    Lower-level variant that avoids torch dependency. Used for testing.

    Args:
        data_bytes: Raw bytes of f32 little-endian data.
        num_floats: Number of float32 values.
        shm: A DriftShm instance.

    Returns:
        bytes: The averaged f32 data.
    """
    shm.write_floats(data_bytes, num_floats)

    op_id = _next_op_id()
    sys.stdout.write(f"DRIFT_ALLREDUCE {op_id} {num_floats}\n")
    sys.stdout.flush()

    response = sys.stdin.readline().strip()
    expected = f"DRIFT_ALLREDUCE_DONE {op_id}"
    if response != expected:
        raise RuntimeError(
            f"unexpected IPC response: {response!r}, expected {expected!r}"
        )

    return shm.read_floats(num_floats)
