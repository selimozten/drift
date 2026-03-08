"""Shared memory module for Python-side gradient IPC with the Rust node."""

import struct
from multiprocessing.shared_memory import SharedMemory

# Header layout (64 bytes):
#   0..3   magic = 0x44524654 ("DRFT")
#   4..7   version = 1 (u32 LE)
#   8..15  total_size (u64 LE)
#   16..63 reserved
HEADER_SIZE = 64
MAGIC = 0x44524654
VERSION = 1


class DriftShm:
    """Opens an existing POSIX shared memory region created by the Rust node.

    The Rust side creates and owns the shm lifecycle. Python only opens
    and mmaps it for zero-copy gradient transfer.
    """

    def __init__(self, name: str):
        self._shm = SharedMemory(name=name, create=False)
        self._validate_header()

    def _validate_header(self):
        buf = self._shm.buf
        magic = struct.unpack_from("<I", buf, 0)[0]
        if magic != MAGIC:
            raise ValueError(f"bad shm magic: 0x{magic:08x}, expected 0x{MAGIC:08x}")
        version = struct.unpack_from("<I", buf, 4)[0]
        if version != VERSION:
            raise ValueError(f"bad shm version: {version}, expected {VERSION}")

    @property
    def total_size(self) -> int:
        return struct.unpack_from("<Q", self._shm.buf, 8)[0]

    @property
    def capacity_floats(self) -> int:
        """Maximum number of f32 values that fit in the data region."""
        return (self.total_size - HEADER_SIZE) // 4

    def write_floats(self, data: bytes, num_floats: int):
        """Write raw f32 bytes at offset 64 (data region)."""
        nbytes = num_floats * 4
        if len(data) < nbytes:
            raise ValueError(f"data too short: {len(data)} < {nbytes}")
        if num_floats > self.capacity_floats:
            raise ValueError(f"too many floats: {num_floats} > {self.capacity_floats}")
        self._shm.buf[HEADER_SIZE : HEADER_SIZE + nbytes] = data[:nbytes]

    def read_floats(self, num_floats: int) -> bytes:
        """Read raw f32 bytes from offset 64 (data region)."""
        nbytes = num_floats * 4
        if num_floats > self.capacity_floats:
            raise ValueError(f"too many floats: {num_floats} > {self.capacity_floats}")
        return bytes(self._shm.buf[HEADER_SIZE : HEADER_SIZE + nbytes])

    def close(self):
        self._shm.close()

    def __del__(self):
        try:
            self._shm.close()
        except Exception:
            pass
