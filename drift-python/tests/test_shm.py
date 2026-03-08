"""Tests for drift shared memory module."""

import struct
import pytest
from multiprocessing.shared_memory import SharedMemory

from drift.shm import DriftShm, HEADER_SIZE, MAGIC, VERSION


def _create_test_shm(name: str, num_floats: int = 1024) -> SharedMemory:
    """Create a shm region with valid drift header (mimics Rust side)."""
    size = HEADER_SIZE + num_floats * 4
    shm = SharedMemory(name=name, create=True, size=size)
    # Write header
    struct.pack_into("<I", shm.buf, 0, MAGIC)
    struct.pack_into("<I", shm.buf, 4, VERSION)
    struct.pack_into("<Q", shm.buf, 8, size)
    return shm


class TestDriftShm:
    def test_roundtrip(self, tmp_path):
        owner = _create_test_shm("test-rt")
        try:
            ds = DriftShm("test-rt")
            # Write [1.0, 2.0, 3.0]
            data = struct.pack("<3f", 1.0, 2.0, 3.0)
            ds.write_floats(data, 3)
            out = ds.read_floats(3)
            vals = struct.unpack("<3f", out)
            assert vals == (1.0, 2.0, 3.0)
            ds.close()
        finally:
            owner.close()
            owner.unlink()

    def test_capacity(self):
        owner = _create_test_shm("test-cap", num_floats=256)
        try:
            ds = DriftShm("test-cap")
            assert ds.capacity_floats == 256
            assert ds.total_size == HEADER_SIZE + 256 * 4
            ds.close()
        finally:
            owner.close()
            owner.unlink()

    def test_bad_magic(self):
        shm = SharedMemory(name="test-bad", create=True, size=HEADER_SIZE + 64)
        try:
            struct.pack_into("<I", shm.buf, 0, 0xDEADBEEF)
            struct.pack_into("<I", shm.buf, 4, VERSION)
            with pytest.raises(ValueError, match="bad shm magic"):
                DriftShm("test-bad")
        finally:
            shm.close()
            shm.unlink()

    def test_bad_version(self):
        shm = SharedMemory(name="test-ver", create=True, size=HEADER_SIZE + 64)
        try:
            struct.pack_into("<I", shm.buf, 0, MAGIC)
            struct.pack_into("<I", shm.buf, 4, 99)
            with pytest.raises(ValueError, match="bad shm version"):
                DriftShm("test-ver")
        finally:
            shm.close()
            shm.unlink()

    def test_write_too_many_floats(self):
        owner = _create_test_shm("test-ovf", num_floats=4)
        try:
            ds = DriftShm("test-ovf")
            data = struct.pack("<10f", *range(10))
            with pytest.raises(ValueError, match="too many floats"):
                ds.write_floats(data, 10)
            ds.close()
        finally:
            owner.close()
            owner.unlink()
