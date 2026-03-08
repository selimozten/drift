//! POSIX shared memory for zero-copy gradient transfer between Rust node and Python subprocess.
//!
//! Layout:
//!   Offset 0:   Header (64 bytes)
//!     0..3      magic = 0x44524654 ("DRFT")
//!     4..7      version = 1 (u32 LE)
//!     8..15     total_size (u64 LE)
//!     16..63    reserved (zeros)
//!   Offset 64:  Gradient data (raw f32 little-endian, contiguous)

use anyhow::{bail, Result};
use std::ptr;

const HEADER_SIZE: usize = 64;
const MAGIC: u32 = 0x44524654; // "DRFT"
const VERSION: u32 = 1;

/// Default shared memory size: 256 MB (~64M floats).
pub const DEFAULT_SHM_SIZE: usize = 256 * 1024 * 1024;

pub struct DriftShm {
    name: String,
    ptr: *mut u8,
    size: usize,
}

// Safety: The shm region is only accessed via &self/&mut self methods.
unsafe impl Send for DriftShm {}
unsafe impl Sync for DriftShm {}

impl DriftShm {
    /// Create a new shared memory region with the drift header.
    pub fn create(pid: u32, size: usize) -> Result<Self> {
        let name = format!("/drift-shm-{}", pid);
        Self::create_named(&name, size)
    }

    /// Create a new shared memory region with a specific name.
    pub fn create_named(name: &str, size: usize) -> Result<Self> {
        if size < HEADER_SIZE {
            bail!("shm size must be >= {} bytes", HEADER_SIZE);
        }

        unsafe {
            let c_name = std::ffi::CString::new(name)?;

            // Create and open
            let fd = libc::shm_open(
                c_name.as_ptr(),
                libc::O_CREAT | libc::O_RDWR | libc::O_EXCL,
                0o600,
            );
            if fd < 0 {
                // Try unlinking first in case of stale shm
                libc::shm_unlink(c_name.as_ptr());
                let fd2 = libc::shm_open(
                    c_name.as_ptr(),
                    libc::O_CREAT | libc::O_RDWR | libc::O_EXCL,
                    0o600,
                );
                if fd2 < 0 {
                    bail!("shm_open failed: {}", std::io::Error::last_os_error());
                }
                Self::init_shm(fd2, name, size)
            } else {
                Self::init_shm(fd, name, size)
            }
        }
    }

    unsafe fn init_shm(fd: i32, name: &str, size: usize) -> Result<Self> {
        // Set size
        if libc::ftruncate(fd, size as libc::off_t) < 0 {
            libc::close(fd);
            let c_name = std::ffi::CString::new(name)?;
            libc::shm_unlink(c_name.as_ptr());
            bail!("ftruncate failed: {}", std::io::Error::last_os_error());
        }

        // mmap
        let ptr = libc::mmap(
            ptr::null_mut(),
            size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED,
            fd,
            0,
        );
        libc::close(fd);

        if ptr == libc::MAP_FAILED {
            let c_name = std::ffi::CString::new(name)?;
            libc::shm_unlink(c_name.as_ptr());
            bail!("mmap failed: {}", std::io::Error::last_os_error());
        }

        let ptr = ptr as *mut u8;

        // Zero the header region
        ptr::write_bytes(ptr, 0, HEADER_SIZE);

        // Write header
        let magic_bytes = MAGIC.to_le_bytes();
        ptr::copy_nonoverlapping(magic_bytes.as_ptr(), ptr, 4);
        let version_bytes = VERSION.to_le_bytes();
        ptr::copy_nonoverlapping(version_bytes.as_ptr(), ptr.add(4), 4);
        let size_bytes = (size as u64).to_le_bytes();
        ptr::copy_nonoverlapping(size_bytes.as_ptr(), ptr.add(8), 8);

        Ok(DriftShm {
            name: name.to_string(),
            ptr,
            size,
        })
    }

    /// The POSIX shm name (e.g. "/drift-shm-12345").
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The name for Python's SharedMemory (without leading "/").
    /// Python's multiprocessing.shared_memory adds the "/" prefix internally.
    pub fn python_name(&self) -> &str {
        self.name.strip_prefix('/').unwrap_or(&self.name)
    }

    /// Maximum number of f32 values that fit in the data region.
    pub fn capacity_floats(&self) -> usize {
        (self.size - HEADER_SIZE) / 4
    }

    /// Write gradient data to the data region (offset 64).
    pub fn write_gradient(&self, data: &[f32]) -> Result<()> {
        if data.len() > self.capacity_floats() {
            bail!(
                "gradient too large: {} floats > {} capacity",
                data.len(),
                self.capacity_floats()
            );
        }
        unsafe {
            let dst = self.ptr.add(HEADER_SIZE);
            let src = data.as_ptr() as *const u8;
            ptr::copy_nonoverlapping(src, dst, data.len() * 4);
        }
        Ok(())
    }

    /// Read gradient data from the data region (offset 64).
    pub fn read_gradient(&self, num_floats: usize) -> Result<Vec<f32>> {
        if num_floats > self.capacity_floats() {
            bail!(
                "read too large: {} floats > {} capacity",
                num_floats,
                self.capacity_floats()
            );
        }
        let mut result = vec![0f32; num_floats];
        unsafe {
            let src = self.ptr.add(HEADER_SIZE);
            let dst = result.as_mut_ptr() as *mut u8;
            ptr::copy_nonoverlapping(src, dst, num_floats * 4);
        }
        Ok(result)
    }
}

impl Drop for DriftShm {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.size);
            if let Ok(c_name) = std::ffi::CString::new(self.name.as_str()) {
                libc::shm_unlink(c_name.as_ptr());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_read_write() {
        let shm = DriftShm::create_named("/drift-test-rw", 4096).unwrap();
        assert_eq!(shm.name(), "/drift-test-rw");

        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        shm.write_gradient(&data).unwrap();

        let result = shm.read_gradient(5).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_header() {
        let shm = DriftShm::create_named("/drift-test-hdr", 4096).unwrap();
        unsafe {
            let magic = u32::from_le_bytes([
                *shm.ptr,
                *shm.ptr.add(1),
                *shm.ptr.add(2),
                *shm.ptr.add(3),
            ]);
            assert_eq!(magic, MAGIC);

            let version = u32::from_le_bytes([
                *shm.ptr.add(4),
                *shm.ptr.add(5),
                *shm.ptr.add(6),
                *shm.ptr.add(7),
            ]);
            assert_eq!(version, VERSION);

            let total_size = u64::from_le_bytes([
                *shm.ptr.add(8),
                *shm.ptr.add(9),
                *shm.ptr.add(10),
                *shm.ptr.add(11),
                *shm.ptr.add(12),
                *shm.ptr.add(13),
                *shm.ptr.add(14),
                *shm.ptr.add(15),
            ]);
            assert_eq!(total_size, 4096);
        }
    }

    #[test]
    fn test_capacity() {
        let shm = DriftShm::create_named("/drift-test-cap", 4096).unwrap();
        // 4096 - 64 = 4032 bytes = 1008 floats
        assert_eq!(shm.capacity_floats(), 1008);
    }

    #[test]
    fn test_overflow() {
        let shm = DriftShm::create_named("/drift-test-ovf", HEADER_SIZE + 16).unwrap();
        assert_eq!(shm.capacity_floats(), 4);
        let big = vec![0f32; 5];
        assert!(shm.write_gradient(&big).is_err());
    }

    #[test]
    fn test_overwrite() {
        let shm = DriftShm::create_named("/drift-test-ow", 4096).unwrap();
        shm.write_gradient(&[1.0, 2.0, 3.0]).unwrap();
        shm.write_gradient(&[4.0, 5.0, 6.0]).unwrap();
        let result = shm.read_gradient(3).unwrap();
        assert_eq!(result, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_create_by_pid() {
        let shm = DriftShm::create(99999, 4096).unwrap();
        assert_eq!(shm.name(), "/drift-shm-99999");
    }

    #[test]
    fn test_drop_cleans_up() {
        let name = "/drift-test-drop";
        {
            let _shm = DriftShm::create_named(name, 4096).unwrap();
        } // dropped here
        // Should be able to create again (unlinked on drop)
        let _shm = DriftShm::create_named(name, 4096).unwrap();
    }
}
