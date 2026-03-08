//! Integration test: Rust creates shm, spawns helper Python script,
//! exchanges one allreduce round-trip, verifies data integrity.

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::ptr;

const HEADER_SIZE: usize = 64;
const MAGIC: u32 = 0x44524654;
const VERSION: u32 = 1;

/// Minimal shm helper for tests (doesn't depend on drift-cli internals).
struct TestShm {
    name: String,
    ptr: *mut u8,
    size: usize,
}

unsafe impl Send for TestShm {}

impl TestShm {
    fn create(name: &str, size: usize) -> Self {
        unsafe {
            let c_name = std::ffi::CString::new(name).unwrap();
            // Clean up any stale shm
            libc::shm_unlink(c_name.as_ptr());

            let fd = libc::shm_open(
                c_name.as_ptr(),
                libc::O_CREAT | libc::O_RDWR | libc::O_EXCL,
                0o600,
            );
            assert!(fd >= 0, "shm_open failed");

            libc::ftruncate(fd, size as libc::off_t);
            let p = libc::mmap(
                ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            );
            libc::close(fd);
            assert_ne!(p, libc::MAP_FAILED);

            let ptr = p as *mut u8;
            // Write header
            ptr::write_bytes(ptr, 0, HEADER_SIZE);
            ptr::copy_nonoverlapping(MAGIC.to_le_bytes().as_ptr(), ptr, 4);
            ptr::copy_nonoverlapping(VERSION.to_le_bytes().as_ptr(), ptr.add(4), 4);
            ptr::copy_nonoverlapping((size as u64).to_le_bytes().as_ptr(), ptr.add(8), 8);

            TestShm {
                name: name.to_string(),
                ptr,
                size,
            }
        }
    }

    fn read_floats(&self, n: usize) -> Vec<f32> {
        let mut result = vec![0f32; n];
        unsafe {
            ptr::copy_nonoverlapping(
                self.ptr.add(HEADER_SIZE),
                result.as_mut_ptr() as *mut u8,
                n * 4,
            );
        }
        result
    }

    fn write_floats(&self, data: &[f32]) {
        unsafe {
            ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                self.ptr.add(HEADER_SIZE),
                data.len() * 4,
            );
        }
    }
}

impl Drop for TestShm {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.size);
            if let Ok(c_name) = std::ffi::CString::new(self.name.as_str()) {
                libc::shm_unlink(c_name.as_ptr());
            }
        }
    }
}

#[test]
fn test_rust_python_allreduce_roundtrip() {
    let shm_name = "/drift-test-py-ipc";
    // Python's SharedMemory expects name without leading "/"
    let python_shm_name = &shm_name[1..];
    let num_floats = 4usize;
    let shm = TestShm::create(shm_name, HEADER_SIZE + num_floats * 4);

    // Find the helper script relative to this test file
    let helper_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("helper_allreduce.py");

    let mut child = Command::new("python3")
        .arg(&helper_path)
        .env("DRIFT_SHM_NAME", python_shm_name)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn python3");

    let mut child_stdin = child.stdin.take().unwrap();
    let child_stdout = child.stdout.take().unwrap();
    let child_stderr = child.stderr.take().unwrap();
    let mut reader = BufReader::new(child_stdout);

    // Read DRIFT_ALLREDUCE from child
    let mut line = String::new();
    reader.read_line(&mut line).unwrap();
    let line = line.trim();
    assert!(
        line.starts_with("DRIFT_ALLREDUCE"),
        "expected DRIFT_ALLREDUCE, got: {}",
        line
    );

    let parts: Vec<&str> = line.split_whitespace().collect();
    let op_id: u64 = parts[1].parse().unwrap();
    let n: usize = parts[2].parse().unwrap();
    assert_eq!(n, num_floats);

    // Verify Python wrote [1.0, 2.0, 3.0, 4.0] to shm
    let gradient = shm.read_floats(num_floats);
    assert_eq!(gradient, vec![1.0f32, 2.0, 3.0, 4.0]);

    // "All-reduce": multiply by 3 (simulate result from 3-node ring)
    let averaged: Vec<f32> = gradient.iter().map(|v| v * 3.0).collect();
    shm.write_floats(&averaged);

    // Send DRIFT_ALLREDUCE_DONE
    write!(child_stdin, "DRIFT_ALLREDUCE_DONE {}\n", op_id).unwrap();
    child_stdin.flush().unwrap();

    // Wait for child and verify result
    drop(child_stdin);
    drop(reader);
    let mut stderr_output = String::new();
    {
        use std::io::Read;
        let mut stderr_reader = BufReader::new(child_stderr);
        stderr_reader.read_to_string(&mut stderr_output).unwrap();
    }
    let status = child.wait().unwrap();
    assert!(status.success(), "child failed: {:?}", status);

    let stderr = &stderr_output;
    let first_line = stderr.lines().next().unwrap_or("");
    assert_eq!(
        first_line, "RESULT 3.0 6.0 9.0 12.0",
        "unexpected result: {}",
        first_line
    );
}
