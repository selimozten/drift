/// Ring all-reduce implementation for gradient synchronization.
///
/// Each node holds a gradient buffer. The ring all-reduce works in two phases:
///   1. Scatter-reduce: each node sends a chunk to its right neighbor, receives
///      from its left neighbor, and accumulates. After N-1 steps, each node
///      holds the fully reduced version of one chunk.
///   2. All-gather: each node sends its fully reduced chunk around the ring
///      so all nodes end up with the complete reduced result.
///
/// For the network layer, we represent gradients as f32 slices serialized
/// to bytes. The coordinator assigns ring positions to nodes.

/// Split a gradient buffer into N equal chunks (last chunk may be slightly larger).
pub fn chunk_ranges(total_len: usize, n: usize) -> Vec<(usize, usize)> {
    if n == 0 {
        return vec![];
    }
    let chunk_size = total_len / n;
    let mut ranges = Vec::with_capacity(n);
    let mut offset = 0;
    for i in 0..n {
        let end = if i == n - 1 {
            total_len
        } else {
            offset + chunk_size
        };
        ranges.push((offset, end));
        offset = end;
    }
    ranges
}

/// Accumulate (sum) a received chunk into the local buffer.
pub fn accumulate(local: &mut [f32], received: &[f32]) {
    for (l, r) in local.iter_mut().zip(received.iter()) {
        *l += *r;
    }
}

/// Average the buffer by dividing each element by the number of nodes.
pub fn average(buf: &mut [f32], n: usize) {
    let scale = 1.0 / n as f32;
    for v in buf.iter_mut() {
        *v *= scale;
    }
}

/// Serialize f32 slice to bytes (little-endian).
pub fn f32_to_bytes(data: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

/// Deserialize bytes to f32 vec (little-endian).
pub fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

/// Perform a local all-reduce across multiple gradient buffers (for testing
/// and single-node multi-GPU scenarios). Returns the averaged gradient.
pub fn local_allreduce(gradients: &[Vec<f32>]) -> Vec<f32> {
    if gradients.is_empty() {
        return vec![];
    }
    let len = gradients[0].len();
    let mut result = vec![0.0f32; len];
    for grad in gradients {
        accumulate(&mut result, grad);
    }
    average(&mut result, gradients.len());
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_ranges() {
        let ranges = chunk_ranges(100, 3);
        assert_eq!(ranges, vec![(0, 33), (33, 66), (66, 100)]);
    }

    #[test]
    fn test_chunk_ranges_even() {
        let ranges = chunk_ranges(100, 4);
        assert_eq!(ranges, vec![(0, 25), (25, 50), (50, 75), (75, 100)]);
    }

    #[test]
    fn test_chunk_ranges_single() {
        let ranges = chunk_ranges(100, 1);
        assert_eq!(ranges, vec![(0, 100)]);
    }

    #[test]
    fn test_accumulate() {
        let mut a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        accumulate(&mut a, &b);
        assert_eq!(a, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_average() {
        let mut buf = vec![10.0, 20.0, 30.0];
        average(&mut buf, 2);
        assert_eq!(buf, vec![5.0, 10.0, 15.0]);
    }

    #[test]
    fn test_f32_roundtrip() {
        let data = vec![1.0f32, -2.5, 3.14, 0.0, f32::MAX];
        let bytes = f32_to_bytes(&data);
        let decoded = bytes_to_f32(&bytes);
        assert_eq!(data, decoded);
    }

    #[test]
    fn test_local_allreduce() {
        let grads = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let result = local_allreduce(&grads);
        assert_eq!(result, vec![4.0, 5.0, 6.0]); // average
    }

    #[test]
    fn test_local_allreduce_two_nodes() {
        let grads = vec![vec![2.0, 4.0], vec![6.0, 8.0]];
        let result = local_allreduce(&grads);
        assert_eq!(result, vec![4.0, 6.0]);
    }
}
