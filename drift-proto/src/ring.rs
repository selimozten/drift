use crate::allreduce::{
    accumulate, average, bytes_to_f32, chunk_ranges, compress_sparse, decompress_sparse,
    f32_to_bytes,
};
use crate::{read_message, write_message, DriftMessage, GradientChunk, ReducePhase};

/// State for a single ring all-reduce operation.
pub struct RingState {
    pub rank: usize,
    pub world_size: usize,
    pub buffer: Vec<f32>,
    ranges: Vec<(usize, usize)>,
}

impl RingState {
    pub fn new(rank: usize, world_size: usize, gradient: Vec<f32>) -> Self {
        let ranges = chunk_ranges(gradient.len(), world_size);
        Self {
            rank,
            world_size,
            buffer: gradient,
            ranges,
        }
    }

    /// Index of the chunk to send during scatter-reduce iteration `iter`.
    fn scatter_send_idx(&self, iter: usize) -> usize {
        (self.rank + self.world_size - iter) % self.world_size
    }

    /// Index of the chunk to receive during scatter-reduce iteration `iter`.
    fn scatter_recv_idx(&self, iter: usize) -> usize {
        (self.rank + self.world_size - iter - 1) % self.world_size
    }

    /// Returns the (start, end) range and chunk index to send in scatter-reduce iteration `iter`.
    pub fn scatter_chunk_to_send(&self, iter: usize) -> (usize, usize, usize) {
        let idx = self.scatter_send_idx(iter);
        let (start, end) = self.ranges[idx];
        (start, end, idx)
    }

    /// Returns the (start, end) range and chunk index to receive in scatter-reduce iteration `iter`.
    pub fn scatter_chunk_to_recv(&self, iter: usize) -> (usize, usize, usize) {
        let idx = self.scatter_recv_idx(iter);
        let (start, end) = self.ranges[idx];
        (start, end, idx)
    }

    /// Accumulate received data into the correct chunk for scatter-reduce.
    pub fn apply_scatter(&mut self, iter: usize, received: &[f32]) {
        let (start, end, _) = self.scatter_chunk_to_recv(iter);
        accumulate(&mut self.buffer[start..end], received);
    }

    /// Index of the chunk to send during all-gather iteration `iter`.
    fn gather_send_idx(&self, iter: usize) -> usize {
        (self.rank + self.world_size - iter + 1) % self.world_size
    }

    /// Index of the chunk to receive during all-gather iteration `iter`.
    fn gather_recv_idx(&self, iter: usize) -> usize {
        (self.rank + self.world_size - iter) % self.world_size
    }

    /// Returns the (start, end) range and chunk index to send in all-gather iteration `iter`.
    pub fn gather_chunk_to_send(&self, iter: usize) -> (usize, usize, usize) {
        let idx = self.gather_send_idx(iter);
        let (start, end) = self.ranges[idx];
        (start, end, idx)
    }

    /// Returns the (start, end) range and chunk index to receive in all-gather iteration `iter`.
    pub fn gather_chunk_to_recv(&self, iter: usize) -> (usize, usize, usize) {
        let idx = self.gather_recv_idx(iter);
        let (start, end) = self.ranges[idx];
        (start, end, idx)
    }

    /// Overwrite the correct chunk with gathered data.
    pub fn apply_gather(&mut self, iter: usize, received: &[f32]) {
        let (start, end, _) = self.gather_chunk_to_recv(iter);
        self.buffer[start..end].copy_from_slice(received);
    }

    /// Divide all elements by world_size to get the average.
    pub fn finalize(&mut self) {
        average(&mut self.buffer, self.world_size);
    }

    /// Get the resulting averaged gradient.
    pub fn result(&self) -> &[f32] {
        &self.buffer
    }
}

/// Check if >50% of values are zero.
fn is_sparse(data: &[f32]) -> bool {
    let zeros = data.iter().filter(|&&v| v == 0.0).count();
    zeros > data.len() / 2
}

/// Encode a chunk: compress if sparse, otherwise raw bytes.
fn encode_chunk(data: &[f32]) -> (Vec<u8>, bool) {
    if is_sparse(data) {
        (compress_sparse(data), true)
    } else {
        (f32_to_bytes(data), false)
    }
}

/// Decode a chunk: decompress if compressed, otherwise raw.
fn decode_chunk(gc: &GradientChunk, expected_len: usize) -> Vec<f32> {
    if gc.compressed {
        decompress_sparse(&gc.data, expected_len)
    } else {
        bytes_to_f32(&gc.data)
    }
}

/// Run the scatter-reduce phase over QUIC streams.
/// `send_right` sends to the right neighbor, `recv_left` receives from the left.
pub async fn run_scatter_reduce(
    state: &mut RingState,
    step: u64,
    send_right: &mut iroh::endpoint::SendStream,
    recv_left: &mut iroh::endpoint::RecvStream,
) -> anyhow::Result<()> {
    let n = state.world_size;
    for iter in 0..(n - 1) {
        let (start, end, chunk_idx) = state.scatter_chunk_to_send(iter);
        let (data, compressed) = encode_chunk(&state.buffer[start..end]);

        let send_msg = DriftMessage::GradientChunk(GradientChunk {
            step,
            chunk_index: chunk_idx as u32,
            phase: ReducePhase::ScatterReduce,
            compressed,
            data,
        });

        // Concurrent send and recv to prevent deadlock
        let (send_result, recv_result) = tokio::join!(
            write_message(send_right, &send_msg),
            read_message(recv_left),
        );
        send_result?;
        let msg = recv_result?;

        match msg {
            DriftMessage::GradientChunk(gc) => {
                let (recv_start, recv_end, _) = state.scatter_chunk_to_recv(iter);
                let received = decode_chunk(&gc, recv_end - recv_start);
                state.apply_scatter(iter, &received);
            }
            other => anyhow::bail!("expected GradientChunk during scatter, got {}", other),
        }
    }
    Ok(())
}

/// Run the all-gather phase over QUIC streams.
pub async fn run_allgather(
    state: &mut RingState,
    step: u64,
    send_right: &mut iroh::endpoint::SendStream,
    recv_left: &mut iroh::endpoint::RecvStream,
) -> anyhow::Result<()> {
    let n = state.world_size;
    for iter in 0..(n - 1) {
        let (start, end, chunk_idx) = state.gather_chunk_to_send(iter);
        let (data, compressed) = encode_chunk(&state.buffer[start..end]);

        let send_msg = DriftMessage::GradientChunk(GradientChunk {
            step,
            chunk_index: chunk_idx as u32,
            phase: ReducePhase::AllGather,
            compressed,
            data,
        });

        let (send_result, recv_result) = tokio::join!(
            write_message(send_right, &send_msg),
            read_message(recv_left),
        );
        send_result?;
        let msg = recv_result?;

        match msg {
            DriftMessage::GradientChunk(gc) => {
                let (recv_start, recv_end, _) = state.gather_chunk_to_recv(iter);
                let received = decode_chunk(&gc, recv_end - recv_start);
                state.apply_gather(iter, &received);
            }
            other => anyhow::bail!("expected GradientChunk during gather, got {}", other),
        }
    }
    Ok(())
}

/// Execute a full ring all-reduce over QUIC streams.
/// Returns the averaged gradient buffer.
pub async fn ring_allreduce(
    mut state: RingState,
    step: u64,
    send_right: &mut iroh::endpoint::SendStream,
    recv_left: &mut iroh::endpoint::RecvStream,
) -> anyhow::Result<Vec<f32>> {
    if state.world_size <= 1 {
        state.finalize();
        return Ok(state.buffer);
    }

    run_scatter_reduce(&mut state, step, send_right, recv_left).await?;
    run_allgather(&mut state, step, send_right, recv_left).await?;
    state.finalize();
    Ok(state.buffer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allreduce::local_allreduce;

    /// Simulate a full ring all-reduce in-process for 3 nodes.
    #[test]
    fn test_ring_allreduce_3_nodes() {
        let gradients = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
            vec![100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0],
        ];

        let expected = local_allreduce(&gradients);
        let n = 3;

        let mut states: Vec<RingState> = gradients
            .into_iter()
            .enumerate()
            .map(|(rank, grad)| RingState::new(rank, n, grad))
            .collect();

        // Scatter-reduce: N-1 iterations
        for iter in 0..(n - 1) {
            // Each node sends its chunk to the right, receives from the left
            let mut chunks_to_send: Vec<Vec<f32>> = Vec::new();
            for state in &states {
                let (start, end, _) = state.scatter_chunk_to_send(iter);
                chunks_to_send.push(state.buffer[start..end].to_vec());
            }

            // Node i receives from node (i-1) % n (its left neighbor)
            for i in 0..n {
                let left = if i == 0 { n - 1 } else { i - 1 };
                let received = chunks_to_send[left].clone();
                states[i].apply_scatter(iter, &received);
            }
        }

        // All-gather: N-1 iterations
        for iter in 0..(n - 1) {
            let mut chunks_to_send: Vec<Vec<f32>> = Vec::new();
            for state in &states {
                let (start, end, _) = state.gather_chunk_to_send(iter);
                chunks_to_send.push(state.buffer[start..end].to_vec());
            }

            for i in 0..n {
                let left = if i == 0 { n - 1 } else { i - 1 };
                let received = chunks_to_send[left].clone();
                states[i].apply_gather(iter, &received);
            }
        }

        // Finalize
        for state in &mut states {
            state.finalize();
        }

        // All nodes should have the same result equal to local_allreduce
        for (rank, state) in states.iter().enumerate() {
            let result = state.result();
            for (j, (a, b)) in result.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-4,
                    "rank {} element {}: got {} expected {}",
                    rank,
                    j,
                    a,
                    b
                );
            }
        }
    }

    #[test]
    fn test_ring_allreduce_2_nodes() {
        let gradients = vec![vec![2.0, 4.0, 6.0, 8.0], vec![10.0, 20.0, 30.0, 40.0]];
        let expected = local_allreduce(&gradients);
        let n = 2;

        let mut states: Vec<RingState> = gradients
            .into_iter()
            .enumerate()
            .map(|(rank, grad)| RingState::new(rank, n, grad))
            .collect();

        // Scatter-reduce
        for iter in 0..(n - 1) {
            let mut chunks: Vec<Vec<f32>> = Vec::new();
            for state in &states {
                let (start, end, _) = state.scatter_chunk_to_send(iter);
                chunks.push(state.buffer[start..end].to_vec());
            }
            for i in 0..n {
                let left = if i == 0 { n - 1 } else { i - 1 };
                states[i].apply_scatter(iter, &chunks[left]);
            }
        }

        // All-gather
        for iter in 0..(n - 1) {
            let mut chunks: Vec<Vec<f32>> = Vec::new();
            for state in &states {
                let (start, end, _) = state.gather_chunk_to_send(iter);
                chunks.push(state.buffer[start..end].to_vec());
            }
            for i in 0..n {
                let left = if i == 0 { n - 1 } else { i - 1 };
                states[i].apply_gather(iter, &chunks[left]);
            }
        }

        for state in &mut states {
            state.finalize();
        }

        for state in &states {
            for (a, b) in state.result().iter().zip(expected.iter()) {
                assert!((a - b).abs() < 1e-4);
            }
        }
    }
}
