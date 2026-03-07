pub mod allreduce;

use std::fmt;

use serde::{Deserialize, Serialize};

/// Information about a node's GPU capabilities.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeInfo {
    pub node_id: String,
    pub gpu_name: String,
    pub gpu_vram_mb: u64,
    pub gpu_compute_capability: String,
    pub available: bool,
}

impl fmt::Display for NodeInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} | {} ({} MB VRAM, compute {})",
            &self.node_id[..12.min(self.node_id.len())],
            self.gpu_name,
            self.gpu_vram_mb,
            self.gpu_compute_capability,
        )
    }
}

impl fmt::Display for DriftMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NodeInfo(info) => write!(f, "NodeInfo({})", info),
            Self::TrainConfig(c) => {
                write!(f, "TrainConfig(epochs={}, batch={})", c.epochs, c.batch_size)
            }
            Self::ShardAssignment(s) => {
                write!(f, "ShardAssignment(node={}, idx={})", &s.node_id[..12.min(s.node_id.len())], s.shard_index)
            }
            Self::TrainProgress(p) => {
                write!(f, "TrainProgress(step={}, loss={:.4})", p.step, p.loss)
            }
            Self::GradientPayload(g) => {
                write!(f, "GradientPayload(step={}, {} bytes)", g.step, g.data.len())
            }
            Self::CheckpointInfo(c) => write!(f, "CheckpointInfo(step={})", c.step),
            Self::Ping => write!(f, "Ping"),
            Self::Pong => write!(f, "Pong"),
            Self::Heartbeat { uptime_secs } => write!(f, "Heartbeat({}s)", uptime_secs),
            Self::TrainComplete => write!(f, "TrainComplete"),
            Self::RingConfig(r) => {
                write!(f, "RingConfig(rank={}, world={})", r.rank, r.world_size)
            }
            Self::GradientChunk(g) => {
                write!(f, "GradientChunk(step={}, chunk={}, {} bytes)", g.step, g.chunk_index, g.data.len())
            }
        }
    }
}

/// Training configuration sent from coordinator to nodes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainConfig {
    pub model_path: String,
    pub dataset_path: String,
    pub batch_size: u32,
    pub learning_rate: f64,
    pub epochs: u32,
}

/// Shard assignment for a specific node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShardAssignment {
    pub node_id: String,
    pub shard_index: u32,
    pub shard_start: u64,
    pub shard_end: u64,
}

impl ShardAssignment {
    pub fn size(&self) -> u64 {
        self.shard_end.saturating_sub(self.shard_start)
    }
}

/// Training progress report from a node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainProgress {
    pub node_id: String,
    pub epoch: u32,
    pub step: u64,
    pub loss: f64,
    pub throughput_samples_per_sec: f64,
}

/// Gradient payload for all-reduce synchronization.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GradientPayload {
    pub node_id: String,
    pub step: u64,
    pub data: Vec<u8>,
}

/// Checkpoint metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CheckpointInfo {
    pub step: u64,
    pub path: String,
    pub nodes_contributed: Vec<String>,
}

/// Ring topology configuration sent to each node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RingConfig {
    pub rank: u32,
    pub world_size: u32,
    pub left_peer_id: String,
    pub right_peer_id: String,
}

/// Phase of the ring all-reduce algorithm.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ReducePhase {
    ScatterReduce,
    AllGather,
}

/// A gradient chunk sent between ring neighbors.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GradientChunk {
    pub step: u64,
    pub chunk_index: u32,
    pub phase: ReducePhase,
    pub compressed: bool,
    pub data: Vec<u8>,
}

/// All messages exchanged between drift nodes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DriftMessage {
    NodeInfo(NodeInfo),
    TrainConfig(TrainConfig),
    ShardAssignment(ShardAssignment),
    TrainProgress(TrainProgress),
    GradientPayload(GradientPayload),
    CheckpointInfo(CheckpointInfo),
    Ping,
    Pong,
    /// Periodic heartbeat with uptime in seconds.
    Heartbeat { uptime_secs: u64 },
    /// Coordinator signals training is complete.
    TrainComplete,
    /// Ring topology configuration for a node.
    RingConfig(RingConfig),
    /// Gradient chunk sent between ring neighbors.
    GradientChunk(GradientChunk),
}

/// ALPN protocol identifier for drift.
pub const DRIFT_ALPN: &[u8] = b"drift/0";

/// Maximum allowed message size (64 MB).
pub const MAX_MESSAGE_SIZE: usize = 64 * 1024 * 1024;

/// Serialize a DriftMessage to bytes (length-prefixed JSON).
pub fn encode_message(msg: &DriftMessage) -> anyhow::Result<Vec<u8>> {
    let json = serde_json::to_vec(msg)?;
    let len = (json.len() as u32).to_be_bytes();
    let mut buf = Vec::with_capacity(4 + json.len());
    buf.extend_from_slice(&len);
    buf.extend_from_slice(&json);
    Ok(buf)
}

/// Read a length-prefixed JSON message from a recv stream.
pub async fn read_message(recv: &mut iroh::endpoint::RecvStream) -> anyhow::Result<DriftMessage> {
    let mut len_buf = [0u8; 4];
    recv.read_exact(&mut len_buf).await?;
    let len = u32::from_be_bytes(len_buf) as usize;

    if len > MAX_MESSAGE_SIZE {
        anyhow::bail!("message too large: {} bytes", len);
    }

    let mut buf = vec![0u8; len];
    recv.read_exact(&mut buf).await?;
    let msg: DriftMessage = serde_json::from_slice(&buf)?;
    Ok(msg)
}

/// Write a length-prefixed JSON message to a send stream.
pub async fn write_message(
    send: &mut iroh::endpoint::SendStream,
    msg: &DriftMessage,
) -> anyhow::Result<()> {
    let bytes = encode_message(msg)?;
    send.write_all(&bytes).await?;
    Ok(())
}
