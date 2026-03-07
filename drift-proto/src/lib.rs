use std::fmt;

use serde::{Deserialize, Serialize};

/// Information about a node's GPU capabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
        }
    }
}

/// Training configuration sent from coordinator to nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainConfig {
    pub model_path: String,
    pub dataset_path: String,
    pub batch_size: u32,
    pub learning_rate: f64,
    pub epochs: u32,
}

/// Shard assignment for a specific node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardAssignment {
    pub node_id: String,
    pub shard_index: u32,
    pub shard_start: u64,
    pub shard_end: u64,
}

/// Training progress report from a node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainProgress {
    pub node_id: String,
    pub epoch: u32,
    pub step: u64,
    pub loss: f64,
    pub throughput_samples_per_sec: f64,
}

/// Gradient payload for all-reduce synchronization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientPayload {
    pub node_id: String,
    pub step: u64,
    pub data: Vec<u8>,
}

/// Checkpoint metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointInfo {
    pub step: u64,
    pub path: String,
    pub nodes_contributed: Vec<String>,
}

/// All messages exchanged between drift nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DriftMessage {
    NodeInfo(NodeInfo),
    TrainConfig(TrainConfig),
    ShardAssignment(ShardAssignment),
    TrainProgress(TrainProgress),
    GradientPayload(GradientPayload),
    CheckpointInfo(CheckpointInfo),
    Ping,
    Pong,
}

/// ALPN protocol identifier for drift.
pub const DRIFT_ALPN: &[u8] = b"drift/0";

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

    if len > 64 * 1024 * 1024 {
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
