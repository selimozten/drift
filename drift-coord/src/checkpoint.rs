use drift_proto::CheckpointInfo;
use std::path::{Path, PathBuf};
use tracing::info;

/// Manages checkpoint storage for the training run.
pub struct CheckpointManager {
    output_dir: PathBuf,
    checkpoints: Vec<CheckpointInfo>,
}

impl CheckpointManager {
    pub fn new(output_dir: impl AsRef<Path>) -> Self {
        Self {
            output_dir: output_dir.as_ref().to_path_buf(),
            checkpoints: Vec::new(),
        }
    }

    /// Record a checkpoint.
    pub fn record(&mut self, step: u64, nodes: Vec<String>) {
        let path = self
            .output_dir
            .join(format!("checkpoint-step-{}.pt", step))
            .to_string_lossy()
            .to_string();

        info!(step, path = %path, "checkpoint recorded");

        self.checkpoints.push(CheckpointInfo {
            step,
            path,
            nodes_contributed: nodes,
        });
    }

    pub fn latest(&self) -> Option<&CheckpointInfo> {
        self.checkpoints.last()
    }
}
