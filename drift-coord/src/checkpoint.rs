use drift_proto::CheckpointInfo;
use std::path::{Path, PathBuf};
use tracing::{info, warn};

/// Manages checkpoint storage for the training run.
pub struct CheckpointManager {
    output_dir: PathBuf,
    checkpoints: Vec<CheckpointInfo>,
    save_interval_steps: u64,
    last_saved_step: u64,
}

impl CheckpointManager {
    pub fn new(output_dir: impl AsRef<Path>) -> Self {
        Self {
            output_dir: output_dir.as_ref().to_path_buf(),
            checkpoints: Vec::new(),
            save_interval_steps: 100,
            last_saved_step: 0,
        }
    }

    pub fn with_interval(mut self, interval: u64) -> Self {
        self.save_interval_steps = interval;
        self
    }

    /// Check if a checkpoint should be saved at this step.
    pub fn should_save(&self, step: u64) -> bool {
        step > 0 && step - self.last_saved_step >= self.save_interval_steps
    }

    /// Record a checkpoint and write metadata to disk.
    pub fn record(&mut self, step: u64, nodes: Vec<String>) -> CheckpointInfo {
        let path = self
            .output_dir
            .join(format!("checkpoint-step-{}.pt", step))
            .to_string_lossy()
            .to_string();

        let ckpt = CheckpointInfo {
            step,
            path: path.clone(),
            nodes_contributed: nodes,
        };

        // Write metadata JSON alongside the checkpoint
        let meta_path = self
            .output_dir
            .join(format!("checkpoint-step-{}.json", step));
        match serde_json::to_string_pretty(&ckpt) {
            Ok(json) => {
                if let Err(e) = std::fs::write(&meta_path, &json) {
                    warn!(step, "failed to write checkpoint metadata: {}", e);
                } else {
                    info!(step, path = %path, "checkpoint saved");
                }
            }
            Err(e) => warn!(step, "failed to serialize checkpoint: {}", e),
        }

        self.last_saved_step = step;
        self.checkpoints.push(ckpt.clone());

        // Write a "latest" symlink/file for easy resume
        let latest_path = self.output_dir.join("latest.json");
        let _ = std::fs::write(&latest_path, serde_json::to_string_pretty(&ckpt).unwrap_or_default());

        ckpt
    }

    pub fn latest(&self) -> Option<&CheckpointInfo> {
        self.checkpoints.last()
    }

    /// Load the latest checkpoint from disk for resuming.
    pub fn load_latest(output_dir: impl AsRef<Path>) -> Option<CheckpointInfo> {
        let latest_path = output_dir.as_ref().join("latest.json");
        let data = std::fs::read_to_string(&latest_path).ok()?;
        serde_json::from_str(&data).ok()
    }

    pub fn checkpoint_count(&self) -> usize {
        self.checkpoints.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_save_respects_interval() {
        let mgr = CheckpointManager::new("/tmp/test").with_interval(50);
        assert!(!mgr.should_save(0));
        assert!(!mgr.should_save(49));
        assert!(mgr.should_save(50));
        assert!(mgr.should_save(100));
    }

    #[test]
    fn record_updates_last_saved() {
        let dir = std::env::temp_dir().join("drift-test-ckpt");
        std::fs::create_dir_all(&dir).ok();
        let mut mgr = CheckpointManager::new(&dir).with_interval(10);

        assert!(mgr.should_save(10));
        mgr.record(10, vec!["node-a".into()]);
        assert!(!mgr.should_save(15));
        assert!(mgr.should_save(20));

        // Verify metadata was written
        let meta = dir.join("checkpoint-step-10.json");
        assert!(meta.exists());

        let latest = dir.join("latest.json");
        assert!(latest.exists());

        // Cleanup
        std::fs::remove_dir_all(&dir).ok();
    }
}
