use drift_proto::TrainProgress;
use std::collections::HashMap;
use tracing::{info, warn};

/// Tracks training progress and node health.
pub struct Monitor {
    progress: HashMap<String, TrainProgress>,
    disconnected: Vec<String>,
}

impl Monitor {
    pub fn new() -> Self {
        Self {
            progress: HashMap::new(),
            disconnected: Vec::new(),
        }
    }

    /// Update progress for a node.
    pub fn update_progress(&mut self, progress: TrainProgress) {
        info!(
            node_id = %progress.node_id,
            epoch = progress.epoch,
            step = progress.step,
            loss = progress.loss,
            throughput = progress.throughput_samples_per_sec,
            "training progress"
        );
        self.progress.insert(progress.node_id.clone(), progress);
    }

    /// Mark a node as disconnected.
    pub fn mark_disconnected(&mut self, node_id: &str) {
        warn!(node_id, "node disconnected");
        self.disconnected.push(node_id.to_string());
        self.progress.remove(node_id);
    }

    /// Print a summary of current training status.
    pub fn print_status(&self) {
        println!("--- Training Status ---");
        if self.progress.is_empty() {
            println!("  No active nodes.");
        }
        for (node_id, p) in &self.progress {
            println!(
                "  {} | epoch {} step {} | loss {:.4} | {:.1} samples/s",
                &node_id[..12.min(node_id.len())],
                p.epoch,
                p.step,
                p.loss,
                p.throughput_samples_per_sec,
            );
        }
        if !self.disconnected.is_empty() {
            println!("  Disconnected: {}", self.disconnected.join(", "));
        }
        println!("-----------------------");
    }
}
