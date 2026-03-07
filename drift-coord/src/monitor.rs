use drift_proto::TrainProgress;
use std::collections::HashMap;
use std::time::Instant;
use tracing::{info, warn};

/// Tracks training progress and node health.
pub struct Monitor {
    progress: HashMap<String, TrainProgress>,
    last_seen: HashMap<String, Instant>,
    disconnected: Vec<String>,
}

impl Monitor {
    pub fn new() -> Self {
        Self {
            progress: HashMap::new(),
            last_seen: HashMap::new(),
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
        self.last_seen
            .insert(progress.node_id.clone(), Instant::now());
        self.progress.insert(progress.node_id.clone(), progress);
    }

    /// Record that we heard from a node (heartbeat or any message).
    pub fn touch(&mut self, node_id: &str) {
        self.last_seen
            .insert(node_id.to_string(), Instant::now());
    }

    /// Mark a node as disconnected.
    pub fn mark_disconnected(&mut self, node_id: &str) {
        warn!(node_id, "node disconnected");
        self.disconnected.push(node_id.to_string());
        self.progress.remove(node_id);
        self.last_seen.remove(node_id);
    }

    /// Return node IDs that haven't been heard from in `timeout` seconds.
    pub fn stale_nodes(&self, timeout_secs: u64) -> Vec<String> {
        let now = Instant::now();
        self.last_seen
            .iter()
            .filter(|(id, last)| {
                now.duration_since(**last).as_secs() > timeout_secs
                    && !self.disconnected.contains(id)
            })
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Number of active (non-disconnected) nodes.
    pub fn active_count(&self) -> usize {
        self.last_seen.len()
    }

    /// Print a summary of current training status.
    pub fn print_status(&self) {
        println!("--- Training Status ---");
        if self.progress.is_empty() {
            println!("  No active nodes.");
        }
        for (node_id, p) in &self.progress {
            let stale = self
                .last_seen
                .get(node_id)
                .map(|t| t.elapsed().as_secs())
                .unwrap_or(0);
            let stale_marker = if stale > 30 { " [stale]" } else { "" };
            println!(
                "  {} | epoch {} step {} | loss {:.4} | {:.1} samples/s{}",
                &node_id[..12.min(node_id.len())],
                p.epoch,
                p.step,
                p.loss,
                p.throughput_samples_per_sec,
                stale_marker,
            );
        }
        if !self.disconnected.is_empty() {
            let short: Vec<&str> = self
                .disconnected
                .iter()
                .map(|id| &id[..12.min(id.len())])
                .collect();
            println!("  Disconnected: {}", short.join(", "));
        }
        println!("-----------------------");
    }
}
