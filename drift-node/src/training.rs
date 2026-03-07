use anyhow::Result;
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, BufReader};
use tracing::{error, info};

/// Spawn a Python training subprocess with the given config.
pub async fn spawn_training(
    script: &str,
    model_path: &str,
    dataset_path: &str,
    batch_size: u32,
    learning_rate: f64,
    shard_index: u32,
    shard_start: u64,
    shard_end: u64,
) -> Result<tokio::process::Child> {
    info!(
        script,
        shard_index, shard_start, shard_end, "spawning training subprocess"
    );

    let mut child = tokio::process::Command::new("python")
        .arg(script)
        .arg("--model-path")
        .arg(model_path)
        .arg("--dataset-path")
        .arg(dataset_path)
        .arg("--batch-size")
        .arg(batch_size.to_string())
        .arg("--learning-rate")
        .arg(learning_rate.to_string())
        .arg("--shard-index")
        .arg(shard_index.to_string())
        .arg("--shard-start")
        .arg(shard_start.to_string())
        .arg("--shard-end")
        .arg(shard_end.to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    // Stream stdout in background
    if let Some(stdout) = child.stdout.take() {
        tokio::spawn(async move {
            let reader = BufReader::new(stdout);
            let mut lines = reader.lines();
            while let Ok(Some(line)) = lines.next_line().await {
                info!(target: "training", "{}", line);
            }
        });
    }

    // Stream stderr in background
    if let Some(stderr) = child.stderr.take() {
        tokio::spawn(async move {
            let reader = BufReader::new(stderr);
            let mut lines = reader.lines();
            while let Ok(Some(line)) = lines.next_line().await {
                error!(target: "training", "{}", line);
            }
        });
    }

    Ok(child)
}
