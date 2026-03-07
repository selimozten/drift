use anyhow::Result;
use drift_proto::{read_message, write_message, DriftMessage, NodeInfo, DRIFT_ALPN};
use iroh::Endpoint;
use tracing::{error, info, warn};

/// Detect GPUs via nvidia-smi. Returns empty vec if unavailable.
async fn detect_gpus() -> Vec<(String, u64, String)> {
    let output = tokio::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .await;

    match output {
        Ok(o) if o.status.success() => {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .filter(|l| !l.trim().is_empty())
                .filter_map(|line| {
                    let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                    if parts.len() >= 3 {
                        Some((
                            parts[0].to_string(),
                            parts[1].parse::<u64>().unwrap_or(0),
                            parts[2].to_string(),
                        ))
                    } else {
                        None
                    }
                })
                .collect()
        }
        _ => vec![],
    }
}

pub async fn join(name: Option<String>) -> Result<()> {
    let gpus = detect_gpus().await;
    let (gpu_name, gpu_vram, gpu_cc) = gpus.first().cloned().unwrap_or((
        "CPU-only (no GPU detected)".to_string(),
        0,
        "0.0".to_string(),
    ));
    let total_vram: u64 = if gpus.is_empty() {
        0
    } else {
        gpus.iter().map(|g| g.1).sum()
    };

    let endpoint = Endpoint::builder()
        .alpns(vec![DRIFT_ALPN.to_vec()])
        .bind()
        .await?;

    let node_id = endpoint.id();
    let display_name = name.unwrap_or_else(|| node_id.to_string()[..12].to_string());

    println!("drift node started");
    println!("  Node ID:  {}", node_id);
    println!("  Name:     {}", display_name);
    if gpus.len() <= 1 {
        println!("  GPU:      {} ({} MB VRAM)", gpu_name, gpu_vram);
    } else {
        println!("  GPUs:     {} devices ({} MB total VRAM)", gpus.len(), total_vram);
        for (i, (name, vram, _)) in gpus.iter().enumerate() {
            println!("    [{}] {} ({} MB)", i, name, vram);
        }
    }
    println!();
    println!("Share your Node ID with the coordinator to join training.");
    println!("Waiting for connections...");

    let node_info_msg = DriftMessage::NodeInfo(NodeInfo {
        node_id: node_id.to_string(),
        gpu_name,
        gpu_vram_mb: total_vram.max(gpu_vram),
        gpu_compute_capability: gpu_cc,
        available: true,
    });

    let accept_loop = async {
        loop {
            let incoming = match endpoint.accept().await {
                Some(incoming) => incoming,
                None => {
                    info!("endpoint closed");
                    break;
                }
            };

            let node_info = node_info_msg.clone();
            tokio::spawn(async move {
                match incoming.await {
                    Ok(conn) => {
                        if let Err(e) = handle_connection(conn, node_info).await {
                            error!("connection error: {}", e);
                        }
                    }
                    Err(e) => error!("accept error: {}", e),
                }
            });
        }
    };

    tokio::select! {
        _ = accept_loop => {}
        _ = tokio::signal::ctrl_c() => {
            println!();
            println!("shutting down...");
        }
    }

    endpoint.close().await;
    Ok(())
}

async fn handle_connection(
    conn: iroh::endpoint::Connection,
    node_info_msg: DriftMessage,
) -> Result<()> {
    let remote = conn.remote_id();
    info!(%remote, "coordinator connected");

    let (mut send, mut recv) = conn.accept_bi().await?;

    // Wait for initial Ping
    let msg = read_message(&mut recv).await?;
    if !matches!(msg, DriftMessage::Ping) {
        anyhow::bail!("expected Ping, got {}", msg);
    }

    // Send node info
    write_message(&mut send, &node_info_msg).await?;
    info!("sent node info");

    // State for training
    let mut train_config = None;
    let mut shard = None;

    loop {
        match read_message(&mut recv).await {
            Ok(msg) => match msg {
                DriftMessage::Ping => {
                    write_message(&mut send, &DriftMessage::Pong).await?;
                }
                DriftMessage::TrainConfig(config) => {
                    info!(
                        model = %config.model_path,
                        epochs = config.epochs,
                        "received training config"
                    );
                    train_config = Some(config);
                }
                DriftMessage::ShardAssignment(s) => {
                    info!(shard_index = s.shard_index, size = s.size(), "received shard");
                    shard = Some(s);

                    // Both config and shard received — start training
                    if let Some(ref config) = train_config {
                        let shard_ref = shard.as_ref().unwrap();
                        info!("starting training");
                        run_training(config, shard_ref, &mut send).await?;
                    }
                }
                DriftMessage::Heartbeat { .. } => {
                    write_message(
                        &mut send,
                        &DriftMessage::Heartbeat { uptime_secs: 0 },
                    )
                    .await?;
                }
                DriftMessage::TrainComplete => {
                    info!("training complete");
                    break;
                }
                other => {
                    info!(%other, "received message");
                }
            },
            Err(e) => {
                warn!("connection closed: {}", e);
                break;
            }
        }
    }

    Ok(())
}

/// Execute training and stream progress back to coordinator.
async fn run_training(
    config: &drift_proto::TrainConfig,
    shard: &drift_proto::ShardAssignment,
    send: &mut iroh::endpoint::SendStream,
) -> Result<()> {
    use std::process::Stdio;
    use tokio::io::{AsyncBufReadExt, BufReader};

    // Build env vars for the training subprocess
    let env_vars = [
        ("DRIFT_MODEL_PATH", config.model_path.clone()),
        ("DRIFT_DATASET_PATH", config.dataset_path.clone()),
        ("DRIFT_BATCH_SIZE", config.batch_size.to_string()),
        ("DRIFT_LEARNING_RATE", config.learning_rate.to_string()),
        ("DRIFT_EPOCHS", config.epochs.to_string()),
        ("DRIFT_SHARD_INDEX", shard.shard_index.to_string()),
        ("DRIFT_SHARD_START", shard.shard_start.to_string()),
        ("DRIFT_SHARD_END", shard.shard_end.to_string()),
        ("DRIFT_SHARD_SIZE", shard.size().to_string()),
        ("DRIFT_NODE_ID", shard.node_id.clone()),
    ];

    // Look for a training script. If none provided, just simulate progress.
    // In real usage, the coordinator would send the script path in TrainConfig.
    let script_path = std::path::Path::new(&config.model_path);

    if script_path.extension().is_some_and(|ext| ext == "py") && script_path.exists() {
        // Run actual Python training script
        let mut child = tokio::process::Command::new("python")
            .arg(&config.model_path)
            .envs(env_vars.iter().map(|(k, v)| (*k, v.as_str())))
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;

        // Parse stdout for progress lines: "DRIFT_PROGRESS epoch step loss throughput"
        if let Some(stdout) = child.stdout.take() {
            let reader = BufReader::new(stdout);
            let mut lines = reader.lines();
            while let Ok(Some(line)) = lines.next_line().await {
                if let Some(progress) = parse_progress_line(&line, &shard.node_id) {
                    write_message(send, &DriftMessage::TrainProgress(progress)).await?;
                } else {
                    info!(target: "training", "{}", line);
                }
            }
        }

        let status = child.wait().await?;
        info!(exit_code = status.code(), "training subprocess finished");
    } else {
        // Simulate training progress for testing
        info!("no training script found, simulating progress");
        simulate_training(config, shard, send).await?;
    }

    Ok(())
}

/// Parse a progress line from the training script.
/// Expected format: "DRIFT_PROGRESS <epoch> <step> <loss> <throughput>"
fn parse_progress_line(line: &str, node_id: &str) -> Option<drift_proto::TrainProgress> {
    let line = line.trim();
    if !line.starts_with("DRIFT_PROGRESS") {
        return None;
    }
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 5 {
        return None;
    }
    Some(drift_proto::TrainProgress {
        node_id: node_id.to_string(),
        epoch: parts[1].parse().ok()?,
        step: parts[2].parse().ok()?,
        loss: parts[3].parse().ok()?,
        throughput_samples_per_sec: parts[4].parse().ok()?,
    })
}

/// Simulate training progress for testing without a real training script.
async fn simulate_training(
    config: &drift_proto::TrainConfig,
    shard: &drift_proto::ShardAssignment,
    send: &mut iroh::endpoint::SendStream,
) -> Result<()> {
    let steps_per_epoch = (shard.size() / config.batch_size as u64).max(1);
    let mut loss = 2.5_f64;

    for epoch in 0..config.epochs {
        for step in 0..steps_per_epoch {
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            loss *= 0.98; // simulate decreasing loss

            let progress = drift_proto::TrainProgress {
                node_id: shard.node_id.clone(),
                epoch,
                step: epoch as u64 * steps_per_epoch + step,
                loss,
                throughput_samples_per_sec: config.batch_size as f64 * 10.0,
            };
            write_message(send, &DriftMessage::TrainProgress(progress)).await?;
        }
    }

    Ok(())
}

pub async fn status() -> Result<()> {
    let gpus = detect_gpus().await;

    // Driver version
    if let Ok(output) = tokio::process::Command::new("nvidia-smi")
        .args(["--query-gpu=driver_version", "--format=csv,noheader"])
        .output()
        .await
    {
        if output.status.success() {
            let v = String::from_utf8_lossy(&output.stdout);
            let v = v.trim();
            if !v.is_empty() {
                println!("  Driver: {}", v);
            }
        }
    }

    println!("drift node status");
    println!("---");
    if gpus.is_empty() {
        println!("  GPUs: none detected");
    } else {
        let total_vram: u64 = gpus.iter().map(|g| g.1).sum();
        for (i, (name, vram, cc)) in gpus.iter().enumerate() {
            println!("  GPU {}: {} ({} MB VRAM, compute {})", i, name, vram, cc);
        }
        if gpus.len() > 1 {
            println!("  Total: {} MB VRAM across {} devices", total_vram, gpus.len());
        }
    }

    Ok(())
}
