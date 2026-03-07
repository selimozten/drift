use anyhow::Result;
use drift_proto::{
    read_message, write_message, DriftMessage, NodeInfo, DRIFT_ALPN, DRIFT_RING_ALPN,
};
use iroh::{Endpoint, PublicKey};
use std::str::FromStr;
use std::sync::Arc;
use tokio::sync::Mutex;
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
        .alpns(vec![DRIFT_ALPN.to_vec(), DRIFT_RING_ALPN.to_vec()])
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
            let ep = endpoint.clone();
            tokio::spawn(async move {
                match incoming.await {
                    Ok(conn) => {
                        if let Err(e) = handle_connection(conn, node_info, ep).await {
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

/// Ring connection streams for gradient synchronization.
struct RingStreams {
    send_right: iroh::endpoint::SendStream,
    recv_left: iroh::endpoint::RecvStream,
}

async fn handle_connection(
    conn: iroh::endpoint::Connection,
    node_info_msg: DriftMessage,
    endpoint: Endpoint,
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
    let mut ring_config = None;
    let ring_streams: Arc<Mutex<Option<RingStreams>>> = Arc::new(Mutex::new(None));

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
                }
                DriftMessage::RingConfig(rc) => {
                    info!(rank = rc.rank, world_size = rc.world_size, "received ring config");
                    ring_config = Some(rc);
                }
                DriftMessage::StartRing => {
                    info!("StartRing received, establishing ring connections");
                    if let Some(ref rc) = ring_config {
                        // Connect to right neighbor and accept from left
                        let streams = establish_ring(
                            &endpoint,
                            rc,
                        )
                        .await?;
                        *ring_streams.lock().await = Some(streams);
                        info!("ring connections established");

                        // Now start training if config and shard are ready
                        // (shard was already received before StartRing)
                    }

                    // Start training with ring streams
                    if let Some(ref config) = train_config {
                        info!("starting training with gradient sync");
                        run_training(
                            config,
                            &ring_config.as_ref().expect("ring config"),
                            &mut send,
                            &mut recv,
                            &ring_streams,
                        )
                        .await?;
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

/// Establish ring connections: connect to right neighbor and accept from left.
async fn establish_ring(
    endpoint: &Endpoint,
    ring_config: &drift_proto::RingConfig,
) -> Result<RingStreams> {
    let right_key = PublicKey::from_str(&ring_config.right_peer_id)?;

    // Connect to right neighbor (we send to right)
    // Accept from left neighbor (we receive from left)
    // Use try_join to do both concurrently — left neighbor is connecting to us simultaneously

    let connect_right = async {
        let conn = endpoint.connect(right_key, DRIFT_RING_ALPN).await?;
        let (send, _recv) = conn.open_bi().await?;
        // Write a small handshake so the bi-stream is visible on the other side
        let mut s = send;
        write_message(&mut s, &DriftMessage::Ping).await?;
        info!("connected to right neighbor");
        Ok::<_, anyhow::Error>(s)
    };

    let accept_left = async {
        // Accept incoming ring connection from left neighbor
        loop {
            let incoming = endpoint
                .accept()
                .await
                .ok_or_else(|| anyhow::anyhow!("endpoint closed while waiting for left neighbor"))?;
            let conn = incoming.await?;
            let alpn = conn.alpn();
            if alpn.as_ref() == DRIFT_RING_ALPN {
                let (_send, mut recv) = conn.accept_bi().await?;
                // Read the handshake ping
                let msg = read_message(&mut recv).await?;
                if !matches!(msg, DriftMessage::Ping) {
                    anyhow::bail!("expected Ping from left neighbor, got {}", msg);
                }
                info!("accepted ring connection from left neighbor");
                return Ok::<_, anyhow::Error>(recv);
            }
            // Not a ring connection, ignore
        }
    };

    let (send_right, recv_left) = tokio::try_join!(connect_right, accept_left)?;

    Ok(RingStreams {
        send_right,
        recv_left,
    })
}

/// Execute training and stream progress back to coordinator.
async fn run_training(
    config: &drift_proto::TrainConfig,
    ring_config: &drift_proto::RingConfig,
    coord_send: &mut iroh::endpoint::SendStream,
    coord_recv: &mut iroh::endpoint::RecvStream,
    ring_streams: &Arc<Mutex<Option<RingStreams>>>,
) -> Result<()> {
    // Simulate training with gradient sync
    info!("running simulated training with gradient sync");
    simulate_training(config, ring_config, coord_send, coord_recv, ring_streams).await?;
    Ok(())
}

/// Parse a progress line from the training script.
/// Expected format: "DRIFT_PROGRESS <epoch> <step> <loss> <throughput>"
#[allow(dead_code)]
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

/// Simulate training progress with gradient synchronization.
async fn simulate_training(
    config: &drift_proto::TrainConfig,
    ring_config: &drift_proto::RingConfig,
    coord_send: &mut iroh::endpoint::SendStream,
    coord_recv: &mut iroh::endpoint::RecvStream,
    ring_streams: &Arc<Mutex<Option<RingStreams>>>,
) -> Result<()> {
    use drift_proto::ring::{ring_allreduce, RingState};

    let steps_per_epoch = 5u64;
    let gradient_size = 100usize; // mock gradient with 100 floats
    let mut loss = 2.5_f64;

    for epoch in 0..config.epochs {
        for step_in_epoch in 0..steps_per_epoch {
            let global_step = epoch as u64 * steps_per_epoch + step_in_epoch;

            tokio::time::sleep(std::time::Duration::from_millis(50)).await;

            // Generate mock gradient data
            let gradient: Vec<f32> = (0..gradient_size)
                .map(|i| (ring_config.rank as f32 + 1.0) * (i as f32 + 1.0))
                .collect();

            // Barrier sync with coordinator
            write_message(
                coord_send,
                &DriftMessage::BarrierSync {
                    step: global_step,
                    node_id: format!("rank-{}", ring_config.rank),
                },
            )
            .await?;

            // Wait for BarrierReady
            loop {
                let msg = read_message(coord_recv).await?;
                match msg {
                    DriftMessage::BarrierReady { step } if step == global_step => break,
                    DriftMessage::Ping => {
                        write_message(coord_send, &DriftMessage::Pong).await?;
                    }
                    _ => {}
                }
            }

            // Run ring all-reduce if we have ring streams
            let mut streams = ring_streams.lock().await;
            if let Some(ref mut rs) = *streams {
                let state = RingState::new(
                    ring_config.rank as usize,
                    ring_config.world_size as usize,
                    gradient,
                );
                let _averaged = ring_allreduce(
                    state,
                    global_step,
                    &mut rs.send_right,
                    &mut rs.recv_left,
                )
                .await?;
                info!(step = global_step, "gradient sync complete");
                println!(
                    "DRIFT_GRADIENT {} {}",
                    global_step, gradient_size
                );
            }
            drop(streams);

            loss *= 0.98;
            let progress = drift_proto::TrainProgress {
                node_id: format!("rank-{}", ring_config.rank),
                epoch,
                step: global_step,
                loss,
                throughput_samples_per_sec: config.batch_size as f64 * 10.0,
            };
            write_message(coord_send, &DriftMessage::TrainProgress(progress)).await?;
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
