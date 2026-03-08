use anyhow::Result;
use drift_proto::{
    read_message, write_message, DriftMessage, NodeInfo, DRIFT_ALPN, DRIFT_RING_ALPN,
};
use iroh::{Endpoint, PublicKey};
use std::str::FromStr;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::Mutex;
use tracing::{error, info, warn};

use crate::ipc::{self, PythonMessage};
use crate::shm::{DriftShm, DEFAULT_SHM_SIZE};

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
    let mut shard_assignment = None;
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
                    shard_assignment = Some(s);
                }
                DriftMessage::RingConfig(rc) => {
                    info!(rank = rc.rank, world_size = rc.world_size, "received ring config");
                    ring_config = Some(rc);
                }
                DriftMessage::StartRing => {
                    info!("StartRing received, establishing ring connections");
                    if let Some(ref rc) = ring_config {
                        if rc.world_size > 1 {
                            // Connect to right neighbor and accept from left
                            let streams = establish_ring(
                                &endpoint,
                                rc,
                            )
                            .await?;
                            *ring_streams.lock().await = Some(streams);
                            info!("ring connections established");
                        } else {
                            info!("single-node training, skipping ring establishment");
                        }
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
                            shard_assignment.as_ref(),
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
/// Dispatches to real Python training if model_path is a .py file, otherwise simulates.
async fn run_training(
    config: &drift_proto::TrainConfig,
    ring_config: &drift_proto::RingConfig,
    coord_send: &mut iroh::endpoint::SendStream,
    coord_recv: &mut iroh::endpoint::RecvStream,
    ring_streams: &Arc<Mutex<Option<RingStreams>>>,
    shard: Option<&drift_proto::ShardAssignment>,
) -> Result<()> {
    if config.model_path.ends_with(".py") {
        if !std::path::Path::new(&config.model_path).exists() {
            anyhow::bail!(
                "Python training script not found: {}. Check the --model-path argument.",
                config.model_path
            );
        }
        info!(script = %config.model_path, "launching real Python training");
        run_real_training(config, ring_config, coord_send, coord_recv, ring_streams, shard).await
    } else {
        info!("running simulated training with gradient sync");
        simulate_training(config, ring_config, coord_send, coord_recv, ring_streams).await
    }
}

/// Run real Python training via subprocess with shared memory IPC.
async fn run_real_training(
    config: &drift_proto::TrainConfig,
    ring_config: &drift_proto::RingConfig,
    coord_send: &mut iroh::endpoint::SendStream,
    coord_recv: &mut iroh::endpoint::RecvStream,
    ring_streams: &Arc<Mutex<Option<RingStreams>>>,
    shard: Option<&drift_proto::ShardAssignment>,
) -> Result<()> {
    use drift_proto::ring::{ring_allreduce, RingState};
    use std::process::Stdio;

    // 1. Create shared memory
    let shm = DriftShm::create(std::process::id(), DEFAULT_SHM_SIZE)?;
    let shm_name = shm.name().to_string();
    info!(shm = %shm_name, "created shared memory");

    // 2. Spawn Python subprocess with piped stdio and env vars
    // Python's SharedMemory expects name without leading "/"
    let python_shm_name = shm.python_name().to_string();
    let master_port = 29500 + (std::process::id() % 1000);
    let mut cmd = tokio::process::Command::new("python3");
    cmd.arg(&config.model_path)
        .env("DRIFT_SHM_NAME", &python_shm_name)
        .env("DRIFT_RANK", ring_config.rank.to_string())
        .env("DRIFT_WORLD_SIZE", ring_config.world_size.to_string())
        .env("DRIFT_BATCH_SIZE", config.batch_size.to_string())
        .env("DRIFT_LEARNING_RATE", config.learning_rate.to_string())
        .env("DRIFT_EPOCHS", config.epochs.to_string())
        .env("MASTER_ADDR", "127.0.0.1")
        .env("MASTER_PORT", master_port.to_string());

    if let Some(dataset_path) = std::env::var_os("DRIFT_DATASET_PATH") {
        cmd.env("DRIFT_DATASET_PATH", dataset_path);
    }

    if let Some(s) = shard {
        cmd.env("DRIFT_SHARD_INDEX", s.shard_index.to_string())
            .env("DRIFT_SHARD_START", s.shard_start.to_string())
            .env("DRIFT_SHARD_END", s.shard_end.to_string());
    }

    let mut child = cmd
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()?;

    let child_stdin = child.stdin.take().expect("piped stdin");
    let child_stdout = child.stdout.take().expect("piped stdout");
    let mut stdin_writer = child_stdin;
    let mut stdout_reader = BufReader::new(child_stdout).lines();

    // 3. Wait for DRIFT_READY with timeout
    let ready_deadline = tokio::time::timeout(
        std::time::Duration::from_secs(60),
        async {
            while let Some(line) = stdout_reader.next_line().await? {
                if matches!(ipc::parse_python_line(&line), PythonMessage::Ready) {
                    return Ok::<_, anyhow::Error>(());
                }
            }
            anyhow::bail!("Python subprocess exited before sending DRIFT_READY")
        },
    )
    .await;

    match ready_deadline {
        Ok(Ok(())) => info!("Python subprocess ready"),
        Ok(Err(e)) => {
            let _ = child.kill().await;
            return Err(e);
        }
        Err(_) => {
            let _ = child.kill().await;
            anyhow::bail!("Python subprocess did not send DRIFT_READY within 60s — check stderr for errors");
        }
    }

    // 4. IPC loop: read lines from child stdout, process commands
    // DDP sends multiple allreduce calls per step (one per gradient bucket).
    // We only barrier-sync with coordinator on the first bucket per step.
    let mut last_barrier_step: Option<u64> = None;

    while let Some(line) = stdout_reader.next_line().await? {
        let msg = ipc::parse_python_line(&line);
        match msg {
            PythonMessage::Ready => {
                // Already handled above, but harmless
            }
            PythonMessage::Allreduce { op_id, num_floats } => {
                info!(op_id, num_floats, "allreduce request from Python");

                // Read gradient from shm
                let gradient = shm.read_gradient(num_floats)?;

                // Barrier sync with coordinator (only once per step, not per bucket).
                // Use op_id as an approximation — first bucket of each step triggers barrier.
                let needs_barrier = last_barrier_step.map_or(true, |s| op_id > s);
                if needs_barrier {
                    write_message(
                        coord_send,
                        &DriftMessage::BarrierSync {
                            step: op_id,
                            node_id: format!("rank-{}", ring_config.rank),
                        },
                    )
                    .await?;

                    loop {
                        let coord_msg = read_message(coord_recv).await?;
                        match coord_msg {
                            DriftMessage::BarrierReady { step } if step == op_id => break,
                            DriftMessage::Ping => {
                                write_message(coord_send, &DriftMessage::Pong).await?;
                            }
                            _ => {}
                        }
                    }
                    last_barrier_step = Some(op_id);
                }

                // Run ring all-reduce
                let mut streams = ring_streams.lock().await;
                let averaged = if let Some(ref mut rs) = *streams {
                    let state = RingState::new(
                        ring_config.rank as usize,
                        ring_config.world_size as usize,
                        gradient,
                    );
                    ring_allreduce(state, op_id, &mut rs.send_right, &mut rs.recv_left).await?
                } else {
                    // Single node: just average with self (no-op)
                    gradient
                };
                drop(streams);

                // Write result back to shm
                shm.write_gradient(&averaged)?;

                // Signal Python that allreduce is done
                let response = format!("{}\n", ipc::format_allreduce_done(op_id));
                stdin_writer.write_all(response.as_bytes()).await?;
                stdin_writer.flush().await?;

                info!(op_id, "allreduce complete");
            }
            PythonMessage::Progress { epoch, step, loss, throughput } => {
                let progress = drift_proto::TrainProgress {
                    node_id: format!("rank-{}", ring_config.rank),
                    epoch,
                    step,
                    loss,
                    throughput_samples_per_sec: throughput,
                };
                write_message(coord_send, &DriftMessage::TrainProgress(progress)).await?;
            }
            PythonMessage::Done => {
                info!("Python training complete");
                break;
            }
            PythonMessage::Unknown(line) => {
                // Log non-protocol lines from Python
                if !line.is_empty() {
                    info!(line, "python output");
                }
            }
        }
    }

    // 5. Wait for child exit with timeout
    match tokio::time::timeout(std::time::Duration::from_secs(30), child.wait()).await {
        Ok(Ok(status)) => {
            if !status.success() {
                warn!(code = ?status.code(), "Python subprocess exited with error");
            }
        }
        Ok(Err(e)) => {
            warn!("error waiting for Python subprocess: {}", e);
        }
        Err(_) => {
            warn!("Python subprocess did not exit within 30s, killing");
            let _ = child.kill().await;
        }
    }

    Ok(())
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
