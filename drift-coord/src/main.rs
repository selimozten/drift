use drift_coord::{checkpoint, monitor, scheduler};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use drift_proto::{
    read_message, write_message, DriftMessage, NodeInfo, TrainConfig, DRIFT_ALPN,
};
use iroh::{Endpoint, PublicKey};
use std::str::FromStr;
use std::time::{Duration, Instant};
use tracing::{error, info, warn};

#[derive(Parser)]
#[command(name = "drift-coord", version, about = "P2P distributed training coordinator")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start a training run across peer nodes
    Train {
        /// Comma-separated list of peer node IDs
        #[arg(long, value_delimiter = ',')]
        peers: Vec<String>,

        /// Path to training config YAML
        #[arg(long, default_value = "train.yaml")]
        config: String,

        /// Path to model
        #[arg(long, default_value = "model.pt")]
        model_path: String,

        /// Path to dataset
        #[arg(long, default_value = "data/")]
        dataset_path: String,

        /// Batch size per node
        #[arg(long, default_value = "32")]
        batch_size: u32,

        /// Learning rate
        #[arg(long, default_value = "0.001")]
        learning_rate: f64,

        /// Number of epochs
        #[arg(long, default_value = "10")]
        epochs: u32,

        /// Total dataset size in bytes (for shard calculation)
        #[arg(long, default_value = "1000000")]
        dataset_size: u64,

        /// Checkpoint output directory
        #[arg(long, default_value = "checkpoints/")]
        checkpoint_dir: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Train {
            peers,
            config: _,
            model_path,
            dataset_path,
            batch_size,
            learning_rate,
            epochs,
            dataset_size,
            checkpoint_dir,
        } => {
            train(
                peers,
                model_path,
                dataset_path,
                batch_size,
                learning_rate,
                epochs,
                dataset_size,
                checkpoint_dir,
            )
            .await
        }
    }
}

async fn train(
    peer_ids: Vec<String>,
    model_path: String,
    dataset_path: String,
    batch_size: u32,
    learning_rate: f64,
    epochs: u32,
    dataset_size: u64,
    checkpoint_dir: String,
) -> Result<()> {
    if peer_ids.is_empty() {
        anyhow::bail!("no peers specified. Use --peers <node_id1>,<node_id2>");
    }

    let started = Instant::now();

    println!("drift coordinator starting");
    println!("  Peers: {}", peer_ids.len());

    // Create iroh endpoint
    let endpoint = Endpoint::builder()
        .alpns(vec![DRIFT_ALPN.to_vec()])
        .bind()
        .await?;

    let coord_id = endpoint.id();
    info!(%coord_id, "coordinator endpoint bound");

    let train_config = TrainConfig {
        model_path,
        dataset_path,
        batch_size,
        learning_rate,
        epochs,
    };

    let mut _checkpoint_mgr = checkpoint::CheckpointManager::new(&checkpoint_dir);
    let mut monitor = monitor::Monitor::new();

    // Connect to each peer and collect node info
    let mut node_infos: Vec<NodeInfo> = Vec::new();
    let mut connections = Vec::new();

    for peer_id_str in &peer_ids {
        let public_key = PublicKey::from_str(peer_id_str)
            .with_context(|| format!("invalid node ID: {}", peer_id_str))?;

        println!("  Connecting to {}...", &peer_id_str[..12.min(peer_id_str.len())]);

        let conn = tokio::time::timeout(
            Duration::from_secs(30),
            endpoint.connect(public_key, DRIFT_ALPN),
        )
        .await
        .with_context(|| format!("connection to {} timed out after 30s", peer_id_str))?
        .with_context(|| format!("failed to connect to {}", peer_id_str))?;

        info!(peer = %peer_id_str, "connected to peer");

        let (mut send, mut recv) = conn.open_bi().await?;

        // Send initial Ping to trigger the bi stream on the node side
        write_message(&mut send, &DriftMessage::Ping).await?;

        // Read node info from peer
        let msg = read_message(&mut recv).await?;
        match msg {
            DriftMessage::NodeInfo(info) => {
                println!(
                    "  Connected: {} | {} ({} MB VRAM)",
                    &info.node_id[..12.min(info.node_id.len())],
                    info.gpu_name,
                    info.gpu_vram_mb
                );
                node_infos.push(info);
            }
            other => {
                warn!(?other, "expected NodeInfo, got something else");
                continue;
            }
        }

        connections.push((send, recv));
    }

    if node_infos.is_empty() {
        anyhow::bail!("no peers responded with node info");
    }

    // Calculate shard assignments
    let assignments = scheduler::assign_shards(&node_infos, dataset_size);

    let total_vram: u64 = node_infos.iter().map(|n| n.gpu_vram_mb).sum();
    let connect_elapsed = started.elapsed();

    println!();
    println!("All peers connected in {:.1}s", connect_elapsed.as_secs_f64());
    println!();
    println!("Starting training:");
    println!("  Nodes:         {}", node_infos.len());
    println!("  Total VRAM:    {} MB", total_vram);
    println!("  Epochs:        {}", epochs);
    println!("  Batch size:    {} (per node)", batch_size);
    println!("  Learning rate: {}", learning_rate);
    println!("  Dataset:       {} bytes ({} shards)", dataset_size, assignments.len());
    println!();

    // Send training config and shard assignments to each peer
    for (i, (send, _recv)) in connections.iter_mut().enumerate() {
        // Send training config
        write_message(send, &DriftMessage::TrainConfig(train_config.clone())).await?;

        // Send shard assignment
        write_message(
            send,
            &DriftMessage::ShardAssignment(assignments[i].clone()),
        )
        .await?;

        info!(node = %node_infos[i].node_id, "sent config and shard assignment");
    }

    // Monitor training progress
    println!("Monitoring training progress (Ctrl+C to stop)...");
    println!();

    // Listen for progress updates from all peers
    let mut handles = Vec::new();
    for (i, (_send, mut recv)) in connections.into_iter().enumerate() {
        let node_id = node_infos[i].node_id.clone();
        let handle = tokio::spawn(async move {
            loop {
                match read_message(&mut recv).await {
                    Ok(DriftMessage::TrainProgress(progress)) => {
                        println!(
                            "  [{}] epoch {} step {} | loss {:.4} | {:.1} samples/s",
                            &node_id[..12.min(node_id.len())],
                            progress.epoch,
                            progress.step,
                            progress.loss,
                            progress.throughput_samples_per_sec,
                        );
                    }
                    Ok(DriftMessage::Pong) => {
                        info!(node = %node_id, "received pong");
                    }
                    Ok(other) => {
                        info!(?other, "received message from node");
                    }
                    Err(e) => {
                        warn!(node = %node_id, "peer disconnected: {}", e);
                        break;
                    }
                }
            }
            node_id
        });
        handles.push(handle);
    }

    // Wait for all peers to finish or disconnect
    for handle in handles {
        match handle.await {
            Ok(node_id) => {
                monitor.mark_disconnected(&node_id);
            }
            Err(e) => {
                error!("task join error: {}", e);
            }
        }
    }

    monitor.print_status();
    println!("Training complete.");
    endpoint.close().await;

    Ok(())
}
