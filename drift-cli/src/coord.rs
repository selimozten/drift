use anyhow::{Context, Result};
use drift_proto::{
    read_message, write_message, DriftMessage, NodeInfo, TrainConfig, DRIFT_ALPN,
};
use iroh::{Endpoint, PublicKey};
use std::str::FromStr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use tracing::{error, info, warn};

pub async fn train(
    peer_ids: Vec<String>,
    _script: Option<String>,
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

    let endpoint = Endpoint::builder()
        .alpns(vec![DRIFT_ALPN.to_vec()])
        .bind()
        .await?;

    info!(coord_id = %endpoint.id(), "coordinator endpoint bound");

    let train_config = TrainConfig {
        model_path,
        dataset_path,
        batch_size,
        learning_rate,
        epochs,
    };

    // Connect to each peer and collect node info
    let mut node_infos: Vec<NodeInfo> = Vec::new();
    let mut connections = Vec::new();

    for peer_id_str in &peer_ids {
        let public_key = PublicKey::from_str(peer_id_str)
            .with_context(|| format!("invalid node ID: {}", peer_id_str))?;

        println!(
            "  Connecting to {}...",
            &peer_id_str[..12.min(peer_id_str.len())]
        );

        let conn = tokio::time::timeout(
            Duration::from_secs(30),
            endpoint.connect(public_key, DRIFT_ALPN),
        )
        .await
        .with_context(|| format!("connection to {} timed out after 30s", peer_id_str))?
        .with_context(|| format!("failed to connect to {}", peer_id_str))?;

        info!(peer = %peer_id_str, "connected");

        let (mut send, mut recv) = conn.open_bi().await?;

        // Initiate protocol
        write_message(&mut send, &DriftMessage::Ping).await?;

        let msg = read_message(&mut recv).await?;
        match msg {
            DriftMessage::NodeInfo(info) => {
                println!("  Connected: {}", info);
                node_infos.push(info);
            }
            other => {
                warn!(%other, "expected NodeInfo, skipping peer");
                continue;
            }
        }

        connections.push((send, recv));
    }

    if node_infos.is_empty() {
        anyhow::bail!("no peers responded with node info");
    }

    // Calculate shard assignments
    let assignments = assign_shards(&node_infos, dataset_size);
    let total_vram: u64 = node_infos.iter().map(|n| n.gpu_vram_mb).sum();

    println!();
    println!(
        "All peers connected in {:.1}s",
        started.elapsed().as_secs_f64()
    );
    println!();
    println!("Starting training:");
    println!("  Nodes:         {}", node_infos.len());
    println!("  Total VRAM:    {} MB", total_vram);
    println!("  Epochs:        {}", epochs);
    println!("  Batch size:    {} (per node)", batch_size);
    println!("  Learning rate: {}", learning_rate);
    println!(
        "  Dataset:       {} bytes ({} shards)",
        dataset_size,
        assignments.len()
    );
    println!();

    // Send config and shard assignments
    for (i, (send, _recv)) in connections.iter_mut().enumerate() {
        write_message(send, &DriftMessage::TrainConfig(train_config.clone())).await?;
        write_message(
            send,
            &DriftMessage::ShardAssignment(assignments[i].clone()),
        )
        .await?;
        info!(node = %node_infos[i].node_id, "sent config and shard");
    }

    // Create checkpoint dir
    tokio::fs::create_dir_all(&checkpoint_dir).await.ok();

    println!("Monitoring training progress (Ctrl+C to stop)...");
    println!();

    // Track active nodes and progress
    let active_nodes = Arc::new(Mutex::new(
        node_infos.iter().map(|n| n.node_id.clone()).collect::<Vec<_>>(),
    ));
    let train_start = Instant::now();

    // Spawn a listener per peer
    let mut handles = Vec::new();
    for (i, (_send, mut recv)) in connections.into_iter().enumerate() {
        let node_id = node_infos[i].node_id.clone();
        let active = active_nodes.clone();

        let handle = tokio::spawn(async move {
            let mut last_step = 0u64;
            let mut last_loss = 0.0f64;

            loop {
                match read_message(&mut recv).await {
                    Ok(DriftMessage::TrainProgress(p)) => {
                        println!(
                            "  [{}] epoch {} step {} | loss {:.4} | {:.1} samples/s",
                            &node_id[..12.min(node_id.len())],
                            p.epoch,
                            p.step,
                            p.loss,
                            p.throughput_samples_per_sec,
                        );
                        last_step = p.step;
                        last_loss = p.loss;
                    }
                    Ok(DriftMessage::Pong) => {}
                    Ok(DriftMessage::Heartbeat { uptime_secs }) => {
                        info!(node = %node_id, uptime_secs, "heartbeat");
                    }
                    Ok(other) => {
                        info!(%other, "message from node");
                    }
                    Err(e) => {
                        warn!(node = %node_id, "disconnected: {}", e);
                        break;
                    }
                }
            }

            // Mark node as disconnected
            active.lock().await.retain(|id| id != &node_id);
            (node_id, last_step, last_loss)
        });
        handles.push(handle);
    }

    // Wait for all peers to finish
    let mut results = Vec::new();
    for handle in handles {
        match handle.await {
            Ok(result) => results.push(result),
            Err(e) => error!("task error: {}", e),
        }
    }

    let elapsed = train_start.elapsed();
    println!();
    println!("--- Training Summary ---");
    println!("  Duration: {:.1}s", elapsed.as_secs_f64());
    for (node_id, last_step, last_loss) in &results {
        println!(
            "  [{}] final step {} | final loss {:.4}",
            &node_id[..12.min(node_id.len())],
            last_step,
            last_loss,
        );
    }
    println!("  Checkpoints: {}", checkpoint_dir);
    println!("------------------------");

    endpoint.close().await;
    Ok(())
}

/// Compute shard assignments weighted by VRAM.
fn assign_shards(
    nodes: &[NodeInfo],
    total_dataset_size: u64,
) -> Vec<drift_proto::ShardAssignment> {
    if nodes.is_empty() {
        return vec![];
    }

    let total_vram: u64 = nodes.iter().map(|n| n.gpu_vram_mb).sum();
    let use_equal = total_vram == 0;

    let mut assignments = Vec::with_capacity(nodes.len());
    let mut offset = 0u64;

    for (i, node) in nodes.iter().enumerate() {
        let shard_size = if use_equal {
            if i < nodes.len() - 1 {
                total_dataset_size / nodes.len() as u64
            } else {
                total_dataset_size - offset
            }
        } else if i < nodes.len() - 1 {
            (total_dataset_size as f64 * node.gpu_vram_mb as f64 / total_vram as f64) as u64
        } else {
            total_dataset_size - offset
        };

        let shard_end = offset + shard_size;
        assignments.push(drift_proto::ShardAssignment {
            node_id: node.node_id.clone(),
            shard_index: i as u32,
            shard_start: offset,
            shard_end,
        });
        offset = shard_end;
    }

    assignments
}
