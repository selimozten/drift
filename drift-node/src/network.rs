use anyhow::Result;
use drift_proto::{read_message, write_message, DriftMessage, DRIFT_ALPN};
use iroh::Endpoint;
use tracing::{info, warn};

/// Create and bind an iroh endpoint configured for drift.
pub async fn create_endpoint() -> Result<Endpoint> {
    let endpoint = Endpoint::builder()
        .alpns(vec![DRIFT_ALPN.to_vec()])
        .bind()
        .await?;

    let node_id = endpoint.id();
    info!(%node_id, "endpoint bound");

    Ok(endpoint)
}

/// Accept a connection and handle the message exchange.
/// Protocol: coordinator sends Ping first, node responds with NodeInfo,
/// then processes further messages from coordinator.
pub async fn handle_connection(
    conn: iroh::endpoint::Connection,
    node_info_msg: DriftMessage,
) -> Result<()> {
    let remote = conn.remote_id();
    info!(%remote, "accepted connection from coordinator");

    let (mut send, mut recv) = conn.accept_bi().await?;

    // Wait for initial Ping from coordinator (triggers the bi stream)
    let msg = read_message(&mut recv).await?;
    match msg {
        DriftMessage::Ping => {
            info!("received initial ping from coordinator");
        }
        other => {
            anyhow::bail!("expected initial Ping, got {:?}", other);
        }
    }

    // Send our node info
    write_message(&mut send, &node_info_msg).await?;
    info!("sent node info to coordinator");

    // Process messages from coordinator
    loop {
        match read_message(&mut recv).await {
            Ok(msg) => match msg {
                DriftMessage::Ping => {
                    info!("received ping, sending pong");
                    write_message(&mut send, &DriftMessage::Pong).await?;
                }
                DriftMessage::TrainConfig(config) => {
                    info!(
                        model = %config.model_path,
                        dataset = %config.dataset_path,
                        epochs = config.epochs,
                        "received training config"
                    );
                }
                DriftMessage::ShardAssignment(shard) => {
                    info!(
                        shard_index = shard.shard_index,
                        start = shard.shard_start,
                        end = shard.shard_end,
                        size = shard.size(),
                        "received shard assignment"
                    );
                }
                DriftMessage::Heartbeat { .. } => {
                    write_message(
                        &mut send,
                        &DriftMessage::Heartbeat { uptime_secs: 0 },
                    )
                    .await?;
                }
                DriftMessage::TrainComplete => {
                    info!("training complete signal received");
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
