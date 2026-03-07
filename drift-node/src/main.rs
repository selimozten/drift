mod gpu;
mod network;
mod training;

use anyhow::Result;
use clap::{Parser, Subcommand};
use drift_proto::{DriftMessage, NodeInfo};
use tracing::{error, info};

#[derive(Parser)]
#[command(name = "drift-node", version, about = "P2P distributed training node")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Join the drift swarm and wait for a coordinator
    Join {
        /// Optional human-readable name for this node
        #[arg(long)]
        name: Option<String>,
    },
    /// Show node status
    Status,
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
        Commands::Join { name } => join(name).await,
        Commands::Status => status().await,
    }
}

async fn join(name: Option<String>) -> Result<()> {
    // Detect GPUs
    let gpus = gpu::detect_gpus().await?;
    let gpu_info = gpus.first().cloned().unwrap_or_else(gpu::placeholder_gpu);

    // Create iroh endpoint
    let endpoint = network::create_endpoint().await?;
    let node_id = endpoint.id();

    let display_name = name.unwrap_or_else(|| node_id.to_string()[..12].to_string());
    println!("drift node started");
    println!("  Node ID:  {}", node_id);
    println!("  Name:     {}", display_name);
    println!("  GPU:      {} ({} MB VRAM)", gpu_info.name, gpu_info.vram_mb);
    println!();
    println!("Share your Node ID with the coordinator to join training.");
    println!("Waiting for connections...");

    let node_info_msg = DriftMessage::NodeInfo(NodeInfo {
        node_id: node_id.to_string(),
        gpu_name: gpu_info.name,
        gpu_vram_mb: gpu_info.vram_mb,
        gpu_compute_capability: gpu_info.compute_capability,
        available: true,
    });

    // Accept incoming connections until Ctrl+C
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
                        if let Err(e) = network::handle_connection(conn, node_info).await {
                            error!("connection handler error: {}", e);
                        }
                    }
                    Err(e) => {
                        error!("failed to accept connection: {}", e);
                    }
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

async fn status() -> Result<()> {
    let gpus = gpu::detect_gpus().await?;

    println!("drift node status");
    println!("---");
    if gpus.is_empty() {
        println!("  GPUs: none detected");
    } else {
        for (i, gpu) in gpus.iter().enumerate() {
            println!(
                "  GPU {}: {} ({} MB VRAM, compute {})",
                i, gpu.name, gpu.vram_mb, gpu.compute_capability
            );
        }
    }

    Ok(())
}
