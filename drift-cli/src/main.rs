mod node;
mod coord;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "drift",
    version,
    about = "P2P distributed training. Plug your GPU into the mesh."
)]
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

    /// Start a training run across peer nodes
    Train {
        /// Comma-separated list of peer node IDs
        #[arg(long, value_delimiter = ',')]
        peers: Vec<String>,

        /// Path to the training script
        #[arg(long)]
        script: Option<String>,

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

        /// Resume from the latest checkpoint in checkpoint_dir
        #[arg(long, default_value = "false")]
        resume: bool,
    },

    /// Show local GPU status
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
        Commands::Join { name } => node::join(name).await,
        Commands::Train {
            peers,
            script,
            model_path,
            dataset_path,
            batch_size,
            learning_rate,
            epochs,
            dataset_size,
            checkpoint_dir,
            resume,
        } => {
            coord::train(
                peers,
                script,
                model_path,
                dataset_path,
                batch_size,
                learning_rate,
                epochs,
                dataset_size,
                checkpoint_dir,
                resume,
            )
            .await
        }
        Commands::Status => node::status().await,
    }
}
