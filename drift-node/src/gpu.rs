use anyhow::Result;
use tracing::{info, warn};

#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub name: String,
    pub vram_mb: u64,
    pub compute_capability: String,
}

/// Detect GPUs by running nvidia-smi.
pub async fn detect_gpus() -> Result<Vec<GpuInfo>> {
    let output = tokio::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .await;

    match output {
        Ok(output) if output.status.success() => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let gpus: Vec<GpuInfo> = stdout
                .lines()
                .filter(|line| !line.trim().is_empty())
                .filter_map(|line| {
                    let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                    if parts.len() >= 3 {
                        let vram = parts[1].parse::<u64>().unwrap_or(0);
                        Some(GpuInfo {
                            name: parts[0].to_string(),
                            vram_mb: vram,
                            compute_capability: parts[2].to_string(),
                        })
                    } else {
                        warn!("unexpected nvidia-smi output line: {}", line);
                        None
                    }
                })
                .collect();

            if gpus.is_empty() {
                warn!("nvidia-smi returned no GPUs");
            } else {
                for gpu in &gpus {
                    info!(
                        name = %gpu.name,
                        vram_mb = gpu.vram_mb,
                        compute = %gpu.compute_capability,
                        "detected GPU"
                    );
                }
            }

            Ok(gpus)
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            warn!("nvidia-smi failed: {}", stderr);
            Ok(vec![])
        }
        Err(e) => {
            warn!("nvidia-smi not found: {} (no NVIDIA GPU detected)", e);
            Ok(vec![])
        }
    }
}

/// Return a placeholder GPU for systems without NVIDIA GPUs (e.g. development).
pub fn placeholder_gpu() -> GpuInfo {
    GpuInfo {
        name: "CPU-only (no GPU detected)".to_string(),
        vram_mb: 0,
        compute_capability: "0.0".to_string(),
    }
}
