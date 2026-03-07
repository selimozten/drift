use drift_proto::{NodeInfo, ShardAssignment};
use tracing::info;

/// Compute shard assignments based on GPU VRAM.
/// Nodes with more VRAM get proportionally larger shards.
pub fn assign_shards(nodes: &[NodeInfo], total_dataset_size: u64) -> Vec<ShardAssignment> {
    if nodes.is_empty() {
        return vec![];
    }

    // Weight by VRAM. If all VRAM is 0, distribute equally.
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

        info!(
            node_id = %node.node_id,
            shard_index = i,
            start = offset,
            end = shard_end,
            "assigned shard"
        );

        assignments.push(ShardAssignment {
            node_id: node.node_id.clone(),
            shard_index: i as u32,
            shard_start: offset,
            shard_end,
        });

        offset = shard_end;
    }

    assignments
}
