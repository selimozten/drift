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

/// Redistribute shards when a node drops out. Remaining nodes absorb
/// the dropped node's data range proportionally by VRAM.
pub fn redistribute_shards(
    _current: &[ShardAssignment],
    remaining_nodes: &[NodeInfo],
    total_dataset_size: u64,
) -> Vec<ShardAssignment> {
    if remaining_nodes.is_empty() {
        return vec![];
    }

    // If only one node left, give it everything
    if remaining_nodes.len() == 1 {
        return vec![ShardAssignment {
            node_id: remaining_nodes[0].node_id.clone(),
            shard_index: 0,
            shard_start: 0,
            shard_end: total_dataset_size,
        }];
    }

    // Reassign using VRAM weighting over the remaining nodes
    assign_shards(remaining_nodes, total_dataset_size)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: &str, vram: u64) -> NodeInfo {
        NodeInfo {
            node_id: id.to_string(),
            gpu_name: "GPU".to_string(),
            gpu_vram_mb: vram,
            gpu_compute_capability: "8.0".to_string(),
            available: true,
        }
    }

    #[test]
    fn equal_shards_when_no_vram() {
        let nodes = vec![node("a", 0), node("b", 0)];
        let shards = assign_shards(&nodes, 1000);
        assert_eq!(shards.len(), 2);
        assert_eq!(shards[0].size(), 500);
        assert_eq!(shards[1].size(), 500);
    }

    #[test]
    fn weighted_shards_by_vram() {
        let nodes = vec![node("a", 8000), node("b", 24000)];
        let shards = assign_shards(&nodes, 32000);
        assert_eq!(shards.len(), 2);
        assert_eq!(shards[0].size(), 8000);
        assert_eq!(shards[1].size(), 24000);
    }

    #[test]
    fn single_node_gets_full_dataset() {
        let nodes = vec![node("a", 16000)];
        let shards = assign_shards(&nodes, 50000);
        assert_eq!(shards.len(), 1);
        assert_eq!(shards[0].size(), 50000);
    }

    #[test]
    fn empty_nodes_returns_empty() {
        let shards = assign_shards(&[], 1000);
        assert!(shards.is_empty());
    }

    #[test]
    fn redistribute_after_node_drop() {
        let nodes = vec![node("a", 8000), node("b", 12000), node("c", 24000)];
        let original = assign_shards(&nodes, 100000);
        assert_eq!(original.len(), 3);

        // Node "b" drops out
        let remaining = vec![node("a", 8000), node("c", 24000)];
        let new_shards = redistribute_shards(&original, &remaining, 100000);
        assert_eq!(new_shards.len(), 2);

        // Shards should still cover the entire dataset
        let total: u64 = new_shards.iter().map(|s| s.size()).sum();
        assert_eq!(total, 100000);

        // Node c should get proportionally more
        assert!(new_shards[1].size() > new_shards[0].size());
    }

    #[test]
    fn redistribute_to_single_node() {
        let nodes = vec![node("a", 8000), node("b", 12000)];
        let original = assign_shards(&nodes, 50000);
        let remaining = vec![node("a", 8000)];
        let new_shards = redistribute_shards(&original, &remaining, 50000);
        assert_eq!(new_shards.len(), 1);
        assert_eq!(new_shards[0].size(), 50000);
    }

    #[test]
    fn shards_cover_entire_dataset() {
        let nodes = vec![node("a", 8000), node("b", 12000), node("c", 24000)];
        let shards = assign_shards(&nodes, 100000);
        let total: u64 = shards.iter().map(|s| s.size()).sum();
        assert_eq!(total, 100000);
        // Verify contiguous
        for i in 1..shards.len() {
            assert_eq!(shards[i].shard_start, shards[i - 1].shard_end);
        }
    }
}
