# drift

P2P distributed training. Plug your GPU into the mesh.

## What is this?

drift lets you train models across consumer GPUs on different networks. No cloud account, no VPN, no static IPs. Each machine joins the swarm by public key using [iroh](https://github.com/n0-computer/iroh) for peer-to-peer connectivity over QUIC.

Think of it as a decentralized Slurm for indie researchers.

## How it works

```
Machine A (RTX 3090)              Machine B (RTX 4090)
  $ drift-node join                 $ drift-node join
  > Node ID: abc123...              > Node ID: def456...
  > GPU: RTX 3090 (24576 MB)       > GPU: RTX 4090 (24564 MB)
  > Waiting for connections...      > Waiting for connections...

Machine A (coordinator):
  $ drift-coord train --peers abc123,def456 --epochs 10
  > Connected: abc123 | RTX 3090 (24576 MB VRAM)
  > Connected: def456 | RTX 4090 (24564 MB VRAM)
  > Starting training (2 nodes, shards weighted by VRAM)
```

## Architecture

```
+------------------+         QUIC/iroh          +------------------+
|   drift-coord    | <========================> |   drift-node     |
|                  |   (ALPN: drift/0)          |                  |
| - Connects to    |    Ping -> NodeInfo ->     | - Joins swarm    |
|   peer nodes     |    TrainConfig ->          | - Detects GPU    |
| - Shards dataset |    ShardAssignment ->      | - Runs training  |
| - Assigns ring   |    RingConfig ->           | - Reports back   |
| - Barrier sync   |    StartRing ->            | - Ring all-reduce|
|                  |    <- BarrierSync          |                  |
|                  |    BarrierReady ->          |                  |
|                  |    <- TrainProgress        |                  |
+------------------+                            +------------------+
                                                   |          ^
                                                   v          |
                                          +------------------+
                                          |  node <-> node   |
                                          | (ALPN: drift-ring/0)
                                          |  GradientChunk   |
                                          |  ring all-reduce |
                                          +------------------+
        |
        v
  drift-proto (shared message types, ring state machine, ALPN, framing)
```

All traffic is encrypted end-to-end via QUIC. NAT hole-punching is handled automatically by iroh, with relay fallback.

## Project structure

```
drift/
  Cargo.toml              # Workspace
  drift-node/             # Node binary
    src/
      main.rs             # CLI: join, status
      gpu.rs              # GPU detection (nvidia-smi)
      network.rs          # iroh endpoint, connection handling
      training.rs         # Python training subprocess
  drift-coord/            # Coordinator binary
    src/
      main.rs             # CLI: train
      scheduler.rs        # Shard assignment by GPU capability
      checkpoint.rs       # Checkpoint management
      monitor.rs          # Health monitoring, progress display
  drift-cli/              # Unified CLI binary
    src/
      main.rs             # CLI: join, train, status
      node.rs             # Node logic (GPU, training)
      coord.rs            # Coordinator logic (sharding, monitoring)
  drift-proto/            # Shared protocol
    src/
      lib.rs              # Message types, framing, ALPN
      allreduce.rs        # Ring all-reduce primitives
      ring.rs             # Ring state machine + async QUIC all-reduce
    tests/
      integration.rs      # Full handshake test
      training.rs         # End-to-end training pipeline
      stress.rs           # Bulk message and gradient tests
      ring_connect.rs     # Ring connectivity test (3-node)
      ring_allreduce.rs   # Ring all-reduce over QUIC + stress tests
  examples/
    mock_train.py          # Mock training script for testing
    train.yaml             # Example training config
```

## Quick start

### Requirements

- Rust 1.75+
- NVIDIA GPU with drivers installed (optional, runs in CPU-only mode without)

### Build

```sh
cargo build --release
```

### Run a node

```sh
# Using the unified CLI
./target/release/drift join --name my-gpu-box

# Or the standalone binary
./target/release/drift-node join --name my-gpu-box
```

### Start training (coordinator)

```sh
./target/release/drift train \
  --peers <node_id_1>,<node_id_2> \
  --model-path model.pt \
  --dataset-path ./data \
  --epochs 10 \
  --batch-size 32
```

### Resume from checkpoint

```sh
./target/release/drift train \
  --peers <node_id_1>,<node_id_2> \
  --resume \
  --checkpoint-dir checkpoints/
```

### Check GPU status

```sh
./target/release/drift status
```

### Debug logging

```sh
RUST_LOG=debug ./target/release/drift join
```

## Protocol

Messages are length-prefixed JSON over QUIC bidirectional streams.

### Coordinator-Node (ALPN: `drift/0`)

1. Coordinator connects to node, sends `Ping`
2. Node responds with `NodeInfo` (GPU name, VRAM, compute capability)
3. Coordinator sends `TrainConfig`, `ShardAssignment`, and `RingConfig`
4. Coordinator sends `StartRing` — nodes establish peer-to-peer ring
5. During training, nodes send `BarrierSync` per step, coordinator replies `BarrierReady`
6. Node streams `TrainProgress` updates back

### Ring All-Reduce (ALPN: `drift-ring/0`)

Nodes form a logical ring (0 -> 1 -> 2 -> ... -> N-1 -> 0). Each node connects to its right neighbor and accepts from its left. Gradient synchronization uses the standard ring all-reduce algorithm:

1. **Scatter-reduce** (N-1 iterations): each node sends one chunk to the right, receives from the left, and accumulates. After this phase, each node holds the fully-reduced version of one chunk.
2. **All-gather** (N-1 iterations): each node sends its reduced chunk around the ring so all nodes end up with the complete result.
3. **Finalize**: divide by N to get the average.

Sparse gradients (>50% zeros) are automatically compressed before sending.

## Milestones

- [x] Cargo workspace, iroh connectivity, message protocol
- [x] GPU detection, node capability announcement
- [x] Coordinator: peer management, shard scheduling
- [x] Integration test: full handshake over local QUIC
- [x] Unified CLI binary (`drift join`, `drift train`, `drift status`)
- [x] Training execution with progress streaming
- [x] Ring all-reduce primitives for gradient sync
- [x] Sparse gradient compression
- [x] Heartbeat loop and stale node detection
- [x] Checkpointing: periodic save with resume support
- [x] Fault tolerance: shard redistribution on node drops
- [x] Stress tests for bulk messages and gradient payloads
- [x] Gradient sync: ring all-reduce over QUIC streams
- [ ] Python bridge: PyTorch DDP backend
- [ ] Benchmarks vs standard DDP

## License

MIT
