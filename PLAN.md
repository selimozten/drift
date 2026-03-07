# drift

P2P distributed training swarm. Plug your GPU into the mesh.

## Why This Exists

Distributed training today requires either expensive cloud clusters or complex networking setup (static IPs, VPNs, SSH tunnels). drift uses iroh for peer-to-peer connectivity — each machine joins the swarm by public key. No IPs, no VPN, no cloud account needed. A decentralized Slurm for indie researchers with consumer GPUs.

## How It Works

```
Machine A (RTX 3090)          Machine B (RTX 4090)
    $ drift join                  $ drift join
    > Node ID: abc123...          > Node ID: def456...
    > Waiting for peers...        > Waiting for peers...

Machine A:
    $ drift train --config train.yaml --peers def456
    > Connected to def456 via QUIC
    > Sharding dataset...
    > Training started (2 GPUs across 2 nodes)
```

## Architecture

### Networking Layer (iroh)
- Each node gets a public key identity (no IP addresses)
- iroh handles NAT hole-punching automatically
- Falls back to relay servers if direct connection fails
- All traffic encrypted via QUIC

### Node Discovery
- Join by explicit peer ID (paste the key)
- Optional: discovery via shared secret / swarm name
- Node announces GPU capabilities on connect (VRAM, compute capability, bandwidth)

### Training Coordination
- Leader election: node that initiates `drift train` is the coordinator
- Data sharding: coordinator splits dataset across nodes
- Gradient sync: all-reduce over iroh QUIC streams
- Checkpointing: coordinator collects and saves checkpoints
- Fault tolerance: if a node drops, redistribute its shard

### Supported Training Modes
- Data parallel (DDP): same model, different data shards
- Pipeline parallel: model split across nodes (future)
- Hybrid: combination of data + pipeline (future)

## Core Components

### drift-node (Rust binary)
- Single binary per machine
- Manages GPU detection (CUDA/ROCm)
- Runs iroh endpoint for P2P connectivity
- Receives training commands from coordinator
- Executes PyTorch training subprocess
- Streams gradients and metrics back

### drift-coord (Rust binary or mode)
- Coordinator role (can also be a training node)
- Manages peer connections and health checks
- Distributes training config and data shards
- Orchestrates gradient synchronization
- Collects metrics and checkpoints

### Python Bridge
- Thin Python library that training scripts import
- Handles gradient serialization/deserialization
- Hooks into PyTorch DDP backend
- Minimal changes needed to existing training scripts:

```python
import drift

# Replace torch.distributed init
drift.init(config="train.yaml")

# Rest of training code stays the same
model = DDP(model, backend="drift")
```

## Tech Stack

- Rust for the core binary (node + coordinator)
- iroh (n0-computer/iroh) for P2P networking
- QUIC for transport (via iroh)
- PyTorch integration via custom DDP backend
- Optional: gRPC or msgpack for control plane messages

## Project Structure

```
drift/
  README.md
  Cargo.toml
  drift-node/             # Node binary
    src/
      main.rs
      gpu.rs              # GPU detection
      network.rs          # iroh networking
      training.rs         # Training subprocess management
      sync.rs             # Gradient synchronization
  drift-coord/            # Coordinator
    src/
      main.rs
      scheduler.rs        # Task scheduling
      checkpoint.rs       # Checkpoint management
      monitor.rs          # Health monitoring
  drift-py/               # Python bridge
    pyproject.toml
    drift/
      __init__.py
      backend.py          # PyTorch DDP backend
      config.py           # Training config
  proto/                  # Protocol definitions
    messages.proto
  examples/
    train_mnist.py        # Simple example
    train_llm.py          # LLM fine-tuning example
    config.yaml           # Example config
  tests/
```

## Milestones

1. Repo setup, Cargo workspace, basic iroh connectivity between 2 nodes
2. GPU detection + node capability announcement
3. Coordinator: peer management, health checks
4. Data sharding: split dataset across nodes
5. Gradient sync: all-reduce over QUIC
6. Python bridge: PyTorch DDP backend integration
7. Checkpointing: save/resume across the swarm
8. Fault tolerance: handle node drops gracefully
9. CLI polish: `drift join`, `drift train`, `drift status`
10. Example: fine-tune a small model across 2+ consumer GPUs
11. Benchmarks: compare throughput vs standard DDP over ethernet

## Key Design Decisions

- **Why Rust?** Performance-critical networking and GPU coordination. No GC pauses during gradient sync.
- **Why iroh?** NAT traversal solved, public key identity, QUIC built-in. Don't reinvent networking.
- **Why not just use DeepSpeed/FSDP?** Those assume you have the network figured out. drift solves the networking problem — it can use DeepSpeed/FSDP underneath.
- **Data parallel first.** Pipeline parallel is harder and less useful for the target audience (indie researchers with 2-4 GPUs).

## Open Questions

- What's the overhead of gradient sync over QUIC vs raw TCP/RDMA?
- Can iroh's relay servers handle gradient traffic, or do we need direct connections?
- Should the Python bridge be a PyTorch DDP backend or a higher-level wrapper?
- How to handle heterogeneous GPUs (3090 + 4090) efficiently?

## Success Criteria

- Two consumer GPUs on different networks can train a model together
- Less than 20% overhead vs same GPUs on the same LAN with standard DDP
- Single binary install, no configuration beyond peer keys
- Works behind NAT without any port forwarding
