#!/usr/bin/env python3
"""CIFAR-10 training script using drift DDP backend.

Launched by the Rust node as a subprocess. Communicates gradient data
via shared memory and control messages via stdin/stdout.

Works on CPU (no GPU required).
"""

import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset

# Add drift-python to path if running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "drift-python"))

import drift


class SimpleCNN(nn.Module):
    """Small CNN for CIFAR-10 (3x32x32 -> 10 classes)."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_synthetic_data(num_samples, batch_size):
    """Create synthetic CIFAR-10-shaped data for testing without downloading."""
    images = torch.randn(num_samples, 3, 32, 32)
    labels = torch.randint(0, 10, (num_samples,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def main():
    epochs = int(os.environ.get("DRIFT_EPOCHS", "3"))
    batch_size = int(os.environ.get("DRIFT_BATCH_SIZE", "32"))

    # Initialize drift (opens shm, registers DDP backend, prints DRIFT_READY)
    drift.init()

    rank = drift.rank()
    world_size = drift.world_size()

    print(f"[train_cifar] rank={rank}/{world_size} epochs={epochs} batch={batch_size}",
          file=sys.stderr, flush=True)

    # Model + DDP wrapper with drift communication hook
    model = SimpleCNN()
    model = DDP(model)
    drift.register(model)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # Synthetic data (real CIFAR-10 would need torchvision)
    num_samples = batch_size * 10  # 10 steps per epoch
    loader = make_synthetic_data(num_samples, batch_size)

    for epoch in range(epochs):
        running_loss = 0.0
        step_count = 0
        t0 = time.time()

        for step, (images, labels) in enumerate(loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()  # DDP calls allreduce here
            optimizer.step()

            running_loss += loss.item()
            step_count += 1
            global_step = epoch * len(loader) + step

            elapsed = time.time() - t0
            throughput = (step_count * batch_size) / max(elapsed, 1e-6)

            # Report progress to Rust node
            sys.stdout.write(
                f"DRIFT_PROGRESS {epoch} {global_step} {loss.item():.6f} {throughput:.1f}\n"
            )
            sys.stdout.flush()

        avg_loss = running_loss / max(step_count, 1)
        print(f"[train_cifar] epoch {epoch}: avg_loss={avg_loss:.4f}",
              file=sys.stderr, flush=True)

    sys.stdout.write("DRIFT_DONE\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
