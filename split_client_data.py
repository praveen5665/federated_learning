"""
Split training data into Non-IID client datasets for Federated Learning.
Uses Dirichlet distribution for realistic Non-IID quantity skew.
Supports any number of clients (1-10).
"""

import argparse
import glob
import os
import numpy as np
import pandas as pd

# Create directories if they don't exist
os.makedirs("data/processed", exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-clients", type=int, default=10, help="Number of clients (1-10)")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5,
                        help="Dirichlet concentration parameter (lower = more skew, default: 0.5)")
    return parser.parse_args()


args = parse_args()
num_clients = max(1, min(args.num_clients, 10))

print("Loading training data...")
train_data = pd.read_csv("data/processed/train_data.csv")

print(f"Total training samples: {len(train_data)}")
print(f"Features: {train_data.shape[1] - 1}")  # Excluding label column
print(f"\nSplitting data into {num_clients} clients with Non-IID distribution...")
print(f"Dirichlet alpha: {args.dirichlet_alpha} (lower = more heterogeneous)")

# Shuffle data to ensure randomness
train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Generate Non-IID split ratios using Dirichlet distribution
# Works for ANY number of clients, not just 10
np.random.seed(42)
split_ratios = np.random.dirichlet([args.dirichlet_alpha] * num_clients)

# Ensure minimum 2% per client to avoid empty partitions
min_ratio = 0.02
split_ratios = np.clip(split_ratios, min_ratio, None)
split_ratios = split_ratios / split_ratios.sum()  # Re-normalize

print(f"Generated Non-IID ratios: {[f'{r:.3f}' for r in split_ratios]}")
print(f"  Min share: {split_ratios.min()*100:.1f}%, Max share: {split_ratios.max()*100:.1f}%")

# Cleanup old client files
for old_file in glob.glob("data/processed/client*_data.csv"):
    os.remove(old_file)

# Calculate split indices
split_indices = []
current_idx = 0
for i, ratio in enumerate(split_ratios):
    next_idx = current_idx + int(len(train_data) * ratio)
    if i == len(split_ratios) - 1:
        next_idx = len(train_data)  # last client gets remainder
    split_indices.append((current_idx, next_idx))
    current_idx = next_idx

# Split and save client data
for client_id in range(1, num_clients + 1):
    start_idx, end_idx = split_indices[client_id - 1]
    client_data = train_data.iloc[start_idx:end_idx].copy()
    filename = f"data/processed/client{client_id}_data.csv"
    client_data.to_csv(filename, index=False)

    print(f"Client {client_id}: {len(client_data)} samples ({len(client_data)/len(train_data)*100:.1f}%)")
    print(f"  Saved to {filename}")

print(f"\n{'='*50}")
print("Client data split completed!")
print(f"{'='*50}")
print(f"Total clients: {num_clients}")
print(f"Total samples distributed: {sum(len(pd.read_csv(f'data/processed/client{i}_data.csv')) for i in range(1, num_clients + 1))}")
print(f"Original training samples: {len(train_data)}")

