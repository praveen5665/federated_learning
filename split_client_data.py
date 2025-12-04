"""
Split training data into Non-IID client datasets for Federated Learning
Each client gets a different slice of data to simulate real-world Non-IID distribution
"""

import pandas as pd
import numpy as np
import os

# Create directories if they don't exist
os.makedirs("data/processed", exist_ok=True)

print("Loading training data...")
train_data = pd.read_csv("data/processed/train_data.csv")

print(f"Total training samples: {len(train_data)}")
print(f"Features: {train_data.shape[1] - 1}")  # Excluding label column

# Number of clients
num_clients = 5

# Create Non-IID distribution by splitting data unevenly
# Each client gets a different portion of the data
print(f"\nSplitting data into {num_clients} clients with Non-IID distribution...")

# Shuffle data to ensure randomness
train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Create uneven splits to simulate Non-IID
# Client 1: 30% of data
# Client 2: 25% of data
# Client 3: 20% of data
# Client 4: 15% of data
# Client 5: 10% of data
split_ratios = [0.30, 0.25, 0.20, 0.15, 0.10]

# Ensure ratios sum to 1.0
if len(split_ratios) != num_clients:
    # If num_clients changes, create equal splits
    split_ratios = [1.0 / num_clients] * num_clients

# Calculate split indices
split_indices = []
current_idx = 0
for i, ratio in enumerate(split_ratios):
    next_idx = current_idx + int(len(train_data) * ratio)
    if i == len(split_ratios) - 1:
        # Last client gets all remaining data
        next_idx = len(train_data)
    split_indices.append((current_idx, next_idx))
    current_idx = next_idx

# Split and save client data
for client_id in range(1, num_clients + 1):
    start_idx, end_idx = split_indices[client_id - 1]
    client_data = train_data.iloc[start_idx:end_idx].copy()
    
    # Save client data
    filename = f"data/processed/client{client_id}_data.csv"
    client_data.to_csv(filename, index=False)
    
    print(f"Client {client_id}: {len(client_data)} samples ({len(client_data)/len(train_data)*100:.1f}%)")
    print(f"  Saved to {filename}")

print(f"\n{'='*50}")
print("Client data split completed!")
print(f"{'='*50}")
print(f"Total clients: {num_clients}")
print(f"Total samples distributed: {sum([len(pd.read_csv(f'data/processed/client{i}_data.csv')) for i in range(1, num_clients + 1)])}")
print(f"Original training samples: {len(train_data)}")

