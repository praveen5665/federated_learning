# Federated Learning for IoT Threat Detection using One-Class SVM

This project implements a **Federated Learning** framework for **IoT intrusion detection** using **One-Class SVM**. The system enables multiple IoT edge devices (clients) to collaboratively train an anomaly detection model without sharing raw data, preserving privacy while detecting network threats.

## 📋 Project Overview

### Problem Statement
IoT devices are vulnerable to various cyber attacks. Traditional centralized machine learning approaches require collecting data from all devices, which raises privacy concerns and bandwidth limitations.

### Solution
Federated Learning allows each IoT client to:
- Train locally on its own data
- Share only model parameters (not raw data)
- Benefit from collective learning across all clients

### Algorithm: One-Class SVM
One-Class SVM is ideal for anomaly detection because:
- Trained only on "normal" traffic patterns
- Detects anomalies as deviations from learned normal behavior
- No need for labeled attack data during training

---

## 📁 Project Structure

project/
├── server.py # Flower FL server with FedAvg aggregation
├── client.py # Flower FL client with One-Class SVM
├── preprocess_data.py # Data preprocessing & feature selection
├── split_client_data.py # Non-IID data distribution to clients
├── generate_metrics.py # Visualization & metrics generation
├── start_clients.bat # Windows batch script to start clients
├── README_FEDERATED.md # This documentation
├── data/
│ ├── raw/ # Original CIC-IoT-2023 dataset CSV files
│ └── processed/
│ ├── train_data.csv # Preprocessed training data
│ ├── test_data.csv # Preprocessed test data
│ ├── client1_data.csv # Client 1 training data (30%)
│ ├── client2_data.csv # Client 2 training data (25%)
│ ├── client3_data.csv # Client 3 training data (20%)
│ ├── client4_data.csv # Client 4 training data (15%)
│ ├── client5_data.csv # Client 5 training data (10%)
│ ├── selected_features.csv
│ └── feature_importance_all.csv
└── results/
└── experiment_*.json # Training results per experiment


---

## 🔧 Installation

### Prerequisites
```bash
pip install flwr pandas numpy scikit-learn matplotlib seaborn joblib

This directory contains the federated learning implementation for IoT intrusion detection using One-Class SVM.


Required Packages
Package	Version	Purpose
flwr	>=1.0.0	Flower Federated Learning
pandas	>=1.3.0	Data manipulation
numpy	>=1.21.0	Numerical operations
scikit-learn	>=1.0.0	One-Class SVM & metrics
matplotlib	>=3.4.0	Visualization
seaborn	>=0.11.0	Enhanced plots

### Files

- `preprocess_data.py` - Data preprocessing and feature selection
- `split_client_data.py` - Splits training data into Non-IID client datasets
- `server.py` - Flower server for federated learning
- `client.py` - Flower client for One-Class SVM training
- `run_federated_learning.sh` - Script to start the server

## 🚀 Setup Instructions

### Step 1: Preprocess Data
```bash
python preprocess_data.py
```

This will:
- Load all CSV files from `data/raw/`
- Perform feature selection (20 features)
- Create `train_data.csv` and `test_data.csv`

### Step 2: Split Data for Clients
```bash
python split_client_data.py
```

This creates Non-IID client datasets:
- `client1_data.csv` (30% of data)
- `client2_data.csv` (25% of data)
- `client3_data.csv` (20% of data)
- `client4_data.csv` (15% of data)
- `client5_data.csv` (10% of data)

### Step 3: Start Federated Learning

**Terminal 1 - Start Server (runs continuously):**
```bash
python server.py
```
The server will run indefinitely and wait for clients. Press `Ctrl+C` to stop.

**Optional: Run for specific number of rounds:**
```bash
python server.py 10  # Run for 10 rounds
```

**Terminal 2 - Start Client 1:**
```bash
python client.py 1
```

**Terminal 3 - Start Client 2:**
```bash
python client.py 2
```

**Terminal 4 - Start Client 3 (optional):**
```bash
python client.py 3
```

(Repeat for more clients as needed)

**Note:** The server remains active after each round. Clients can disconnect and reconnect, and training will continue.

## ⚙️ How It Works

1. **Server**: Aggregates model parameters from all clients using FedAvg strategy
   - Server runs **continuously** and remains active after each round
   - Clients can disconnect and reconnect at any time
   - Training continues as long as the server is running
2. **Clients**: Each client trains One-Class SVM on its local Non-IID data
3. **Rounds**: Training rounds continue indefinitely (or until manually stopped)
4. **Results**: Evaluation metrics are saved to `results.json` after each round

## 📂 Output Files

- `results.json` - Training and evaluation results for each round
- `data/processed/client{1-5}_data.csv` - Client-specific training data
- `data/processed/selected_features.csv` - List of 20 selected features

## 📝 Notes

- One-Class SVM aggregation in federated learning is complex. This implementation uses a practical approach where clients retrain on local data each round.
- The server aggregates parameters, but clients use their local data for training.
- Minimum 2 clients required for training to start.

