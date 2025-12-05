# 🛡️ Federated Learning for IoT Intrusion Detection System

A comprehensive **Federated Learning** framework for detecting network intrusions in IoT devices using **Autoencoder-based anomaly detection** with advanced features including **FP16 quantization**, **dynamic weighting**, and a **graphical user interface**.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Running the System](#running-the-system)
  - [Option 1: Using GUI Dashboard](#option-1-using-gui-dashboard-recommended)
  - [Option 2: Using Terminal](#option-2-using-terminal)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Results and Metrics](#results-and-metrics)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)

---

## 🌟 Overview

This project implements a privacy-preserving machine learning system for IoT network security. Instead of collecting data from all devices to a central location (which raises privacy concerns), our federated learning approach allows each IoT device to:

- ✅ **Train locally** on its own data
- ✅ **Share only model updates** (not raw data)
- ✅ **Benefit from collective intelligence** across all devices
- ✅ **Preserve privacy** and reduce bandwidth requirements

### 🎯 Problem Statement

IoT devices are vulnerable to various cyber attacks (DDoS, Brute Force, Reconnaissance, etc.). Traditional centralized ML approaches require collecting sensitive network traffic data, which raises privacy concerns and requires significant bandwidth.

### 💡 Solution

**Federated Learning + Autoencoder Anomaly Detection**

- **Autoencoder Neural Network**: Learns normal traffic patterns and detects anomalies as deviations
- **Federated Learning**: Enables collaborative training without data sharing
- **FP16 Quantization**: Reduces model size by 2x for resource-constrained IoT devices
- **Dynamic Weighting**: Assigns higher weights to better-performing clients during aggregation

---

## ✨ Key Features

### 🔐 Privacy-Preserving Learning
- Data never leaves local devices
- Only encrypted model parameters are shared
- Compliant with data privacy regulations

### 🤖 Advanced Machine Learning
- **Autoencoder Architecture**: 20 → 12 → 6 → 12 → 20 (lightweight ~2K params)
- **Dynamic Aggregation**: Performance-based client weighting
- **Non-IID Data Support**: Handles heterogeneous data distributions

### 📦 Model Compression
- **FP16 Post-Training Quantization**: 2x model size reduction
- **Minimal Accuracy Loss**: <0.1% performance impact
- **Ideal for Edge Devices**: ~4KB model size (FP16)

### 🖥️ User-Friendly Interface
- **GUI Dashboard**: Start/stop server and clients with one click
- **Real-time Monitoring**: Live logs and status updates
- **Results Visualization**: Automated metrics generation

### 📊 Comprehensive Evaluation
- 8 performance metrics tracked per round
- Publication-ready visualizations (PNG + PDF)
- Confusion matrices and ROC curves

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Federated Learning Server                 │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  • Global Model Aggregation (FedAvg + Dynamic Weights) │ │
│  │  • FP16 Quantization/Dequantization                    │ │
│  │  • Client Performance Tracking                         │ │
│  │  • Threshold Synchronization                           │ │
│  └────────────────────────────────────────────────────────┘ │
└──────────────┬──────────────────────────────┬───────────────┘
               │                              │
    ┌──────────┴──────────┐      ┌──────────┴──────────┐
    │  Client 1 (30%)     │      │  Client 2 (25%)     │
    │  ┌───────────────┐  │      │  ┌───────────────┐  │
    │  │  Autoencoder  │  │      │  │  Autoencoder  │  │
    │  │  Local Train  │  │      │  │  Local Train  │  │
    │  └───────────────┘  │      │  └───────────────┘  │
    └─────────────────────┘      └─────────────────────┘
               │                              │
    ┌──────────┴──────────┐      ┌──────────┴──────────┐
    │  Client 3 (20%)     │      │  Client 4 (15%)     │
    └─────────────────────┘      └─────────────────────┘
               │
    ┌──────────┴──────────┐
    │  Client 5 (10%)     │
    └─────────────────────┘
```

**Training Flow:**
1. Server initializes global autoencoder model
2. Clients receive global parameters
3. Each client trains on local Non-IID data
4. Clients send quantized weight updates (FP16)
5. Server aggregates using dynamic weights
6. Process repeats for N rounds

---

## 🔧 Prerequisites

### System Requirements
- **OS**: Windows 10/11, Linux, or macOS
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 5GB free space for dataset and results
- **Network**: For multi-device deployment (optional)

### Required Python Packages

```bash
# Core FL Framework
flwr>=1.7.0

# Deep Learning
tensorflow>=2.13.0
keras>=2.13.0

# Data Processing
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Utilities
joblib>=1.3.0
```

---

## 📥 Installation

### Step 1: Clone or Download Project

```bash
# If using Git
git clone <repository-url>
cd federated-iot-ids

# Or download and extract ZIP file
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt

# Or install manually:
pip install flwr tensorflow pandas numpy scikit-learn matplotlib seaborn joblib
```

### Step 4: Verify Installation

```bash
python -c "import flwr; import tensorflow; print('✅ Installation successful!')"
```

---

## 📊 Dataset Preparation

This project uses the **CIC-IoT-2023 Dataset** for intrusion detection.

### Download Dataset

1. Visit: [CIC-IoT-2023 Dataset](https://www.unb.ca/cic/datasets/iotdataset-2023.html)
2. Download CSV files (7 attack scenarios + benign traffic)
3. Place all `.csv` files in `data/raw/` directory

### Dataset Structure

```
data/
├── raw/
│   ├── part-00000-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv
│   ├── part-00001-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv
│   └── ... (more CSV files)
└── processed/  (will be created automatically)
```

### Preprocess Data

**Run preprocessing script:**

```bash
python preprocess_data.py
```

**What this does:**
- ✅ Loads all raw CSV files
- ✅ Separates benign (normal) and attack traffic
- ✅ Performs feature selection using Random Forest (selects top 20 features)
- ✅ Normalizes features using StandardScaler
- ✅ Creates training set (normal data only) and test set (10K benign + 10K attack)
- ✅ Saves processed files and scaler

**Output files:**
- `data/processed/train_data.csv` - Training data (benign only)
- `data/processed/test_data.csv` - Test data (balanced: 10K benign + 10K attack)
- `data/processed/selected_features.csv` - List of 20 selected features
- `data/processed/feature_importance_all.csv` - Feature importance scores
- `scaler.pkl` - Fitted StandardScaler for normalization

**Expected output:**
```
Found 8 CSV file(s) in data/raw
Total samples: 500,000+
Training: 400,000 benign samples (ONLY normal data)
Test: 20,000 samples (10K benign + 10K attack)
Selected 20 features saved
```

### Split Data for Clients

**Create Non-IID client datasets:**

```bash
python split_client_data.py
```

**What this does:**
- ✅ Splits training data into 5 clients with **Non-IID distribution**
- ✅ Simulates real-world heterogeneous data distribution
- ✅ Each client gets different data proportions

**Data Distribution:**
- Client 1: 30% of training data
- Client 2: 25% of training data
- Client 3: 20% of training data
- Client 4: 15% of training data
- Client 5: 10% of training data

**Output files:**
- `data/processed/client1_data.csv` through `client5_data.csv`

---

## 🚀 Running the System

You have two options to run the federated learning system:

---

### **Option 1: Using GUI Dashboard** (⭐ Recommended)

The easiest way to run the system with a user-friendly interface.

#### Start the Dashboard

```bash
python fl_dashboard.py
```

#### Dashboard Features

📊 **Overview Tab:**
- Real-time status of server and all clients
- Quick actions: Start Full System, Stop All, View Results

🖥️ **Server Tab:**
- Configure number of training rounds
- Start/stop server
- Real-time server logs

👥 **Clients Tab:**
- Individual tabs for each client (1-5)
- Start/stop clients independently
- Client-specific logs

📝 **All Logs Tab:**
- Combined view of all server and client logs
- Color-coded by source
- Timestamped entries

📈 **Results Tab:**
- View latest experiment results
- Generate metrics and visualizations
- Open results folder

#### Using the Dashboard

**1. Start Server:**
- Go to "Server" tab
- Set number of rounds (e.g., 10)
- Click "▶ Start Server"
- Wait for "Waiting for clients..." message

**2. Start Clients:**
- Go to "Clients" tab
- Navigate to "Client 1" tab
- Click "▶ Start Client 1"
- Repeat for clients 2, 3, 4, 5

**OR use Quick Start:**
- Go to "Overview" tab
- Click "🚀 Start Full System"
- Server + all clients start automatically

**3. Monitor Training:**
- Watch real-time logs in respective tabs
- See status indicators turn green (● Running)
- Training progresses through specified rounds

**4. Generate Results:**
- After training completes
- Click "📊 Generate Metrics" in Results tab
- View metrics and visualizations

---

### **Option 2: Using Terminal** (Advanced Users)

For users who prefer command-line interface.

#### Terminal 1: Start Server

```bash
python server_autoencoder.py 10
```
Replace `10` with desired number of rounds.

**Expected output:**
```
============================================================
AUTOENCODER FEDERATED LEARNING SERVER
  Model: Lightweight Neural Network (~2K params)
  Rounds: 10
============================================================

Autoencoder Experiment ID: 20240115_143022
Quantization: FP16 (16-bit) Enabled
Dynamic Weighting: Hybrid (Accuracy:40%, F1:30%, Loss:20%, AUC:10%)

[OK] Starting Federated Learning Server...
     Waiting for clients to connect...
```

Keep this terminal running!

#### Terminals 2-6: Start Clients

Open separate terminal windows for each client:

**Terminal 2 - Client 1:**
```bash
python client_autoencoder.py 1
```

**Terminal 3 - Client 2:**
```bash
python client_autoencoder.py 2
```

**Terminal 4 - Client 3:**
```bash
python client_autoencoder.py 3
```

**Terminal 5 - Client 4:**
```bash
python client_autoencoder.py 4
```

**Terminal 6 - Client 5:**
```bash
python client_autoencoder.py 5
```

**Expected client output:**
```
==================================================
CLIENT 1 - Autoencoder Federated Learning
==================================================
Training samples: 120000
Features: 20
Data normalized for neural network
==================================================

Autoencoder created:
  Total params: 2088
  Model size (FP32): ~8.17 KB
  Model size (FP16): ~4.08 KB
  Compression: 2.00x
```

#### Monitor Training

Server terminal shows progress:
```
Round 1 - Autoencoder Aggregation
  Clients: 2
  Avg training loss: 0.002341
  Global threshold: 0.004523
  Avg epochs trained: 8.5

Round 1 - Evaluation
  Accuracy: 0.9234
  F1-Score: 0.8967
  Detection Rate: 0.8823
  AUC-ROC: 0.9456
```

#### Generate Metrics

After training completes:

```bash
python generate_metrics.py
```

---

## 📁 Project Structure

```
federated-iot-ids/
├── 📄 README.md                    # This file
├── 📄 requirements.txt              # Python dependencies
│
├── 🖥️ Server & Client Scripts
│   ├── server_autoencoder.py       # FL server with FP16 & dynamic weights
│   ├── client_autoencoder.py       # FL client with autoencoder
│   ├── fl_dashboard.py             # GUI dashboard application
│
├── 🔧 Utility Scripts
│   ├── preprocess_data.py          # Data preprocessing & feature selection
│   ├── split_client_data.py        # Non-IID data distribution
│   ├── generate_metrics.py         # Results visualization
│   ├── quantization_utils.py       # FP16 quantization functions
│   ├── dynamic_weighting.py        # Performance-based aggregation
│
├── 📊 Data
│   ├── data/
│   │   ├── raw/                    # Original CSV files (CIC-IoT-2023)
│   │   └── processed/              # Processed datasets
│   │       ├── train_data.csv
│   │       ├── test_data.csv
│   │       ├── client1_data.csv
│   │       ├── client2_data.csv
│   │       ├── client3_data.csv
│   │       ├── client4_data.csv
│   │       ├── client5_data.csv
│   │       ├── selected_features.csv
│   │       └── feature_importance_all.csv
│   │
│   └── scaler.pkl                  # Fitted StandardScaler
│
└── 📈 Results
    └── results/
        ├── experiment_YYYYMMDD_HHMMSS.json
        ├── convergence_plot_*.png
        ├── learning_curve_*.png
        ├── confusion_matrix_*.png
        ├── detection_metrics_*.png
        ├── radar_comparison_*.png
        ├── weight_evolution_*.png
        ├── client_distribution_*.png
        └── metrics_comparison_*.csv
```

---

## ⚙️ Configuration

### Server Configuration

Edit [`server_autoencoder.py`](server_autoencoder.py):

```python
# Line 68-77: Dynamic weighting parameters
self.weight_calculator = DynamicWeightCalculator(
    alpha=0.4,   # Accuracy weight (40%)
    beta=0.3,    # F1-score weight (30%)
    gamma=0.2,   # Loss weight (20%)
    delta=0.1,   # AUC-ROC weight (10%)
    min_weight=0.05,    # Minimum weight per client
    smoothing=0.7       # EMA smoothing factor
)

# Line 297-302: Server settings
strategy = AutoencoderStrategy(
    min_fit_clients=2,        # Minimum clients to start
    min_available_clients=2,  # Minimum clients available
    fraction_fit=1.0,         # Fraction of clients per round
    fraction_evaluate=1.0     # Fraction for evaluation
)
```

### Client Configuration

Edit [`client_autoencoder.py`](client_autoencoder.py):

```python
# Line 69-90: Autoencoder architecture
def create_autoencoder(input_dim=20):
    encoder_input = keras.Input(shape=(input_dim,))
    encoded = keras.layers.Dense(12, activation='relu', name='encoder_1')(encoder_input)
    encoded = keras.layers.Dropout(0.2)(encoded)
    encoded = keras.layers.Dense(6, activation='relu', name='bottleneck')(encoded)
    
    decoded = keras.layers.Dense(12, activation='relu', name='decoder_1')(encoded)
    decoded = keras.layers.Dropout(0.2)(decoded)
    decoded = keras.layers.Dense(input_dim, activation='linear', name='decoder_output')(decoded)

# Line 157-162: Training parameters
history = self.model.fit(
    self.X_train, self.X_train,
    epochs=20,              # Maximum epochs
    batch_size=32,          # Batch size
    validation_split=0.1,   # Validation split
    callbacks=[early_stopping]  # Early stopping (patience=3)
)
```

### Data Split Configuration

Edit [`split_client_data.py`](split_client_data.py):

```python
# Line 14: Number of clients
num_clients = 5

# Line 21-25: Data distribution ratios
split_ratios = [0.30, 0.25, 0.20, 0.15, 0.10]  # Must sum to 1.0
```

---

## 📊 Results and Metrics

### Automatic Metrics Generation

After training, run:

```bash
python generate_metrics.py
```

### Generated Visualizations

1. **Convergence Plot** (`convergence_plot_*.png`)
   - All metrics over training rounds
   - Shows improvement trends

2. **Learning Curve** (`learning_curve_*.png`)
   - Accuracy and F1-score progression
   - Precision and recall trends

3. **Weight Evolution** (`weight_evolution_*.png`)
   - Heatmap of client contributions over rounds
   - Shows dynamic weighting in action

4. **Detection Metrics** (`detection_metrics_*.png`)
   - Bar chart comparing first vs last round
   - All 6 key metrics side-by-side

5. **Confusion Matrix** (`confusion_matrix_*.png`)
   - Final round prediction breakdown
   - True/False Positives/Negatives

6. **Radar Comparison** (`radar_comparison_*.png`)
   - Spider chart comparing first vs last round
   - Visualizes overall improvement

7. **Client Distribution** (`client_distribution_*.png`)
   - Bar + pie charts of data distribution
   - Shows Non-IID split

### Metrics Tracked

| Metric | Description | Target |
|--------|-------------|--------|
| **Accuracy** | Overall correctness | >0.90 |
| **Precision** | Of predicted attacks, % correct | >0.85 |
| **Recall** | Of actual attacks, % detected | >0.85 |
| **F1-Score** | Harmonic mean of precision/recall | >0.85 |
| **AUC-ROC** | Area under ROC curve | >0.90 |
| **Specificity** | True negative rate | >0.90 |

### Results Files

```
results/
├── experiment_20240115_143022.json     # Complete training log
├── convergence_plot_20240115_143022.png
├── convergence_plot_20240115_143022.pdf
├── learning_curve_20240115_143022.png
├── ...
└── metrics_comparison_20240115_143022.csv
```

**JSON Structure:**
```json
{
  "experiment_id": "20240115_143022",
  "model_type": "Autoencoder_FP16_DynamicWeights",
  "quantization_enabled": true,
  "dynamic_weighting_enabled": true,
  "num_clients": 5,
  "rounds": [
    {
      "round": 1,
      "accuracy": 0.8523,
      "f1_score": 0.8234,
      "confusion_matrix": [[9234, 766], [1123, 8877]],
      "client_contributions": {
        "client_1": 0.32,
        "client_2": 0.28,
        ...
      }
    },
    ...
  ]
}
```

---

## 🔬 Technical Details

### Autoencoder Architecture

```
Input (20 features)
    ↓
Encoder Layer 1: Dense(12, ReLU) + Dropout(0.2)
    ↓
Bottleneck: Dense(6, ReLU)  ← Compressed representation
    ↓
Decoder Layer 1: Dense(12, ReLU) + Dropout(0.2)
    ↓
Output: Dense(20, Linear)  ← Reconstruction
```

**Total Parameters:** ~2,088
**Model Size:** 
- FP32: ~8.17 KB
- FP16: ~4.08 KB (50% reduction)

### Anomaly Detection Mechanism

1. **Training Phase:**
   - Autoencoder learns to reconstruct normal traffic patterns
   - Trained only on benign data
   - Minimizes reconstruction error (MSE)

2. **Detection Phase:**
   - Calculate reconstruction error for test samples
   - Set threshold at 95th percentile of training errors
   - Samples with error > threshold = anomalies (attacks)

3. **Threshold Synchronization:**
   - Server calculates global threshold (weighted average)
   - Shared with all clients for consistent detection

### FP16 Quantization

**Process:**
1. Train model in FP32 (32-bit floating point)
2. Convert weights to FP16 (16-bit) after training
3. Send FP16 weights to server (bandwidth reduction)
4. Server aggregates in FP32 precision
5. Quantize aggregated model back to FP16

**Benefits:**
- 2x model size reduction
- 2x bandwidth savings
- Minimal accuracy loss (<0.1%)
- Ideal for IoT edge devices

### Dynamic Weight Aggregation

**Standard FedAvg:**
```
w_global = Σ(n_i / N) × w_i
```
Where `n_i` = number of samples, `N` = total samples

**Dynamic Weighted Aggregation:**
```
w_global = Σ(α_i × w_i)
```
Where `α_i` is calculated as:
```
α_i = (0.4 × accuracy + 0.3 × f1_score + 0.2 × loss_inv + 0.1 × auc_roc)
```

**Advantages:**
- Rewards high-performing clients
- Penalizes poor-performing clients
- Reduces impact of noisy/poisoned data
- Improves convergence speed

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Import Error: `No module named 'flwr'`

**Solution:**
```bash
pip install flwr tensorflow pandas numpy scikit-learn matplotlib seaborn
```

#### 2. No Raw Data Files Found

**Error:** `FileNotFoundError: No CSV files found in data/raw`

**Solution:**
- Download CIC-IoT-2023 dataset
- Place all `.csv` files in `data/raw/` directory
- Ensure files are not in subfolders

#### 3. Server Won't Start / Connection Refused

**Solution:**
```bash
# Check if port 8080 is already in use
# Windows:
netstat -ano | findstr :8080

# Linux/Mac:
lsof -i :8080

# Kill the process or change port in server_autoencoder.py (line 309)
```

#### 4. Client Data Not Found

**Error:** `Client data file not found: data/processed/client1_data.csv`

**Solution:**
```bash
python split_client_data.py
```

#### 5. Out of Memory Error

**Solution:**
```python
# Edit client_autoencoder.py line 159
# Reduce batch size:
batch_size=16,  # Changed from 32
```

#### 6. GUI Not Starting

**Solution:**
```bash
# Install tkinter (if missing)
# Ubuntu/Debian:
sudo apt-get install python3-tk

# Fedora:
sudo dnf install python3-tkinter

# macOS (should be pre-installed):
# Reinstall Python from python.org
```

#### 7. Slow Training

**Possible causes:**
- CPU-only training (TensorFlow not detecting GPU)
- Large dataset
- Too many epochs

**Solutions:**
```python
# Check GPU availability:
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

# Reduce epochs in client_autoencoder.py line 157:
epochs=10,  # Changed from 20
```

#### 8. Results Not Generating

**Error:** `No experiment results found in results/ directory`

**Solution:**
- Ensure training has completed at least 1 round
- Check if `results/` directory exists
- Verify JSON files are being created during training

### Debug Mode

Enable verbose logging:

```python
# Add to beginning of server_autoencoder.py or client_autoencoder.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@software{federated_iot_ids_2024,
  title={Federated Learning for IoT Intrusion Detection},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo},
  note={Autoencoder-based anomaly detection with FP16 quantization and dynamic weighting}
}
```

**Dataset Citation:**
```bibtex
@article{neto2023ciciot2023,
  title={CIC-IoT-2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment},
  author={Neto, Euclides Carlos Pinto and Dadkhah, Sajjad and Ferreira, Raphael and Zohourian, Ali and Lu, Rongxing and Ghorbani, Ali A},
  journal={Sensors},
  volume={23},
  number={13},
  pages={5941},
  year={2023},
  publisher={MDPI}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📞 Support

For issues, questions, or suggestions:

- 📧 Email: your.email@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/your-repo/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/your-repo/discussions)

---

## 🙏 Acknowledgments

- **CIC (Canadian Institute for Cybersecurity)** for the IoT-2023 dataset
- **Flower Framework** team for the excellent FL library
- **TensorFlow** team for the deep learning framework
- All contributors and users of this project

---

## 📝 Changelog

### Version 1.0.0 (2024-01-15)
- ✅ Initial release
- ✅ Autoencoder-based anomaly detection
- ✅ FP16 quantization support
- ✅ Dynamic weight aggregation
- ✅ GUI dashboard
- ✅ Comprehensive metrics generation

---

**⭐ If you find this project useful, please consider starring the repository!**

---

*Last Updated: January 2024*