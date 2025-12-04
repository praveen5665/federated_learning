"""
Federated Learning Client using Autoencoder for IoT Anomaly Detection
Lightweight neural network approach for edge devices
"""

import flwr as fl
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    accuracy_score, roc_auc_score, confusion_matrix,
    average_precision_score
)
import sys
import os
import time
import warnings
warnings.filterwarnings('ignore')

# Disable GPU if not available (for IoT devices)
tf.config.set_visible_devices([], 'GPU')

# Load selected features
selected_features = pd.read_csv("data/processed/selected_features.csv")['selected_features'].tolist()

# ...existing argument parsing code...
if len(sys.argv) < 2:
    print("Usage: python client_autoencoder.py <client_id>")
    sys.exit(1)

client_id = int(sys.argv[1])
client_data_file = f"data/processed/client{client_id}_data.csv"

if not os.path.exists(client_data_file):
    print(f"Error: Client data file not found: {client_data_file}")
    sys.exit(1)

print(f"\n{'='*50}")
print(f"CLIENT {client_id} - Autoencoder Federated Learning")
print(f"{'='*50}")

# Load data
data = pd.read_csv(client_data_file)
X_train = data[selected_features].values.astype(np.float32)
y_train = data['label'].values

print(f"Training samples: {len(X_train)}")
print(f"Features: {X_train.shape[1]}")

# Normalize data (important for neural networks)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

print(f"Data normalized for neural network")
print(f"{'='*50}\n")


def create_autoencoder(input_dim=20):
    """
    Create lightweight autoencoder for IoT devices
    Total params: ~2000 (very small!)
    """
    # Encoder
    encoder_input = keras.Input(shape=(input_dim,))
    encoded = keras.layers.Dense(12, activation='relu', name='encoder_1')(encoder_input)
    encoded = keras.layers.Dense(6, activation='relu', name='bottleneck')(encoded)
    
    # Decoder
    decoded = keras.layers.Dense(12, activation='relu', name='decoder_1')(encoded)
    decoded = keras.layers.Dense(input_dim, activation='linear', name='decoder_output')(decoded)
    
    # Full autoencoder
    autoencoder = keras.Model(encoder_input, decoded, name='autoencoder')
    
    # Compile with MSE loss (reconstruction error)
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse'
    )
    
    return autoencoder


class AutoencoderClient(fl.client.NumPyClient):
    """Flower client with Autoencoder for anomaly detection"""
    
    def __init__(self, client_id, X_train, y_train, scaler):
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.scaler = scaler
        self.model = create_autoencoder(input_dim=X_train.shape[1])
        self.threshold = None  # Will be set after training
        self.current_round = 0
        
        # Load test data
        test_data = pd.read_csv("data/processed/test_data.csv")
        self.X_test = test_data[selected_features].values.astype(np.float32)
        self.X_test_scaled = scaler.transform(self.X_test)
        self.y_test = test_data['label'].values
        
        print(f"Autoencoder created:")
        print(f"  Total params: {self.model.count_params()}")
        print(f"  Model size: ~{self.model.count_params() * 4 / 1024:.2f} KB")
        self.model.summary()
    
    def get_parameters(self, config):
        """Return model weights"""
        return self.model.get_weights()
    
    def set_parameters(self, parameters):
        """Set model weights from server"""
        self.model.set_weights(parameters)
        print(f"  Applied {len(parameters)} weight matrices from server")
    
    def fit(self, parameters, config):
        """Train autoencoder on normal data"""
        self.current_round += 1
        start_time = time.time()
        
        print(f"\n--- Client {self.client_id} - Round {self.current_round} Training ---")
        
        # Apply global weights
        if parameters:
            self.set_parameters(parameters)
        
        # Train only on normal data (label=1)
        # Autoencoder learns to reconstruct normal patterns
        history = self.model.fit(
            self.X_train, self.X_train,  # Input = Output (reconstruction)
            epochs=10,
            batch_size=32,
            verbose=0,
            validation_split=0.1
        )
        
        training_time = time.time() - start_time
        final_loss = history.history['loss'][-1]
        
        # Calculate reconstruction error on training data
        train_reconstructions = self.model.predict(self.X_train, verbose=0)
        train_mse = np.mean(np.square(self.X_train - train_reconstructions), axis=1)
        
        # Set threshold at 95th percentile of training errors
        self.threshold = np.percentile(train_mse, 95)
        
        print(f"  Training completed in {training_time:.2f}s")
        print(f"  Final loss: {final_loss:.6f}")
        print(f"  Anomaly threshold: {self.threshold:.6f}")
        
        return self.get_parameters(config={}), len(self.X_train), {
            "training_time": training_time,
            "final_loss": float(final_loss),
            "threshold": float(self.threshold)
        }
    
    def evaluate(self, parameters, config):
        """Evaluate autoencoder on test data"""
        print(f"\n--- Client {self.client_id} - Round {self.current_round} Evaluation ---")
        
        if parameters:
            self.set_parameters(parameters)
        
        # If threshold not set yet (before first training), calculate a default one
        if self.threshold is None:
            print(f"  Threshold not set yet - calculating default threshold...")
            train_reconstructions = self.model.predict(self.X_train, verbose=0)
            train_mse = np.mean(np.square(self.X_train - train_reconstructions), axis=1)
            self.threshold = np.percentile(train_mse, 95)
            print(f"  Default threshold: {self.threshold:.6f}")
        
        # Get reconstructions
        test_reconstructions = self.model.predict(self.X_test_scaled, verbose=0)
        
        # Calculate reconstruction errors (MSE per sample)
        reconstruction_errors = np.mean(np.square(self.X_test_scaled - test_reconstructions), axis=1)
        
        # Predict: error > threshold = anomaly (-1), else normal (1)
        predictions = np.where(reconstruction_errors > self.threshold, -1, 1)
        
        # Debug info
        n_predicted_anomaly = np.sum(predictions == -1)
        print(f"  Reconstruction error range: [{reconstruction_errors.min():.6f}, {reconstruction_errors.max():.6f}]")
        print(f"  Threshold: {self.threshold:.6f}")
        print(f"  Predicted anomalies: {n_predicted_anomaly}/{len(predictions)}")
        
        # Convert to binary for metrics
        y_pred_binary = (predictions == -1).astype(int)
        y_true_binary = (self.y_test == -1).astype(int)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_true_binary, y_pred_binary, reconstruction_errors)
        
        # Loss = average reconstruction error
        loss = float(np.mean(reconstruction_errors))
        
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print(f"  Detection Rate: {metrics['recall']:.4f}")
        
        return loss, len(self.X_test_scaled), metrics
    
    def _calculate_metrics(self, y_true, y_pred, reconstruction_errors):
        """Calculate comprehensive metrics"""
        # ...existing code from client.py _calculate_metrics...
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        try:
            auc_roc = roc_auc_score(y_true, reconstruction_errors)
            auc_pr = average_precision_score(y_true, reconstruction_errors)
        except:
            auc_roc = 0.5
            auc_pr = 0.0
        
        mcc_num = (tp * tn) - (fp * fn)
        mcc_den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        mcc = mcc_num / mcc_den if mcc_den > 0 else 0
        
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "specificity": float(specificity),
            "auc_roc": float(auc_roc),
            "auc_pr": float(auc_pr),
            "mcc": float(mcc),
            "balanced_accuracy": float((recall + specificity) / 2),
            "detection_rate": float(recall),
            "false_alarm_rate": float(fpr),
            "false_negative_rate": float(fn / (fn + tp) if (fn + tp) > 0 else 0),
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
        }


def main():
    client = AutoencoderClient(client_id, X_train_scaled, y_train, scaler)
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=client.to_client()
    )


if __name__ == "__main__":
    main()
