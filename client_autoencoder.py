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
from quantization_utils import quantize_weights_fp16, dequantize_weights_fp16

warnings.filterwarnings('ignore')

# Disable GPU if not available (for IoT devices)
tf.config.set_visible_devices([], 'GPU')

# Load selected features
selected_features = pd.read_csv("data/processed/selected_features.csv")['selected_features'].tolist()

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
    Create lightweight autoencoder for IoT devices with regularization
    Total params: ~2000 (very small!)
    """
    encoder_input = keras.Input(shape=(input_dim,))
    encoded = keras.layers.Dense(12, activation='relu', name='encoder_1')(encoder_input)
    encoded = keras.layers.Dropout(0.2)(encoded)
    encoded = keras.layers.Dense(6, activation='relu', name='bottleneck')(encoded)
    
    decoded = keras.layers.Dense(12, activation='relu', name='decoder_1')(encoded)
    decoded = keras.layers.Dropout(0.2)(decoded)
    decoded = keras.layers.Dense(input_dim, activation='linear', name='decoder_output')(decoded)
    
    autoencoder = keras.Model(encoder_input, decoded, name='autoencoder')
    
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse'
    )
    
    return autoencoder


class AutoencoderClient(fl.client.NumPyClient):
    """Flower client with Autoencoder for anomaly detection with FP16 quantization"""
    
    def __init__(self, client_id, X_train, y_train, scaler):
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.scaler = scaler
        self.model = create_autoencoder(input_dim=X_train.shape[1])
        self.threshold = None
        self.current_round = 0
        self.global_threshold = None
        self.use_quantization = True
        
        # Load test data
        test_data = pd.read_csv("data/processed/test_data.csv")
        self.X_test = test_data[selected_features].values.astype(np.float32)
        self.X_test_scaled = scaler.transform(self.X_test)
        self.y_test = test_data['label'].values
        
        print(f"Autoencoder created:")
        print(f"  Total params: {self.model.count_params()}")
        original_size = self.model.count_params() * 4 / 1024
        quantized_size = self.model.count_params() * 2 / 1024
        print(f"  Model size (FP32): ~{original_size:.2f} KB")
        print(f"  Model size (FP16): ~{quantized_size:.2f} KB")
        print(f"  Compression: {original_size/quantized_size:.2f}x")
        self.model.summary()
    
    def get_parameters(self, config):
        """Return model weights (quantized to FP16)"""
        weights = self.model.get_weights()
        if self.use_quantization:
            weights, _ = quantize_weights_fp16(weights)
        return weights
    
    def set_parameters(self, parameters):
        """Set model weights from server (dequantize from FP16)"""
        if self.use_quantization:
            parameters = dequantize_weights_fp16(parameters)
        self.model.set_weights(parameters)
        print(f"  Applied {len(parameters)} weight matrices from server (FP16 -> FP32)")
    
    def fit(self, parameters, config):
        """Train autoencoder on normal data"""
        self.current_round += 1
        start_time = time.time()
        
        print(f"\n--- Client {self.client_id} - Round {self.current_round} Training ---")
        
        if parameters:
            self.set_parameters(parameters)
        
        if 'global_threshold' in config:
            self.global_threshold = config['global_threshold']
            print(f"  Using global threshold: {self.global_threshold:.6f}")
        
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            min_delta=0.0001
        )
        
        history = self.model.fit(
            self.X_train, self.X_train,
            epochs=20,
            batch_size=32,
            verbose=0,
            validation_split=0.1,
            callbacks=[early_stopping]
        )
        
        training_time = time.time() - start_time
        final_loss = history.history['loss'][-1]
        
        train_reconstructions = self.model.predict(self.X_train, verbose=0)
        train_mse = np.mean(np.square(self.X_train - train_reconstructions), axis=1)
        
        local_threshold = np.percentile(train_mse, 95)
        
        if self.global_threshold is not None:
            self.threshold = self.global_threshold
            print(f"  Using global threshold: {self.threshold:.6f}")
        else:
            self.threshold = local_threshold
            print(f"  Using local threshold: {self.threshold:.6f}")
        
        print(f"  Training completed in {training_time:.2f}s")
        print(f"  Final loss: {final_loss:.6f}")
        print(f"  Epochs trained: {len(history.history['loss'])}")
        
        # Sends client_id in both fit() and evaluate()
        return self.get_parameters(config={}), len(self.X_train), {
            "client_id": self.client_id,  # ✅ ADDED MISSING COMMA HERE
            "training_time": training_time,
            "final_loss": float(final_loss),
            "threshold": float(local_threshold),
            "epochs_trained": len(history.history['loss'])
        }
    
    def evaluate(self, parameters, config):
        """Evaluate autoencoder on test data"""
        print(f"\n--- Client {self.client_id} - Round {self.current_round} Evaluation ---")
        
        if parameters:
            self.set_parameters(parameters)
        
        if self.threshold is None:
            print(f"  Threshold not set yet - calculating default threshold...")
            train_reconstructions = self.model.predict(self.X_train, verbose=0)
            train_mse = np.mean(np.square(self.X_train - train_reconstructions), axis=1)
            self.threshold = np.percentile(train_mse, 95)
            print(f"  Default threshold: {self.threshold:.6f}")
        
        test_reconstructions = self.model.predict(self.X_test_scaled, verbose=0)
        reconstruction_errors = np.mean(np.square(self.X_test_scaled - test_reconstructions), axis=1)
        
        predictions = np.where(reconstruction_errors > self.threshold, -1, 1)
        
        n_predicted_anomaly = np.sum(predictions == -1)
        print(f"  Reconstruction error range: [{reconstruction_errors.min():.6f}, {reconstruction_errors.max():.6f}]")
        print(f"  Threshold: {self.threshold:.6f}")
        print(f"  Predicted anomalies: {n_predicted_anomaly}/{len(predictions)}")
        
        y_pred_binary = (predictions == -1).astype(int)
        y_true_binary = (self.y_test == -1).astype(int)
        
        metrics = self._calculate_metrics(y_true_binary, y_pred_binary, reconstruction_errors)
        
        # ✅ FIX: Send client_id in evaluation metrics too
        metrics["client_id"] = self.client_id  # ✅ int
        
        loss = float(np.mean(reconstruction_errors))
        
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print(f"  Detection Rate: {metrics['recall']:.4f}")
        
        return loss, len(self.X_test_scaled), metrics
    
    def _calculate_metrics(self, y_true, y_pred, reconstruction_errors):
        """Calculate comprehensive metrics"""
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
