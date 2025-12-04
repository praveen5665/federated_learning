"""
Data Preprocessing Script for CIC-IoT-2023 Dataset
This script prepares the dataset for One-Class SVM (anomaly detection).
Training: Only normal/benign data (outliers will be classified as attacks)
Test: Both normal and attack data for evaluation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import glob

# Create directories if they don't exist
os.makedirs("data/processed", exist_ok=True)

print("Loading dataset...")
# Find all CSV files in the raw folder
raw_folder = "data/raw"
csv_files = glob.glob(os.path.join(raw_folder, "*.csv"))
csv_files.sort()  # Sort for consistent ordering

if len(csv_files) == 0:
    raise FileNotFoundError(f"No CSV files found in {raw_folder}")

print(f"Found {len(csv_files)} CSV file(s) in {raw_folder}")

# Load all CSV files and combine them
dataframes = []
for i, csv_file in enumerate(csv_files, 1):
    filename = os.path.basename(csv_file)
    print(f"Reading from {filename}...")
    df = pd.read_csv(csv_file)
    print(f"  Loaded {len(df)} samples from file {i}")
    dataframes.append(df)

# Combine all datasets
print("Combining all datasets...")
df_all = pd.concat(dataframes, ignore_index=True)
print(f"Total samples after combining: {len(df_all)}")

print("Separating benign and attack data...")
# Separate BenignTraffic from all attack types
df_all_benign = df_all[df_all['label'] == 'BenignTraffic'].copy().reset_index(drop=True)
df_all_attack = df_all[df_all['label'] != 'BenignTraffic'].copy().reset_index(drop=True)

print(f"Found {len(df_all_benign)} benign samples and {len(df_all_attack)} attack samples")

# For One-Class SVM: Training uses ONLY normal data, Test uses both normal and attack
print("Preparing data for One-Class SVM (training on normal data only)...")

# For test set: 10K benign + 10K attack = 20K total
n_benign_test = min(10000, len(df_all_benign))  # 10K benign for testing
n_attack_test = min(10000, len(df_all_attack))  # 10K attack for testing

# Sample benign test data
df_benign_test = df_all_benign.sample(n=n_benign_test, random_state=43).reset_index(drop=True)

# Sample attack test data
df_attack_test = df_all_attack.sample(n=n_attack_test, random_state=42).reset_index(drop=True)

# Use remaining benign samples for training (exclude test samples)
test_benign_indices = df_benign_test.index
df_benign_train = df_all_benign.drop(df_all_benign.sample(n=n_benign_test, random_state=43).index).reset_index(drop=True)

print(f"Reserved {n_benign_test} benign samples for test evaluation")
print(f"Reserved {n_attack_test} attack samples for test evaluation")

# Convert labels: BenignTraffic -> 1 (Normal), Attacks -> -1 (Anomaly)
df_benign_train['label'] = 1  # 1 = Normal
df_benign_test['label'] = 1   # 1 = Normal
df_attack_test['label'] = -1  # -1 = Anomaly

print(f"Training: {len(df_benign_train)} benign samples (ONLY normal data for One-Class SVM)")
print(f"Test: {len(df_benign_test)} benign + {len(df_attack_test)} attack = {len(df_benign_test) + len(df_attack_test)} total samples")

print("Selecting features...")
# Select the 46 feature columns (excluding the 'label' column)
# The dataset has 47 columns total: 46 features + 1 label
all_cols = df_benign_train.columns.tolist()
cols = [col for col in all_cols if col != 'label']

# Verify we have 46 features
if len(cols) != 46:
    print(f"Warning: Expected 46 features, found {len(cols)}. Using all available features.")
    # If we have more than 46, take the first 46
    if len(cols) > 46:
        cols = cols[:46]
        print(f"Using first 46 features: {cols[:5]}... (showing first 5)")

print(f"Initial features available: {len(cols)}")
print(f"Feature names: {cols[:5]}... (showing first 5)")

# Create test set for feature selection (mixed normal + attack)
print("\n" + "="*50)
print("FEATURE SELECTION using Random Forest")
print("="*50)
print("Preparing mixed test data for feature selection...")
if len(df_benign_test) > 0:
    test_mixed_for_selection = pd.concat([df_benign_test, df_attack_test], ignore_index=True)
else:
    # If no benign test samples, use only attack samples
    test_mixed_for_selection = df_attack_test.copy()

# Prepare data for Random Forest feature selection
X_mixed = test_mixed_for_selection[cols].copy()
X_mixed = X_mixed.fillna(0)
y_mixed = test_mixed_for_selection['label'].copy()

print(f"Using mixed test data with {len(X_mixed)} samples for feature selection")
print(f"  - Normal (label=1): {len(y_mixed[y_mixed==1])}")
print(f"  - Anomaly (label=-1): {len(y_mixed[y_mixed==-1])}")

# Train Random Forest for feature selection
print("\nTraining Random Forest for feature importance analysis...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_mixed, y_mixed)

# Get feature importances
importances = rf.feature_importances_
feature_names = X_mixed.columns

# Create a DataFrame to store results
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Select top 20 features
n_selected_features = 20
selected_features = feature_importance_df.head(n_selected_features)['Feature'].tolist()

print(f"\n--- TOP {n_selected_features} SELECTED FEATURES ---")
print(feature_importance_df.head(n_selected_features))

# Save feature importance results
print("\nSaving feature importance results...")
feature_importance_df.to_csv("data/processed/feature_importance_all.csv", index=False)
pd.DataFrame(selected_features, columns=['selected_features']).to_csv("data/processed/selected_features.csv", index=False)
print(f"Selected {n_selected_features} features saved to data/processed/selected_features.csv")

# Visualize top features
print("\nCreating feature importance visualization...")
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(n_selected_features))
plt.title(f'Top {n_selected_features} Most Important Features for IoT Intrusion Detection')
plt.xlabel('Importance Score')
plt.ylabel('Feature Name')
plt.tight_layout()
plt.savefig("feature_importance_graph.png", dpi=300, bbox_inches='tight')
print("Feature importance graph saved to 'feature_importance_graph.png'")

# Update cols to use only selected features
cols = selected_features
print(f"\nUsing {len(cols)} selected features for preprocessing")

# For One-Class SVM: Training uses ONLY normal/benign data
print("\nPreparing training data (normal data only)...")
df_train = df_benign_train.copy()  # Only benign data for training

# Extract features using selected features only
X_train = df_train[cols].copy()

# Handle any missing values
X_train = X_train.fillna(0)

print("Normalizing features (fitted on normal data only)...")
# Normalize features - scaler fitted ONLY on normal data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Save the scaler for later use in the live demo!
print("Saving scaler...")
joblib.dump(scaler, 'scaler.pkl')
print("Scaler saved to scaler.pkl")

# Create processed training data with labels (all should be 1 = Normal)
train_data = pd.DataFrame(X_train_scaled, columns=cols)
train_data['label'] = df_train['label'].values

# Save processed training CSV
print("Saving processed training data (normal data only)...")
train_data.to_csv("data/processed/train_data.csv", index=False)
print(f"Training data saved: {len(train_data)} normal samples")

# Create test set (already prepared above)
print("Preparing test set...")
if len(df_benign_test) > 0:
    test_mixed = pd.concat([df_benign_test, df_attack_test], ignore_index=True)
else:
    # If no benign test samples, use only attack samples
    test_mixed = df_attack_test.copy()

# Extract and normalize test features
X_test = test_mixed[cols].copy()
X_test = X_test.fillna(0)
X_test_scaled = scaler.transform(X_test)

# Create processed test data with labels
test_data = pd.DataFrame(X_test_scaled, columns=cols)
test_data['label'] = test_mixed['label'].values

# Save processed test CSV
print("Saving processed test data...")
test_data.to_csv("data/processed/test_data.csv", index=False)
print(f"Test data saved: {len(test_data)} samples")

print("\n" + "="*50)
print("Preprocessing completed successfully for One-Class SVM!")
print("="*50)
print(f"Training samples: {len(train_data)} (ONLY normal data)")
print(f"  - Normal (label=1): {len(train_data[train_data['label']==1])}")
print(f"  - Anomaly (label=-1): {len(train_data[train_data['label']==-1])}")
print(f"Test samples: {len(test_data)} (normal + attack)")
print(f"  - Normal (label=1): {len(test_data[test_data['label']==1])}")
print(f"  - Anomaly (label=-1): {len(test_data[test_data['label']==-1])}")
print(f"Features used: {len(cols)} (selected from Random Forest)")
print(f"Selected features: {cols[:5]}... (showing first 5)")
print(f"Scaler saved: scaler.pkl (fitted on normal data only)")
print(f"Feature importance saved: data/processed/feature_importance_all.csv")
print(f"Selected features list saved: data/processed/selected_features.csv")
print("="*50)
print("Note: One-Class SVM will be trained on normal data only.")
print("      Outliers detected during testing will be classified as attacks.")
print("      Feature selection was performed using Random Forest on mixed test data.")
print("="*50)

