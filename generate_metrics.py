"""
Generate comprehensive metrics and visualizations for research paper
Updated for SGDOneClassSVM with learning curve visualization
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import os

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def ensure_results_directory():
    """Ensure results directory exists"""
    Path("results").mkdir(parents=True, exist_ok=True)

def load_results():
    """Load the latest experiment results"""
    result_files = glob.glob("results/experiment_*.json")
    if not result_files:
        raise FileNotFoundError("No experiment results found in results/ directory")
    
    latest_file = max(result_files, key=os.path.getctime)
    print(f"Loading results from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    return pd.DataFrame(data['rounds']), data['experiment_id'], data

def generate_convergence_plot(df, experiment_id):
    """Generate convergence plot showing all metrics over rounds"""
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'auc_roc']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for metric in metrics:
        if metric in df.columns:
            ax.plot(df['round'], df[metric], marker='o', label=metric.replace('_', ' ').title())
    
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Convergence Over Federated Learning Rounds', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(f'results/convergence_plot_{experiment_id}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'results/convergence_plot_{experiment_id}.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Saved: convergence_plot")

def generate_learning_curve(df, experiment_id):
    """Generate learning curve showing training progress"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy and Loss
    ax1.plot(df['round'], df['accuracy'], 'b-o', label='Accuracy', linewidth=2)
    ax1.set_xlabel('Round', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12, color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_title('Learning Curve: Accuracy', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # F1 Score progression
    ax2.plot(df['round'], df['f1_score'], 'g-o', label='F1 Score', linewidth=2)
    ax2.plot(df['round'], df['precision'], 'r--^', label='Precision', linewidth=1.5)
    ax2.plot(df['round'], df['recall'], 'm--s', label='Recall', linewidth=1.5)
    ax2.set_xlabel('Round', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Learning Curve: Detection Metrics', fontsize=13, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results/learning_curve_{experiment_id}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'results/learning_curve_{experiment_id}.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Saved: learning_curve")

def generate_weight_evolution(full_data, experiment_id):
    """Generate weight evolution heatmap"""
    rounds_data = full_data.get('rounds', [])
    if not rounds_data:
        print("⚠ Warning: No rounds data available - skipping weight evolution plot")
        return
    
    # Check if any round has client contributions
    has_contributions = any('client_contributions' in rd for rd in rounds_data)
    if not has_contributions:
        print("⚠ Warning: No client contribution data available - skipping weight evolution plot")
        return
    
    # Extract client contributions
    client_ids = []
    contributions = []
    
    for round_data in rounds_data:
        if 'client_contributions' in round_data:
            contribs = round_data['client_contributions']
            if not client_ids:
                client_ids = list(contribs.keys())
            contributions.append([contribs.get(cid, 0) for cid in client_ids])
    
    if not contributions:
        print("⚠ Warning: No contributions data to plot")
        return
    
    contrib_matrix = np.array(contributions).T
    
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(contrib_matrix, aspect='auto', cmap='YlOrRd')
    
    ax.set_yticks(range(len(client_ids)))
    ax.set_yticklabels(client_ids)
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Client ID', fontsize=12)
    ax.set_title('Client Weight Contributions Evolution', fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Contribution Weight', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f'results/weight_evolution_{experiment_id}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'results/weight_evolution_{experiment_id}.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Saved: weight_evolution")

def generate_detection_metrics_plot(df, experiment_id):
    """Generate bar plot comparing detection metrics"""
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'auc_roc']
    first_round = df.iloc[0]
    last_round = df.iloc[-1]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width/2, [first_round[m] for m in metrics], width, 
                   label='Round 1', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, [last_round[m] for m in metrics], width,
                   label=f'Round {int(last_round["round"])}', color='#e74c3c', alpha=0.8)
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Detection Metrics: First vs Last Round', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'results/detection_metrics_{experiment_id}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'results/detection_metrics_{experiment_id}.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Saved: detection_metrics")

def generate_confusion_matrix_plot(df, experiment_id):
    """Generate confusion matrix visualization"""
    last_round = df.iloc[-1]
    
    if 'confusion_matrix' not in last_round or last_round['confusion_matrix'] is None:
        print("⚠ Warning: No confusion matrix data available - skipping confusion matrix plot")
        return
    
    cm = np.array(last_round['confusion_matrix'])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign', 'Attack'],
                yticklabels=['Benign', 'Attack'],
                cbar_kws={'label': 'Count'}, ax=ax)
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(f'Confusion Matrix - Round {int(last_round["round"])}', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'results/confusion_matrix_{experiment_id}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'results/confusion_matrix_{experiment_id}.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Saved: confusion_matrix")

def generate_metrics_comparison_table(df, experiment_id):
    """Generate metrics comparison table"""
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'auc_roc']
    
    first_round = df.iloc[0]
    last_round = df.iloc[-1]
    
    comparison_data = {
        'Metric': [m.replace('_', ' ').title() for m in metrics],
        'Round 1': [first_round[m] for m in metrics],
        f'Round {int(last_round["round"])}': [last_round[m] for m in metrics],
        'Improvement': [last_round[m] - first_round[m] for m in metrics],
        'Improvement %': [(last_round[m] - first_round[m]) * 100 for m in metrics]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save to CSV
    comparison_df.to_csv(f'results/metrics_comparison_{experiment_id}.csv', index=False)
    
    # Create table visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=comparison_df.round(4).values,
                    colLabels=comparison_df.columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Color header
    for i in range(len(comparison_df.columns)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Performance Metrics Comparison', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(f'results/metrics_table_{experiment_id}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'results/metrics_table_{experiment_id}.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Saved: metrics_table")
    
    return comparison_df

def generate_client_distribution_plot(experiment_id):
    """Generate client data distribution plot"""
    client_files = glob.glob("data/processed/client*_data.csv")
    
    if not client_files:
        print("⚠ Warning: No client data files found")
        return
    
    client_sizes = []
    client_names = []
    
    for file in sorted(client_files):
        try:
            df = pd.read_csv(file)
            client_sizes.append(len(df))
            client_names.append(Path(file).stem.replace('_data', '').replace('client', 'Client '))
        except Exception as e:
            print(f"⚠ Warning: Could not read {file}: {e}")
    
    if not client_sizes:
        print("⚠ Warning: No valid client data")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar plot
    bars = ax1.bar(client_names, client_sizes, color='#3498db', alpha=0.7)
    ax1.set_xlabel('Client', fontsize=12)
    ax1.set_ylabel('Number of Samples', fontsize=12)
    ax1.set_title('Data Distribution Across Clients', fontsize=13, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    # Pie chart
    ax2.pie(client_sizes, labels=client_names, autopct='%1.1f%%',
           startangle=90, colors=sns.color_palette("husl", len(client_sizes)))
    ax2.set_title('Client Data Distribution (%)', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'results/client_distribution_{experiment_id}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'results/client_distribution_{experiment_id}.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Saved: client_distribution")

def generate_all_round_comparison(df, experiment_id):
    """Generate radar chart comparing first and last round"""
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'auc_roc']
    
    first_round = df.iloc[0]
    last_round = df.iloc[-1]
    
    # Number of metrics
    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    first_values = [first_round[m] for m in metrics]
    first_values += first_values[:1]
    
    last_values = [last_round[m] for m in metrics]
    last_values += last_values[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    ax.plot(angles, first_values, 'o-', linewidth=2, label=f'Round 1', color='#3498db')
    ax.fill(angles, first_values, alpha=0.25, color='#3498db')
    
    ax.plot(angles, last_values, 's-', linewidth=2, label=f'Round {int(last_round["round"])}', color='#e74c3c')
    ax.fill(angles, last_values, alpha=0.25, color='#e74c3c')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title('Performance Improvement: First vs Last Round', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f'results/radar_comparison_{experiment_id}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'results/radar_comparison_{experiment_id}.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Saved: radar_comparison")

def main():
    print("=" * 60)
    print("Generating Paper Metrics and Visualizations")
    print("=" * 60)
    
    try:
        # Ensure results directory exists
        ensure_results_directory()
        
        # Load results
        df, experiment_id, full_data = load_results()
        print(f"\n✓ Loaded experiment: {experiment_id}")
        print(f"✓ Model type: {full_data.get('model_type', 'Unknown')}")
        print(f"✓ Number of rounds: {len(df)}")
        print(f"✓ Number of clients: {full_data.get('num_clients', 'Unknown')}")
        
        print("\n" + "-" * 60)
        print("Generating visualizations...")
        print("-" * 60)
        
        # Generate all plots
        generate_convergence_plot(df, experiment_id)
        generate_learning_curve(df, experiment_id)
        generate_weight_evolution(full_data, experiment_id)
        generate_detection_metrics_plot(df, experiment_id)
        generate_confusion_matrix_plot(df, experiment_id)
        summary_df = generate_metrics_comparison_table(df, experiment_id)
        generate_client_distribution_plot(experiment_id)
        generate_all_round_comparison(df, experiment_id)
        
        print("\n" + "=" * 60)
        print("FINAL METRICS SUMMARY")
        print("=" * 60)
        print(summary_df.to_string(index=False))
        
        # Print improvement summary
        print("\n" + "=" * 60)
        print("FEDERATED LEARNING IMPROVEMENT")
        print("=" * 60)
        
        first_acc = df.iloc[0]['accuracy']
        final_acc = df.iloc[-1]['accuracy']
        print(f"Accuracy:    {first_acc:.4f} → {final_acc:.4f} ({(final_acc-first_acc)*100:+.2f}%)")
        
        first_prec = df.iloc[0]['precision']
        final_prec = df.iloc[-1]['precision']
        print(f"Precision:   {first_prec:.4f} → {final_prec:.4f} ({(final_prec-first_prec)*100:+.2f}%)")
        
        first_rec = df.iloc[0]['recall']
        final_rec = df.iloc[-1]['recall']
        print(f"Recall:      {first_rec:.4f} → {final_rec:.4f} ({(final_rec-first_rec)*100:+.2f}%)")
        
        first_f1 = df.iloc[0]['f1_score']
        final_f1 = df.iloc[-1]['f1_score']
        print(f"F1-Score:    {first_f1:.4f} → {final_f1:.4f} ({(final_f1-first_f1)*100:+.2f}%)")
        
        first_auc = df.iloc[0]['auc_roc']
        final_auc = df.iloc[-1]['auc_roc']
        print(f"AUC-ROC:     {first_auc:.4f} → {final_auc:.4f} ({(final_auc-first_auc)*100:+.2f}%)")
        
        print("\n" + "=" * 60)
        print("✓ All visualizations generated successfully!")
        print(f"✓ Results saved to: results/")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease ensure you have run the federated learning training first.")
        print("Run: python server_autoencoder.py")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()