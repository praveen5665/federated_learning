"""
Federated Learning Server for Autoencoder-based IoT Anomaly Detection
"""

import flwr as fl
import json
import os
import numpy as np
from datetime import datetime
from typing import List, Tuple, Dict, Optional
from flwr.common import Metrics, Parameters, FitRes, EvaluateRes, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from quantization_utils import (
    quantize_weights_fp16, 
    dequantize_weights_fp16, 
    calculate_quantization_error,
    print_quantization_stats
)

os.makedirs("results", exist_ok=True)

# ...existing weighted_average function...
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    if not metrics:
        return {}
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {}
    all_keys = set()
    for _, m in metrics:
        if m:
            all_keys.update(m.keys())
    averaged_metrics = {}
    for key in all_keys:
        values = []
        weights = []
        for num_examples, m in metrics:
            if m and key in m:
                val = m[key]
                if isinstance(val, (int, float, np.number)) and not np.isnan(val):
                    values.append(float(val))
                    weights.append(num_examples)
        if values:
            averaged_metrics[key] = float(np.average(values, weights=weights))
    return averaged_metrics


class AutoencoderStrategy(fl.server.strategy.FedAvg):
    """FedAvg strategy optimized for Autoencoder with FP16 Quantization"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.round_results = []
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.global_threshold = None
        self.use_quantization = True  # Enable quantization
        self.quantization_stats = []  # Track quantization metrics
        print(f"Autoencoder Experiment ID: {self.experiment_id}")
        print(f"Quantization: FP16 (16-bit) Enabled")
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, float]]:
        """Aggregate neural network weights with quantization"""
        if not results:
            return None, {}
        
        # Extract weights from all clients (dequantize if needed)
        all_weights_list = []
        for _, fit_res in results:
            weights = parameters_to_ndarrays(fit_res.parameters)
            # Dequantize from FP16 to FP32 for aggregation
            if self.use_quantization:
                weights = dequantize_weights_fp16(weights)
            all_weights_list.append(weights)
        
        all_num_examples = [fit_res.num_examples for _, fit_res in results]
        
        # Weighted average (FedAvg) in FP32
        total_examples = sum(all_num_examples)
        aggregated_weights = []
        
        # Track client contributions
        client_contributions = {}
        for idx, (client_proxy, fit_res) in enumerate(results):
            client_id = f"client_{idx + 1}"
            contribution = fit_res.num_examples / total_examples
            client_contributions[client_id] = float(contribution)
        
        for layer_idx in range(len(all_weights_list[0])):
            layer_weights = [weights[layer_idx] * (num / total_examples) 
                           for weights, num in zip(all_weights_list, all_num_examples)]
            aggregated_weights.append(np.sum(layer_weights, axis=0))
        
        # Quantize aggregated weights to FP16
        if self.use_quantization:
            original_weights = aggregated_weights.copy()
            aggregated_weights, quant_stats = quantize_weights_fp16(aggregated_weights)
            
            # Calculate quantization error
            error_metrics = calculate_quantization_error(original_weights, aggregated_weights)
            quant_stats.update(error_metrics)
            self.quantization_stats.append(quant_stats)
            
            print_quantization_stats(quant_stats, error_metrics)
        
        # Calculate global threshold (weighted average)
        thresholds = [fit_res.metrics.get("threshold", 0) for _, fit_res in results]
        weights = [fit_res.num_examples for _, fit_res in results]
        self.global_threshold = float(np.average(thresholds, weights=weights))
        
        # Stats
        avg_loss = np.mean([fit_res.metrics.get("final_loss", 0) for _, fit_res in results])
        avg_epochs = np.mean([fit_res.metrics.get("epochs_trained", 10) for _, fit_res in results])
        
        print(f"\n{'='*60}")
        print(f"Round {server_round} - Autoencoder Aggregation")
        print(f"  Clients: {len(results)}")
        print(f"  Avg training loss: {avg_loss:.6f}")
        print(f"  Global threshold: {self.global_threshold:.6f}")
        print(f"  Avg epochs trained: {avg_epochs:.1f}")
        print(f"  Weight matrices: {len(aggregated_weights)}")
        if self.use_quantization:
            print(f"  Model size (FP16): {quant_stats['quantized_size_kb']:.2f} KB")
            print(f"  Compression: {quant_stats['compression_ratio']:.2f}x")
        print(f"{'='*60}")
        
        # Store client contributions for this round
        self.current_round_contributions = client_contributions
        
        return ndarrays_to_parameters(aggregated_weights), {
            "num_clients": len(results),
            "avg_loss": float(avg_loss),
            "global_threshold": self.global_threshold,
            "quantized": self.use_quantization
        }
    
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ):
        """Configure the next round of training"""
        config = {}
        
        # Send global threshold to clients after first round
        if self.global_threshold is not None:
            config["global_threshold"] = self.global_threshold
        
        # Call parent's configure_fit
        return super().configure_fit(server_round, parameters, client_manager)
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, float]]:
        """Aggregate evaluation results"""
        if not results:
            return None, {}
        
        total_examples = sum(res.num_examples for _, res in results)
        weighted_loss = sum(res.loss * res.num_examples for _, res in results) / total_examples
        
        client_metrics = [(res.num_examples, res.metrics) for _, res in results if res.metrics]
        aggregated_metrics = weighted_average(client_metrics)
        
        # Calculate aggregated confusion matrix
        total_tp = sum(res.metrics.get('true_positives', 0) for _, res in results if res.metrics)
        total_tn = sum(res.metrics.get('true_negatives', 0) for _, res in results if res.metrics)
        total_fp = sum(res.metrics.get('false_positives', 0) for _, res in results if res.metrics)
        total_fn = sum(res.metrics.get('false_negatives', 0) for _, res in results if res.metrics)
        
        confusion_matrix_data = [[int(total_tn), int(total_fp)], [int(total_fn), int(total_tp)]]
        
        result_entry = {
            "round": server_round,
            "timestamp": datetime.now().isoformat(),
            "num_clients": len(results),
            "total_test_samples": total_examples,
            "loss": float(weighted_loss),
            "confusion_matrix": confusion_matrix_data,
            "client_contributions": getattr(self, 'current_round_contributions', {}),
            "quantization_stats": self.quantization_stats[-1] if self.quantization_stats else {},
            **{k: float(v) if isinstance(v, (int, float, np.number)) else v 
               for k, v in aggregated_metrics.items()}
        }
        
        self.round_results.append(result_entry)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Round {server_round} - Evaluation")
        print(f"  Accuracy: {aggregated_metrics.get('accuracy', 0):.4f}")
        print(f"  F1-Score: {aggregated_metrics.get('f1_score', 0):.4f}")
        print(f"  Detection Rate: {aggregated_metrics.get('detection_rate', 0):.4f}")
        print(f"  AUC-ROC: {aggregated_metrics.get('auc_roc', 0):.4f}")
        print(f"{'='*60}\n")
        
        # Save results
        results_data = {
            "experiment_id": self.experiment_id,
            "model_type": "Autoencoder_FP16_Quantized",
            "quantization_enabled": self.use_quantization,
            "quantization_type": "FP16_Post_Training",
            "rounds": self.round_results
        }
        with open(f"results/experiment_autoencoder_{self.experiment_id}.json", 'w') as f:
            json.dump(results_data, f, indent=2)
        
        return weighted_loss, aggregated_metrics


def main():
    import sys
    num_rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    
    print(f"\n{'='*60}")
    print("AUTOENCODER FEDERATED LEARNING SERVER")
    print(f"  Model: Lightweight Neural Network (~2K params)")
    print(f"  Rounds: {num_rounds}")
    print(f"{'='*60}\n")
    
    strategy = AutoencoderStrategy(
        min_fit_clients=2,
        min_available_clients=2,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
