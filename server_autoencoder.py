"""
Federated Learning Server for Autoencoder-based IoT Anomaly Detection
With proper client identification and synchronization
"""

import flwr as fl
import json
import os
import numpy as np
import time
from datetime import datetime
from typing import List, Tuple, Dict, Optional
from flwr.common import Metrics, Parameters, FitRes, EvaluateRes, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from quantization_utils import (
    quantize_weights_fp16, 
    dequantize_weights_fp16, 
    calculate_quantization_error,
    print_quantization_stats
)
from dynamic_weighting import DynamicWeightCalculator, print_dynamic_weights_summary

os.makedirs("results", exist_ok=True)


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
    """FedAvg strategy with FP16 Quantization, Dynamic Weighting, and Client Synchronization"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.round_results = []
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.global_threshold = None
        self.use_quantization = True
        self.quantization_stats = []
        
        # Dynamic weighting setup
        self.use_dynamic_weights = True
        self.weight_calculator = DynamicWeightCalculator(
            alpha=0.4,   # 40% weight on accuracy
            beta=0.3,    # 30% weight on F1-score
            gamma=0.2,   # 20% weight on loss
            delta=0.1,   # 10% weight on AUC-ROC
            min_weight=0.05,
            smoothing=0.7
        )
        self.client_performance = {}  # Track performance per client
        self.weight_history = []  # Track weight evolution
        self.known_clients = set()  # Track all clients ever seen
        
        print(f"Autoencoder Experiment ID: {self.experiment_id}")
        print(f"Quantization: FP16 (16-bit) Enabled")
        print(f"Dynamic Weighting: Hybrid (Accuracy:40%, F1:30%, Loss:20%, AUC:10%)")
        print(f"Client Synchronization: 20 second wait before first round")
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, float]]:
        """Aggregate neural network weights with proper client identification"""
        if not results:
            return None, {}
        
        # ✅ FIX: Extract client IDs from metrics (not from order!)
        all_weights_list = []
        all_num_examples = []
        client_ids = []
        
        for client_proxy, fit_res in results:
            weights = parameters_to_ndarrays(fit_res.parameters)
            if self.use_quantization:
                weights = dequantize_weights_fp16(weights)
            all_weights_list.append(weights)
            all_num_examples.append(fit_res.num_examples)
            
            # Read client_id with validation
            client_id = fit_res.metrics.get("client_id")
            if client_id is None:
                # Fallback with warning
                client_id = len(client_ids) + 1
                print(f"⚠️  WARNING: Client didn't send ID, assigning temporary ID: {client_id}")
            
            client_id = f"client_{client_id}"
            client_ids.append(client_id)
            self.known_clients.add(client_id)
        
        print(f"\n{'='*60}")
        print(f"Round {server_round} - Participating Clients: {sorted(client_ids)}")
        print(f"  Total clients ever seen: {len(self.known_clients)}")
        print(f"{'='*60}")
        
        # Calculate dynamic weights if enabled and past first round
        if self.use_dynamic_weights and server_round > 1 and self.client_performance:
            client_results = []
            for idx, client_id in enumerate(client_ids):
                if client_id in self.client_performance:
                    metrics = self.client_performance[client_id]
                    client_results.append((client_id, all_num_examples[idx], metrics))
                else:
                    # New client joining - use default weight
                    print(f"  ⚠ New client {client_id} detected - using default weight")
            
            if client_results:
                dynamic_weights_dict = self.weight_calculator.calculate_dynamic_weights(client_results)
                
                # Assign weights (including default for new clients)
                dynamic_weights = []
                for idx, cid in enumerate(client_ids):
                    if cid in dynamic_weights_dict:
                        dynamic_weights.append(dynamic_weights_dict[cid])
                    else:
                        # New client gets average weight
                        avg_weight = 1.0 / len(client_ids)
                        dynamic_weights.append(avg_weight)
                        print(f"  → {cid}: {avg_weight:.4f} (new client - default weight)")
                
                rationales = []
                for client_id in client_ids:
                    if client_id in self.client_performance:
                        rationale = self.weight_calculator.get_weight_rationale(
                            client_id,
                            self.client_performance[client_id],
                            dynamic_weights_dict.get(client_id, 0)
                        )
                        rationales.append(rationale)
                
                print_dynamic_weights_summary(dynamic_weights_dict, rationales, server_round)
                
                self.weight_history.append({
                    "round": server_round,
                    "weights": dynamic_weights_dict,
                    "rationales": rationales
                })
            else:
                # All clients are new - use FedAvg
                total_examples = sum(all_num_examples)
                dynamic_weights = [n / total_examples for n in all_num_examples]
        else:
            # Use standard FedAvg for first round
            total_examples = sum(all_num_examples)
            dynamic_weights = [n / total_examples for n in all_num_examples]
            print(f"  Using FedAvg (Round {server_round}): Data-weighted averaging")
        
        # Aggregate with dynamic weights
        aggregated_weights = []
        for layer_idx in range(len(all_weights_list[0])):
            layer_weights = [weights[layer_idx] * weight 
                           for weights, weight in zip(all_weights_list, dynamic_weights)]
            aggregated_weights.append(np.sum(layer_weights, axis=0))
        
        # Quantize aggregated weights to FP16
        if self.use_quantization:
            original_weights = aggregated_weights.copy()
            aggregated_weights, quant_stats = quantize_weights_fp16(aggregated_weights)
            error_metrics = calculate_quantization_error(original_weights, aggregated_weights)
            quant_stats.update(error_metrics)
            self.quantization_stats.append(quant_stats)
            print_quantization_stats(quant_stats, error_metrics)
        
        # Calculate global threshold (weighted average)
        thresholds = [fit_res.metrics.get("threshold", 0) for _, fit_res in results]
        weights_for_threshold = [fit_res.num_examples for _, fit_res in results]
        self.global_threshold = float(np.average(thresholds, weights=weights_for_threshold))
        
        # Stats
        avg_loss = np.mean([fit_res.metrics.get("final_loss", 0) for _, fit_res in results])
        avg_epochs = np.mean([fit_res.metrics.get("epochs_trained", 10) for _, fit_res in results])
        
        print(f"\n{'='*60}")
        print(f"Round {server_round} - Autoencoder Aggregation")
        print(f"  Clients: {len(results)}")
        print(f"  Weighting: {'Dynamic (Performance-Based)' if self.use_dynamic_weights and server_round > 1 else 'FedAvg (Data Size)'}")
        print(f"  Avg training loss: {avg_loss:.6f}")
        print(f"  Global threshold: {self.global_threshold:.6f}")
        print(f"  Avg epochs trained: {avg_epochs:.1f}")
        print(f"  Weight matrices: {len(aggregated_weights)}")
        if self.use_quantization:
            print(f"  Model size (FP16): {quant_stats['quantized_size_kb']:.2f} KB")
            print(f"  Compression: {quant_stats['compression_ratio']:.2f}x")
        print(f"{'='*60}")
        
        # Store client contributions
        client_contributions = {client_ids[i]: float(dynamic_weights[i]) 
                              for i in range(len(client_ids))}
        self.current_round_contributions = client_contributions
        
        return ndarrays_to_parameters(aggregated_weights), {
            "num_clients": len(results),
            "avg_loss": float(avg_loss),
            "global_threshold": self.global_threshold,
            "quantized": self.use_quantization,
            "dynamic_weights": self.use_dynamic_weights
        }
    
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ):
        """Configure the next round of training"""
        config = {}
        
        # Send global threshold to clients after first round
        if self.global_threshold is not None:
            config["global_threshold"] = self.global_threshold
        
        # Get configuration from parent
        fit_config = super().configure_fit(server_round, parameters, client_manager)
        
        # Return list of (ClientProxy, FitIns) tuples
        return [(client, fl.common.FitIns(parameters, config)) for client, _ in fit_config] if fit_config else []
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, float]]:
        """Aggregate evaluation results and update client performance tracking"""
        if not results:
            return None, {}
        
        # ✅ FIX: Store individual client performance using actual client_id
        for client_proxy, eval_res in results:
            # Read client_id from metrics
            client_id = eval_res.metrics.get("client_id") if eval_res.metrics else None
            if client_id is not None:
                client_id = f"client_{client_id}"
            else:
                client_id = f"client_{getattr(client_proxy, 'cid', 'unknown')}"
            
            if eval_res.metrics:
                self.client_performance[client_id] = {
                    'accuracy': eval_res.metrics.get('accuracy', 0),
                    'f1_score': eval_res.metrics.get('f1_score', 0),
                    'loss': eval_res.loss,
                    'auc_roc': eval_res.metrics.get('auc_roc', 0)
                }
        
        # Aggregate evaluation results
        total_examples = sum(res.num_examples for _, res in results)
        weighted_loss = sum(res.loss * res.num_examples for _, res in results) / total_examples
        
        client_metrics = [(res.num_examples, res.metrics) for _, res in results if res.metrics]
        aggregated_metrics = weighted_average(client_metrics)
        
        # Calculate aggregated confusion matrix - FIXED
        total_tp = sum(res.metrics.get('true_positives', 0) for _, res in results if res.metrics)
        total_tn = sum(res.metrics.get('true_negatives', 0) for _, res in results if res.metrics)  # ✅ FIXED
        total_fp = sum(res.metrics.get('false_positives', 0) for _, res in results if res.metrics)  # ✅ FIXED
        total_fn = sum(res.metrics.get('false_negatives', 0) for _, res in results if res.metrics)  # ✅ FIXED
        
        confusion_matrix_data = [[int(total_tn), int(total_fp)], [int(total_fn), int(total_tp)]]
        
        weight_info = {}
        if self.weight_history:
            weight_info = self.weight_history[-1]
        
        result_entry = {
            "round": server_round,
            "timestamp": datetime.now().isoformat(),
            "num_clients": len(results),
            "total_test_samples": total_examples,
            "loss": float(weighted_loss),
            "confusion_matrix": confusion_matrix_data,
            "client_contributions": getattr(self, 'current_round_contributions', {}),
            "dynamic_weight_info": weight_info,
            "quantization_stats": self.quantization_stats[-1] if self.quantization_stats else {},
            **{k: float(v) if isinstance(v, (int, float, np.number)) else v 
               for k, v in aggregated_metrics.items()}
        }
        
        self.round_results.append(result_entry)
        
        # Print summary - FIXED SYNTAX ERROR
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
            "model_type": "Autoencoder_FP16_DynamicWeights",
            "quantization_enabled": self.use_quantization,
            "quantization_type": "FP16_Post_Training",
            "dynamic_weighting_enabled": self.use_dynamic_weights,
            "weighting_params": {
                "alpha": self.weight_calculator.alpha,
                "beta": self.weight_calculator.beta,
                "gamma": self.weight_calculator.gamma,
                "delta": self.weight_calculator.delta
            },
            "num_clients": len(self.known_clients),
            "rounds": self.round_results,
            "weight_evolution": self.weight_history
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
    
    # REMOVED: 20 second wait - let clients connect naturally
    # Flower server starts immediately and waits for clients
    
    strategy = AutoencoderStrategy(
        min_fit_clients=2,
        min_available_clients=2,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    
    print("[OK] Starting Federated Learning Server...")
    print("     Waiting for clients to connect...\n")
    
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
