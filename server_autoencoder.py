# -*- coding: utf-8 -*-
"""
Federated Learning Server for Autoencoder-based IoT Anomaly Detection
With proper client identification and synchronization
"""

import sys
import io
# Force UTF-8 encoding for Windows console
if sys.platform == 'win32': 
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import flwr as fl
import json
import os
import numpy as np
import time
from datetime import datetime
from typing import List, Tuple, Dict, Optional
from flwr.common import (
    Metrics, Parameters, FitRes, EvaluateRes, FitIns,
    ndarrays_to_parameters, parameters_to_ndarrays
)
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
            alpha=0.4,
            beta=0.3,
            gamma=0.2,
            delta=0.1,
            min_weight=0.05,
            smoothing=0.7
        )
        self.client_performance = {}
        self.weight_history = []
        self.known_clients = set()

        # -------- FedBuff state --------
        self.fedbuff_enabled = True
        self.fedbuff_buffer_size = int(os.getenv("FEDBUFF_K", "5"))
        self.fedbuff_min_updates = int(os.getenv("FEDBUFF_MIN_UPDATES", "2"))
        self.fedbuff_staleness_alpha = float(os.getenv("FEDBUFF_STALENESS_ALPHA", "0.5"))
        self.fedbuff_server_lr = float(os.getenv("FEDBUFF_ETA", "0.8"))

        self.update_buffer = []  # queued client updates
        self.global_version = 0  # server model version
        self.current_global_weights = None
        self.current_round_contributions = {}
        # -------------------------------

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ):
        """Configure the next round of training and inject server version/config."""
        client_instructions = super().configure_fit(server_round, parameters, client_manager)

        fit_config = {"server_version": self.global_version}
        if self.global_threshold is not None:
            fit_config["global_threshold"] = float(self.global_threshold)

        updated_instructions = []
        for client_proxy, fit_ins in client_instructions:
            merged = dict(fit_ins.config) if fit_ins.config is not None else {}
            merged.update(fit_config)
            updated_instructions.append((client_proxy, FitIns(fit_ins.parameters, merged)))

        return updated_instructions

    def _print_async_status(self, version: int, buffered: int, applied: int, avg_staleness: float) -> None:
        """Emit a parseable async status line for dashboard."""
        fedbuff_flag = 1 if getattr(self, "fedbuff_enabled", False) else 0
        print(
            f"[ASYNC] fedbuff={fedbuff_flag} "
            f"version={version} buffered={buffered} applied={applied} "
            f"avg_staleness={avg_staleness:.4f}"
        )

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, float]]:
        """FedBuff-style buffered aggregation with optional dynamic + staleness weights."""
        if not results:
            if self.current_global_weights is None:
                return None, {}
            if self.use_quantization:
                out_weights, _ = quantize_weights_fp16(self.current_global_weights)
            else:
                out_weights = self.current_global_weights
            self._print_async_status(
                version=int(getattr(self, "global_version", 0)),
                buffered=int(len(getattr(self, "update_buffer", []))),
                applied=0,
                avg_staleness=0.0,
            )
            return ndarrays_to_parameters(out_weights), {
                "buffered_updates": float(len(self.update_buffer)),
                "applied_updates": 0.0,
                "global_version": float(self.global_version),
            }

        # 1) Push incoming updates into buffer
        for client_proxy, fit_res in results:
            weights = parameters_to_ndarrays(fit_res.parameters)
            if self.use_quantization:
                weights = dequantize_weights_fp16(weights)

            raw_client_id = fit_res.metrics.get("client_id") if fit_res.metrics else None
            if raw_client_id is None:
                raw_client_id = getattr(client_proxy, "cid", "unknown")
            client_id = f"client_{raw_client_id}"
            self.known_clients.add(client_id)

            base_version = int((fit_res.metrics or {}).get("base_version", self.global_version))

            self.update_buffer.append({
                "client_id": client_id,
                "weights": weights,
                "num_examples": int(fit_res.num_examples),
                "base_version": base_version,
            })

        # 2) Consume up to K buffered updates
        consume_n = min(len(self.update_buffer), self.fedbuff_buffer_size)

        # CASE: buffer not yet ready to apply
        if consume_n < self.fedbuff_min_updates and self.current_global_weights is not None:
            if self.use_quantization:
                out_weights, _ = quantize_weights_fp16(self.current_global_weights)
            else:
                out_weights = self.current_global_weights
            self._print_async_status(
                version=int(self.global_version),
                buffered=int(len(self.update_buffer)),
                applied=0,
                avg_staleness=0.0,
            )
            return ndarrays_to_parameters(out_weights), {
                "buffered_updates": float(len(self.update_buffer)),
                "applied_updates": 0.0,
                "global_version": float(self.global_version),
            }

        selected = [self.update_buffer.pop(0) for _ in range(consume_n)]

        client_ids = [u["client_id"] for u in selected]
        all_num_examples = [u["num_examples"] for u in selected]
        all_weights_list = [u["weights"] for u in selected]

        # 3) Base weights (dynamic if available, else data-weighted)
        if (
            self.use_dynamic_weights
            and server_round > 1
            and all(cid in self.client_performance for cid in client_ids)
        ):
            client_results_for_weights = [
                (cid, n_examples, self.client_performance[cid])
                for cid, n_examples in zip(client_ids, all_num_examples)
            ]
            dyn_dict = self.weight_calculator.calculate_dynamic_weights(
                client_results_for_weights,
                use_data_size=True,
            )
            base_weights = [dyn_dict.get(cid, 0.0) for cid in client_ids]

            if sum(base_weights) <= 0:
                total_examples = max(sum(all_num_examples), 1)
                base_weights = [n / total_examples for n in all_num_examples]

            rationales = []
            for cid in client_ids:
                rationales.append(
                    self.weight_calculator.get_weight_rationale(
                        cid, self.client_performance[cid], dyn_dict.get(cid, 0.0)
                    )
                )
            print_dynamic_weights_summary(dyn_dict, rationales, server_round)
            self.weight_history.append({
                "round": server_round,
                "weights": dyn_dict,
                "rationales": rationales
            })
        else:
            total_examples = max(sum(all_num_examples), 1)
            base_weights = [n / total_examples for n in all_num_examples]

        # 4) Staleness scaling
        staleness_factors = []
        for u in selected:
            stale = max(0, self.global_version - int(u["base_version"]))
            staleness_factors.append(1.0 / (1.0 + self.fedbuff_staleness_alpha * stale))

        fedbuff_weights = [bw * sf for bw, sf in zip(base_weights, staleness_factors)]
        s = sum(fedbuff_weights)
        if s <= 0:
            fedbuff_weights = [1.0 / len(fedbuff_weights)] * len(fedbuff_weights)
        else:
            fedbuff_weights = [w / s for w in fedbuff_weights]

        # 5) Aggregate selected buffered updates
        buffered_agg = []
        for layer_idx in range(len(all_weights_list[0])):
            layer_sum = np.zeros_like(all_weights_list[0][layer_idx], dtype=np.float32)
            for w_i, alpha_i in zip(all_weights_list, fedbuff_weights):
                layer_sum += w_i[layer_idx] * alpha_i
            buffered_agg.append(layer_sum)

        # 6) Server update step
        if self.current_global_weights is None:
            new_global = buffered_agg
        else:
            eta = self.fedbuff_server_lr
            new_global = [
                (1.0 - eta) * g + eta * b
                for g, b in zip(self.current_global_weights, buffered_agg)
            ]

        self.current_global_weights = new_global
        self.global_version += 1
        self.current_round_contributions = {
            cid: float(w) for cid, w in zip(client_ids, fedbuff_weights)
        }

        # 7) Quantize for transport
        if self.use_quantization:
            out_weights, quant_stats = quantize_weights_fp16(new_global)
            quant_error = calculate_quantization_error(new_global, out_weights)

            # Warn if quantization error is too high
            mean_rel_err = quant_error.get('mean_relative_error_percent', 0)
            if mean_rel_err > 1.0:
                print(f"  [WARNING] High quantization error: {mean_rel_err:.4f}% "
                      f"(threshold: 1.0%). Consider using FP32 for this round.")

            self.quantization_stats.append({
                "round": server_round,
                "compression_ratio": quant_stats.get("compression_ratio", 1.0),
                "error": quant_error,
                "buffered_updates": consume_n,
            })
            out_params = ndarrays_to_parameters(out_weights)
        else:
            out_params = ndarrays_to_parameters(new_global)

        avg_staleness = float(np.mean([max(0, self.global_version - 1 - u["base_version"]) for u in selected]))

        # Emit async telemetry for dashboard parsing
        self._print_async_status(
            version=int(self.global_version),
            buffered=int(len(self.update_buffer)),
            applied=int(consume_n),
            avg_staleness=avg_staleness,
        )

        return out_params, {
            "buffered_updates": float(len(self.update_buffer)),
            "applied_updates": float(consume_n),
            "global_version": float(self.global_version),
            "avg_staleness": avg_staleness,
        }
    
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
        
        # Calculate aggregated confusion matrix - ✅ FIXED
        total_tp = sum(res.metrics.get('true_positives', 0) for _, res in results if res.metrics)
        total_tn = sum(res.metrics.get('true_negatives', 0) for _, res in results if res.metrics)   # ✅ FIXED
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
