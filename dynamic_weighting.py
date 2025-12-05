"""
Dynamic Weight Calculation for Performance-Based Federated Aggregation
Implements hybrid strategy combining multiple performance metrics
"""

import numpy as np
from typing import Dict, List, Tuple


class DynamicWeightCalculator:
    """
    Calculate dynamic weights based on client performance metrics
    Uses hybrid approach combining accuracy, F1-score, loss, and AUC-ROC
    """
    
    def __init__(self, 
                 alpha=0.4,  # Accuracy weight
                 beta=0.3,   # F1-score weight
                 gamma=0.2,  # Loss weight (inverted)
                 delta=0.1,  # AUC-ROC weight
                 min_weight=0.05,  # Minimum weight per client (fairness)
                 smoothing=0.7):   # EMA smoothing factor
        """
        Initialize dynamic weight calculator
        
        Args:
            alpha, beta, gamma, delta: Weights for each metric (should sum to 1.0)
            min_weight: Minimum weight threshold to ensure fairness
            smoothing: Exponential moving average factor for stability
        """
        assert abs(alpha + beta + gamma + delta - 1.0) < 0.01, "Weights must sum to 1.0"
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.min_weight = min_weight
        self.smoothing = smoothing
        
        # Track historical performance
        self.performance_history = {}
        
    def calculate_performance_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate composite performance score from metrics
        
        Args:
            metrics: Dictionary with 'accuracy', 'f1_score', 'loss', 'auc_roc'
        
        Returns:
            performance_score: Weighted combination of metrics
        """
        accuracy = metrics.get('accuracy', 0.5)
        f1_score = metrics.get('f1_score', 0.5)
        loss = metrics.get('loss', 1.0)
        auc_roc = metrics.get('auc_roc', 0.5)
        
        # Invert loss (lower is better)
        loss_score = 1.0 / (1.0 + loss) if loss > 0 else 0.5
        
        # Weighted combination
        score = (
            self.alpha * accuracy +
            self.beta * f1_score +
            self.gamma * loss_score +
            self.delta * auc_roc
        )
        
        return float(score)
    
    def calculate_dynamic_weights(
        self,
        client_results: List[Tuple[str, int, Dict[str, float]]],
        use_data_size: bool = True
    ) -> Dict[str, float]:
        """
        Calculate dynamic weights for all clients
        
        Args:
            client_results: List of (client_id, num_examples, metrics)
            use_data_size: Whether to factor in data size
        
        Returns:
            weights: Dictionary mapping client_id to weight
        """
        if not client_results:
            return {}
        
        client_scores = {}
        
        # Calculate performance score for each client
        for client_id, num_examples, metrics in client_results:
            # Calculate raw performance score
            perf_score = self.calculate_performance_score(metrics)
            
            # Apply EMA smoothing if historical data exists
            if client_id in self.performance_history:
                prev_score = self.performance_history[client_id]
                perf_score = self.smoothing * perf_score + (1 - self.smoothing) * prev_score
            
            # Store for future smoothing
            self.performance_history[client_id] = perf_score
            
            # Combine with data size if enabled
            if use_data_size:
                client_scores[client_id] = perf_score * num_examples
            else:
                client_scores[client_id] = perf_score
        
        # Normalize to get weights
        total_score = sum(client_scores.values())
        weights = {}
        
        for client_id, score in client_scores.items():
            weight = score / total_score if total_score > 0 else 1.0 / len(client_results)
            # Apply minimum weight threshold for fairness
            weight = max(weight, self.min_weight)
            weights[client_id] = float(weight)
        
        # Re-normalize after applying minimum threshold
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def get_weight_rationale(
        self,
        client_id: str,
        metrics: Dict[str, float],
        weight: float
    ) -> Dict[str, float]:
        """
        Get detailed rationale for weight assignment
        
        Args:
            client_id: Client identifier
            metrics: Performance metrics
            weight: Assigned weight
        
        Returns:
            rationale: Dictionary with breakdown
        """
        accuracy = metrics.get('accuracy', 0)
        f1_score = metrics.get('f1_score', 0)
        loss = metrics.get('loss', 0)
        auc_roc = metrics.get('auc_roc', 0)
        
        return {
            "client_id": client_id,
            "accuracy": float(accuracy),
            "f1_score": float(f1_score),
            "loss": float(loss),
            "auc_roc": float(auc_roc),
            "performance_score": float(self.calculate_performance_score(metrics)),
            "final_weight": float(weight),
            "contribution_factors": {
                "accuracy_contribution": float(self.alpha * accuracy),
                "f1_contribution": float(self.beta * f1_score),
                "loss_contribution": float(self.gamma / (1.0 + loss)),
                "auc_contribution": float(self.delta * auc_roc)
            }
        }


def print_dynamic_weights_summary(
    weights: Dict[str, float],
    rationales: List[Dict],
    round_num: int
):
    """Pretty print dynamic weight summary"""
    print(f"\n{'='*60}")
    print(f"DYNAMIC WEIGHTS - Round {round_num}")
    print(f"{'='*60}")
    
    for rationale in sorted(rationales, key=lambda x: x['final_weight'], reverse=True):
        client_id = rationale['client_id']
        weight = rationale['final_weight']
        perf_score = rationale['performance_score']
        
        print(f"\n{client_id.upper()}:")
        print(f"  Performance Score: {perf_score:.4f}")
        print(f"  Final Weight:      {weight:.4f} ({weight*100:.1f}%)")
        print(f"  Metrics:")
        print(f"    Accuracy:  {rationale['accuracy']:.4f}")
        print(f"    F1-Score:  {rationale['f1_score']:.4f}")
        print(f"    Loss:      {rationale['loss']:.4f}")
        print(f"    AUC-ROC:   {rationale['auc_roc']:.4f}")
    
    print(f"\n{'='*60}\n")
