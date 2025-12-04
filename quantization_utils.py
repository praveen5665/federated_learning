"""
Quantization utilities for Federated Learning Autoencoder
Implements FP16 (16-bit) post-training quantization
"""

import numpy as np
from typing import List, Tuple


def quantize_weights_fp16(weights: List[np.ndarray]) -> Tuple[List[np.ndarray], dict]:
    """
    Quantize model weights from FP32 to FP16
    
    Args:
        weights: List of numpy arrays (FP32)
    
    Returns:
        quantized_weights: List of numpy arrays (FP16)
        stats: Dictionary with quantization statistics
    """
    quantized_weights = []
    original_size = 0
    quantized_size = 0
    
    for weight in weights:
        original_size += weight.nbytes
        # Convert to FP16
        quantized = weight.astype(np.float16)
        quantized_weights.append(quantized)
        quantized_size += quantized.nbytes
    
    compression_ratio = original_size / quantized_size if quantized_size > 0 else 1.0
    
    stats = {
        "original_size_kb": original_size / 1024,
        "quantized_size_kb": quantized_size / 1024,
        "compression_ratio": compression_ratio,
        "size_reduction_percent": (1 - 1/compression_ratio) * 100
    }
    
    return quantized_weights, stats


def dequantize_weights_fp16(quantized_weights: List[np.ndarray]) -> List[np.ndarray]:
    """
    Dequantize model weights from FP16 back to FP32
    
    Args:
        quantized_weights: List of numpy arrays (FP16)
    
    Returns:
        weights: List of numpy arrays (FP32)
    """
    return [weight.astype(np.float32) for weight in quantized_weights]


def calculate_quantization_error(original: List[np.ndarray], 
                                 quantized: List[np.ndarray]) -> dict:
    """
    Calculate error introduced by quantization
    
    Args:
        original: Original FP32 weights
        quantized: Quantized FP16 weights
    
    Returns:
        error_metrics: Dictionary with error statistics
    """
    errors = []
    relative_errors = []
    
    for orig, quant in zip(original, quantized):
        quant_fp32 = quant.astype(np.float32)
        diff = np.abs(orig - quant_fp32)
        errors.append(np.mean(diff))
        
        # Relative error (avoid division by zero)
        denominator = np.abs(orig) + 1e-10
        rel_error = np.mean(diff / denominator)
        relative_errors.append(rel_error)
    
    return {
        "mean_absolute_error": float(np.mean(errors)),
        "max_absolute_error": float(np.max(errors)),
        "mean_relative_error_percent": float(np.mean(relative_errors) * 100),
        "max_relative_error_percent": float(np.max(relative_errors) * 100)
    }


def print_quantization_stats(stats: dict, error_metrics: dict = None):
    """Pretty print quantization statistics"""
    print(f"\n{'='*50}")
    print("QUANTIZATION STATISTICS")
    print(f"{'='*50}")
    print(f"  Original Size:    {stats['original_size_kb']:.2f} KB")
    print(f"  Quantized Size:   {stats['quantized_size_kb']:.2f} KB")
    print(f"  Compression:      {stats['compression_ratio']:.2f}x")
    print(f"  Size Reduction:   {stats['size_reduction_percent']:.1f}%")
    
    if error_metrics:
        print(f"\n  Quantization Error:")
        print(f"    Mean Error:     {error_metrics['mean_absolute_error']:.6f}")
        print(f"    Max Error:      {error_metrics['max_absolute_error']:.6f}")
        print(f"    Mean Rel Error: {error_metrics['mean_relative_error_percent']:.4f}%")
    
    print(f"{'='*50}\n")
