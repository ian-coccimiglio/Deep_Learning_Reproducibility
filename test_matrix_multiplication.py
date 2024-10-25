#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 11:27:28 2024

@author: iancoccimiglio
"""

import numpy as np
import torch
import hashlib

def test_precision(size=1000, seed=42):
    """
    Test numerical precision differences across hardware by performing matrix operations
    and computing a hash of the results for comparison across machines.
    
    Args:
        size (int): Size of the square matrices
        seed (int): Random seed for reproducibility
    
    Returns:
        dict: Contains results and their hashes for comparison
    """
    # Set seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Generate identical random matrices
    A = np.random.randn(size, size).astype(np.float32)
    B = np.random.randn(size, size).astype(np.float32)
    
    # CPU computations
    cpu_torch = torch.from_numpy(A) @ torch.from_numpy(B)
    cpu_numpy = A @ B
    
    # GPU computations (if available)
    gpu_result = None
    if torch.cuda.is_available():
        gpu_A = torch.from_numpy(A).cuda()
        gpu_B = torch.from_numpy(B).cuda()
        gpu_result = (gpu_A @ gpu_B).cpu()
    
    # Convert results to numpy for consistent comparison
    results = {
        'cpu_torch': cpu_torch.numpy(),
        'cpu_numpy': cpu_numpy,
    }
    if gpu_result is not None:
        results['gpu'] = gpu_result.numpy()
    
    # Compute hashes for exact comparison across machines
    hashes = {}
    for name, result in results.items():
        # Convert to bytes in a consistent way
        result_bytes = result.tobytes()
        # Compute SHA-256 hash
        hash_value = hashlib.sha256(result_bytes).hexdigest()
        hashes[name] = hash_value
        
    # Compute some basic statistics
    differences = {}
    baseline = results['cpu_numpy']
    for name, result in results.items():
        if name != 'cpu_numpy':
            diff = np.abs(result - baseline)
            differences[f'{name}_vs_numpy'] = {
                'max_diff': diff.max(),
                'mean_diff': diff.mean(),
                'std_diff': diff.std()
            }
    
    return {
        'results': results,
        'hashes': hashes,
        'differences': differences
    }

def print_comparison(output):
    """Print the comparison results in a readable format."""
    print("Result Hashes (for cross-machine comparison):")
    for name, hash_value in output['hashes'].items():
        print(f"{name}: {hash_value}")
    
    print("\nDifferences from NumPy baseline:")
    for name, stats in output['differences'].items():
        print(f"\n{name}:")
        for stat_name, value in stats.items():
            print(f"  {stat_name}: {value:.2e}")

# def print_lean_comparison(results):
#     results
# Run the test
if __name__ == "__main__":
    # Test with different sizes to see if differences scale
    for size in [100, 1024, 1025]:
        results = test_precision(size=size)
        print(f"Testing with matrix size: {size}x{size}: {results['differences']['cpu_torch_vs_numpy']['mean_diff']}")
        # print_comparison(results)
