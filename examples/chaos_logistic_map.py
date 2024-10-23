#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 21:49:42 2024

@author: iancoccimiglio
"""

def logistic_map(x, iterations=100):
    """
    Implements the logistic map, a classic example of a chaotic system.
    f(x) = rx(1-x) where r = 3.99 (chosen for chaotic behavior)
    
    Parameters:
        x (float): Initial value between 0 and 1
        iterations (int): Number of iterations to perform
    
    Returns:
        float: Final value after iterations
    """
    if not 0 <= x <= 1:
        raise ValueError("Input x must be between 0 and 1")
    
    r = 3.99  # Parameter that leads to chaos
    value = x
    
    # Iterate the map
    for _ in range(iterations):
        value = r * value * (1 - value)
    
    return value

print(logistic_map(0.3))
print(logistic_map((0.2+0.1)))