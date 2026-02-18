#!/usr/bin/env python3
"""
Spectral Stability Probability Analysis
---------------------------------------
This script investigates the robustness of spectral stability in Small-World networks
with mixed-sign couplings. It performs a Monte Carlo simulation to estimate the
probability that a random network configuration satisfies the stability condition
(all non-zero eigenvalues <= 0) as a function of the fraction of inhibitory (negative) edges.

Usage:
    python3 probability_stability.py
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy.linalg as la
import os

def analyze_stability_probability():
    print("Running Spectral Stability Probability Analysis...")
    np.random.seed(42)  # Ensure reproducibility
    
    # Parameters
    N = 20              # Number of nodes
    k = 4               # Nearest neighbors
    p = 0.1             # Rewiring probability (Small-World)
    trials_per_frac = 50 
    fractions = np.linspace(0, 0.5, 11) # 0% to 50% inhibition
    
    probabilities = []
    
    print(f" Simulating {len(fractions)} levels of inhibition with {trials_per_frac} trials each...")
    
    for f in fractions:
        stable_count = 0
        for _ in range(trials_per_frac):
            # 1. Generate random small-world graph topology
            G = nx.watts_strogatz_graph(n=N, k=k, p=p)
            adj = nx.to_numpy_array(G)
            edges = list(G.edges())
            
            # 2. Assign negative weights to a fraction 'f' of edges
            num_neg = int(len(edges) * f)
            W = adj.copy()
            if num_neg > 0:
                neg_idx = np.random.choice(len(edges), num_neg, replace=False)
                for idx in neg_idx:
                    u, v = edges[idx]
                    W[u, v] = -0.5  # Inhibitory weight
                    W[v, u] = -0.5  # Symmetric
            
            # 3. Construct Laplacian-like Coupling Matrix (Diffusive)
            C = W.copy()
            for i in range(N):
                # Diagonal elements ensure row sum is zero (diffusive coupling condition)
                C[i, i] = -np.sum(W[i, :]) + W[i, i]
            
            # 4. Check Spectral Stability Condition
            # Condition: All non-zero eigenvalues must have Re(lambda) <= 0
            eig = la.eigvals(C)
            max_real_eig = np.max(eig.real)
            
            # Allow for small numerical error (epsilon)
            if max_real_eig <= 1e-5:
                stable_count += 1
                
        prob = stable_count / trials_per_frac
        probabilities.append(prob)
        print(f"  Fraction {f*100:5.1f}% neg edges -> Probability: {prob:.2f}")

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(fractions * 100, probabilities, 'bo-', linewidth=2, markersize=8, label='Spectral Stability Probability')
    
    # Aesthetics
    plt.xlabel('Percentage of Inhibitory Edges (%)', fontsize=12)
    plt.ylabel('Probability of Stable Configuration', fontsize=12)
    plt.title('Statistical Robustness of Mixed-Sign Networks', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axhline(0.1, color='r', linestyle=':', label='Biological Plausibility Threshold') 
    plt.legend()
    plt.tight_layout()
    
    output_file = 'probability_stability.png'
    plt.savefig(output_file, dpi=300)
    print(f"\nAnalysis complete. Plot saved to: {os.path.abspath(output_file)}")

if __name__ == "__main__":
    analyze_stability_probability()
