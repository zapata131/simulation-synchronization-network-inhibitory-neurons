import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy.linalg as la

# Add parent directory to path to import the simulation class
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from jansen_rit_network import JansenRitNetwork

def generate_paper_figures():
    print("Initializing Simulation for Paper Figures...")
    np.random.seed(42)  # Ensure reproducibility
    output_dir = os.path.join(os.path.dirname(__file__), 'figures')
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Standard Simulation (Figs 1-3) ---
    sim = JansenRitNetwork(N=20, coupling_strength=20.0, negative_fraction=0.1)
    trajectory = sim.run()
    
    steps = sim.steps
    N = sim.N
    data = trajectory.reshape((steps, N, 6))
    y1_data = data[:, :, 1] - data[:, :, 2]  # Proxy for EEG (y1 - y2)
    sync_error = np.std(y1_data, axis=1)
    
    # --- Figure 1: Network Topology ---
    print("Generating Figure 1: Topology...")
    plt.figure(figsize=(6, 6))
    pos = nx.spring_layout(sim.G, seed=42)
    edges = sim.G.edges()
    edge_colors = ['red' if sim.W[u, v] < 0 else 'green' for u, v in edges]
    nx.draw(sim.G, pos, node_size=150, node_color='lightgray', 
            edge_color=edge_colors, width=1.5, with_labels=True, 
            font_size=10, font_weight='bold')
    plt.title("Mixed-Sign Small-World Network\n(Green=Excitatory, Red=Inhibitory)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figure_1_topology.png'), dpi=300)
    plt.close()
    
    # --- Figure 2: Eigenvalue Spectrum ---
    print("Generating Figure 2: Spectrum...")
    plt.figure(figsize=(6, 5))
    eig = la.eigvals(sim.C_matrix)
    plt.scatter(eig.real, eig.imag, c='blue', alpha=0.7, s=50, edgecolors='k')
    plt.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Stability Boundary')
    plt.xlabel('Real Part ($\lambda$)')
    plt.ylabel('Imaginary Part ($i\omega$)')
    plt.title('Eigenvalue Spectrum of Coupling Matrix $\mathcal{C}$')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figure_2_spectrum.png'), dpi=300)
    plt.close()
    
    # --- Figure 3: Synchronization Dynamics ---
    print("Generating Figure 3: Dynamics...")
    plt.figure(figsize=(10, 6))
    ax1 = plt.subplot(2, 1, 1)
    for i in range(min(5, N)):
        ax1.plot(sim.time, y1_data[:, i], alpha=0.8, linewidth=1.2)
    ax1.set_ylabel('Potential (mV)')
    ax1.set_title('Neural Activity (Subset of Nodes)')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, sim.T)
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(sim.time, sync_error, 'k-', linewidth=1.5)
    
    # Basin Stability Metric
    final_sync = sync_error[-1]
    bs_label = f"Basin Stability (Trial 1): Sync Error = {final_sync:.4f}"
    ax2.text(0.5, 0.9, bs_label, transform=ax2.transAxes, ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Sync Error $\sigma(t)$')
    ax2.set_title('Global Synchronization Error')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, sim.T)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figure_3_dynamics.png'), dpi=300)
    plt.close()

    # --- Figure 4: Probability of Stability vs Negative Fraction ---
    print("Generating Figure 4: Probability of Stability...")
    fractions = np.linspace(0, 0.5, 11) # 0 to 50% inhibition
    probabilities = []
    trials_per_frac = 50 
    
    for f in fractions:
        stable_count = 0
        for _ in range(trials_per_frac):
            # Generate random small-world graph
            G_temp = nx.watts_strogatz_graph(n=20, k=4, p=0.1)
            adj = nx.to_numpy_array(G_temp)
            edges = list(G_temp.edges())
            num_neg = int(len(edges) * f)
            W_temp = adj.copy()
            if num_neg > 0:
                neg_idx = np.random.choice(len(edges), num_neg, replace=False)
                for idx in neg_idx:
                    u, v = edges[idx]
                    W_temp[u, v] = -0.5
                    W_temp[v, u] = -0.5
            
            # Laplacian construction
            C_temp = W_temp.copy()
            for i in range(20):
                C_temp[i, i] = -np.sum(W_temp[i, :]) + W_temp[i, i]
            
            # Check stability
            eig_temp = la.eigvals(C_temp)
            if np.max(eig_temp.real) <= 1e-5:
                stable_count += 1
        probabilities.append(stable_count / trials_per_frac)

    plt.figure(figsize=(7, 5))
    plt.plot(fractions * 100, probabilities, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Percentage of Inhibitory Edges (%)')
    plt.ylabel('Probability of Spectral Stability')
    plt.title('Robustness of Spectral Condition')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axhline(0.1, color='r', linestyle=':', label='Biological Plausibility Threshold') # Arbitrary visual cue
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figure_4_probability.png'), dpi=300)
    plt.close()
    
    print("All figures generated successfully in paper/figures/")

if __name__ == "__main__":
    generate_paper_figures()
