import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy.linalg as la
import os
from jansen_rit_full import JansenRitFull

def run_monte_carlo_analysis(M=100):
    print(f"Starting Monte Carlo Basin Stability Analysis (M={M})...")
    np.random.seed(42)
    output_dir = "paper/figures_full"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the scaled network
    sim = JansenRitFull(N=100, coupling_strength=30.0) # slightly higher coupling for N=100 stability
    
    final_errors = []
    sync_times = []
    num_sync = 0
    sync_threshold = 0.1 
    
    # Run trials
    for i in range(M):
        print(f"Trial {i+1}/{M}...")
        trajectory = sim.run()
        sync_error = sim.calculate_sync_error(trajectory)
        final_errors.append(sync_error[-1])
        
        # Calculate time to convergence (first time error < threshold)
        sync_idx = np.where(sync_error < sync_threshold)[0]
        if len(sync_idx) > 0:
            t_sync = sim.time[sync_idx[0]]
            sync_times.append(t_sync)
            num_sync += 1
        else:
            sync_times.append(sim.T) # Didn't sync
            
    basin_stability = num_sync / M
    print(f"Basin Stability Score: {basin_stability:.4f} ({num_sync}/{M} trials synchronized)")
    print(f"Mean Convergence Time: {np.mean(sync_times):.4f}s")
    
    # --- Visualization ---
    
    # 1. Topology (Dale's Law visualization)
    print("Generating Figure 1: Scaled Topology...")
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(sim.G, seed=42)
    nodes_E = [n for n in range(sim.N) if sim.node_types[n] > 0]
    nodes_I = [n for n in range(sim.N) if sim.node_types[n] < 0]
    
    nx.draw_networkx_nodes(sim.G, pos, nodelist=nodes_E, node_color='green', label='Excitatory', node_size=100)
    nx.draw_networkx_nodes(sim.G, pos, nodelist=nodes_I, node_color='red', label='Inhibitory', node_size=100)
    
    # Draw edges with transparency
    nx.draw_networkx_edges(sim.G, pos, alpha=0.1, edge_color='gray')
    
    plt.title(f"Scaled Neural Network (N=100) with Dale's Law\n(Green=E, Red=I)")
    plt.legend()
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'figure_1_full_topology.png'), dpi=300)
    plt.close()
    
    # 2. Eigenvalue Spectrum
    print("Generating Figure 2: Large Spectrum...")
    plt.figure(figsize=(8, 6))
    eig = la.eigvals(sim.C_matrix)
    plt.scatter(eig.real, eig.imag, c='darkblue', alpha=0.5, s=20, edgecolors='none')
    plt.axvline(0, color='red', linestyle='--', linewidth=1.5)
    plt.xlabel('Real Part ($\lambda$)')
    plt.ylabel('Imaginary Part ($i\omega$)')
    plt.title('Eigenvalue Spectrum (N=100, Dale-Compliant)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'figure_2_full_spectrum.png'), dpi=300)
    plt.close()
    
    # 3. Convergence Time Histogram
    print("Generating Figure 3: Convergence Time Distribution...")
    plt.figure(figsize=(8, 5))
    plt.hist(sync_times, bins=15, color='salmon', edgecolor='black', alpha=0.7)
    plt.axvline(np.mean(sync_times), color='blue', linestyle='--', label=f'Mean = {np.mean(sync_times):.3f}s')
    plt.xlabel('Time to Synchronization $T_{sync}$ (s)')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Convergence Times (M={M} Trials)\nThreshold = {sync_threshold}')
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.savefig(os.path.join(output_dir, 'figure_3_full_basin.png'), dpi=300)
    plt.close()
    
    # 4. Representative Dynamics (last trial)
    print("Generating Figure 4: Representative Dynamics...")
    plt.figure(figsize=(10, 6))
    sim_last = JansenRitFull(N=100, coupling_strength=30.0)
    sim_last.C_matrix = sim.C_matrix # same matrix
    traj = sim_last.run()
    error = sim_last.calculate_sync_error(traj)
    
    plt.subplot(2, 1, 1)
    data = traj.reshape((sim_last.steps, sim_last.N, 6))
    y1_y2 = data[:, :, 1] - data[:, :, 2]
    plt.plot(sim_last.time, y1_y2[:, :10], alpha=0.6) # plot 10 nodes
    plt.ylabel('Activity (mV)')
    plt.title('Neural Activity (N=100, Dale-Compliant)')
    
    plt.subplot(2, 1, 2)
    plt.semilogy(sim_last.time, error + 1e-12, 'k-') # Log scale for error
    plt.xlabel('Time (s)')
    plt.ylabel('Sync Error (Log)')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'figure_4_full_dynamics.png'), dpi=300)
    plt.close()

def run_noise_robustness_analysis():
    print("Starting Noise Robustness Analysis...")
    output_dir = "paper/figures_full"
    os.makedirs(output_dir, exist_ok=True)
    
    sim = JansenRitFull(N=100, coupling_strength=30.0)
    noise_levels = [0, 5, 10, 20, 30, 40, 50]
    avg_errors = []
    
    for sigma in noise_levels:
        print(f"Testing Noise Sigma = {sigma}...")
        trajectory = sim.run(noise_sigma=sigma)
        error = sim.calculate_sync_error(trajectory)
        avg_errors.append(np.mean(error[int(len(error)*0.8):])) # mean of last 20%
        
    plt.figure(figsize=(8, 5))
    plt.plot(noise_levels, avg_errors, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Noise Intensity $\sigma_{noise}$ (mV)')
    plt.ylabel('Residual Sync Error')
    plt.title('Noise Robustness (N=100, Dale-Compliant)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'figure_5_full_noise.png'), dpi=300)
    plt.close()

def run_pathology_analysis():
    print("Starting Pathological Bifurcation Analysis...")
    output_dir = "paper/figures_full"
    os.makedirs(output_dir, exist_ok=True)
    
    # We sweep coupling strength K until max(Re(eig)) > 0
    k_range = np.linspace(10, 150, 20)
    max_eigs = []
    final_errors = []
    
    # Use one stable matrix found previously for sweep
    sim = JansenRitFull(N=100, coupling_strength=30.0)
    base_C = sim.C_matrix.copy()
    
    for k in k_range:
        sim.K = k # Update coupling strength
        # The effective eigenvalues are K * eigenvalues(C_matrix)
        # However, MSF stability also depends on the local dynamics Jacobian.
        # For simplicity in this demo, we'll show how K pushes the "Spectral Bound" 
        # relative to the system's internal stability.
        
        eigs = np.linalg.eigvals(base_C)
        # alpha_k = K * sigma_k. We track the 'critical' mode.
        # In many systems, stability is lost when K*Re(sigma_k) exceeds a threshold
        max_eigs.append(k * np.max(np.real(eigs)))
        
        trajectory = sim.run()
        error = sim.calculate_sync_error(trajectory)
        final_errors.append(np.mean(error[-100:]))
        
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    ax1.set_xlabel('Coupling Strength $K$')
    ax1.set_ylabel('Max $\text{Re}(\lambda)$ (Stability)', color='tab:blue')
    ax1.plot(k_range, max_eigs, 'b-', linewidth=2, label='Spectral Bound')
    ax1.axhline(0, color='red', linestyle='--')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Oscillation Amplitude / Sync Error', color='tab:red')
    ax2.plot(k_range, final_errors, 'r--', marker='o', alpha=0.6, label='Dynamics Status')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    plt.title('Bifurcation to Hypersynchrony (The Failure Point)')
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figure_6_full_pathology.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
    # For a quicker review, we can start with M=50 or M=100
    run_monte_carlo_analysis(M=100)
    run_noise_robustness_analysis()
    run_pathology_analysis()
