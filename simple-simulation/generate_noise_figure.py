import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from jansen_rit_network import JansenRitNetwork, sigmoid, A, a, C2 # p_input is local in dynamics

class NoisyJansenRitNetwork(JansenRitNetwork):
    def __init__(self, noise_std=0.0, **kwargs):
        super().__init__(**kwargs)
        self.noise_std = noise_std
        self.noise_grid = None

    def run(self):
        # Pre-generate noise (Zero-Order Hold)
        np.random.seed(42)
        self.noise_grid = np.random.normal(0, self.noise_std, size=(self.steps, self.N))
        
        return super().run()

    def dynamics(self, state, t):
        # Override dynamics to include pre-generated noise
        X = state.reshape((self.N, 6))
        dX = np.zeros_like(X)
        
        # Determine current step index
        step_idx = int(round(t / self.dt))
        if step_idx >= self.steps:
            step_idx = self.steps - 1
            
        current_noise = self.noise_grid[step_idx]

        # Standard input specific to Jansen Rit
        p_in = 120.0 + 50.0 * np.sin(2 * np.pi * 10 * t)
        
        # Coupling term
        coupling_input = self.K * (self.C_matrix @ X[:, 1])

        # Constants from imported module
        from jansen_rit_network import A, B, a, b, C, C2, C4, sigmoid

        for i in range(self.N):
            y0, y1, y2, y3, y4, y5 = X[i]
            
            s_y1_y2 = sigmoid(y1 - y2)
            s_y0 = sigmoid(y0)
            s_y1 = sigmoid(y1)
            
            dX[i, 0] = y3
            dX[i, 3] = A * a * s_y1_y2 - 2 * a * y3 - a * a * y0
            
            dX[i, 1] = y4
            # Add noise to input
            dX[i, 4] = A * a * (p_in + current_noise[i] + C2 * s_y0 + coupling_input[i]) - 2 * a * y4 - a * a * y1
            
            dX[i, 2] = y5
            dX[i, 5] = B * b * (C4 * s_y1) - 2 * b * y5 - b * b * y2
            
        return dX.flatten()

def generate_noise_figure():
    print("Generating Figure 5: Noise Robustness...")
    output_dir = os.path.join(os.path.dirname(__file__), 'figures')
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Sweep Noise Levels
    noise_levels = np.linspace(0, 50, 11) # 0 to 50 mV std dev (significant noise!)
    sync_errors = []
    
    print("Running noise sweep...")
    for sigma in noise_levels:
        sim = NoisyJansenRitNetwork(N=20, coupling_strength=20.0, negative_fraction=0.1, noise_std=sigma)
        traj = sim.run()
        steps = sim.steps
        data = traj.reshape((steps, sim.N, 6))
        y1_data = data[:, :, 1] - data[:, :, 2]
        
        # Calculate error in last 1 second
        last_1s_idx = int(1.0 / sim.dt)
        local_sync_error = np.std(y1_data[-last_1s_idx:, :], axis=1).mean()
        sync_errors.append(local_sync_error)
        print(f"  Noise Std: {sigma:.1f} mV -> Sync Error: {local_sync_error:.4f}")

    # 2. Run Single High-Noise Simulation for Time Series
    high_noise = 25.0
    print(f"Running single simulation with noise={high_noise} for visualization...")
    sim = NoisyJansenRitNetwork(N=20, coupling_strength=20.0, negative_fraction=0.1, noise_std=high_noise)
    traj = sim.run()
    steps = sim.steps
    data = traj.reshape((steps, sim.N, 6))
    y1_data = data[:, :, 1] - data[:, :, 2]
    time = np.linspace(0, sim.T, steps)

    # 3. Plotting (2-Panel)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    
    # Panel 1: Time Series at High Noise
    for i in range(min(5, sim.N)):
        ax1.plot(time, y1_data[:, i], alpha=0.6, label=f'Node {i}')
    ax1.set_xlim(2.0, 3.0) # Zoom in on 1 second
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Output (mV)')
    ax1.set_title(f'Synchronized Trajectories under Noise ($\sigma={high_noise}$ mV)')
    # ax1.legend(loc='upper right', fontsize='small')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Error Sweep
    ax2.plot(noise_levels, sync_errors, 'ro-', linewidth=2)
    ax2.set_xlabel('Noise Standard Deviation (mV)')
    ax2.set_ylabel('Mean Synchronization Error (mV)')
    ax2.set_title('Robustness Sweep: Error vs Noise Intensity')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figure_5_noise.png'), dpi=300)
    plt.close()
    print("Figure 5 (Enhanced) generated.")

if __name__ == "__main__":
    generate_noise_figure()
