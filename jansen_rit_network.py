import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import networkx as nx

# Jansen-Rit Model Parameters
A = 3.25
B = 22.0
a = 100.0
b = 50.0
C = 135.0
C1 = C
C2 = 0.8 * C
C3 = 0.25 * C
C4 = 0.25 * C
v0 = 6.0
e0 = 2.5
r = 0.56

def sigmoid(v):
    # Clip exponent to avoid overflow
    val = r * (v0 - v)
    val = np.clip(val, -100, 100) # prevent overflow
    return 2 * e0 / (1 + np.exp(val))

class JansenRitNetwork:
    def __init__(self, N=20, coupling_strength=1.0, p_rewire=0.1, negative_fraction=0.1):
        self.N = N
        self.K = coupling_strength
        self.dt = 0.0005 # Integration step size (s) - smaller for stability
        self.T = 5.0     # Total simulation time (s)
        self.steps = int(self.T / self.dt)
        self.time = np.linspace(0, self.T, self.steps)
        
        # Attempt to generate a stable coupling matrix
        for attempt in range(100):
            # Initialize Network Topology
            self.G = nx.watts_strogatz_graph(n=N, k=4, p=p_rewire)
            self.adj_matrix = nx.to_numpy_array(self.G)
            
            # Introduce Negative Couplings
            edges = list(self.G.edges())
            num_negative = int(len(edges) * negative_fraction)
            if num_negative > 0:
                negative_edges_indices = np.random.choice(len(edges), num_negative, replace=False)
                
                self.W = self.adj_matrix.copy()
                for idx in negative_edges_indices:
                    u, v = edges[idx]
                    self.W[u, v] = -0.5 # Weaker inhibition
                    self.W[v, u] = -0.5
            else:
                self.W = self.adj_matrix.copy()
                
            # Construct Laplacian-like Coupling Matrix (Diffusive)
            # Row sum MUST be zero for diffusive coupling
            self.C_matrix = self.W.copy()
            for i in range(N):
                self.C_matrix[i, i] = -np.sum(self.W[i, :]) + self.W[i, i]
                
            # Check Stability
            eig = la.eigvals(self.C_matrix)
            max_real_eig = np.max(eig.real)
            
            # We want max_real_eig <= 0 (allow small epsilon for float error)
            if max_real_eig < 1e-5:
                print(f"Stable matrix found on attempt {attempt+1}. Max Real Eig: {max_real_eig:.4f}")
                break
        else:
             print("WARNING: Could not find stable matrix after 100 attempts. Using last one.")
             print(f"Max Real Eig: {max_real_eig:.4f}")

        # Verify Row Sums
        row_sums = np.sum(self.C_matrix, axis=1)
            
        # Verify Row Sums
        row_sums = np.sum(self.C_matrix, axis=1)
        assert np.allclose(row_sums, 0), f"Coupling matrix rows must sum to zero! {row_sums}"
        
        # Check Expected Stability
        eig = la.eigvals(self.C_matrix)
        max_real_eig = np.max(eig.real)
        print(f"Coupling Matrix Max Real Eigenvalue: {max_real_eig:.4f}")
        if max_real_eig > 1e-5:
             print("WARNING: Coupling matrix has positive eigenvalues. System WILL be unstable!")

    def dynamics(self, state, t):
        """
        y0: Excitatory Interneurons Output
        y1: Pyramidal Output (Main)
        y2: Inhibitory Interneurons Output
        y3, y4, y5: Derivatives
        """
        X = state.reshape((self.N, 6))
        dX = np.zeros_like(X)
        
        # Input (Standard Jansen-Rit)
        p_input = 120.0 + 50.0 * np.sin(2 * np.pi * 10 * t)
        
        # Diffusive coupling term
        coupling_input = self.K * (self.C_matrix @ X[:, 1])

        for i in range(self.N):
            y0, y1, y2, y3, y4, y5 = X[i]
            
            s_y1_y2 = sigmoid(y1 - y2)
            s_y0 = sigmoid(y0)
            s_y1 = sigmoid(y1)
            
            # Derivatives
            dX[i, 0] = y3
            dX[i, 3] = A * a * s_y1_y2 - 2 * a * y3 - a * a * y0
            
            dX[i, 1] = y4
            dX[i, 4] = A * a * (p_input + C2 * s_y0 + coupling_input[i]) - 2 * a * y4 - a * a * y1
            
            dX[i, 2] = y5
            dX[i, 5] = B * b * (C4 * s_y1) - 2 * b * y5 - b * b * y2
            
        return dX.flatten()

    def run(self):
        # Initial conditions: Random small perturbations around 0
        state = np.random.normal(0, 0.1, size=(self.N * 6))
        
        # Integrate using simple RK4
        print(f"Starting integration for {self.T} seconds...")
        trajectory = np.zeros((self.steps, self.N * 6))
        trajectory[0] = state
        
        for k in range(self.steps - 1):
            t = self.time[k]
            y = trajectory[k]
            
            k1 = self.dynamics(y, t)
            k2 = self.dynamics(y + 0.5 * self.dt * k1, t + 0.5 * self.dt)
            k3 = self.dynamics(y + 0.5 * self.dt * k2, t + 0.5 * self.dt)
            k4 = self.dynamics(y + self.dt * k3, t + self.dt)
            
            trajectory[k + 1] = y + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            
        print("Integration complete.")
        return trajectory

    def analyze_stability(self):
        # Calculate Eigenvalues of Matrix C
        eigenvalues = la.eigvals(self.C_matrix)
        return eigenvalues

    def plot_results(self, trajectory):
        # Extract y1 - Pyramidal Output (Main variable)
        # Reshape to (Steps, N, 6) -> Take index 1
        data = trajectory.reshape((self.steps, self.N, 6))
        y1_data = data[:, :, 1] - data[:, :, 2] # Main variable often approximated as y1-y2 (membrane potential)
        
        # Calculate Global Synchronization Error (Std Dev across nodes)
        sync_error = np.std(y1_data, axis=1)
        print(f"Final Synchronization Error: {sync_error[-1]:.6f}")
        print(f"Mean Synchronization Error (last 1s): {np.mean(sync_error[-int(1.0/self.dt):]):.6f}")

        fig = plt.figure(figsize=(15, 12))
        
        # 1. Coupling Matrix
        ax1 = plt.subplot(2, 2, 1)
        im = ax1.imshow(self.C_matrix, cmap='coolwarm', interpolation='nearest')
        plt.colorbar(im, ax=ax1)
        ax1.set_title("Coupling Matrix (C)\n(Red=Pos, Blue=Neg)")
        
        # 2. Eigenvalue Spectrum
        ax2 = plt.subplot(2, 2, 2)
        eig = self.analyze_stability()
        ax2.scatter(eig.real, eig.imag, color='blue')
        ax2.axvline(0, color='red', linestyle='--')
        ax2.set_xlabel('Real Part')
        ax2.set_ylabel('Imaginary Part')
        ax2.set_title('Eigenvalue Spectrum of Coupling Matrix')
        ax2.grid(True)
        
        # 3. Network Graph
        ax3 = plt.subplot(2, 2, 3)
        pos = nx.spring_layout(self.G)
        edges = self.G.edges()
        colors = []
        for u, v in edges:
            if self.W[u, v] < 0:
                colors.append('red') # Inhibition
            else:
                colors.append('green') # Excitation
        nx.draw(self.G, pos, ax=ax3, node_size=100, edge_color=colors, width=1.5, with_labels=True, font_size=8)
        ax3.set_title(f"Network Topology (N={self.N}, k=4, p=0.1)")
        
        # 4. Time Series
        ax4 = plt.subplot(2, 2, 4)
        for i in range(min(5, self.N)): # Plot first 5 nodes
            ax4.plot(self.time, y1_data[:, i], alpha=0.7, label=f'Node {i}')
        ax4.plot(self.time, sync_error, 'k--', linewidth=2, label='Sync Error (Std)')
        ax4.set_xlabel('Time (s)')
        ax4.set_title('Neural Activity (y1 - y2)')
        ax4.legend(loc='upper right', fontsize='small')
        
        plt.tight_layout()
        plt.savefig('jansen_rit_simulation.png', dpi=150)
        print("Plot saved to jansen_rit_simulation.png")

if __name__ == "__main__":
    # Create and run simulation
    # Try a configuration that is likely to be stable (or at least checkable)
    np.random.seed(42) # Reproducibility
    sim = JansenRitNetwork(N=20, coupling_strength=20.0, negative_fraction=0.1) 
    traj = sim.run()
    sim.plot_results(traj)
