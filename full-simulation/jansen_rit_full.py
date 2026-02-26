import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import networkx as nx

# Jansen-Rit Model Parameters (Standard)
A, B = 3.25, 22.0
a, b = 100.0, 50.0
C = 135.0
C1, C2, C3, C4 = C, 0.8*C, 0.25*C, 0.25*C
v0, e0, r = 6.0, 2.5, 0.56

def sigmoid(v):
    val = np.clip(r * (v0 - v), -100, 100)
    return 2 * e0 / (1 + np.exp(val))

class JansenRitFull:
    """
    Scaled Jansen-Rit Network (N=100) with Dale's Law compliance.
    """
    def __init__(self, N=100, coupling_strength=20.0, inhibitory_fraction=0.2):
        self.N = N
        self.K = coupling_strength
        self.dt = 0.0005 
        self.T = 2.0     # Reduced time for efficiency in Monte Carlo
        self.steps = int(self.T / self.dt)
        self.time = np.linspace(0, self.T, self.steps)
        
        # Enforce Dale's Law: Assign node types
        self.node_types = np.ones(N) # 1 for Excitatory, -1 for Inhibitory
        num_inhibitory = int(N * inhibitory_fraction)
        inhibitory_indices = np.random.choice(N, num_inhibitory, replace=False)
        self.node_types[inhibitory_indices] = -1
        
        # Search for a stable matrix
        for attempt in range(200):
            # Topology: Small-World (to mimic cortical connectivity)
            self.G = nx.watts_strogatz_graph(n=N, k=6, p=0.1)
            adj = nx.to_numpy_array(self.G)
            
            # Construct weight matrix W respecting Dale's Law
            # W[i, j] is the connection from j to i
            self.W = np.zeros((N, N))
            for j in range(N):
                # All outgoing edges from j have the same sign
                sign = self.node_types[j]
                weight = 1.0 if sign > 0 else -0.5 # Scale factors
                self.W[:, j] = adj[:, j] * weight
                
            # Construct Diffusive Coupling Matrix (Laplacian-like)
            # Row sum must be zero: C[i, i] = -sum_{j!=i} W[i, j]
            self.C_matrix = self.W.copy()
            for i in range(N):
                self.C_matrix[i, i] = 0 # reset diagonal
                self.C_matrix[i, i] = -np.sum(self.C_matrix[i, :])
            
            # Spectral Stability Check (Master Stability Function)
            eig = la.eigvals(self.C_matrix)
            max_real_eig = np.max(eig.real)
            
            # We need max_real_eig <= 0 (with tolerance)
            if max_real_eig < 1e-5:
                print(f"Stable Dale-compliant matrix found (Attempt {attempt+1}). Max Real Eig: {max_real_eig:.6f}")
                break
        else:
            print(f"WARNING: No stable matrix found. Max Real Eig: {max_real_eig:.4f}")

    def dynamics(self, state, t):
        X = state.reshape((self.N, 6))
        dX = np.zeros_like(X)
        p_input = 200.0 # Constant input to induce oscillations
        
        # Coupling using diffusive logic
        coupling = self.K * (self.C_matrix @ X[:, 1])
        
        # Batch sigmoids for speed
        s_y1_y2 = sigmoid(X[:, 1] - X[:, 2])
        s_y0 = sigmoid(X[:, 0])
        s_y1 = sigmoid(X[:, 1])
        
        dX[:, 0] = X[:, 3]
        dX[:, 3] = A * a * s_y1_y2 - 2 * a * X[:, 3] - a * a * X[:, 0]
        
        dX[:, 1] = X[:, 4]
        dX[:, 4] = A * a * (p_input + C2 * s_y0 + coupling) - 2 * a * X[:, 4] - a * a * X[:, 1]
        
        dX[:, 2] = X[:, 5]
        dX[:, 5] = B * b * (C4 * s_y1) - 2 * b * X[:, 5] - b * b * X[:, 2]
        
        return dX.flatten()

    def run(self, initial_state=None, noise_sigma=0.0):
        if initial_state is None:
            # Completely heterogeneous random states
            initial_state = np.random.normal(0, 5.0, size=(self.N * 6))
            
        trajectory = np.zeros((self.steps, self.N * 6))
        trajectory[0] = initial_state
        
        # Pre-calculate noise for speed if needed, or do it per step
        for k in range(self.steps - 1):
            t = self.time[k]
            y = trajectory[k]
            k1 = self.dynamics(y, t)
            k2 = self.dynamics(y + 0.5 * self.dt * k1, t + 0.5 * self.dt)
            k3 = self.dynamics(y + 0.5 * self.dt * k2, t + 0.5 * self.dt)
            k4 = self.dynamics(y + self.dt * k3, t + self.dt)
            
            # Deterministic update
            next_y = y + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            
            # Additive Noise (AWGN) - applied to pyramidal potential y1
            if noise_sigma > 0:
                noise = np.random.normal(0, noise_sigma * np.sqrt(self.dt), size=self.N)
                # Apply noise specifically to the y1 derivative index (index 1, 7, 13...) 
                # or just add to the state directly for simplicity in this mean-field context
                noise_state = np.zeros(self.N * 6)
                noise_state[1::6] = noise # Apply to y1
                next_y += noise_state
                
            trajectory[k + 1] = next_y
            
        return trajectory

    def calculate_sync_error(self, trajectory):
        data = trajectory.reshape((self.steps, self.N, 6))
        y1_y2 = data[:, :, 1] - data[:, :, 2]
        sync_error = np.std(y1_y2, axis=1)
        return sync_error
