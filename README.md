# Synchronization in Complex Networks with Negative Couplings: From Chaos to Neuroscience

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Scientific Computing](https://img.shields.io/badge/SciPy-Stack-orange?style=flat-square&logo=scipy&logoColor=white)](https://scipy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)](https://github.com/zapata131/simulation-chaotic-systems-negative-couplings)

> **Simulating the emergence of synchronization in complex networks with inhibitory (negative) interactions.**

This repository explores the counter-intuitive phenomenon where networks can achieve global synchronization even in the presence of antagonisms, competition, or inhibition. Originally demonstrating this effect in chaotic **Lorenz oscillators**, we have extended the framework to **Biological Neural Mass Models**, simulating how inhibitory synaptic connections contribute to brain network dynamics.

![Jansen-Rit Simulation](jansen_rit_simulation.png)
*Figure 1: Synchronization of a Jensen-Rit neural network with 10% inhibitory connections.*

## 🧠 Biological Modelling: Neural Mass Networks

Our primary focus is applying the "negative coupling" concept to neuroscience. In brain networks, **inhibitory interneurons** play a crucial role in regulating activity. We model this using the **Jansen-Rit Neural Mass Model**, where each node represents a cortical column.

### Key Capabilities
- **Cortical Column Simulation**: Implements the standard Jansen-Rit differential equations (Pyramidal, Excitatory, Inhibitory populations).
- **Inhibitory Synaptic Connections**: Simulates long-range inhibition by introducing **negative weights** ($A_{ij} < 0$) in the coupling matrix.
- **Spectral Stability Search**: An adaptive algorithm generates Small-World topologies that satisfy the spectral stability condition (all non-zero eigenvalues $\lambda \le 0$) even with mixed-sign couplings.
- **Perfect Synchronization**: Demonstrates that networks can synchronize (Error $\approx 0$) with significant fractions of inhibitory links.

### Run the Simulation
```bash
python3 jansen_rit_network.py
```
*Output: Generates `jansen_rit_simulation.png` visualizing the coupling matrix spectrum, network topology, and neural activity.*

### Results & Interpretation

The simulation reveals a critical insight for biological networks: **Inhibition does not preclude synchronization; in fact, it structures it.**

- **Spectral Stability in E-I Networks**: The eigenvalue plot (top-right) shows that even with ~10% negative edges (red lines in the topology graph), the spectral stability condition is met (all non-zero eigenvalues $\lambda \le 0$). This confirms that the *net* connectivity remains stable despite local antagonisms.
- **Emergence of Coherent Rhythms**: The time-series plot (bottom-right) shows the neural activity of 20 cortical columns. Despite the inhibitory connections providing "negative feedback," the system converges to a unified rhythm (global synchronization error $\rightarrow 0$).
- **Implications for Brain Dynamics**:
    - **E-I Balance**: This reflects the delicate Excitatory-Inhibitory balance crucial for healthy brain function.
    - **Cognitive Binding**: Synchronization is hypothesized to be the mechanism behind binding different sensory features into a coherent percept. Our results suggest this binding can occur robustly even in the presence of required inhibition (e.g., lateral inhibition for sharpening signals).
    - **Pathology**: It implies that seizures (hypersynchronization) might not just be "too much excitation" but potentially a failure of the *network topology* to maintain spectral stability, regardless of the sign of individual connections.

---

## 🌪️ Foundational Theory: Chaotic Lorenz Systems

This work is built upon the theoretical findings of **Solís-Perales & Zapata (2013)**, which challenged the traditional assumption that cooperative (positive) links are necessary for synchronization.

### The Relaxed Stability Condition
Classical network theory often requires non-negative off-diagonal elements in the Laplacian. We show that this is **not necessary**. The critical requirement is that the **eigenvalues of the coupling matrix** remain in the stable region of the Master Stability Function.

$$ \dot{\mathbf{x}}_i = \mathbf{f}(\mathbf{x}_i) + C \sum_{j=1}^{N} A_{ij} (\mathbf{x}_j - \mathbf{x}_i) $$

We validate this using a network of **Lorenz Oscillators**, where $A_{ij}$ can be negative.

### Run the Chaotic Simulation
```bash
# Verify the foundational theory with Lorenz oscillators
python3 lorenz_network.py
```

## 🔑 Stability Criteria for Synchronization

The core contribution of this work is defining *when* a network with negative couplings can synchronize.

### Standard Condition (Traditional)
Standard network theory typically requires **cooperative interactions**:
$$ A_{ij} \ge 0 \quad \forall i \neq j $$
This ensures the coupling matrix has the properties of a standard Laplacian (M-matrix), guaranteeing stability if the coupling strength is sufficient.

### Relaxed Condition (Our Method)
We demonstrate that non-negativity is **not necessary**. Synchronization is preserved if the **Spectral Stability Condition** is met:

> **Condition**: The eigenvalues $\lambda_k$ of the coupling matrix $\mathcal{C}$ must lie within the stability region of the Master Stability Function (MSF) for the dynamical system.
>
> $$ \text{Re}(\lambda_k) < 0 \quad \text{for } k=2 \dots N $$

In practice, this means we can have negative links ($A_{ij} < 0$) as long as the "net" connectivity remains diffusive and stable. For Lorenz and Jansen-Rit systems, this often implies that the second largest eigenvalue $\tilde{\lambda}_2$ must satisfy:
$$ |\tilde{\lambda}_2| \ge |\bar{d}| $$
Where $\bar{d}$ is the maximum Lyapunov exponent (divergence) of the isolated node dynamics.

## 💻 Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/zapata131/simulation-chaotic-systems-negative-couplings.git
    cd simulation-chaotic-systems-negative-couplings
    ```

2.  **Install scientific dependencies**:
    ```bash
    pip install -r requirements.txt
    ```



## 📚 References

> **G. Solís-Perales and J. L. Zapata**, "Synchronization of Complex Networks with Negative Couplings," *2013 International Conference on Physics and Control (PHYSCON 2013)*, San Luis Potosí, Mexico, Aug. 2013.

---
*Maintained by [Zapata131](https://github.com/zapata131)*