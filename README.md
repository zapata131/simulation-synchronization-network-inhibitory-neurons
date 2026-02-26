# Spectral Stability & Synchronization in Mixed-Sign Neural Networks (Scaled Edition)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Scientific Computing](https://img.shields.io/badge/SciPy-Stack-orange?style=flat-square&logo=scipy&logoColor=white)](https://scipy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Rigorous-brightgreen?style=flat-square)](https://github.com/zapata131/simulation-chaotic-systems-negative-couplings)

> **Simulating the emergence of synchronization in large-scale ($N=100$) networks with biologically realistic inhibition (Dale's Law).**

This repository explores how networks with inhibitory (negative) connections can achieve global synchronization. We demonstrate that provided the topology satisfies a **Relaxed Spectral Stability Condition**, global coherence is a robust attractor even in networks with 20% inhibitory populations.

![Jansen-Rit Scaled Simulation](paper/figures_full/figure_4_full_dynamics.png)
*Figure 1: Robust synchronization of a 100-node Dale-compliant neural network (Basin Stability = 1.0).*

## 🧠 Biological Modelling: Dale-Compliant Networks

This project applies the "negative coupling" concept to the study of cortical networks, enforcing **Dale's Law**:
- **Node Classification**: Each neuron is either Excitatory or Inhibitory, and its output sign is fixed for all synapses.
- **Structural Proportions**: We utilize an 80/20 E-I ratio, mimicking mammalian cortical architecture.
- **Large Scale**: Validated on high-dimensional networks ($N=100$) to move beyond small-system artifacts.

### Key Capabilities
- **Cortical Mass Simulation**: High-fidelity implementation of scaled Jansen-Rit ODEs.
- **Dale-Compliant Topology**: Realistic mixed-sign networks where inhibition is a node property, not just an edge weight.
- **Statistical Basin Stability**: Monte Carlo evidence ($M=100$ trials) proving that synchronization is a near-universal attractor.

### Run the Simulation
```bash
# Full Edition (N=100, Dale's Law, Monte Carlo)
python3 full-simulation/generate_figures_full.py

# Simple Edition (N=20 baseline)
python3 simple-simulation/generate_figures.py
```
*Output: Generates statistical visualizations in `paper/figures_full/`.*

### Research Insights

Our simulations reveal a critical mechanism: **Inhibition does not preclude synchronization; it structures it.**

- **Binding by Inhibition**: Contrary to the intuition that inhibition segregates activity, our results show that **inhibitory interneurons can actively bind functional assemblies**. 
- **Statistical Robustness**: The system demonstrates 100% synchronization success ($S=1.0$) across heterogeneous initial states, provided the global topology remains spectrally stable ($\text{Re}(\lambda) \le 0$).

> **Read the full paper**: The complete scientific paper, including mathematical proofs and extensive citations, is available in the `paper/` directory.

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

### Analysis: Statistical Robustness
We provide a standalone tool to quantify how rare stable configurations are. This script runs a Monte Carlo simulation to estimate the probability of finding a spectrally stable network as the fraction of inhibitory edges increases.

```bash
# Run the robustness analysis
python3 probability_stability.py
```
*Output: Generates `probability_stability.png`, plotting the likelihood of stability vs. inhibition percentage.*

### Analysis: Noise Robustness
We also verify that synchronization survives in the presence of biologically realistic noise (Additive White Gaussian Noise).

```bash
# Run the noise robustness check
python3 paper/generate_noise_figure.py
```
*Output: Generates `figure_5_noise.png`, showing synchronization error vs noise intensity.*

![Noise Robustness](noise_robustness.png)

## 🔑 Stability Criteria for Synchronization

The core contribution of this work is defining *when* a network with negative couplings can synchronize.

### Standard Condition (Traditional)
Standard network theory typically requires **cooperative interactions**:
$$ A_{ij} \ge 0 \quad \forall i \neq j $$
This ensures the coupling matrix has the properties of a standard Laplacian (M-matrix), guaranteeing stability if the coupling strength is sufficient.

We demonstrate that non-negativity is **not necessary**. Synchronization is preserved if the **Spectral Stability Condition** is met:

> [!IMPORTANT]
> **Condition**: The eigenvalues $\lambda_k$ of the coupling matrix $\mathcal{C}$ must lie within the stability region of the Master Stability Function (MSF) for the dynamical system.
>
> $$ \text{Re}(\lambda_k) < 0 \quad \text{for } \quad k=2 \dots N $$

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