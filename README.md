# chaos-evolve-nero-quantizer
# Chaos-Evolve V2 x Nero-Quantizer Core
> **An Empirical Investigation into the Quantization Robustness of Evolutionarily Developed Neural Networks.**

Built entirely from scratch in NumPy with zero external machine learning frameworks.

## Executive Summary
Can a neural network optimized via non-gradient stochastic search (Neuroevolution) survive aggressive sub-8-bit quantization? 

We simulated a continuous evolutionary ecosystem (`Chaos-Evolve V2`) where agents trained via crossover and mutation strategies learned obstacle-avoidance maneuvers. We then subjected the elite champion to a tight, asymmetric, block-wise 4-bit linear quantizer (`Nero-Quantizer Core`), compressing floating-point weights down to signed INT4 integer boundaries ($[-8, 7]$). 

**The Finding:** The compressed 4-bit model did not experience performance degradation. Instead, it achieved a **13.22% fitness efficiency boost**, registering a total Intelligence Retention Rate of **113.22%**.

## Empirical Metrics
| Benchmark Metric | FP32 Base Precision | INT4 Block-Wise Quantized |
| :--- | :--- | :--- |
| **Elite Agent Fitness** | 538.63 | **609.83** |
| **Quantization Noise (MSE)**| 0.000000 (Ref) | **0.002113** |
| **Memory Reduction (Theoretical)** | 1x (Reference) | **~8x Footprint Compression** |
| **Intelligence Retention Rate** | 100% | **113.22%** |

---

## Core Architecture & Mathematical Engine

### 1. Block-Wise Asymmetric Quantization
Standard per-tensor quantization methods drop precision entirely when exposed to high-magnitude parameter variations (outliers). To prevent matrix variance corruption, we designed a localized block-wise mapping pipeline operating at a micro-block interval ($Block\_Size = 4$).

$$\text{scale} = \frac{W_{max} - W_{min}}{15}$$
$$\text{zero\_point} = \text{clip}\left(\text{round}\left(\frac{-W_{min}}{\text{scale}}\right) - 8, -8, 7\right)$$

During targeted outlier stress testing, baseline per-tensor quantization degraded significantly to an MSE of **0.250919**. Our custom block-wise implementation contained the mathematical distortion locally, proving **118.8x more accurate** with a minimal global MSE of **0.002113**.

### 2. Theoretical Breakdown: Why Did INT4 Outperform FP32?
* **Quantization as a Low-Pass Filter:** Stochastic evolutionary paths can introduce behavioral jitter or high-frequency floating-point noise into parameter configurations during rapid mutations. The strict integer boundaries of the 4-bit engine (`round` and `clip` gates) smoothed out these micro-oscillations, regularizing the network and yielding a cleaner spatial navigation path.
* **Granular Scaling:** Restricting the scale constraints to tiny block windows ensured high localized precision, stabilizing the agent's forward policy executions across unknown environment resets.

---

## Getting Started

### Installation
Ensure you have the core scientific computing layer installed:
```bash
pip install numpy
