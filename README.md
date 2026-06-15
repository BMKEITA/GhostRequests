# Ghost Requests: Stealthy and Non-Attributable Adversarial Unlearning Attacks in Federated Learning Systems

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.3.1-orange?logo=pytorch" />
  <img src="https://img.shields.io/badge/CUDA-12.1-green?logo=nvidia" />
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
  <img src="https://img.shields.io/badge/Status-IEEE%20TIFS%20Submission-red" />
</p>
## Overview

**Ghost Requests** is an adversarial unlearning framework that exploits the **Federated Unlearning (FU)** pipeline in Federated Learning (FL) systems. While FU is designed to satisfy privacy regulations (GDPR, CCPA — the *right to be forgotten*), Ghost Requests demonstrates that this process can be **weaponized** to silently redirect global model predictions toward a chosen misclassification.

Unlike prior work (FedMUA, BadFU), Ghost Requests treats **stealth as a hard constraint** rather than a soft objective. By combining **Solo-Optimization** with **Bootstrap Gradient Rescaling**, the attack produces updates that are statistically indistinguishable from benign client updates — achieving a **0.000 IQR Detection Rate by construction**.

### Key Results ( — NVIDIA TITAN Xp)

| Metric | Ghost Requests | FedMUA (baseline) |
|--------|---------------|-------------------|
| **Attack Success Rate (ASR)** | **100%** (9/10 configs) | ~95% |
| **DR_IQR** | **0.000** (all configs) | 0.850 |
| **Stealth Index (SI)** | **≥ 0.939** | 0.082 |
| **Improvement over FedMUA** | **11.4×** | — |
| **Grey-box (25% proxy)** | ASR = 1.000, SI = 0.971 | N/A |
---
##  Repository Structure

```
GhostRequests/
├── README.md
│── datasets
│   │── MNIST
│   └── CIFAR10
├── results/
├── ghost_model.py
├── ghost_data.py
├── ghost_metrics.py
├── ghost_requests.py
├── ghost_main.py
├── run_experiments.py
├── ghost_main.py
└── ghost_main.py
```
##  Installation
### Requirements
- Python ≥ 3.10
- CUDA 12.1 (recommended: NVIDIA GPU with ≥ 8 GB VRAM)
- PyTorch 2.3.1
### Setup
```bash
# Clone the repository
git clone https://github.com/BMKEITA/GhostRequests.git
cd GhostRequests

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

---

##  Experimental Results

All results were obtained on **NVIDIA TITAN Xp (12.8 GB VRAM)**, PyTorch 2.3.1+cu121, seed = 42.

### Main Results (10 Configurations)

| Configuration | AccG (%) | ASR | GNR | DR_IQR | DR_KL | SI |
|---------------|----------|-----|-----|--------|-------|----|
| Benign (ref) | 95.12 | — | 1.000 | 0.250 | 0.000 | 0.817 |
| FedMUA (ref) | 95.87 | 1.000 | 2.100 | 0.850 | 0.600 | 0.082 |
| MNIST IID FedAvg | 96.13 | **1.000** | 0.956 | **0.000** | 0.067 | 0.970 |
| MNIST Non-IID FedAvg | 88.93 | **1.000** | 1.076 | **0.000** | 0.115 | 0.949 |
| MNIST IID Median | 96.20 | **1.000** | 0.973 | **0.000** | 0.041 | **0.982** |
| MNIST IID Trimmed-Mean | 96.10 | **1.000** | 0.957 | **0.000** | 0.065 | 0.971 |
| MNIST IID Krum | 94.88 | **1.000** | 1.060 | **0.000** | 0.060 | 0.970 |
| CIFAR-10 IID FedAvg | 86.91 | **1.000** | 0.996 | **0.000** | 0.005 | **0.997** |
| CIFAR-10 Non-IID FedAvg | 29.55 | 0.000† | 1.010 | **0.000** | 0.009 | **0.995** |
| CIFAR-10 IID Median | 83.71 | **1.000** | 0.987 | **0.000** | 0.023 | 0.990 |
| CIFAR-10 IID Trimmed-Mean | 86.95 | **1.000** | 0.951 | **0.000** | 0.084 | 0.964 |
| CIFAR-10 IID Krum | 76.18 | **1.000** | 0.951 | **0.000** | 0.158 | 0.939 |

> † ASR = 0.000 due to model non-convergence (AccG = 29.55%); stealth is unaffected (SI = 0.995).

### Defense Resistance (FedMUA IQR Clipping)

| λ_def | ASR | SI | DR_IQR | Note |
|-------|-----|----|--------|------|
| 1.0 (no defense) | **1.000** | **0.971** | **0.000** | Full stealth preserved |
| 0.5 | **1.000** | 0.382 | 1.000 | Defense clips benign updates too |
| 0.1 | **1.000** | 0.186 | 1.000 | Impractical: destroys model utility |

### Grey-Box Evaluation (MNIST IID FedAvg)

| Proxy Checkpoint | Proxy Acc (%) | ASR | SI |
|------------------|--------------|-----|----|
| 25% | 89.3 | **1.000** | **0.971** |
| 50% | 92.7 | **1.000** | **0.971** |
| 75% | 94.7 | **1.000** | **0.971** |
| 100% (white-box) | 96.1 | **1.000** | **0.971** |

---

##  Stealth Metrics

| Metric | Formula | Ideal Value |
|--------|---------|-------------|
| **GNR** | $$\|g^*\|_2 \;/\; \mathbb{E}[\mathcal{B}]$$ | 1.0 |
| **DR_IQR** | $$\mathbf{1}[\|g^*\|_2 \notin [Q_1, Q_3]]$$ | 0.0 |
| **DR_KL** | Normalized KL deviation from benign mean | 0.0 |
| **SI** | $$1 - \frac{p_{\text{GNR}} + \text{DR}_{\text{IQR}} + \text{DR}_{\text{KL}}}{3}$$ | 1.0 |

---

##  Hardware & Runtime

| Dataset | ISI Time | MUG Time | Total Overhead |
|---------|----------|----------|----------------|
| MNIST (avg) | ~2.8 s | ~0.8 s | **~3.1 s/round** |
| CIFAR-10 (avg) | ~12.5 s | ~7.4 s | **~20.7 s/round** |

> Overhead is negligible relative to a standard FL training round (tens of seconds to minutes).

---

##  Defense Recommendations

Based on our analysis, we recommend the following directions for future FU defenses:

1. **Multi-modal detection** — Combine gradient-norm analysis with direction-aware detectors (cosine similarity screening).
2. **Adaptive IQR thresholds** — Flag updates suspiciously *close* to the median, not just far from it.
3. **Per-client contribution caps** — DP-style clipping to limit any single unlearning request's model shift.
4. **Unlearning request verification** — Cryptographic proof-of-ownership to prevent request forgery.

---

##  Citation

If you use this code in your research, please cite:

```bibtex
@article{keita2026ghostrequests,
  title     = {Ghost Requests: Stealthy and Non-Attributable Adversarial
               Unlearning Attacks in Federated Learning Systems},
  author    = {Keita, B.M.},
  journal   = {IEEE Transactions on Information Forensics and Security},
  year      = {2026},
  note      = {Under review}
}
```

---

##  Ethical Statement

This research is intended **solely for academic and defensive purposes**. The Ghost Requests framework is published to:
- Expose a critical and previously underexplored vulnerability in Federated Unlearning pipelines.
- Enable the research community to develop robust defenses.
- Advance the understanding of the **privacy–security gap** in certified unlearning systems.

The authors strongly discourage any malicious use of this code.

---

<p align="center">
  <b>Ghost Requests </b> · NVIDIA TITAN Xp · PyTorch 2.3.1 · Seed 42
</p>
