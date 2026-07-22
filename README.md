


# PML-DDT: Dual Dynamic Thresholds for Partial Multi-label Learning

Official reproducibility repository for the paper:

> **Note:** The full training and evaluation code will be released in this repository upon acceptance of the paper.
---

## 1. Overview

PML-DDT is a two-stage framework for Partial Multi-label Learning (PML):

1. **Stage 1 — Confidence-weighted warm-up:** Each candidate label is weighted by the model's own predicted probability (with stop-gradient), i.e., $L_{\text{warmup}} = \sum_i \sum_j p_{ij} \cdot \ell_{\text{BCE}}$, which suppresses noisy candidates during early training.
2. **Stage 2 — Dual dynamic threshold (DDT) disambiguation:** An upper threshold $\tau_{\text{upper}}$ selects reliable positive labels and a lower threshold $\tau_{\text{lower}}$ selects reliable negative labels; labels in between are treated as ambiguous and masked. Both thresholds are progressively relaxed by a linear schedule, together with a dynamic asymmetric loss (dynamic ASL) and a mean-constraint regularizer $L_{\text{mc}}$ (margin $\delta = 0.5$), giving the total loss $L_{\text{total}} = L_{\text{dis}} + \lambda_{\text{mc}} \cdot L_{\text{mc}}$.

---


## 2. Synthetic Noise Generation

### 3.1 Random false-positive noise (main protocol)

For each training instance, every **irrelevant (non-candidate) label** is independently flipped into the candidate set with probability $\rho$ (Bernoulli), using the RNG determined by the seed. Noise rates: $\rho \in \{0.1, 0.2, 0.4, 0.6, 0.8\}$.

### 3.2 Instance-dependent noise (supplementary)

A ResNet-50 model is pre-trained with ground-truth labels and used to score each sample; the top-$k$ highest-scoring negative labels are injected as false positives, where $k = r \times$ (#ground-truth labels of the sample), with $r \in \{0.5, 1.0, 1.5\}$.

---

## 3. Complete Implementation Configurations

### 3.1 Per-dataset threshold initialization & update schedule

The thresholds follow the **linear schedule** of Eqs. (12)–(13): starting from $(\tau_{\text{upper\_init}}, \tau_{\text{lower\_init}})$, they are updated every $n$ epochs by a fixed step until epoch $T_{\text{max}}$, reaching $(\tau_{\text{upper\_final}}, \tau_{\text{lower\_final}})$.

| Hyperparameter | VOC 2007 | MS-COCO | NUS-WIDE |
| :--- | :--- | :--- | :--- |
| **$\tau_{\text{upper}}$ (init)** | TBD (e.g., 0.90) | TBD | TBD |
| **$\tau_{\text{lower}}$ (init)** | TBD (e.g., 0.05) | TBD | TBD |
| **$\tau_{\text{upper}}$ (final)**| TBD | TBD | TBD |
| **$\tau_{\text{lower}}$ (final)**| TBD | TBD | TBD |
| **Update interval $n$ (epochs)** | TBD | TBD | TBD |
| **Max schedule epoch $T_{\text{max}}$** | TBD | TBD | TBD |
| **Step size per update** | TBD | TBD | TBD |

> **Sensitivity:** As shown in Fig. 7 of the paper, performance is stable for $\tau_{\text{upper\_init}} \in [0.90, 0.95]$ and $\tau_{\text{lower\_init}} \in [0, 0.1]$; Table VIII shows that alternative schedules (exponential / logarithmic / polynomial) perform comparably, indicating the gain stems from the dual-threshold mechanism itself rather than a particular schedule.

---

### 3.2 Random Seeds

*   **Seed set:** $S = \{0, 1, 2, 3, 4\}$.
*   Each seed jointly controls: (i) synthetic-noise generation, (ii) network initialization, (iii) data-loader shuffling, and (iv) data augmentation.

---

## 4. Expected Results

Mean mAP (%) over 3 runs (see paper Tables I–III for full results with std, CF1, and OF1):

| Dataset | $\rho=0.1$ | $\rho=0.2$ | $\rho=0.4$ | $\rho=0.6$ | $\rho=0.8$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **VOC 2007** | TBD | TBD | TBD | TBD | TBD |
| **MS-COCO** | TBD | TBD | TBD | TBD | TBD |
| **NUS-WIDE** | TBD | TBD | TBD | TBD | TBD |

---

## 5. Contact

For questions, please open an issue or contact: `jkguo0508@163.com`.

```
