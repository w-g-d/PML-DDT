

```markdown
# PML-DDT: Dual Dynamic Thresholds for Partial Multi-label Learning

Official reproducibility repository for the paper:

> **PML-DDT: Dual Dynamic Thresholds for Partial Multi-label Learning**
> Ke Wang, Jinkui Guo, Jiaming Chang, Xize Sun, Hong Ye, Zhaohong Jia
> *Submitted to IEEE Transactions on Multimedia (TMM)*

This repository provides the **complete implementation configurations, random seeds, noise-generation protocol, and the per-dataset threshold initialization and update schedules** required to fully reproduce all results reported in the paper and its supplementary material.

> **Note:** The full training and evaluation code will be released in this repository upon acceptance of the paper. The configurations documented below are self-contained and sufficient to reproduce all experiments.

---

## 1. Overview

PML-DDT is a two-stage framework for Partial Multi-label Learning (PML):

1. **Stage 1 — Confidence-weighted warm-up:** Each candidate label is weighted by the model's own predicted probability (with stop-gradient), i.e., $L_{\text{warmup}} = \sum_i \sum_j p_{ij} \cdot \ell_{\text{BCE}}$, which suppresses noisy candidates during early training.
2. **Stage 2 — Dual dynamic threshold (DDT) disambiguation:** An upper threshold $\tau_{\text{upper}}$ selects reliable positive labels and a lower threshold $\tau_{\text{lower}}$ selects reliable negative labels; labels in between are treated as ambiguous and masked. Both thresholds are progressively relaxed by a linear schedule, together with a dynamic asymmetric loss (dynamic ASL) and a mean-constraint regularizer $L_{\text{mc}}$ (margin $\delta = 0.5$), giving the total loss $L_{\text{total}} = L_{\text{dis}} + \lambda_{\text{mc}} \cdot L_{\text{mc}}$.

---

## 2. Environment

| Item | Version |
| :--- | :--- |
| **OS** | TBD (e.g., Ubuntu 20.04) |
| **Python** | TBD (e.g., 3.9) |
| **PyTorch** | TBD |
| **CUDA / cuDNN** | TBD |
| **GPU** | TBD (e.g., 1× NVIDIA RTX 3090, 24 GB) |

**Install dependencies:**

```bash
conda create -n pmlddt python=3.9
conda activate pmlddt
pip install -r requirements.txt
```

---

## 3. Data Preparation

We evaluate on three multi-label image benchmarks:

| Dataset | #Classes | Train / Test split | Source |
| :--- | :--- | :--- | :--- |
| **PASCAL VOC 2007** | 20 | official trainval / test | [Link](http://host.robots.ox.ac.uk/pascal/VOC/) |
| **MS-COCO 2014** | 80 | official train / val | [Link](https://cocodataset.org) |
| **NUS-WIDE** | 81 | official split | [Link](https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html) |

Place the datasets as follows (or edit the `data_root` field in the configs):

```text
data/
├── voc2007/
├── coco2014/
└── nuswide/
```

---

## 4. Random Seeds & Statistical Protocol

*   **Seed set:** $S = \{0, 1, 2, 3, 4\}$.
*   Each seed jointly controls: (i) synthetic-noise generation, (ii) network initialization, (iii) data-loader shuffling, and (iv) data augmentation.
*   **Main comparison tables (Tables I–III):** mean ± std over **3 independent runs** (seeds $\{0, 1, 2\}$).
*   **Significance analysis (Table IX):** **5 independent runs** (seeds $\{0, 1, 2, 3, 4\}$); paired t-tests match our method and each baseline on the identical noisy dataset generated under the same seed.

---

## 5. Synthetic Noise Generation

### 5.1 Random false-positive noise (main protocol)

For each training instance, every **irrelevant (non-candidate) label** is independently flipped into the candidate set with probability $\rho$ (Bernoulli), using the RNG determined by the seed. Noise rates: $\rho \in \{0.1, 0.2, 0.4, 0.6, 0.8\}$.

The generated noisy candidate sets are **deterministic given (dataset, $\rho$, seed)** and are **shared identically across all compared methods** to guarantee fair comparison and valid pairing.

```bash
python tools/generate_noise.py --dataset coco --rho 0.4 --seed 0
```

### 5.2 Instance-dependent noise (supplementary)

A ResNet-50 model is pre-trained with ground-truth labels and used to score each sample; the top-$k$ highest-scoring negative labels are injected as false positives, where $k = r \times$ (#ground-truth labels of the sample), with $r \in \{0.5, 1.0, 1.5\}$.

```bash
python tools/generate_noise_instance.py --dataset coco --r 1.0 --seed 0
```

---

## 6. Complete Implementation Configurations

All configurations are also provided as per-dataset YAML files under `configs/` (e.g., `configs/voc2007.yaml`).

### 6.1 Shared configuration

| Item | Value |
| :--- | :--- |
| **Backbone** | TResNet-L (ImageNet-pretrained) |
| **Input resolution** | TBD (e.g., 448 × 448) |
| **Optimizer** | TBD (e.g., Adam) |
| **Learning rate / schedule** | TBD (e.g., 1e-4, 1-cycle) |
| **Weight decay** | TBD |
| **Batch size** | TBD |
| **Total epochs** | TBD |
| **Warm-up (Stage 1) epochs** | TBD |
| **Data augmentation** | Cutout, RandAugment (TBD parameters) |
| **Dynamic ASL** | $\gamma_+ =$ TBD; $\gamma_-$ init = TBD, decayed per Eq. (11) |
| **Mean-constraint regularizer** | margin $\delta = 0.5$; $\lambda_{\text{mc}} =$ TBD |
| **EMA** | TBD (if used) |

### 6.2 Per-dataset threshold initialization & update schedule

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

## 7. Training & Evaluation

Train on a given dataset / noise rate / seed:

```bash
python train.py --config configs/coco2014.yaml --rho 0.4 --seed 0
```

Evaluate (mAP / CF1 / OF1):

```bash
python evaluate.py --config configs/coco2014.yaml --checkpoint <path>
```

Reproduce a full table (all noise rates × seeds):

```bash
bash scripts/run_coco_all.sh
```

Reproduce the significance analysis (Table IX):

```bash
python tools/significance_test.py --dataset coco --rho 0.6 --runs 5
```

---

## 8. Expected Results

Mean mAP (%) over 3 runs (see paper Tables I–III for full results with std, CF1, and OF1):

| Dataset | $\rho=0.1$ | $\rho=0.2$ | $\rho=0.4$ | $\rho=0.6$ | $\rho=0.8$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **VOC 2007** | TBD | TBD | TBD | TBD | TBD |
| **MS-COCO** | TBD | TBD | TBD | TBD | TBD |
| **NUS-WIDE** | TBD | TBD | TBD | TBD | TBD |

---

## 9. Citation

```bibtex
@article{wang2026pmlddt,
  title   = {PML-DDT: Dual Dynamic Thresholds for Partial Multi-label Learning},
  author  = {Wang, Ke and Guo, Jinkui and Chang, Jiaming and Sun, Xize and Ye, Hong and Jia, Zhaohong},
  journal = {IEEE Transactions on Multimedia},
  year    = {2026},
  note    = {under review}
}
```

## 10. Contact

For questions, please open an issue or contact: `jkguo0508@163.com`.

```
