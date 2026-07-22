


# PML-DDT: Dual Dynamic Thresholds for Partial Multi-label Learning

Official reproducibility repository for the paper:

> **Note:** The full training and evaluation code will be released in this repository upon acceptance of the paper.
---

## 1. Overview

This repository implements PML-DDT, a robust label disambiguation framework based on dual dynamic thresholds. It adaptively selects highly reliable positive and negative labels from candidate sets, effectively suppressing noise interference. As training progresses, more increasingly challenging samples are incorporated to enhance the model’s generalization ability through a progressive learning strategy.
---



## 2. Complete Implementation Configurations

### 2.1 Per-dataset threshold initialization & update schedule

The thresholds follow the **linear schedule** of Eqs. (12)–(13): starting from $(\tau_{\text{upper}}^{\text{init}}, \tau_{\text{lower}}^{\text{init}})$, they are updated every epoch by a fixed step until epoch $T_{\text{max}}$, reaching $(\tau_{\text{upper}}^{\text{final}}, \tau_{\text{lower}}^{\text{final}})$.



#### PASCAL VOC


| Noise rate | $\tau_{\text{upper}}^{\text{init}}$ | $\tau_{\text{lower}}^{\text{init}}$ | Threshold update epochs | ASL ($\gamma^-$) |
| :---: | :---: | :---: | :---: | :---: |
| 0.1 | 0.9 | 0.0 | 60 | 5
| 0.2 | 0.9 | 0.0 | 60 | 5
| 0.4 | 0.9 | 0.0 | 60 | 4
| 0.6 | 0.9 | 0.0 | 60 | 3
| 0.8 | 0.95 | 0.0 | 60 | 2

#### MS-COCO and NUS-WIDE

| Noise rate | $\tau_{\text{upper}}^{\text{init}}$ | $\tau_{\text{lower}}^{\text{init}}$ | Threshold update epochs | ASL ($\gamma^-$) |
| :---: | :---: | :---: | :---: | :---: |
| 0.1 | 0.8 | 0.0 | 30 | 5
| 0.2 | 0.8 | 0.0 | 30 | 5
| 0.4 | 0.8 | 0.0 | 30 | 4
| 0.6 | 0.8 | 0.0 | 30 | 3
| 0.8 | 0.85 | 0.0 | 30 | 2

---

### 2.2 Random Seeds

*   **Seed set:** $S = \{0, 1, 2, 3, 4\}$.
*   Each seed jointly controls: (i) synthetic-noise generation, (ii) network initialization, (iii) data-loader shuffling, and (iv) data augmentation.

---


## 3. Contact

For questions, please open an issue or contact: `jkguo0508@163.com`.

```
