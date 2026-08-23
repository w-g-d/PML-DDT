# PML-DDT: Dual Dynamic Thresholds for Partial Multi-label Learning

PyTorch implementation of PML-DDT

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![Datasets](https://img.shields.io/badge/Datasets-VOC2007%20%7C%20COCO2014%20%7C%20NUS--WIDE-orange)

---

## 1. Overview

> **PML-DDT**: A robust label disambiguation framework based on dual dynamic thresholds.

&emsp;&emsp;This repository implements **PML-DDT**, a robust label disambiguation framework based on dual dynamic thresholds. It adaptively selects highly reliable positive and negative labels from candidate sets, effectively suppressing noise interference. 

&emsp;&emsp;As training progresses, increasingly challenging samples are incorporated to enhance the model’s generalization ability through a progressive learning strategy.

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




## 3. Dataset Structure

<details>
<summary><b>VOC2007</b></summary>

```
<data-path>/VOC/
├── VOCtest_06-Nov-2007/
│   └── VOCdevkit/
│       └── VOC2007/
│           ├── Annotations/
│           ├── ImageSets/
│           ├── JPEGImages/
│           ├── SegmentationClass/
│           └── SegmentationObject/
└── VOCtrainval_06-Nov-2007/
    └── VOCdevkit/
        └── VOC2007/
            └── ... (same structure)
```
</details>

<details>
<summary><b>MS-COCO 2014</b></summary>

```
<data-path>/COCO/
├── train2014/
│   └── COCO_train2014_xxx.jpg
└── val2014/
    └── COCO_val2014_xxx.jpg
```

The annotation files (`train_anno.json`, `val_anno.json`) are already provided in [`dataset/coco/`](dataset/coco/).
</details>

<details>
<summary><b>NUS-WIDE</b></summary>

```
<data-path>/NUSWIDE/
├── ImageList/
│   ├── TrainImagelist.txt
│   └── TestImagelist.txt
├── Groundtruth/
│   ├── TrainTestLabels/
│   │   ├── Labels_airport_Test.txt
│   │   └── ...
│   └── AllLabels/
└── Flickr/
    └── <category>/<image>.jpg
```
</details>


## 4. Contact

For questions, please open an issue or contact: `jkguo0508@163.com`.

