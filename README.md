# FASOLA: Robust Multi-Source-Free Domain Adaptation via Posterior Adjustment and Label Agreement

This project is a PyTorch implementation of **FASOLA**  
(*Robust Multi-Source-Free Domain Adaptation via Posterior Adjustment and Label Agreement*).

FASOLA addresses **Multi-Source-Free Domain Adaptation (MSFDA)** by (i) correcting prediction bias via **Posterior Adjustment** and (ii) prioritizing reliable sources via **Label Agreement**, enabling robust adaptation without access to source data.

---

## Prerequisites

Our implementation is based on Python 3.9 and PyTorch.  
Please refer to `requirements.txt` for the complete package list.

- Python ≥ 3.9
- PyTorch ≥ 1.13.1
- Torchaudio ≥ 0.13.1
- NumPy ≥ 1.24.3
- Pandas ≥ 2.0.3
- Librosa ≥ 0.10.1

---

## Datasets

FASOLA is evaluated on the **DCASE 2020 Task 1A** dataset (Acoustic Scene Classification).  
You can find the raw dataset at the following Zenodo record:

- Dataset: https://zenodo.org/records/3819968

### What `--data_dir` should be

Do **not** assume a custom folder layout.  
Simply set `--data_dir` to the **parent directory that contains the `audio/` folder** from the downloaded dataset.

If your dataset directory contains:

```

<DATA_DIR>/
└── audio/
├── evaluation_setup/
└── meta.csv

````

then you should pass:

```bash
python preprocess_dcase.py --data_dir <DATA_DIR>
````

### Domains used in this project

* The dataset provides domains: `a, b, c, s1, s2, s3, s4, s5, s6`.
* In our experiments, we **evaluate only on** the six device domains: **`s1`–`s6`**.
* For each run, one of `s1`–`s6` is chosen as the **target**, and **all remaining domains except the target** are used as sources (MSFDA protocol).

| Target | Sources (all except target) |
| ------ | --------------------------- |
| s1     | a, b, c, s2, s3, s4, s5, s6 |
| s2     | a, b, c, s1, s3, s4, s5, s6 |
| s3     | a, b, c, s1, s2, s4, s5, s6 |
| s4     | a, b, c, s1, s2, s3, s5, s6 |
| s5     | a, b, c, s1, s2, s3, s4, s6 |
| s6     | a, b, c, s1, s2, s3, s4, s5 |

## Usage

### 1) Preprocessing (raw audio → cached features)

We preprocess raw audio once and cache features for faster training/adaptation.

```bash
python preprocess_dcase.py \
  --data_dir <DATA_DIR>
````

---

### 2) Train Source Models (one model per source domain)

```bash
python train_source_models.py 
```

### 3) Adaptation (FASOLA) on the Target Domain

```bash
python run.py \
  --target s2 \
  --gpu 0
```

---

## Arguments

### General & Data

| Argument       | Example                    | Description                                   |
| -------------- | -------------------------- | --------------------------------------------- |
| `--targets`     | `s2`                       | Target domain (one at a time)                 |
| `--data_dir`   | `data/processed_dcase2020` | Preprocessed feature directory                |
| `--model_dir`  | `saved_models/`            | Directory containing pretrained source models |
| `--batch_size` | `64`                       | Batch size                                    |
| `--seed`       | `42`                       | Random seed                                   |
| `--gpu`        | `0`                        | GPU id                                        |

### FASOLA Hyperparameters

| Argument           | Default | Description                               |
| ------------------ | ------: | ----------------------------------------- |
| `--tau`            |   `2.0` | Temperature for posterior adjustment      |
| `--sharpness`      |  `10.0` | Sharpness for entropy minimization        |
| `--em_iters`       |     `5` | EM iterations for target prior estimation |
| `--momentum_alpha` |   `0.2` | Momentum for prior update                 || `--lr`             |  `0.01` | Learning rate                             |



## Reference
If you use this code, please cite the following paper.
```
@article{yoon2026fasola,
  title={FASOLA: Robust Multi-Source-Free Domain Adaptation via Posterior Adjustment and Label Agreement},
  author={Yoon, hoyoung and Kang, U.},
  booktitle={Interspeech},
  year={2026},
}
```


