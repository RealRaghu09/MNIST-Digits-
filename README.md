# MNIST Handwritten Digits — Project README

> **Dataset Source:** [Kaggle — Handwritten Digits 0-9 by olafkrastovski](https://www.kaggle.com/datasets/olafkrastovski/handwritten-digits-0-9)
---

## Project Overview

This project works with the **MNIST-style handwritten digits dataset** (digits 0–9) downloaded via the KaggleHub API. The pipeline demonstrates how to:

- Download the dataset programmatically using `kagglehub`
- Read and inspect images using **PIL (Pillow)**
- Convert image data into **NumPy arrays** for numerical processing
- Wrap the dataset into a **PyTorch `Dataset`** class for use in training deep learning models

---

## Dependencies


| Library         | Purpose                                              |
|-----------------|------------------------------------------------------|
| `kagglehub`     | Download datasets directly from Kaggle               |
| `Pillow (PIL)`  | Open, resize, and manipulate image files             |
| `numpy`         | Convert images to arrays for numerical operations    |
| `torch`         | Build tensors and integrate with the training loop   |
| `torchvision`   | Transforms and standard dataset utilities            |
| `matplotlib`    | Visualize images and pixel distributions             |

---

## Dataset Download

Use `kagglehub` to automatically download the dataset to your local cache:

```python
import kagglehub

# Downloads dataset and returns the local path
path = kagglehub.dataset_download("olafkrastovski/handwritten-digits-0-9")

print(f"Dataset downloaded to: {path}")
```



## Dataset Structure

After downloading, the dataset typically has the following folder structure:

```
handwritten-digits-0-9/
├── 0/
│   ├── img_0001.png
│   ├── img_0002.png
│   └── ...
├── 1/
│   ├── img_0001.png
│   └── ...
├── 2/
│   └── ...
...
└── 9/
    └── ...
```

Each **subfolder** is named after the digit class (0–9) it contains. Each image is a grayscale PNG, typically **28×28 pixels** — consistent with the classic MNIST format.

---

## Loading Images with PIL


**Common PIL image modes for MNIST:**
- `'L'` — 8-bit grayscale (most common for MNIST)
- `'RGB'` — 3-channel color (if the dataset stores color images)


---


**Why normalize?**
- Raw pixel values range from 0 to 255 (integers)
- Neural networks train more stably with inputs in the range [0.0, 1.0] or [-1.0, 1.0]
- Dividing by 255.0 scales values to [0.0, 1.0]

---

## Building a PyTorch Dataset


**DataLoader parameters:**
- `batch_size=64` — number of samples returned per iteration; adjust based on GPU memory
- `shuffle=True` — randomizes order every epoch (important for training, not for validation)
- `num_workers=2` — number of CPU subprocesses used for parallel data loading

---
