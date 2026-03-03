<div align="center">

# NewsLies

**Arabic Fake News Detection using LSTM**

![Status](https://img.shields.io/badge/STATUS-UNDER_DEVELOPMENT-orange?style=for-the-badge&logo=github)  
[![Live Demo](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-blue?style=flat-square)](https://assem-elqersh.github.io/NewsLies/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

</div>

A deep learning project that classifies Arabic news articles as **credible**, **not credible**, or **undecided**, trained on the Arabic Fake News Dataset (AFND) using a two-layer LSTM model.

> This project is still under development.

---

## Live Demo

Try the model directly in your browser — no installation required:

**[https://assem-elqersh.github.io/NewsLies/](https://assem-elqersh.github.io/NewsLies/)**

All inference runs client-side via WebAssembly (ONNX Runtime Web). No data is sent to any server.

---

## Dataset

**[Arabic Fake News Dataset (AFND)](https://www.kaggle.com/datasets/murtadhayaseen/arabic-fake-news-dataset-afnd)** — Kaggle

| | |
|---|---|
| Total articles | ~606,000 |
| News sources | 134 Arabic websites |
| Labels | `credible` · `not credible` · `undecided` |
| Source split | 52 credible · 51 not credible · 31 undecided |

Each article contains `title`, `text`, and `published date`. Source URLs are anonymized (`source_1` … `source_134`).

Dataset created by Ashwaq Khalil, Moath Jarrah, and Monther Aldwairi.

---

## Project Structure

```
NewsLies/
├── train_tensorflow.py   # TensorFlow/Keras LSTM trainer (CPU only)
├── train_pytorch.py      # PyTorch LSTM trainer (GPU auto-detected)
├── export_model.py       # Export trained model to ONNX for the web demo
├── requirements.txt
├── docs/                 # GitHub Pages inference app
│   ├── index.html
│   ├── js/app.js
│   ├── model.onnx        # Exported ONNX model (~4 MB)
│   ├── vocab.json        # Tokenizer vocabulary
│   └── stopwords.json    # Arabic stopwords
└── data/                 # Dataset (not tracked — download separately)
    └── AFND/
        ├── sources.json
        └── Dataset/
            └── source_N/
                └── scraped_articles.json
```

---

## Setup

### 1. Create environment (Python 3.11 required for TF 2.14)

```bash
conda create -n newslies python=3.11 -y
conda activate newslies
pip install -r requirements.txt
```

For GPU support with PyTorch (recommended — uses your NVIDIA GPU automatically):

```bash
# Check CUDA version with: nvidia-smi
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 2. Download NLTK stopwords

```bash
python -c "import nltk; nltk.download('stopwords')"
```

### 3. Download the dataset

```bash
pip install kaggle
# Place your ~/.kaggle/kaggle.json credentials first
kaggle datasets download -d murtadhayaseen/arabic-fake-news-dataset-afnd -p data --unzip
```

---

## Training

### PyTorch — uses GPU automatically

```bash
python train_pytorch.py --data-dir data/AFND
```

### TensorFlow — CPU only (TF 2.14 requires CUDA 11.x)

```bash
python train_tensorflow.py --data-dir data/AFND
```

Both scripts accept `--data-dir` or the `AFND_DATA_DIR` environment variable. Outputs are saved to `outputs/`.

---

## Model

### Architecture

```
Input tokens (padded to 128)
    ↓
Embedding  (10,000 vocab × 100 dims)
    ↓
SpatialDropout1D  (p = 0.2)
    ↓
LSTM  (64 units, return sequences)
    ↓
LSTM  (64 units)
    ↓
Dense / Linear  (3 classes, softmax)
    ↓
credible | not credible | undecided
```

### Hyperparameters

| Parameter | Value |
|---|---|
| Vocabulary size | 10,000 |
| Sequence length | 128 |
| Embedding dim | 100 |
| LSTM units | 64 |
| Dropout | 0.2 |
| Batch size | 64 |
| Learning rate | 1e-4 |
| Early stopping patience | 2 |

### Text preprocessing

Raw text → lowercase → Arabic stopword removal (NLTK) → ISRI stemming

### Results

| Metric | Value |
|---|---|
| Test accuracy | 70.3% |
| Credible F1 | 0.73 |
| Not Credible F1 | 0.58 |
| Undecided F1 | 0.75 |

Trained on 388k articles, validated on 97k, tested on 121k.

---

## GPU vs CPU

| Script | Device | Notes |
|---|---|---|
| `train_pytorch.py` | GPU (auto-detected) | Works with CUDA 12+ / 13 |
| `train_tensorflow.py` | CPU only | TF 2.14 requires CUDA 11.x |

---

## Deploying the Web Demo

After training, export the model and deploy to GitHub Pages:

```bash
# 1. Export ONNX model + vocab + stopwords to docs/
pip install onnx onnxruntime
python export_model.py

# 2. Commit and push
git add docs/
git commit -m "update web demo"
git push
```

Then enable GitHub Pages: **Settings → Pages → Source: main branch, /docs folder**.

---

## License

MIT License — see [LICENSE](LICENSE).

The AFND dataset does not specify a license. Review the usage terms set by the dataset creators before any commercial use.
