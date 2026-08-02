# MNIST MLP: Classification, Error Analysis & Representation Visualization

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

A PyTorch MLP for MNIST digit classification, paired with a full model
analysis toolkit: confusion matrices, error inspection, PCA/t-SNE
representation visualization, and an exploratory activation-search
experiment.

## Overview

This project trains a 3-layer Multi-Layer Perceptron on the MNIST
handwritten digit dataset, using data augmentation and validation-based
checkpointing. Beyond training, it provides a complete diagnostic suite for
understanding *how* the model classifies digits — not just *how well*.

## Features

- Clean, modular PyTorch implementation with a configurable `Config` dataclass
- Data augmentation (random rotation + crop) for the training set only
- Automatic dataset mean/std computation for correct normalization
- Best-checkpoint saving based on validation loss
- Training/validation loss & accuracy curves
- Test-set confusion matrix
- Inspection of the most confidently *incorrect* predictions
- PCA and t-SNE visualization of learned output and hidden-layer representations
- An "imagined digit" experiment probing the model's decision boundary via random search
- First-layer weight visualization
- GPU-accelerated when CUDA is available, with automatic CPU fallback

## Project Structure

```
mnist-mlp-analysis/
│
├── mnist_mlp_analysis.ipynb   # Main notebook: data, model, training, full analysis suite
├── README.md                  # Project documentation (this file)
├── requirements.txt           # Python dependencies
├── LICENSE                    # MIT License
├── .gitignore                 # Ignored files (data, checkpoints, caches)
├── .data/                     # MNIST dataset (downloaded automatically, gitignored)
├── assets/                    # Images used in this README
└── output/                    # Saved plots / model checkpoints (optional)
```

## Requirements

- Python 3.9+
- PyTorch 2.0+
- Torchvision 0.15+
- scikit-learn 1.3+
- Matplotlib, NumPy, tqdm

See [`requirements.txt`](requirements.txt) for pinned minimum versions.

## Installation

```bash
git clone https://github.com/<your-username>/mnist-mlp-analysis.git
cd mnist-mlp-analysis
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

1. Launch Jupyter:
   ```bash
   jupyter notebook mnist_mlp_analysis.ipynb
   ```
2. Run all cells in order. MNIST will be downloaded automatically into
   `./.data` on first run.
3. Training progress (loss/accuracy per epoch) prints to the notebook
   output, followed by loss/accuracy curves, a confusion matrix, error
   analysis, PCA/t-SNE plots, the imagined-digit experiment, and weight
   visualizations.

> **Note:** the t-SNE steps and the imagined-digit random search are the
> slowest parts of the notebook — expect these cells to take noticeably
> longer than the training loop itself.

## Example Output

The notebook produces several diagnostic visualizations:

- **Loss/accuracy curves** — training vs. validation performance across epochs
- **Confusion matrix** — which digits the model confuses most often
- **Most-incorrect grid** — the test images the model got wrong most confidently
- **PCA/t-SNE scatter plots** — 2D projections of learned representations, colored by digit
- **Imagined digit** — the random-noise image the model most confidently labels as a target digit
- **Weight grid** — first-layer weights reshaped as 28x28 images

## Configuration

All hyperparameters live in a single `Config` dataclass at the top of the
notebook:

| Parameter             | Default | Description                                   |
|------------------------|---------|------------------------------------------------|
| `seed`                 | 1234    | Random seed for reproducibility                |
| `batch_size`           | 64      | Samples per batch                              |
| `valid_ratio`          | 0.9     | Fraction of training data kept for training    |
| `hidden_dim_1`         | 250     | First hidden layer size                        |
| `hidden_dim_2`         | 100     | Second hidden layer size                       |
| `epochs`               | 10      | Training epochs                                |
| `tsne_sample_size`     | 5000    | Number of points used for the t-SNE fit        |
| `imagine_digit`        | 3       | Target digit for the imagined-digit experiment |
| `imagine_iterations`   | 50000   | Random samples tried in that experiment        |

## Future Work

- Replace the MLP with a CNN for a stronger accuracy baseline
- Add a learning-rate scheduler and early stopping
- Save full training-state checkpoints to allow resuming
- Use gradient-based activation maximization instead of random search
- Track experiments with TensorBoard or Weights & Biases

## License

This project is licensed under the [MIT License](LICENSE).

## Author

Developed by **Mehran**.

## Acknowledgements

- LeCun, Y., Cortes, C., & Burges, C. J. C. — The MNIST Database of
  Handwritten Digits.
- Van der Maaten, L., & Hinton, G. (2008) — Visualizing Data using t-SNE.
- The [PyTorch](https://pytorch.org/) and [scikit-learn](https://scikit-learn.org/) teams.
