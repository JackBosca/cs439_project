# LLMs Optimization Landscapes

## 📚 Overview

This project is part of the **EPFL** course *CS-439: Optimization for machine learning*. It investigates the optimization landscapes of large language models (LLMs), specifically focusing on DistilGPT2 fine-tuned on the Wikitext-2 dataset. By analyzing how different training configurations affect convergence and generalization, we visualize and quantify the sharpness and structure of the resulting loss landscapes.

Our primary objectives include:
- Exploring how hyperparameters (batch size, learning rate, weight decay, optimizer) influence the optimization landscape;
- Comparing training with randomly shuffled vs. length-sorted data;
- Visualizing 2D loss landscapes;
- Computing sharpness using two methods:
    - Epsilon sharpness;
    - Hessian sharpness (largest eigenvalue).

## 📦 Dataset

- Wikitext-2 (a small subset): A small dataset from the WikiText family with over 2 million tokens. Suitable for benchmarking language models.
- Preprocessing: Conducted via `data/preprocessing.py`, which tokenizes and chunks text using the HuggingFace tokenizer for DistilGPT2.

## 🤖 Model

- DistilGPT2 from HuggingFace Transformers;
- Loaded and wrapped via `models/models_utils.py`.

## ⚙️ Configurations & Training

Training and experimentation are orchestrated through `optimization/training.py` and driven by configuration files such as `config/base.yaml`.

## 🔧 Tuned Hyperparameters

- Batch Size (bs): Multiple sizes explored;
- Optimizer: SGD and AdamW;
- Learning Rate (lr): Different magnitudes tested;
- Weight Decay (wd): Different regularization strengths;
- Data Order: Random vs. sorted by sequence length;

## 🛠️ Implementation

- Training loop: `training.py`
- Experiment orchestration: `experiments.py`
- Evaluation and metric logging: `evaluation.py`

## 🌄 Loss Landscape Visualization

Implemented in `visualization/visual.py`, our approach visualizes 2D loss landscapes by perturbing model parameters in random orthogonal directions.

We use techniques inspired by:
- Li et al., "Visualizing the Loss Landscape of Neural Nets" (2018);
- Papers in `literature/`.

The visualizations demonstrate the relative flatness or sharpness of minima under different settings.

## 📉 Sharpness Metrics

Implemented in `sharpness/sharpness.py`:

1. Epsilon Sharpness
- Follows the definition: `Sharpness_ε = max_{||δ|| ≤ ε} L(θ + δ) − L(θ)`;
- Evaluated for multiple epsilons per configuration.

2. Hessian Sharpness
- Estimates the top eigenvalue of the Hessian via power iteration, indicating the local curvature at the loss minimum.

Code: `sharpness.py`.

## 📊 Evaluation & Results

- Evaluation routines in `optimization/evaluation.py`;
- Experimental results stored as `.pt` files;
- `results.ipynb`: Final evaluation plots and figures.

## 🗂️ Repository Structure

```
cs439_project/
├── config/                  # YAML configs for training
├── data/                    # Dataset loading & preprocessing
├── literature/              # Relevant academic papers
├── models/                  # Model loading utilities
├── optimization/            # Training & experiment routines
├── sharpness/               # Sharpness metrics
├── tests/                   # Testing scripts and experimental checkpoints
├── utils/                   # Helper functions, config loading, plotting
├── visualization/           # Loss landscape visualizations
├── results.ipynb            # Evaluation and visualization
└── README.md                # This file
```

## 🚀 Running the Project

1. Environment Setup
```
pip install -r requirements.txt
```

2. Explore Results: open and run `results.ipynb` to view evaluation plots and metrics, visualizations of loss landscapes and sharpness metrics.

## 📖 References
Literature in `literature/`.

***

© 2025 — by Jacopo Boscariol, Leonardo De Novellis, Salya Amanda Diallo. Licensed under the [MIT License](./LICENSE).
