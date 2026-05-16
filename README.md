# Temporal Analysis: Time-Series Forecasting

## Project Overview
This project aims to predict the future state in a sequence of temporal data using Deep Learning. By leveraging historical system trajectories, the model learns to identify underlying patterns and system behaviors to perform accurate one-step-ahead forecasting.

## Weekly Progress Roadmap

### Week 1: Foundation & Design
- **Data Exploration:** Conducted an initial review of `train.csv`, `val.csv`, and `test.csv` to identify temporal patterns.
- **Methodology Selection:** Selected a **Sliding Window** approach to transform continuous trajectory sequences into a supervised learning format.
- **Architecture Design:** Outlined a Multilayer Perceptron (MLP) architecture with input mappings optimized to look-back windows.

### Week 2: Component Implementation
- **Data Pipeline:** Developed the `TrajectoryDataset` class in PyTorch to slice chronological observations sequentially.
- **Model Scripting:** Programmed the `MLP` network infrastructure using fully-connected linear sequences and ReLU activation functions.

### Week 3: Training & Hyperparameter Tuning
- **Pipeline Integration:** Built the `train.py` environment executing training, validation scoring, and automated model weight serialization.
- **Optimization:** Experimented with hidden sizes and window dimensions. Settled on a **Window Size of 100** and **Hidden Density of 128**, yielding an optimized Validation MSE of **0.042**.

### Week 4: System Integration & Verification
- **Data Validation:** Verified model processing against complete `train.csv`, `val.csv`, and `test.csv` input matrices.
- **Code Refactoring:** Finalized optimization logic and prepared the repository for production deployment.

## Model Architecture
The production configuration maps as follows:
* **Input Layer:** 100 features (Matching the sliding window lookback length).
* **Hidden Structure:** Two Dense layers consisting of 128 hidden neurons equipped with ReLU activations.
* **Output Node:** 1 unit predicting the $N+1$ sequence trajectory element.

## Repository Structure
* `data/`: Local storage containing `train.csv`, `val.csv`, and `test.csv`.
* `dataset.py`: PyTorch modular index handler containing sliding window dataset mappings.
* `model.py`: Neural network class structure containing standard structural weights.
* `train.py`: Operational pipeline for model optimization, metrics estimation, and parameter checkpointing.
* `model_checkpoint.pth`: Extracted optimized parameter tensors.

## How to Run
1. Ensure your files are placed in the `data/` folder.
2. Run training and validation:
   ```bash
   python train.py
