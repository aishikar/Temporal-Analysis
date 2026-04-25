# Temporal-Analysis

## Project Overview
This project aims to predict the next state in a sequence of temporal data using Deep Learning. By analyzing historical trajectories, the model learns to identify underlying patterns and system behaviors to perform accurate one-step-ahead forecasting.

## Week 1: Foundation & Design
The focus for this week was setting up the theoretical framework for the project.

### 1. Data Strategy
- **Input:** State trajectories from the training set.
- **Transformation:** We are utilizing a sliding window technique. This frames the temporal data such that a sequence of $N$ previous states serves as the input features for the $N+1$ target state.

### 2. Proposed Architecture
The initial model is a **Multilayer Perceptron (MLP)**.
- **Input Layer:** Dimensioned to the sequence window length.
- **Hidden Layers:** Fully connected layers with ReLU activation to learn non-linear temporal relationships.
- **Output Layer:** Single-node output for the predicted next state.

### 3. Tech Stack
- **Framework:** PyTorch
- **Data Handling:** NumPy & Pandas
- **Environment:** Google Colab / Python 3.9+
