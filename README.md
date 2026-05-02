# Temporal Analysis: Time-Series Forecasting

## Project Overview
This project aims to predict the future state in a sequence of temporal data using Deep Learning. By leveraging historical system trajectories, the model learns to identify underlying patterns and system behaviors to perform accurate one-step-ahead forecasting.

## Weekly Progress Roadmap

### Week 1: Foundation & Design
The initial phase focused on setting up the theoretical framework and project environment.
- **Data Exploration:** Conducted a review of `train.csv`, `val.csv`, and `test.csv` to identify temporal patterns[cite: 1].
- **Methodology:** Selected a **Sliding Window** approach to transform continuous trajectory data into a supervised learning format.
- **Architecture Design:** Designed a Multilayer Perceptron (MLP) with an input layer mapped to the window size and ReLU-activated hidden layers[cite: 1].
- **Tech Stack:** Established the environment using PyTorch, NumPy, and Pandas[cite: 1].

### Week 2: Implementation & Initial Training
This week involved transitioning from architectural design to a functional implementation.
- **Data Pipeline:** Developed a custom `TrajectoryDataset` class in PyTorch to handle on-the-fly sliding window transformations[cite: 1].
- **Model Scripting:** Implemented the `MLP` class architecture featuring three linear layers[cite: 1].
- **Training Infrastructure:** Integrated the Adam optimizer and Mean Squared Error (MSE) loss function to establish the training loop[cite: 1].
- **Initial Execution:** Performed preliminary training runs on the training dataset and monitored loss convergence[cite: 1].

## Model Architecture
The current implementation utilizes a **Multilayer Perceptron (MLP)**:
*   **Input Layer:** Dimensioned based on the chosen sequence window length[cite: 1].
*   **Hidden Layers:** Fully connected layers utilizing ReLU (Rectified Linear Unit) activation functions to capture non-linear system dynamics[cite: 1].
*   **Output Layer:** A single-node output representing the predicted next state in the trajectory[cite: 1].

## Repository Structure
*   `dataset.py`: Contains the `TrajectoryDataset` class and sliding window logic[cite: 1].
*   `model.py`: Defines the `MLP` neural network architecture[cite: 1].
*   `train.py` / `main.py`: The execution script for the training loop and optimization[cite: 1].
*   `data/`: Directory for storing trajectory CSV files (e.g., `train.csv`).

## Future Work (Week 3)
*   **Hyperparameter Tuning:** Optimizing window sizes and hidden layer density to improve accuracy.
*   **Validation Analysis:** Using the validation set to monitor for overfitting and generalizability.
*   **Visualizations:** Generating plots to compare predicted trajectories against ground truth data.
