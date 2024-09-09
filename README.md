# Perceptron Multilayer with Random Search for Hyperparameter Optimization

This project implements a **Multilayer Perceptron (MLP)** from scratch in Python using **Numpy** and optimizes the model's hyperparameters using **Random Search** and **K-Fold Cross-Validation**. The goal of the project is to build a flexible MLP and optimize its parameters to achieve the best performance on the **Telco Customer Churn dataset**.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [Logging](#logging)
- [Cross-Validation](#cross-validation)
- [Performance Evaluation](#performance-evaluation)
- [Contributing](#contributing)
- [License](#license)

## Introduction
The project implements a Multilayer Perceptron (MLP) from scratch without relying on high-level libraries such as TensorFlow or PyTorch. The goal is to offer a deep dive into how MLPs work internally, including backpropagation and training over mini-batches.

The key features include:
- A custom-built MLP class that supports multiple hidden layers, batch training, and backpropagation.
- Hyperparameter optimization via Random Search and K-Fold Cross-Validation to tune the learning rate, number of epochs, batch size, and the number of hidden neurons.
- Evaluation of the model on a classification task using metrics such as Log Loss and Accuracy.

## Dataset
This project uses the **Telco Customer Churn** dataset, which contains customer data for a telecommunications company. The goal is to predict customer churn (binary classification: churn or not churn).

- **Features:** Both numerical and categorical columns representing customer details, usage, and contract types.
- **Target:** `Churn` column indicating whether the customer has churned (`Yes` or `No`).

## Model Architecture
The model is a **Multilayer Perceptron (MLP)** with:
- A configurable number of hidden neurons.
- Sigmoid activation functions.
- Binary Cross-Entropy as the loss function.
- Xavier/Glorot initialization for weights.

### Key Components:
- **Feedforward propagation**: Computes the output for the given input batch.
- **Backpropagation**: Updates the weights based on the error using gradient descent.
- **Mini-batch Gradient Descent**: Trains the model using mini-batches for better convergence.

## Installation
To run this project, you'll need to install the required Python packages. This project requires **Python 3.8** or higher.

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/perceptron-multilayer.git
    cd perceptron-multilayer
    ```

2. Create and activate a virtual environment:

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. (Optional) Download the **Telco Customer Churn dataset** from [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn) and place it in the `data/` folder.

## Usage

You can train the model with different data fractions and hyperparameters using the command-line interface.

### Training the model:

To train the model using 100% of the data and perform random search over 5 iterations with 5-fold cross-validation, run the following command:

```bash
python train.py --fraction 1.0
```

You can specify a smaller fraction of the data if you'd like faster training:

```bash
python train.py --fraction 0.1  # Use 10% of the dataset
```

## Project Structure
```
perceptron_multilayer/
├── data/                         # Folder to store datasets (e.g., Telco Customer Churn)
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   
├── grid_search/                  # Scripts related to grid search hyperparameter tuning
├── legacy/                       # Folder for legacy code (older versions, experiments, etc.)
├── perceptron_multilayer_random_search.py     # Implementation of the Perceptron Multilayer class with random search
├── pytorch/                      # Directory for PyTorch use of GPU
├── sklearn/                      # Script to train an MLP using scikit-learn for comparison
├── __init__.py                   # Python package initialization file
├── Dockerfile                    # Docker configuration for containerizing the project
├── train.py                      # Main script for training and running random search for hyperparameter tuning
├── requirements.txt              # Python package dependencies
└── README.md                     # Comprehensive project README file


## Hyperparameter Optimization
Hyperparameters are optimized using **Random Search** over the following parameter space:

- **hidden_size**: Number of neurons in the hidden layer(s). A random integer between 4 and 32.
- **learning_rate**: Learning rate for gradient descent. A random float between 0.001 and 0.1.
- **epochs**: Number of epochs to train the model. A random integer between 500 and 1000.
- **batch_size**: Size of the mini-batches. A random integer between 32 and 64.

These parameters are tuned via Random Search and K-Fold Cross-Validation to find the combination that gives the best performance on the validation set.

### Example output from Random Search:

```
2024-09-09 18:54:42 - INFO - Random search completed. Best parameters:
  {'hidden_size': 14, 'learning_rate': 0.0859, 'epochs': 796, 'batch_size': 42}

2024-09-09 18:54:59 - INFO - Final results:
  Test Log Loss: 0.3981
  Test Accuracy: 80.98%
```

## Logging
Logging is set up using Python's `logging` module. During training, logs are generated to track the progress, including:

- Epoch loss values.
- Hyperparameter search progress.
- Model evaluation metrics (Log Loss and Accuracy).

To modify the logging configuration, edit the `configure_logging` function in `train.py`.

## Cross-Validation
The model is evaluated using **K-Fold Cross-Validation** (default: 5 folds). The cross-validation process splits the dataset into `k` folds and evaluates the model `k` times, ensuring that every data point is used for both training and validation.

## Performance Evaluation
During the evaluation phase, the following metrics are calculated:

- **Log Loss**: Measures how well the model's predicted probabilities match the true labels. Lower is better.
- **Accuracy**: Measures the percentage of correct predictions out of total predictions.

Results are printed for each fold, and the average Log Loss and Accuracy are logged for the final model.