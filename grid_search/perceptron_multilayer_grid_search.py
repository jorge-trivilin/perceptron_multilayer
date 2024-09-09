import numpy as np
import logging
from typing import Optional
import pandas as pd

def configure_logging():
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

class PerceptronMultilayer:
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 learning_rate=0.1,
                 logger=None):
        """
        Inicializa os pesos e bias do Perceptron Multicamadas.
        """
        self.logger = logger or configure_logging()
        self.logger.info(f"Initializing PerceptronMultilayer with input_size={input_size}, \
                         hidden_size={hidden_size}, \
                         output_size={output_size}, \
                         learning_rate={learning_rate}")
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Inicialização dos pesos usando Xavier/Glorot
        self.weights_input_hidden = np.random.randn(self.hidden_size, self.input_size) * np.sqrt(1. / self.input_size)
        self.weights_hidden_output = np.random.randn(self.output_size, self.hidden_size) * np.sqrt(1. / self.hidden_size)
        self.bias_hidden = np.random.randn(self.hidden_size) * np.sqrt(1. / self.input_size)
        self.bias_output = np.random.randn(self.output_size) * np.sqrt(1. / self.hidden_size)
        self.logger.info("Weights and biases initialized")
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def binary_cross_entropy(self, expected_output, y_output):
        epsilon = 1e-8  # Para evitar log(0)
        y_output = np.clip(y_output, epsilon, 1 - epsilon)
        bce = - expected_output * np.log(y_output) + (1 - expected_output) * np.log(1 - y_output)
        return bce
    
    def fit(self, X, y, epochs=1000, batch_size=32):
        """
        Treina o modelo usando propagação para frente e retropropagação.
        """
        self.logger.info(f"Starting training for {epochs} epochs with batch size {batch_size}")
        
        # Converta y para Numpy array, se não for
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            total_loss = 0
            # Embaralhar os dados
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                batch_loss = 0
                for inputs, expected_output in zip(X_batch, y_batch):
                    # Propagação para frente
                    y_hidden, y_output = self.feedfoward(inputs)

                    # Cálculo do erro e da perda
                    error = expected_output - y_output
                    sample_loss = self.binary_cross_entropy(expected_output, y_output)
                    batch_loss += sample_loss
                    
                    # Backpropagation
                    self.backpropagation(error, inputs, y_hidden, y_output)
                
                total_loss += batch_loss
            
            # Exibir a perda a cada 100 épocas
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Average Loss: {total_loss / n_samples}")
        self.logger.info("Training completed")

    def feedfoward(self, inputs):
        v_hidden = np.dot(self.weights_input_hidden, inputs) + self.bias_hidden
        y_hidden = self.sigmoid(v_hidden)
        v_output = np.dot(self.weights_hidden_output, y_hidden) + self.bias_output
        y_output = self.sigmoid(v_output)
        return y_hidden, y_output

    def backpropagation(self, error, inputs, y_hidden, y_output):
        delta_output = error * self.sigmoid_derivative(y_output)
        delta_hidden = np.dot(self.weights_hidden_output.T, delta_output) * self.sigmoid_derivative(y_hidden)

        # Atualização dos pesos e bias
        self.weights_hidden_output += self.learning_rate * np.outer(delta_output, y_hidden)
        self.weights_input_hidden += self.learning_rate * np.outer(delta_hidden, inputs)
        self.bias_hidden += self.learning_rate * delta_hidden
        self.bias_output += self.learning_rate * delta_output

    def predict_proba(self, X):
        """
        Faz previsões probabilísticas para novas entradas
        """
        self.logger.info(f"Making probabilistic predictions for {len(X)} samples")
        probabilities = []
        for inputs in X:
            _, y_output = self.feedfoward(inputs)
            probabilities.append(y_output[0])
        return np.array(probabilities)
    
    def predict(self, X):
        """
        Faz previsões binárias para novas entradas.
        """
        self.logger.info(f"Making binary predictions for {len(X)} samples")
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)
    
    def evaluate(self, X, y):
        """
        Avalia o modelo calculando a acurácia.
        """
        self.logger.info("Evaluating model performance")
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        self.logger.info(f"Model accuracy: {accuracy:.4f}")
        return accuracy