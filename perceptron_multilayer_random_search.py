# perceptron_multilayer.py
import numpy as np
import logging
import pandas as pd

def configure_logging():
    """Configures the logger for the module, if necessary."""
    logger = logging.getLogger(__name__)
    if not logger.hasHandlers():
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
                 logger=None,
                 random_seed=None):
        """
        Initializes the weights and biases of the Multilayer Perceptron.
        """
        self.logger = logger or configure_logging()
        self.logger.info(f"Initializing with input_size={input_size}, hidden_size={hidden_size}, output_size={output_size}, learning_rate={learning_rate}, random_seed={random_seed}")
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Setup the random generator
        self.rng = np.random.default_rng(random_seed)
        
        # Initialize weights using Xavier/Glorot
        self.logger.info("Weights and biases initialized")
        self.weights_input_hidden = self.rng.normal(0, np.sqrt(1. / self.input_size),
                                                    (self.hidden_size, self.input_size))
        self.weights_hidden_output = self.rng.normal(0, np.sqrt(1. / self.hidden_size), 
                                                 (self.output_size, self.hidden_size))
        self.bias_hidden = self.rng.normal(0, np.sqrt(1. / self.input_size), 
                                        (self.hidden_size,))
        self.bias_output = self.rng.normal(0, np.sqrt(1. / self.hidden_size), 
                                        (self.output_size,))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def binary_cross_entropy(self, expected_output, y_output):
        epsilon = 1e-8  # To avoid log(0)
        y_output = np.clip(y_output, epsilon, 1 - epsilon)
        bce = - expected_output * np.log(y_output) + (1 - expected_output) * np.log(1 - y_output)
        return np.mean(bce)
    
    def fit(self, X, y, epochs=1000, batch_size=32):
        """
        Trains the model using forward propagation and backpropagation.
        """
        self.logger.info(f"Starting training for {epochs} epochs with batch size {batch_size}")
        # Adjusting the fit method to work with the new batch structure:
        n_samples = X.shape[0]

        if isinstance(y, pd.Series):
            y = y.to_numpy()
        
        for epoch in range(epochs):
            total_loss = 0
            # Shuffle the data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                # Forward propagation in batch
                y_hidden, y_output = self.feedfoward(X_batch)

                # Calculate the error
                error = y_batch.reshape(-1, 1) - y_output
                batch_loss = np.mean(self.binary_cross_entropy(y_batch, y_output))

                # Backpropagation for the whole batch
                self.backpropagation(error, X_batch, y_hidden, y_output)

                total_loss += batch_loss
            
            # Display loss every 100 epochs
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Average Loss: {total_loss / n_samples}")
        self.logger.info("Training completed")

    def feedfoward(self, inputs):
        # In feedforward, instead of multiplying a single input, we multiply the matrix of inputs.
        # Each row of the input matrix represents a sample from the batch.
        # Here, inputs will be a batch of samples with shape (batch_size, input_size). 
        # Instead of performing vector multiplication, we'll perform matrix multiplications.
        v_hidden = np.dot(inputs, self.weights_input_hidden.T) + self.bias_hidden
        y_hidden = self.sigmoid(v_hidden)
        
        v_output = np.dot(y_hidden, self.weights_hidden_output.T) + self.bias_output
        y_output = self.sigmoid(v_output)

        return y_hidden, y_output

    def backpropagation(self, error, inputs, y_hidden, y_output):
        # In backpropagation, we calculate the error and update weights for all samples in a batch at once
        # inputs (batch_size, input_size), error (batch_size, output_size)
        # inputs is the batch matrix with shape (batch_size, input_size).
        # error is the error calculated for the whole batch.
        # Matrix multiplication (np.dot) allows us to calculate gradients at once, avoiding loops.

        batch_size = inputs.shape[0]

        # delta_output has shape (batch_size, output_size), i.e., (32, 1)
        delta_output = error * self.sigmoid_derivative(y_output)

        # Correct the multiplication so that dimensions align correctly:
        # delta_hidden will have shape (batch_size, hidden_size), i.e., (32, 4)
        delta_hidden = np.dot(delta_output, self.weights_hidden_output) * self.sigmoid_derivative(y_hidden)

        # Update weights of the output layer
        self.weights_hidden_output += self.learning_rate * np.dot(delta_output.T, y_hidden) / batch_size

        # Update weights of the hidden layer
        self.weights_input_hidden += self.learning_rate * np.dot(delta_hidden.T, inputs) / batch_size

        # Update biases
        self.bias_output += self.learning_rate * np.mean(delta_output, axis=0)
        self.bias_hidden += self.learning_rate * np.mean(delta_hidden, axis=0)

    def predict_proba(self, X):
        """
        Makes probabilistic predictions for new inputs
        """
        self.logger.info(f"Making probabilistic predictions for {len(X)} samples")
        probabilities = []
        for inputs in X:
            _, y_output = self.feedfoward(inputs)
            probabilities.append(y_output[0])
        return np.array(probabilities)
    
    def predict(self, X):
        """
        Makes binary predictions for new inputs.
        """
        self.logger.info(f"Making binary predictions for {len(X)} samples")
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)
    
    def evaluate(self, X, y):
        """
        Evaluates the model by calculating accuracy.
        """
        self.logger.info("Evaluating model performance")
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        self.logger.info(f"Model accuracy: {accuracy:.4f}")
        return accuracy
