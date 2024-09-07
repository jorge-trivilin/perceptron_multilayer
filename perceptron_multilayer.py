import numpy as np

class PerceptronMultilayer:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        """
        Inicializa os pesos e bias do Perceptron Multicamadas.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Inicialização dos pesos usando Xavier/Glorot
        self.weights_input_hidden = np.random.randn(self.hidden_size, self.input_size) * np.sqrt(1. / self.input_size)
        self.weights_hidden_output = np.random.randn(self.output_size, self.hidden_size) * np.sqrt(1. / self.hidden_size)
        self.bias_hidden = np.random.randn(self.hidden_size) * np.sqrt(1. / self.input_size)
        self.bias_output = np.random.randn(self.output_size) * np.sqrt(1. / self.hidden_size)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def fit(self, X, y, epochs=1000):
        """
        Treina o modelo usando propagação para frente e retropropagação.
        """
        for epoch in range(epochs):
            total_loss = 0
            for inputs, expected_output in zip(X, y):
                # Propagação para frente
                v_hidden = np.dot(self.weights_input_hidden, inputs) + self.bias_hidden
                y_hidden = self.sigmoid(v_hidden)
                v_output = np.dot(self.weights_hidden_output, y_hidden) + self.bias_output
                y_output = self.sigmoid(v_output)
                
                # Retropropagação
                error = expected_output - y_output
                total_loss += np.mean((expected_output - y_output) ** 2)
                delta_output = error * self.sigmoid_derivative(y_output)
                delta_hidden = np.dot(self.weights_hidden_output.T, delta_output) * self.sigmoid_derivative(y_hidden)
                
                # Atualização dos pesos e bias
                self.weights_hidden_output += self.learning_rate * np.outer(delta_output, y_hidden)
                self.weights_input_hidden += self.learning_rate * np.outer(delta_hidden, inputs)
                self.bias_hidden += self.learning_rate * delta_hidden
                self.bias_output += self.learning_rate * delta_output
            
            # Exibir a perda a cada 100 épocas
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss}")

    def predict_proba(self, X):
        """
        Faz previsões probabilisticas para novas entradas
        """
        probabilities = []
        for inputs in X:
            v_hidden = np.dot(self.weights_input_hidden, inputs) + self.bias_hidden
            y_hidden = self.sigmoid(v_hidden)
            v_output = np.dot(self.weights_hidden_output, y_hidden) + self.bias_output
            y_output = self.sigmoid(v_output)
            probabilities.append(y_output[0]) # Assumindo output_size = 1 
        return np.array(probabilities)
    
    def predict(self, X):
        """
        Faz previsões binárias para novas entradas.
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)
    
    def evaluate(self, X, y):
        """
        Avalia o modelo calculando a acurácia.
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy
