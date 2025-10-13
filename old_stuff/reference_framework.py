import numpy as np

class NeuralNetwork:
    """
    A flexible neural network framework that supports n number of layers.

    Usage:
        nn = NeuralNetwork([2, 4, 3, 1])  # 2 input, 4 hidden, 3 hidden, 1 output
        nn.train(X, y, epochs=10000, learning_rate=0.1)
        predictions = nn.predict(X)
    """

    def __init__(self, layer_dims, activation='sigmoid', seed=None):
        """
        Initialize the neural network with specified layer dimensions.

        Args:
            layer_dims: List of integers specifying neurons in each layer
                       e.g., [2, 4, 3, 1] for input=2, hidden1=4, hidden2=3, output=1
            activation: Activation function to use ('sigmoid' or 'relu')
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)

        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims)
        self.activation = activation

        # Initialize weights and biases for each layer
        self.weights = []
        self.biases = []

        for i in range(self.num_layers - 1):
            w = np.random.uniform(size=(layer_dims[i], layer_dims[i+1]))
            b = np.random.uniform(size=(1, layer_dims[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def sigmoid_derivative(self, x):
        """Derivative of sigmoid"""
        return x * (1 - x)

    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """Derivative of ReLU"""
        return (x > 0).astype(float)

    def activate(self, x):
        """Apply activation function"""
        if self.activation == 'sigmoid':
            return self.sigmoid(x)
        elif self.activation == 'relu':
            return self.relu(x)
        return x

    def activate_derivative(self, x):
        """Apply activation derivative"""
        if self.activation == 'sigmoid':
            return self.sigmoid_derivative(x)
        elif self.activation == 'relu':
            return self.relu_derivative(x)
        return np.ones_like(x)

    def forward(self, X):
        """
        Perform forward propagation through all layers.

        Args:
            X: Input data

        Returns:
            predictions: Final output
            layer_outputs: List of outputs from each layer (for backprop)
        """
        layer_outputs = [X]
        current_input = X

        for i in range(self.num_layers - 1):
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            a = self.activate(z)
            layer_outputs.append(a)
            current_input = a

        return current_input, layer_outputs

    def backward(self, y, layer_outputs, learning_rate):
        """
        Perform backpropagation through all layers.

        Args:
            y: Target outputs
            layer_outputs: Outputs from forward pass (includes input as first element)
            learning_rate: Learning rate for gradient descent
        """
        # Start with output layer error
        delta = (y - layer_outputs[-1]) * self.activate_derivative(layer_outputs[-1])

        # Backpropagate through all layers
        for i in range(self.num_layers - 2, -1, -1):
            # Update weights and biases
            self.weights[i] += learning_rate * np.dot(layer_outputs[i].T, delta)
            self.biases[i] += learning_rate * np.sum(delta, axis=0, keepdims=True)
            # Propagate delta backward
            delta = np.dot(delta, self.weights[i].T) * self.activate_derivative(layer_outputs[i])

    def train(self, X, y, epochs=10000, learning_rate=0.1, verbose=True):
        """
        Train the neural network.

        Args:
            X: Input data
            y: Target outputs
            epochs: Number of training iterations
            learning_rate: Learning rate for gradient descent
            verbose: Print loss during training
        """
        for epoch in range(epochs):
            # Forward propagation
            predictions, layer_outputs = self.forward(X)

            # Calculate loss (MSE)
            error = y - predictions
            loss = np.mean(np.square(error))

            # Backpropagation
            self.backward(y, layer_outputs, learning_rate)

            # Print progress
            if verbose and (epoch % 1000 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch}: Loss = {loss:.6f}")

        return loss

    def predict(self, X):
        """
        Make predictions using the trained network.

        Args:
            X: Input data

        Returns:
            predictions: Network output
        """
        predictions, _ = self.forward(X)
        return predictions


# Example usage: XOR problem
if __name__ == "__main__":
    # Input data and target outputs for XOR problem
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Example 1: Simple network (same as original reference.py)
    print("Example 1: Network with 1 hidden layer [2, 4, 1]")
    print("=" * 50)
    nn1 = NeuralNetwork([2, 4, 1], seed=0)
    nn1.train(X, y, epochs=10000, learning_rate=0.1, verbose=False)
    predictions = nn1.predict(X)
    print("Predicted output:")
    print(predictions)
    print()

    # Example 2: Deeper network with multiple hidden layers
    print("Example 2: Network with 3 hidden layers [2, 8, 6, 4, 1]")
    print("=" * 50)
    nn2 = NeuralNetwork([2, 8, 6, 4, 1], seed=0)
    nn2.train(X, y, epochs=10000, learning_rate=0.1, verbose=False)
    predictions = nn2.predict(X)
    print("Predicted output:")
    print(predictions)
    print()

    # Example 3: Very simple network
    print("Example 3: Network with small hidden layer [2, 2, 1]")
    print("=" * 50)
    nn3 = NeuralNetwork([2, 2, 1], seed=0)
    nn3.train(X, y, epochs=10000, learning_rate=0.1, verbose=False)
    predictions = nn3.predict(X)
    print("Predicted output:")
    print(predictions)
