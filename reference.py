import numpy as np

# 1. Define the network architecture and data
# Input data (X) and target outputs (y) for the XOR problem
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Initialize weights and biases randomly
np.random.seed(0)
input_dim = 2
hidden_dim = 4
output_dim = 1
weights_h = np.random.uniform(size=(input_dim, hidden_dim))  # Weights for hidden layer
bias_h = np.random.uniform(size=(1, hidden_dim))              # Bias for hidden layer
weights_o = np.random.uniform(size=(hidden_dim, output_dim)) # Weights for output layer
bias_o = np.random.uniform(size=(1, output_dim))             # Bias for output layer

# Define activation functions and their derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Define the training loop
learning_rate = 0.1
for _ in range(10000):
    # 2. Forward Propagation
    # Hidden layer
    hidden_layer_input = np.dot(X, weights_h) + bias_h
    hidden_layer_output = sigmoid(hidden_layer_input)
    # Output layer
    output_layer_input = np.dot(hidden_layer_output, weights_o) + bias_o
    predicted_output = sigmoid(output_layer_input)

    # 3. Calculate Loss (Mean Squared Error)
    error = y - predicted_output
    loss = np.mean(np.square(error))

    # 4. Backpropagation
    # Gradient of loss with respect to output predictions
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    
    # Gradient of loss with respect to output layer weights and biases
    d_weights_o = np.dot(hidden_layer_output.T, d_predicted_output)
    #d_bias_o = np.sum(d_predicted_output, axis=0, keepdims=True)
    
    # Gradient of loss with respect to hidden layer output
    d_hidden_layer = np.dot(d_predicted_output, weights_o.T) * sigmoid_derivative(hidden_layer_output)
    
    # Gradient of loss with respect to hidden layer weights and biases
    d_weights_h = np.dot(X.T, d_hidden_layer)
    #d_bias_h = np.sum(d_hidden_layer, axis=0, keepdims=True)
    
    # 5. Optimization (Gradient Descent)
    weights_o += learning_rate * d_weights_o
    #bias_o += learning_rate * d_bias_o
    weights_h += learning_rate * d_weights_h
    #bias_h += learning_rate * d_bias_h

# Print final prediction and loss
final_output = sigmoid(np.dot(sigmoid(np.dot(X, weights_h) + bias_h), weights_o) + bias_o)
print("Predicted output after training:")
print(final_output)
print(f"\nFinal Loss: {loss:.6f}")
