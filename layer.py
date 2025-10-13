import activations
import numpy as np
np.seterr(divide='ignore') ## TODO FIX THIS

class Layer():
    def __init__(
        self,
        input=4,
        output=4,
        activation=activations.tanh,
        derivative=activations.tanh_derivative,
    ):
        self.bias = np.zeros((1, output))
        self.weights = np.random.randn(
            input,
            output,
        ) * np.sqrt(1.0 / input)
        self.activation = activation
        self.derivative = derivative
        self.learn      = 0.001

    ## Forward pass (also the Predict method)
    def forward(self, inputs):
        self.input = inputs
        self.output = self.activation(inputs @ self.weights + self.bias)
        return self.output

    ## Backward Propegation (training the model)
    def backward(self, gradient):
        delta = gradient * self.derivative(self.output)
        self.optimize(delta)
        return delta @ self.weights.T

    ## Update model weights with gradient (adjust error)
    def optimize(self, gradient):
        self.weights -= self.learn * self.input.T @ gradient
        self.bias -= self.learn * np.mean(gradient, axis=0, keepdims=True)

if __name__ == "__main__": main()
