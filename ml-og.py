## TODAY WE ARE WRITING ML FRAMEWORK
## FROM SCRATCH PART 3: Updating our Framework
import numpy as np

EPOCHS=1000
LEARN_RATE=0.01

## XOR Operator
## features (input) (training)
x = np.array([[0,0],[0,1],[1,0],[1,1]])

## labels (output) actual answer
y = np.array([[0  ],[1  ],[1  ],[0  ]])

## Sigmoid 
def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x): return x * (1 - x)

## Model DNN (Deep Neural Network)
class Model():
    def __init__(self, layers):
        self.layers = layers

    def forward(self, inputs):
        for layer in self.layers:
            out = layer.forward(out or inputs)
        return out

    ## Backpropagation
    def backward(self, gradient):
        for layer in self.layers[::-1]:
            gradient = layer.backward(gradient)

    ## Training (fit)
    def train(self, features, labels):
        pass

# Layer 
class Layer():
    def __init__(
        self,
        input=4,
        output=4,
        activation=np.tanh,
        derivitive=lambda x: 1 - np.tanh(x) ** 2,
    ):
        self.bias = np.random.randn(1, output)
        self.weights = np.random.randn(
            input,
            output,
        )
        self.activation = activation
        self.derivitive = derivitive

    ## Forward pass (also the Predict method)
    def forward(self, inputs):
        self.input = inputs
        return self.activation(inputs @ self.weights + self.bias)

    ## Backward Propegation (training the model)
    def backward(self, gradient):
        self.optimize(gradient)
        return np.dot(gradient, self.weights.T) * self.derivitive(self.input) 

    ## Update model weights with gradient (adjust error)
    def optimize(self, gradient):
        self.weights -= LEARN_RATE * self.input.T @ gradient
        self.bias -= LEARN_RATE * np.sum(gradient, axis=0, keepdims=True)

## Model
a = Layer(input=2, output=4)
b = Layer(input=4, output=4)
c = Layer(input=4, output=1)

## Training (FIT) .train() .fit()
for i in range(EPOCHS):
    #print(f"Epoch: {i}")

    ## Forward
    out = a.forward(x)
    out = b.forward(out)
    out = c.forward(out)

    ## Delta (error)
    delta = out - y

    ## loss for monitoring / cost
    loss = np.mean(np.square(delta))
    print(loss)

    ## First gradient
    ## TODO
    ## TODO
    ## TODO simplify ( making a Model class perhaps )
    ## TODO
    ## TODO
    gradient = delta * c.derivitive(out)

    ## Backpropagation
    gradient = c.backward(gradient)
    gradient = b.backward(gradient)
    gradient = a.backward(gradient)

