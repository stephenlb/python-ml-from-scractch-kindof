## TODAY WE ARE WRITING ML FRAMEWORK
## FROM SCRATCH PART 3: Updating our Framework
import numpy as np

EPOCHS=3000
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

    ## Prediction
    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    ## Backpropagation
    def backward(self, gradient):
        for layer in self.layers[::-1]:
            gradient = layer.backward(gradient)

    ## Training (fit)
    def train(self, features, labels):
        for i in range(EPOCHS):
            ## Forward
            out = self.forward(features)

            ## Delta (error)
            delta = out - y

            ## loss for monitoring / cost
            loss = np.mean(np.square(delta))
            if i % 100 == 0: print(f"Loss: {loss}")

            ## Backpropagation and Optimization
            self.backward(delta)

# Layer 
class Layer():
    def __init__(
        self,
        input=4,
        output=4,
        activation=np.tanh,
        derivative=lambda x: 1 - x ** 2,
    ):
        self.bias = np.zeros((1, output))
        self.weights = np.random.randn(
            input,
            output,
        )
        self.activation = activation
        self.derivative = derivative

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
        self.weights -= LEARN_RATE * self.input.T @ gradient
        self.bias -= LEARN_RATE * np.sum(gradient, axis=0, keepdims=True)

### Main
layers = [
    Layer(input=2, output=4),
    Layer(input=4, output=4),
    Layer(input=4, output=1, activation=sigmoid, derivative=sigmoid_derivative),
]
model = Model(layers)
model.train(x, y)

prediction = model.forward(x)
results = np.round(prediction)

print("Results:")
print(prediction)
print(results)
print([y[i][0] == r[0] for i, r in enumerate(results)])
