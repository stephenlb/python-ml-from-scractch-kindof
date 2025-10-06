## TODAY WE ARE WRITING ML FRAMEWORK
## FROM SCRATCH PART 2: BACKPROPEGATION
import numpy as np

EPOCHS=1
LEARN_RATE=0.01

## Artificail Nerual Network
## ANN

## XOR Operator
## features (input) (training)
x = np.array([[0,0],[0,1],[1,0],[1,1]])

## labels (output) expected answer
y = np.array([[0  ],[1  ],[1  ],[0  ]])

## input data MUST be between -1 and +1

##tanh     = -1 to +1
##sigmoid   = 0 to 1

class Layer():
    def __init__(
        self,
        input=4,
        output=4,
        activation=np.tanh,
        derivitive=lambda x: 1 - np.tanh(x) ** 2,
    ):
        self.bias = np.random.rand(output)
        self.weights = np.random.rand(
            input,
            output,
        )
        self.activation = activation
        self.derivitive = derivitive

    def forward(self, inputs):
        self.input = inputs
        return self.activation(inputs @ self.weights + self.bias)

    def backward(self, gradient):
        self.gradient = gradient
        return np.dot(gradient, self.weights.T) * self.derivitive(self.input) 

    def optimze():
        self.weights += LEARN_RATE * self.input.T @ self.gradient
        self.bias += LEARN_RATE * np.sum(self.gradient, axis=0, keepdims=True)

## Model
a = Layer(input=2, output=4)
b = Layer(input=4, output=4)
c = Layer(input=4, output=1)

## Training (FIT) .train() .fit()
for i in range(EPOCHS):
    print(f"Epoch: {i}")

    ## forward
    out = a.forward(x)
    out = b.forward(out)
    out = c.forward(out)
    print(out) ## AI's "answer"

    ## delta (error)
    delta = out - y
    print(delta)

    ## loss for monitoring / cost
    loss = np.mean(np.square(delta))
    ## good loss = below 1.0 approaching 0.1
    ## bad loss = above 1.0 incrase beyond 1
    print(loss)

    ## First gradient
    gradient = delta * c.derivitive(out)
    print(gradient)

    ## Backpropegation
    gradient = c.backward(gradient)
    print(gradient)

    gradient = b.backward(gradient)
    print(gradient)

    gradient = a.backward(gradient)
    print(gradient)

    #b_back = b.backward(delta)
    #a_back = a.backward(delta)


