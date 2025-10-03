## Suryansh - Use Python for
## Bioinformatics and analytics and AI

## TODAY WE ARE WRITING ML FRAMEWORK
## FROM SCRATCH 
import numpy as np

EPOCHS=1
LEARN_RATE=0.01

## XOR Operator
## features (input) (training)
x = np.array([[0,0],[0,1],[1,0],[1,1]])

## labels (output) expected answer
y = np.array([[0  ],[1  ],[1  ],[0  ]])

##tanh     = -1 to +1
##sigmoid   = 0 to 1

class Layer():
    def __init__(
        self,
        input=4,
        output=4,
        activation=np.tanh,
        derivitive=lambda x: 1 - np.tanh(x)**2,
    ):
        self.bias = np.random.rand(output)
        self.weights = np.random.rand(
            input,
            output,
        )
        self.activation = activation
        self.derivitive = derivitive

    def forward(self, inputs):
        out = inputs @ self.weights
        self.input = inputs
        self.output = out
        return self.activation(out)


    def backward(self, loss):
        #d_predicted_output = error * sigmoid_derivative(predicted_output)
        self.gradient       = loss * self.derivitive(self.output)
        #return back

def optimze(layer, back):
    self.weights -= delta * LEARN_RATE
    pass

## Model
a = Layer(input=2, output=4)
b = Layer(input=4, output=4)
c = Layer(input=4, output=1)

## Training (FIT) .train() .fit()
for i in range(EPOCHS):
    print(f"Epoch: {i}")

    ## forward
    a_out = a.forward(x)
    b_out = b.forward(a_out)
    c_out = c.forward(b_out)
    print(c_out)

    ## delta
    delta = c_out - y
    print(delta)

    ## loss
    loss = np.mean(np.square(delta))

    c.backward(loss)
    print(c.gradient)
    #b_back = b.backward(delta)
    #a_back = a.backward(delta)


