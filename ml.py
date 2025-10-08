## TODAY WE ARE WRITING ML FRAMEWORK
## FROM SCRATCH PART 4: Reviewing and Testing
import numpy as np
np.seterr(divide='ignore')

EPOCHS=1000
LEARN_RATE=0.01

def main():
    ## XOR Operator
    ## features (input) (training)
    #x = np.array([[0,0],[0,1],[1,0],[1,1]])

    ## labels (output) actual answer (target)
    #y = np.array([[ 0 ],[ 1 ],[ 1 ],[ 0 ]])
    ## AND Operator
    #y = np.array([[ 0 ],[ 0 ],[ 0 ],[ 1 ]])
    ## OR Operator
    #y = np.array([[ 0 ],[ 1 ],[ 1 ],[ 1 ]])

    import tensorflow
    from tensorflow.keras.datasets import mnist
    (X_train,y_train),(X_test,y_test)=mnist.load_data()
    x = X_train.reshape(60000, 28*28).astype('float32') / 255.0

    #np.eye(10)[y[0]]
    #[ i == y[0] and 1 or 0 for i, v in enumerate([0,0,0,0,0,0,0,0,0]) ]
    #y = np.array([ np.eye(10)[v] for v in y ])
    y = np.array([[ 1 if i == j else 0 for i in range(10)] for j in y_train])

    print("x shape:")
    print(x.shape)
    print("y shape:")
    print(y.shape)

    ## Print first few samples
    x = x[:100]
    y = y[:100]
    print(x)
    print(y)

    ### Main
    layers = [
        Layer(input=784,  output=100),
        Layer(input=100,  output=100),
        Layer(input=100,  output=10, activation=sigmoid, derivative=sigmoid_derivative),
    ]
    print("Initial weights:")
    print(layers[0].weights[:3])
    model = Model(layers)
    model.train(x, y)

    prediction = model.forward(x)
    print("Results:")
    print(prediction)
    results = np.round(prediction)
    print(results)
    print([y[i][0] == r[0] for i, r in enumerate(results)])

## Sigmoid (0, 1)
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
            delta = out - labels

            ## loss for monitoring / cost
            loss = np.mean(np.square(delta))
            print(f"Loss: {loss}")

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
        ) * np.sqrt(1.0 / input)
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
        self.bias -= LEARN_RATE * np.mean(gradient, axis=0, keepdims=True)

main()
