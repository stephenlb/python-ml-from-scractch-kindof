## TODAY WE ARE WRITING ML FRAMEWORK
## FROM SCRATCH PART 5: Batching and Fixes
import numpy as np
np.seterr(divide='ignore') ## TODO FIX THIS

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

    ## Trainging
    x = X_train.reshape(60000, 28*28).astype('float32') / 255.0
    x_test = x[:100] #X_test.reshape(60000, 28*28).astype('float32') / 255.0

    ## Testing
    y = np.array([[ 1 if i == j else 0 for i in range(10)] for j in y_train])
    y_test = y[:100] #np.array([[ 1 if i == j else 0 for i in range(10)] for j in y_test])

    ### Main
    layers = [
        Layer(input=784,  output=100),
        Layer(input=100,  output=100),
        Layer(input=100,  output=10, activation=sigmoid, derivative=sigmoid_derivative),
    ]
    model = Model(layers)
    model.train(x[:100], y[:100], batch=50, epochs=2000, learn=0.001)

    x_test = x[:100]
    y_test = y[:100]
    prediction = model.forward(x_test)
    print("Results:")
    print(prediction)

    results = np.round(prediction)
    print(results)

    display =['✅' if y[i][0] == r[0] else '⛔️' for i, r in enumerate(results)] 
    print(''.join(display))

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
            gradient = layer.backward(gradient, self.learn)

    ## Training (fit)
    def train(self, features, labels, batch=5, epochs=10, learn=0.001):
        ## TODO cleanup variabls
        ## TODO cleanup variabls
        ## TODO cleanup variabls
        ## TODO cleanup variabls
        ## TODO cleanup variabls
        ## TODO cleanup variabls
        ## TODO cleanup variabls
        self.learn = learn
        ## TODO cleanup variabls
        ## TODO cleanup variabls
        ## TODO cleanup variabls
        ## TODO cleanup variabls


        ## TODO - Improve Batching0
        ## TODO - Improve Batching0
        ## TODO - Improve Batching0 even distribution
        ## TODO - Improve Batching0
        ## TODO - Improve Batching0
        for i in range(int(epochs) * int(len(features) // batch)):
            inputs =  np.array([features[n] for n in range(batch)])
            targets = np.array([labels[n]   for n in range(batch)])

            ## Forward
            out = self.forward(inputs)

            ## Delta (error)
            delta = out - targets

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
        ) * np.sqrt(1.0 / input)
        self.activation = activation
        self.derivative = derivative

    ## Forward pass (also the Predict method)
    def forward(self, inputs):
        self.input = inputs
        self.output = self.activation(inputs @ self.weights + self.bias)
        return self.output

    ## Backward Propegation (training the model)
    def backward(self, gradient, learn):
        delta = gradient * self.derivative(self.output)
        self.optimize(delta, learn)
        return delta @ self.weights.T

    ## Update model weights with gradient (adjust error)
    def optimize(self, gradient, learn):
        self.weights -= learn * self.input.T @ gradient
        self.bias -= learn * np.mean(gradient, axis=0, keepdims=True)

if __name__ == "__main__": main()
