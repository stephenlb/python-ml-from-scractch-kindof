## TODAY WE ARE WRITING ML FRAMEWORK
## FROM SCRATCH PART 7: Cleanup and MNIST
import activations
import numpy as np
import pytorch

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
        Layer(input=100,  output=200),
        Layer(input=200,  output=500),
        Layer(input=500,  output=10, activation=activations.sigmoid, derivative=activations.sigmoid_derivative),
    ]
    model = Model(layers)
    model.train(x[:1000], y[:1000], batch=50, epochs=2000, learn=0.001)


    ## Train = model learns on trainging set
    ## Validate = test model on training set

    x_test = x[:100]
    y_test = y[:100]
    predictions = model.forward(x_test)
    print("Results:")
    #print(predictions)

    ## Max value index
    results = []
    for p in predictions:
        maxp = np.argmax(p)
        results.append([ 1 if maxp == i else 0 for i in range(len(p))])

    #results = np.round(prediction)
    print(np.array(results))

    ## ASCII Blocks Descending Size Non-colored
    ## TODO
    ## TODO
    ## TODO
    ## TODO
    blocks = ['█','▓','▒','░',' ']

    display =['✅' if y[i][0] == r[0] else '⛔️' for i, r in enumerate(results)] 
    print(''.join(display))

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
    def train(self, features, labels, batch=5, epochs=10, learn=0.001):
        for layer in self.layers:
            layer.learn = learn

        ## Randomize Batching
        ## "Stochastic Gradient Descent"
        for i in range(epochs):
            stochastic = np.random.permutation(len(features))
            inputs = features[stochastic]
            targets = labels[stochastic]

            for i in range(len(features) // batch):
                input  =  inputs[i * batch : i * batch + batch]
                target = targets[i * batch : i * batch + batch]

                ## Forward
                out = self.forward(input)

                ## Delta (error)
                delta = out - target

                ## COST 
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
