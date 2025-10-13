## TODAY WE ARE WRITING ML FRAMEWORK
## FROM SCRATCH PART 7: Cleanup and MNIST
import activations
import numpy as np
np.seterr(divide='ignore') ## TODO FIX THIS

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
