## TODAY WE ARE WRITING ML FRAMEWORK
## FROM SCRATCH PART 7: Cleanup and MNIST
import activations
import numpy as np

import layer as L
import model as M

import tensorflow
from tensorflow.keras.datasets import mnist

## Load Data
(X_train,y_train),(X_test,y_test) = mnist.load_data()

## Trainging
x = X_train.reshape(60000, 28*28).astype('float32') / 255.0
x_test = x[:100] #X_test.reshape(60000, 28*28).astype('float32') / 255.0

## Testing
y = np.array([[ 1 if i == j else 0 for i in range(10)] for j in y_train])
y_test = y[:100] #np.array([[ 1 if i == j else 0 for i in range(10)] for j in y_test])

### Main
layers = [
    L.Layer(input=784,  output=100),
    L.Layer(input=100,  output=200),
    L.Layer(input=200,  output=500),
    L.Layer(input=500,  output=10, activation=activations.sigmoid, derivative=activations.sigmoid_derivative),
]
model = M.Model(layers)
model.train(x[:1000], y[:1000], batch=50, epochs=2000, learn=0.001)

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
