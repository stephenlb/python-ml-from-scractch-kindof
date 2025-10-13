## TODAY WE ARE WRITING ML FRAMEWORK
## FROM SCRATCH PART 7: Cleanup and MNIST
import activations
import numpy as np

import layer as L
import model as M

## XOR Operator
## features (input) (training)
x = np.array([[0,0],[0,1],[1,0],[1,1]])

## labels (output) actual answer (target)
y = np.array([[ 0 ],[ 1 ],[ 1 ],[ 0 ]])
## AND Operator
#y = np.array([[ 0 ],[ 0 ],[ 0 ],[ 1 ]])
## OR Operator
#y = np.array([[ 0 ],[ 1 ],[ 1 ],[ 1 ]])

### Main
layers = [
    L.Layer(input=2,  output=10),
    L.Layer(input=10,  output=10),
    L.Layer(input=10,  output=1, activation=activations.sigmoid, derivative=activations.sigmoid_derivative),
]

model = M.Model(layers)
model.train(x, y, batch=4, epochs=2000, learn=0.03)
predictions = model.forward(x)
print("Results:")
print(predictions)

results = np.round(predictions)
print(np.array(results))

display =['✅' if y[i][0] == r[0] else '⛔️' for i, r in enumerate(results)] 
print(''.join(display))
