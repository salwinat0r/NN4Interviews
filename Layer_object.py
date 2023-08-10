# Neural Networks from scratch
import numpy as np
import sys
import matplotlib.pyplot as plt

np.random.seed(0)
# output = input*weight + bias
X = [[1, 2, 3, 2.5],
         [2.0, 5.0, -1.0, 2.0],
         [-1.5, 2.7, 3.3, -0.8]]

# dense layer object
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


layer1 = Layer_Dense(4,5) #n_neurons can be anything
layer2 = Layer_Dense(5,2) #n_inputs has to be = n_neurons in the last layer
layer1.forward(X)
layer2.forward(layer1.output)
print(layer2.output)