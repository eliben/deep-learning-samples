import numpy as np

import math_utils
import neural_net


# Create some toy data to check your implementations
input_size = 4
hidden_size = 10
num_classes = 3
num_inputs = 5


# The toy model is a 2-layer network: it has one hidden layer and one output
# layer.
def init_toy_model():
    model = {}
    # Layer 1 has 10 neurons. Each has 4 connections (one from each input).
    # Overall number of weights is 4 * 10, with weight [i][j] being the
    # weight from input i to neuron j in this layer.
    model['W1'] = np.linspace(-0.2, 0.6, num=input_size * hidden_size)
    model['W1'].shape = (input_size, hidden_size)

    # Each neuron in layer 1 has a bias.
    model['b1'] = np.linspace(-0.3, 0.7, num=hidden_size)

    # Layer 2 has 3 neurons (outputs). Each has 10 connections (one from each
    # neuron in the hidden layer). Overall number of weights is 10 * 3, with
    # weight [i][j] being the weight from hidden neuron i to output neuron j.
    model['W2'] = np.linspace(-0.4, 0.1, num=hidden_size * num_classes)
    model['W2'].shape = (hidden_size, num_classes)
    model['b2'] = np.linspace(-0.5, 0.9, num=num_classes)
    return model


def init_toy_data():
    X = np.linspace(-0.2, 0.5, num=num_inputs * input_size)
    X.shape = (num_inputs, input_size)
    y = np.array([0, 1, 2, 2, 1])
    return X, y


model = init_toy_model()
X, y = init_toy_data()

loss, grads = neural_net.two_layer_net(X, model, y, reg=0.1)
print(loss)

import pprint
pprint.pprint(grads)

#print(scores)
#correct_scores = [[-0.5328368, 0.20031504, 0.93346689],
 #[-0.59412164, 0.15498488, 0.9040914 ],
 #[-0.67658362, 0.08978957, 0.85616275],
 #[-0.77092643, 0.01339997, 0.79772637],
 #[-0.89110401, -0.08754544, 0.71601312]]
#print 'Difference between your scores and correct scores:'
#print np.sum(np.abs(scores - correct_scores))
