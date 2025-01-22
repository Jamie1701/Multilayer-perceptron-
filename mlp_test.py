"""
Test Implementation of a Multilayer Perceptron (MLP) Neural Network

This script demonstrates a basic implementation of an MLP for a regression task.
This can be extended for crystal structure feature prediction but accuracy varies.
-- Only really effective on simpler systems so far for classification. 

"""

import numpy as np
from neural_network import NeuralNetwork
import time

# Define the architecture of the neural network
nn_structure = [
    {"input_shape": 2, "output_shape": 5, "activation": "relu"},  # Activation supports: none, sigmoid, relu, softmax
    {"input_shape": 5, "output_shape": 5, "activation": "none"},
    {"input_shape": 5, "output_shape": 5, "activation": "none"},
    {"input_shape": 5, "output_shape": 1, "activation": "relu"},
]

y_train = np.array([
    [0.41, 0.22],
    [0.33, 0.74],
    [0.55, 0.16],
    [0.27, 0.98],
])  # Random training input matrix

y_true = np.array([
    [0.9],
    [1.0],
    [1.1],
    [1.2],
])  

test_input = np.array([
    [0.25, 0.32],
    [0.44, 0.57],
])  

test_y_true = np.array([
    [0.95],
    [1.05],
])  

epochs = 1000
learning_rate = 0.001

nn = NeuralNetwork(nn_structure)

# Training the mlp
for epoch in range(epochs):
    # Perform forward pass and calculate loss
    predictions = nn.forward_pass(y_train)
    loss = nn.last_layer_loss(y_true, loss_type="mse")  # Supported loss types: mse, cross_entropy
    
    # Backpropagation and parameter update
    nn.backward_pass(y_true=y_true)
    nn.update_parameters(learning_rate=learning_rate)
    
    # Print progress every 10 epochs
    if epoch % 10 == 0:
        test_predictions = nn.forward_pass(test_input)
        test_loss = nn.last_layer_loss(test_y_true, loss_type="mse")
        print(f"Epoch: {epoch}/{epochs}, Training Loss: {loss:.6f}, Testing Loss: {test_loss:.6f}")
        print(f"Test Predictions: {test_predictions}")
        time.sleep(0.3)