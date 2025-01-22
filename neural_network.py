import numpy as np


class NeuralNetwork:
    """
    A multilayer perceptron neural network with customisable architecture.
    
    Attributes:
        nn_structure (list): List of dictionaries, each representing a layer's properties.
        weights (dict): Weights of the neural network layers.
        biases (dict): Biases of the neural network layers.
    """
    
    def __init__(self, nn_structure: list):
        """
        Initialise the neural network with given structure.
        
        Args:
            nn_structure (list): List of dictionaries, each containing input shape, output shape, and activation type.
        """
        self.nn_structure = nn_structure
        self.weights = {}
        self.biases = {}

        for i, layer in enumerate(self.nn_structure):
            layer_index = i + 1
            input_shape = layer["input_shape"]
            output_shape = layer["output_shape"]

            self.weights[layer_index] = np.random.randn(input_shape, output_shape) * np.sqrt(2 / input_shape)
            self.biases[layer_index] = np.zeros((1, output_shape))

    def forward_pass(self, input_matrix: np.ndarray) -> np.ndarray:
        """
        Perform a forward pass through the network.
        
        Args:
            input_matrix (np.ndarray): Input data matrix.
        
        Returns:
            np.ndarray: Output of the final layer.
        """
        Z_values = {}
        A_values = {0: input_matrix}
        self.input_matrix_shape = input_matrix.shape[0]

        for i, layer in enumerate(self.nn_structure):
            layer_index = i + 1
            Z = np.dot(input_matrix, self.weights[layer_index]) + self.biases[layer_index]
            Z_values[layer_index] = Z

            activation = layer["activation"]
            if activation == "sigmoid":
                A = 1 / (1 + np.exp(-Z))
            elif activation == "relu":
                A = np.maximum(0, Z)
            elif activation == "none":
                A = Z
            elif activation == "softmax":
                exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
                A = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
            else:
                raise ValueError(f"Unsupported activation function: {activation}")

            A_values[layer_index] = A
            input_matrix = A

        self.Z_values = Z_values
        self.A_values = A_values
        return A_values[len(self.nn_structure)]
    
    def last_layer_loss(self, y_true: np.ndarray, loss_type: str = "mse") -> float:
        """
        Compute the loss for the last layer.
    
        Args:
            y_true (np.ndarray): Ground truth labels.
            loss_type (str): Type of loss function to use ("mse" or "cross_entropy").
        
        Returns:
            float: Computed loss value.
        """
        last_index = len(self.nn_structure)
        Y_hat = self.A_values[last_index]

        if loss_type == "mse":
        # Mean Squared Error Loss
            self.last_error_residual = y_true - Y_hat
            return np.mean(self.last_error_residual ** 2)

        elif loss_type == "cross_entropy":
            # cross-entropy with softmax
            epsilon = 1e-12  # avoid log(0)
            Y_hat_clipped = np.clip(Y_hat, epsilon, 1 - epsilon) 
            cross_entropy_loss = -np.sum(y_true * np.log(Y_hat_clipped)) / y_true.shape[0]
            self.last_error_residual = Y_hat_clipped - y_true
            return cross_entropy_loss

        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    
    def backward_pass(self, y_true):
        """
        Perform backpropagation to compute gradients of weights and biases.
        """
        grad_L_wrt_weight = {}
        grad_L_wrt_biases = {}
        delta_l = {}

        for index, layer in enumerate(self.nn_structure[::-1]):
            layer_index = len(self.nn_structure) - index

            if layer_index == len(self.nn_structure):
                if layer["activation"] == "softmax":
                    delta_l[layer_index] = self.A_values[layer_index] - y_true
                    
                elif layer["activation"] == "sigmoid":
                    delta_l[layer_index] = (-2 / self.A_values[layer_index].shape[0]) * self.last_error_residual
                    
                elif layer["activation"] == "relu":
                    delta_l[layer_index] = (-2 / self.A_values[layer_index].shape[0]) * self.last_error_residual
                    
                elif layer["activation"] == "none":
                    delta_l[layer_index] = self.last_error_residual

            else:
                activation = layer["activation"]
                prev_delta = delta_l[layer_index + 1]
                weight_next = self.weights[layer_index + 1]

                if activation == "sigmoid":
                    delta_l[layer_index] = np.dot(prev_delta, weight_next.T) * (
                        self.A_values[layer_index] * (1 - self.A_values[layer_index]))
                    
                elif activation == "relu":
                    delta_l[layer_index] = np.dot(prev_delta, weight_next.T) * (self.Z_values[layer_index] > 0)
                    
                elif activation == "none":
                    delta_l[layer_index] = np.dot(prev_delta, weight_next.T)

            if layer_index > 1:
                grad_L_wrt_weight[layer_index] = np.dot(self.A_values[layer_index - 1].T, delta_l[layer_index])
                grad_L_wrt_biases[layer_index] = np.sum(delta_l[layer_index], axis=0)
            elif layer_index == 1:  # Input layer
                grad_L_wrt_weight[layer_index] = np.dot(self.A_values[0].T, delta_l[layer_index])
                grad_L_wrt_biases[layer_index] = np.sum(delta_l[layer_index], axis=0)

        self.grad_L_wrt_weight = grad_L_wrt_weight
        self.grad_L_wrt_biases = grad_L_wrt_biases

    def update_parameters(self, learning_rate: float):
        """
        Update the network's weights and biases using computed gradients.
        
        Args:
            learning_rate (float): Learning rate for gradient descent.
        """
        for layer_index in range(1, len(self.nn_structure) + 1):
            self.weights[layer_index] -= (learning_rate * self.grad_L_wrt_weight[layer_index])
            self.biases[layer_index] -= (learning_rate * self.grad_L_wrt_biases[layer_index])
