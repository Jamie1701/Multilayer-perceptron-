'''
As it stands, this model is built with the express purpose of space group prediction for cubic crystal structures.
Crystal structures were obtained from the crystallography open database and represent a variety of different types. 
Be mindful when using MSE loss function as it might not calculate gradients correctly yet. 
'''

import numpy as np

class NeuralNetwork:
    """
    A multilayer perceptron neural network with customisable architecture. 
    Supports batch normalization and dropout.
    Activation functions: sigmoid, relu, softmax.
    Loss functions: mean squared error (mse), cross-entropy. (Currently supports only these)
    ----------
    Attributes:
        nn_structure (list): List of dictionaries, each representing a layer's properties.
        weights (dict): Weights of the neural network layers.
        biases (dict): Biases of the neural network layers.
    ----------
    Methods:
        forward_pass: Perform a forward pass through the network.
        last_layer_loss: Compute the loss for the last layer.
        backward_pass: Perform backpropagation to compute gradients of weights and biases.
        update_parameters: Update the network's weights and biases using computed gradients.
    """
    
    def __init__(self, nn_structure: list):
        """
        Initialise the neural network with given structure. 
        
        Args:
            nn_structure (list): List of dictionaries: 
            input_shape, output_shape, activation, use_batch_norm, dropout_rate and use_dropout.
        """
        self.nn_structure = nn_structure
        self.weights = {}
        self.biases = {}
        self.gamma = {}
        self.beta = {}
        self.moving_mean = {} 
        self.moving_var = {}   
        self.use_batch_norm = {}
        self.use_dropout = {}
        self.dropout_mask = {}
        self.dropout_rate = {}

        for i, layer in enumerate(self.nn_structure):
            layer_index = i + 1
            input_shape = layer["input_shape"]
            output_shape = layer["output_shape"]
            
            activation = layer.get("activation", "relu")
            
            if activation == "relu":
                self.weights[layer_index] = np.random.randn(input_shape, output_shape) * np.sqrt(2 / input_shape)
            elif activation in ["sigmoid", "softmax"]:
                self.weights[layer_index] = np.random.randn(input_shape, output_shape) * np.sqrt(1 / input_shape)
            else:
                self.weights[layer_index] = np.random.randn(input_shape, output_shape) * np.sqrt(1 / input_shape)

            self.biases[layer_index] = np.zeros((1, output_shape))
            
            self.use_batch_norm[layer_index] = layer.get("use_batch_norm", False)
            if self.use_batch_norm[layer_index]:
                self.gamma[layer_index] = np.ones((1, output_shape))
                self.beta[layer_index] = np.zeros((1, output_shape))
                self.moving_mean[layer_index] = np.zeros((1, output_shape))
                self.moving_var[layer_index] = np.zeros((1, output_shape))
                
            self.use_dropout[layer_index] = layer.get("use_dropout", False)
            self.dropout_rate[layer_index] = layer.get("dropout_rate", 0.0)
            
            # if self.use_layer_norm[layer_index]: ### Scrapped in favour of batch norm
            #     self.gamma[layer_index] = np.ones((1, output_shape))
            #     self.beta[layer_index] = np.zeros((1, output_shape))

    def forward_pass(self, input_matrix: np.ndarray, training = True) -> np.ndarray:
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
        epsilon = 1e-8
        momentum = 0.9

        for i, layer in enumerate(self.nn_structure):
            layer_index = i + 1
            Z = np.dot(input_matrix, self.weights[layer_index]) + self.biases[layer_index]
            
            if self.use_batch_norm[layer_index]:
                if training:
                    batch_mean = np.mean(Z, axis=0, keepdims=True)
                    batch_var = np.var(Z, axis=0, keepdims=True)
                    Z_norm = (Z - batch_mean) / np.sqrt(batch_var + epsilon)
                
                    Z = self.gamma[layer_index] * Z_norm + self.beta[layer_index]

                    self.moving_mean[layer_index] = momentum * self.moving_mean[layer_index] + (1 - momentum) * batch_mean
                    self.moving_var[layer_index] = momentum * self.moving_var[layer_index] + (1 - momentum) * batch_var
                else:
                    Z_norm = (Z - self.moving_mean[layer_index]) / np.sqrt(self.moving_var[layer_index] + epsilon)
                    Z = self.gamma[layer_index] * Z_norm + self.beta[layer_index]
            
            Z_values[layer_index] = Z
            
            # Activation functions

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
                raise ValueError(f"Unsupported activation function: {activation} \nSupported activations: sigmoid, relu, none, softmax")
            
            if training and self.use_dropout[layer_index]:
                dropout_mask = np.random.rand(*A.shape) > self.dropout_rate[layer_index]  # Generate mask
                A = A * dropout_mask / (1 - self.dropout_rate[layer_index])
                self.dropout_mask[layer_index] = dropout_mask
            else:
                self.dropout_mask[layer_index] = np.ones_like(A)

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
            epsilon = 1e-9  # avoid log(0)
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
        grad_L_wrt_gamma = {}
        grad_L_wrt_beta = {}
        delta_l = {}
    
        clip_value = 3.0  # Prevent extreme gradients and so contribute to numerical stability

        for index, layer in enumerate(self.nn_structure[::-1]):
            layer_index = len(self.nn_structure) - index

            if layer_index == len(self.nn_structure):
                activation = self.nn_structure[layer_index - 1]["activation"]
            
                if activation == "softmax":
                    delta_l[layer_index] = self.A_values[layer_index] - y_true
            
                else:
                    delta_l[layer_index] = self.A_values[layer_index] - y_true

            else:
                prev_delta = delta_l[layer_index + 1]
                weight_next = self.weights[layer_index + 1]
                delta_l[layer_index] = np.dot(prev_delta, weight_next.T)
                if self.use_dropout.get(layer_index, False):
                    delta_l[layer_index] *= (self.dropout_mask[layer_index] / (1.0 - self.dropout_rate[layer_index]))

                activation = self.nn_structure[layer_index - 1]["activation"]
                Z = self.Z_values[layer_index]
            
                if activation == "sigmoid":
                    delta_l[layer_index] *= (self.A_values[layer_index] * (1 - self.A_values[layer_index]))
                elif activation == "relu":
                    delta_l[layer_index] *= (Z > 0).astype(float)
                elif activation == "none":
                    pass # No activation function
                if self.use_dropout.get(layer_index, False):
                    delta_l[layer_index] *= self.dropout_mask[layer_index] / (1.0 - self.dropout_rate[layer_index])

                delta_l[layer_index] = np.clip(delta_l[layer_index], -clip_value, clip_value)

            # Batch Normalization gradients (if used)
            if self.use_batch_norm.get(layer_index, False):
                batch_mean = np.mean(Z, axis=0, keepdims=True)
                batch_var = np.var(Z, axis=0, keepdims=True) + 1e-8 # Add epsilon for numerical stability
                Z_norm = (Z - batch_mean) / np.sqrt(batch_var)
            
                # Compute gradients for gamma and beta
                grad_L_wrt_gamma[layer_index] = np.sum(delta_l[layer_index] * Z_norm, axis=0, keepdims=True)
                grad_L_wrt_beta[layer_index] = np.sum(delta_l[layer_index], axis=0, keepdims=True)

                # Compute gradient of loss with respect to Z (before normalization)
                N = Z.shape[0]  # Number of samples in the batch
                delta_norm = delta_l[layer_index] * self.gamma[layer_index]
            
                dvar = np.sum(delta_norm * (Z - batch_mean) * -0.5 * np.power(batch_var, -1.5), axis=0, keepdims=True)
                dmean = np.sum(delta_norm * -1 / np.sqrt(batch_var), axis=0, keepdims=True) + dvar * np.sum(-2 * (Z - batch_mean), axis=0, keepdims=True) / N
            
                delta_l[layer_index] = delta_norm / np.sqrt(batch_var) + (dvar * 2 * (Z - batch_mean) / N) + (dmean / N)

                # Clip batch norm gradients
                grad_L_wrt_gamma[layer_index] = np.clip(grad_L_wrt_gamma[layer_index], -clip_value, clip_value)
                grad_L_wrt_beta[layer_index] = np.clip(grad_L_wrt_beta[layer_index], -clip_value, clip_value)

            # Weight and bias gradients
            grad_L_wrt_weight[layer_index] = np.dot(self.A_values[layer_index - 1].T, delta_l[layer_index])
            grad_L_wrt_biases[layer_index] = np.sum(delta_l[layer_index], axis=0)

            # Clip gradients for numerical stability 
            grad_L_wrt_weight[layer_index] = np.clip(grad_L_wrt_weight[layer_index], -clip_value, clip_value)
            grad_L_wrt_biases[layer_index] = np.clip(grad_L_wrt_biases[layer_index], -clip_value, clip_value)

        self.grad_L_wrt_weight = grad_L_wrt_weight
        self.grad_L_wrt_biases = grad_L_wrt_biases
        self.grad_L_wrt_gamma = grad_L_wrt_gamma
        self.grad_L_wrt_beta = grad_L_wrt_beta

    def update_parameters(self, learning_rate: float, epoch: int, decay_rate: float = 0.01):
        """
        Update the network's weights and biases using computed gradients.
        Learning rate decay is also applied. 
        ----------
        Args:
            learning_rate (float): Learning rate for gradient descent.
        """
        
        lr = learning_rate / (1+decay_rate*epoch)
        
        for layer_index in range(1, len(self.nn_structure) + 1):
            self.weights[layer_index] -= (lr * self.grad_L_wrt_weight[layer_index])
            self.biases[layer_index] -= (lr * self.grad_L_wrt_biases[layer_index])
            
            if self.use_batch_norm[layer_index]:
                self.gamma[layer_index] -= (lr * self.grad_L_wrt_gamma[layer_index])
                self.beta[layer_index] -= (lr * self.grad_L_wrt_beta[layer_index])