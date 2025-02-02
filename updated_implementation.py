'''
------------------------------------------------------------------------------------------------------------------------
Example implementation on a real data set of cubic crystal structures for space group prediction.
The data set is obtained from the Crystallography Open Database (COD) and contains approx 3300 entries.
Other features may be able to be preducted from the data set, however, the focus is on space group prediction.
Model still requires hyperparameter optimisation and stabilisation, but it still performs fairly well. 
------------------------------------------------------------------------------------------------------------------------
'''

from neural_network import NeuralNetwork
from data_scaling import X_train, X_test, y_train, y_test
import numpy as np 

# Approx 71% accuracy (max) with epochs = 1000, learning_rate = 0.01, decay_rate = 0.1 with the following architecture.
# Accuracy will fluctuate as a results of the random initialisation of the weights/bias and the random selection of the training/testing data. 
nn_structure = [{"input_shape": 139, "output_shape": 256, "activation": "relu", "use_batch_norm": True, "dropout_rate": 0.0, "use_dropout": False}, # Approx 71% accuracy (max) with epochs = 1000, learning_rate = 0.01, decay_rate = 0.1
                {"input_shape": 256, "output_shape": 256, "activation": "relu", "use_batch_norm": False, "dropout_rate": 0.1, "use_dropout": True},
                {"input_shape": 256, "output_shape": 512, "activation": "relu", "use_batch_norm": False, "dropout_rate": 0.0, "use_dropout": False},
                {"input_shape": 512, "output_shape": 512, "activation": "relu", "use_batch_norm": True, "dropout_rate": 0.1, "use_dropout": True},
                {"input_shape": 512, "output_shape": 512, "activation": "relu", "use_batch_norm": False, "dropout_rate": 0.1, "use_dropout": True},
                {"input_shape": 512, "output_shape": 256, "activation": "relu", "use_batch_norm": False, "dropout_rate": 0.0, "use_dropout": False},
                {"input_shape": 256, "output_shape": 144, "activation": "softmax", "use_batch_norm": False, "dropout_rate": 0.0, "use_dropout": False},
]

epochs = 1000
learning_rate = 0.01
decay_rate = 0.1

nn = NeuralNetwork(nn_structure)

for epoch in range(epochs):
    y_pred = nn.forward_pass(X_train, training=True)

    loss = nn.last_layer_loss(y_train, loss_type="cross_entropy")

    nn.backward_pass(y_train)

    nn.update_parameters(learning_rate, epoch, decay_rate)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")
        
# Evaluation
y_pred_test = nn.forward_pass(X_test, training=False)
y_pred_classes = np.argmax(y_pred_test, axis=1)  # Probability to class label 
y_true_classes = np.argmax(y_test, axis=1) 
accuracy = np.mean(y_pred_classes == y_true_classes) * 100
print(f"Test Set Accuracy: {accuracy:.2f}%")
num_classes = y_test.shape[1] # 144 classes (sg to choose from)
random_preds = np.random.randint(0, num_classes, size=len(y_true_classes))

# Random guess for comparison
accuracy_random = np.mean(random_preds == y_true_classes) * 100
print(f"Test Set Accuracy (Random Guessing): {accuracy_random:.2f}%") # Should be around 0.7% accuracy.