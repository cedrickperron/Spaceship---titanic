import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import Sequential, layers, Input, Model
import numpy as np
import random

import pennylane as qml
import pennylane.numpy as npp
import matplotlib.pyplot as plt

import sys
sys.path.append("../classical_models")
from NeuralNetwork import NeuralNetworkClassifier
sys.path.append("../quantum_models")
from QuantumCircuit import StronglyEntanglingQuantumCircuit, BaseQuantumCircuit, MPSQuantumCircuit

# FILTER OUT SOME WARNING
import warnings
#warnings.filterwarnings("ignore", message="You are casting an input of type complex128 to an incompatible dtype float32.")
warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')


def default_layer(X_train, units, regularizer=None, dropout_rate = 0.3):
    """
    Creates the starting and hidden classical layer of the NN
    """
    return [
                layers.BatchNormalization(),
                layers.Dense(units=units, activation='relu', kernel_regularizer=regularizer, input_shape=[X_train.shape[1]]),
                layers.Dropout(rate=dropout_rate),
                layers.BatchNormalization(),
                layers.Dense(units=units//2, activation='relu', kernel_regularizer=regularizer),
                layers.Dropout(rate=dropout_rate),
                layers.BatchNormalization(),
                layers.Dense(units=units//4, activation='relu', kernel_regularizer=regularizer),
                layers.Dropout(rate=dropout_rate),
                layers.BatchNormalization(),
            ]




class QuantumNeuralNetwork(NeuralNetworkClassifier):
    def __init__(self, X_train, X_test, y_train, y_test):
        super().__init__(X_train, X_test, y_train, y_test)
        self.x = Input(shape=(self.X_train.shape[1],))


    def apply_qlayers(self, x, qm_circuit, concatenate = True, axis = 1, normalize = False):
        """
        Takes as inputs:
        x - Input to pass to the quantum layer
        qm_circuit - quantum circuit from one of the class ( MPSQuantumCircuit, StronglyEntanglingQuantumCircuit)

        """
        n_wires = qm_circuit.n_wires
        n_qm_circuits = x.shape[1] // n_wires
    
        weight_shape = qm_circuit.initialize_parameters().shape

        """ We need to define the qnode dynamically, and cannot use the one inside the class BaseQuantumCircuit """
        dev = qml.device("default.qubit", wires=range(n_wires))
        @qml.qnode(dev, diff_method="backprop", interface="tf")
        def qnode(inputs, weights):
            return qm_circuit.quantum_circuit(inputs, weights, normalize = normalize)
            
        qlayers = []
        for i in range(n_qm_circuits):
            qlayer = qml.qnn.KerasLayer(qnode, weight_shapes =  {'weights': weight_shape}, output_dim=n_wires, trainable=True) # weight_specs = {"weights": {"initializer": "random_uniform"}}
            qlayers.append(qlayer)

        ## Split the input tensor (x) and pass it to the qlayers
        x = layers.Lambda(lambda x: tf.split(x, n_qm_circuits, axis = axis))(x) 
        output = [qlayer(out) for q_layer, out in zip(qlayers, x)]

        if concatenate == True:
            output = layers.Concatenate(axis=axis)(output)

        return output

    def apply_dense_layer(self, x, m, activation = None):
        """
        Apply a Dense Layer (for renormalization)

        Inputs:
        x - inputs of the layer
        m - number of units of the Dense layer (size of the output)
        activation - Activation function of the dense layer (
        
        """
        x = layers.Dense(m, activation = activation)(x)
        return x

    def apply_classical_layers(self, x, layer_list):
        """
        Applies a list of layers to the model
        """
        for layer in layer_list:
            x = layer(x)
        return x

    def create_model(self, inputs, outputs):
        """
        Creates the model
        """
        self.model = Model(inputs = inputs, outputs = outputs)
        return self.model

    def create_default_model(self, m, units = 128, MPS = True, *args, **kwargs):
        """
        Creates the default model (choosing either MPS or StronglyEntanglingLayer)

        """
        x = self.x
        inputs = x
        
        layer_list = default_layer(x, units = units)
        x = self.apply_classical_layers(x, layer_list)  ## Apply the first layers
        x = self.apply_dense_layer(x, m)  ## Apply the normalization layer to compact it into m

        # Create a MPS or a StronglyEntanglingQuantumCircuit
        qm_circuit = MPSQuantumCircuit( *args, **kwargs ) if MPS else StronglyEntanglingQuantumCircuit( *args, **kwargs)

        x = self.apply_qlayers(x, qm_circuit, concatenate = True)

        x = self.apply_dense_layer(x, 1, activation = 'sigmoid')

        self.model = self.create_model(inputs, x)
        return self.model
            

if __name__ == '__main__':
    import pre_processing
    from pre_processing import DataProcessor

    dataset_df = pd.read_csv("./data/train.csv")
    dataset_df = DataProcessor(dataset_df)

    X_train, X_test, y_train, y_test = dataset_df.run_and_split()
    print(X_train.shape)
    n_layers = 5
    n_wires = 5
    m = 40
    from time import time
    s_t = time()
    # Create an instance of QuantumNeuralNetwork
    quantum_nn = QuantumNeuralNetwork(X_train, X_test, y_train, y_test)
    print(time()-s_t)
    s_t = time()
    quantum_nn.model = quantum_nn.create_default_model(m, n_layers=n_layers, n_wires= n_wires, MPS = True)
    print(time()-s_t)
    # Fit the model
    quantum_nn.fit(epochs = 20, batch_size = 100,  learning_rate = 0.01)
    
    print(time()-s_t)
    # Predict
    y_pred = quantum_nn.predict(X_test[:250])

    # Calculate the score
    score = quantum_nn.score(y_test[:250], y_pred)

    print("Accuracy score:", score)



    

    


