import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import Sequential, layers
from sklearn.metrics import accuracy_score
import pickle
# ------------------------------------------------------------ #
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")
# ------------------------------------------------------------ #

data_file = '../../data/train.csv'
save_model_path = "../../results/model_params"

def default_layer(X_train, units, regularizer=None, dropout_rate = 0.3):
    """
    Creates a default layer
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
                layers.Dense(units=units//6, activation='relu', kernel_regularizer=regularizer),
                layers.Dropout(rate=dropout_rate),
                layers.BatchNormalization(),
                layers.Dense(units=units//4, activation='relu', kernel_regularizer=regularizer),
                layers.Dropout(rate=dropout_rate),
                layers.BatchNormalization(),
                layers.Dense(units=units//2, activation='relu', kernel_regularizer=regularizer),
                layers.Dropout(rate=dropout_rate),
                layers.BatchNormalization(),
                layers.Dense(units=round(units), activation='relu', kernel_regularizer=regularizer),
                layers.Dropout(rate=dropout_rate),
                layers.BatchNormalization(),
                layers.Dense(units=1, activation='sigmoid')
            ]


class NeuralNetworkClassifier:
    def __init__(self, X_train, X_test, y_train, y_test, model = None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.accuracy = None
        self.params = None
        self.model = model
        self.history = None

        self.callbacks = keras.callbacks.EarlyStopping(
                patience=50, # Specifies the number of epoch (number of forward steps) with no improvement after which the training stop (So if the validation percentage does not increase for 20 consecutive epochs, it's going to stop the training)
                min_delta=0.001, # This is the minimum change in validation loss (speaks about ability of the system to predict new data) that is required for an epoch to be consider successful
                restore_best_weights=True, # If the training is stopped, we take the best weights (the ones that lead to the lowest validation loss) of all the analysis we did
                )


    def add_layer(self, layers):
        """ Create a layer to the Neural Network

        layers - a list of Keras.layer
        """
        if self.model is None:
            self.model = Sequential()
        for layer in layers:
            self.model.add(layer)
        return self.model

    def create_default_model(self, *args, **kwargs):
        """
        Create the default model
        """
        layer = default_layer(self.X_train, *args, **kwargs)
        self.model = self.add_layer(layer)
        return self.model
    
        
    def fit(self, epochs = 100, batch_size = 512, learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-07):
        """
        Fits the model with the training dataset.
        """
        if self.model is None:
            raise ValueError("Model is None")
        opt = tf.keras.optimizers.Adam(learning_rate= learning_rate, beta_1=beta_1, beta_2 =  beta_2, epsilon =  epsilon) 
        self.model.compile(optimizer = opt, loss="binary_crossentropy", metrics=['binary_accuracy'])
        self.history = self.model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), batch_size=batch_size,
                       epochs=epochs, callbacks=self.callbacks, verbose=2)

        
    def predict(self, X_test=None):
        """
        Predicts the target values for the input data.

        Inputs:
        X_test: array-like, shape (n_samples, n_features), optional (default=None)
            Test data.

        Returns:
        array-like, shape (n_samples,)
            Predicted target values.
        """
        if self.model is None:
            raise ValueError("Model is not fitted yet. Call fit method first.")

        if X_test is None:
            X_test = self.X_test
        return self.model.predict(X_test)

    def score(self, y_true, y_pred):
        """
        Calculates the accuracy score based on true and predicted values.

        Parameters:
        y_true: array-like, shape (n_samples,)
            True target values.
        y_pred: array-like, shape (n_samples,)
            Predicted target values.

        Returns:
        float
            Accuracy score.
        """
        y_pred = (y_pred > 0.5).astype(int)
        return accuracy_score(y_true, y_pred)

    def default_model_best_fit(self, units_list = [128, 256, 512],
                 regularizers_list=[None, keras.regularizers.l1_l2(l1=0.01, l2=0.01), keras.regularizers.l1(0.01), keras.regularizers.l2(0.01)], dropout_rate=0.3,  batch_size = 512):
        """
        Finds the best model based on accuracy score.

        Inputs:
        units_list: list of list of int
            List of hidden layer unit configurations.
        regularizers_list: list of keras.regularizers.Regularizer or None, optional (default=[None])
            List of regularizers to try.
        dropout_rate: float, optional (default=0.3)
            Dropout rate.

        Returns:
        None
        """
        max_accuracy = 0.0
        best_params = None
        best_model = None
        
        for units in units_list:
            for regularizer in regularizers_list:
                self.model = self.create_default_model(units, regularizer, dropout_rate)
                self.fit()
                predictions = self.predict()
                accuracy = self.score(self.y_test, predictions)
                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    best_params = {'units': units, 'regularizer': regularizer, 'dropout_rate': dropout_rate}
                    best_model = self.model

        self.accuracy = max_accuracy
        self.params = best_params
        self.model = best_model

    def print_best_params(self):
        """
        Prints the best hyperparameters found.
        """
        print(f"Best parameters - Units: {self.params['units']}, Regularizer: {self.params['regularizer']}, Dropout Rate: {self.params['dropout_rate']}")
        print(f"Validation accuracy for Neural Network: {self.accuracy:.3f}")

    def summary(self):
        """
        Print the model summary
        """
        print(self.model.summary())

    def get_training_metrics(self, plot = False):
        """
        Returns the training and validation loss, and accuracy from the training history.
        """
        if self.history is None:
            raise ValueError("History is not initialized. Fit the model first.")
        qm_loss = self.history.history['loss']
        qm_val_loss = self.history.history['val_loss']
        qm_binary_accuracy = self.history.history['binary_accuracy']
        qm_val_binary_accuracy = self.history.history['val_binary_accuracy']

        if plot == True:
            qm_history_df = pd.DataFrame(qm_fitting)
            qm_history_df[['loss', 'val_loss']].plot()
            qm_history_df[['accuracy', 'val_accuracy']].plot()
            qm_loss, qm_accuracy = qm_model.evaluate(test_images, test_labels)
            print()
            print(f"Validation Loss: {qm_loss}, Validation Accuracy: {qm_accuracy}")


        return qm_loss, qm_val_loss, qm_binary_accuracy, qm_val_binary_accuracy

    import pickle

    def save_model(self, weights_file, history_file):
        """
        Saves the model, training history, and weights to files.

        Inputs:
        history: dict
            The training history dictionary.
        weights_file: str
            File path to save the model weights.
        history_file: str
            File path to save the training history.
        """
        # Save model weights
        with open(weights_file, 'wb') as f:
            pickle.dump(self.model.get_weights(), f)
        
        # Save training history
        with open(history_file, 'wb') as f:
            pickle.dump(self.history.history, f)

    def load_model(self, weights_file, history_file):
        """
        Loads the model, training history, and weights from files.

        Inputs:
        model: tf.keras.Model
            The model object to load the weights into.
        weights_file: str
            File path to load the model weights from.
        history_file: str
            File path to load the training history from.

        Returns:
        history: dict
            The training history dictionary.
        """
        # Load model weights
        asset
        with open(weights_file, 'rb') as f:
            weights = pickle.load(f)
        self.model.set_weights(weights)
        
        # Load training history
        with open(history_file, 'rb') as f:
            self.history = pickle.load(f)
        
        return self.model, self.history


"""
if __name__ == '__main__':
    import pre_processing
    from pre_processing import DataProcessor

    dataset_df = pd.read_csv("./data/train.csv")
    dataset_df = DataProcessor(dataset_df)

    X_train, X_test, y_train, y_test = dataset_df.run_and_split()

    from time import time
    s_t = time()
    nn_classifier = NeuralNetworkClassifier(X_train, X_test, y_train, y_test)
    print(time()-s_t)
    s_t = time()
    nn_classifier.default_model_best_fit(regularizers_list=[None])
    print(time()-s_t)
    nn_classifier.print_best_params()

    # Example of saving and loading the model
    #nn_classifier.save_model("./checkpoints/qm_model_weights.pkl", "./checkpoints/qm_fitting.pkl")
    #nn_classifier_loaded = NeuralNetworkClassifier(X_train, X_test, y_train, y_test)
    #nn_classifier_loaded.load_model("./checkpoints/qm_model_weights.pkl", "./checkpoints/qm_fitting.pkl")
"""


if __name__ == '__main__':
    from mpi4py import MPI
    import pre_processing
    from pre_processing import DataProcessor

    dataset_df = pd.read_csv(data_file)
    dataset_df = DataProcessor(dataset_df)

    X_train, X_test, y_train, y_test = dataset_df.run_and_split()
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    local_X_train = np.array_split(X_train, size)[rank]
    local_y_train = np.array_split(y_train, size)[rank]

    nn_classifier = NeuralNetworkClassifier(local_X_train, X_test, local_y_train, y_test)
    nn_classifier.best_fit(regularizers_list=[None])

    global_accuracy = comm.gather(nn_classifier.accuracy, root=0)
    global_params = comm.gather(nn_classifier.params, root=0)

    if rank == 0:
        max_accuracy_index = np.argmax(global_accuracy)
        best_accuracy = global_accuracy[max_accuracy_index]
        best_params = global_params[max_accuracy_index]
        print("Best model found across all processes:")
        print(f"Best parameters - Units: {best_params['units']}, Regularizer: {best_params['regularizer']}, Dropout Rate: {best_params['dropout_rate']}")
        print(f"Validation accuracy for Neural Network: {best_accuracy:.3f}")

    nn_classifier.save_model(save_model_path + "qm_model_weights.pkl", save_model_path + "qm_fitting.pkl")

## mpirun -n <num_processes> python neural_network_mpi.py
