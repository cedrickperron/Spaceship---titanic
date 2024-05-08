import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import Sequential, layers
from sklearn.metrics import accuracy_score

# ------------------------------------------------------------ #
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")
# ------------------------------------------------------------ #



early_stopping = keras.callbacks.EarlyStopping(
    patience=20, # Specifies the number of epoch (number of forward steps) with no improvement after which the training stop (So if the validation percentage does not increase for 20 consecutive epochs, it's going to stop the training)
    min_delta=0.001, # This is the minimum change in validation loss (speaks about ability of the system to predict new data) that is required for an epoch to be consider successful
    restore_best_weights=True, # If the training is stopped, we take the best weights (the ones that lead to the lowest validation loss) of all the analysis we did
)

class NeuralNetworkClassifier:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.model = None
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.accuracy = None
        self.params = None

    def fit(self, units, regularizer=None, dropout_rate=0.3):
        """
        Fits the model with the training dataset.

        Parameters:
        units: list of int
            Number of units in each hidden layer.
        regularizer: keras.regularizers.Regularizer or None
            Regularizer to apply to the kernel weights matrix.
        dropout_rate: float
            Dropout rate.

        Returns:
        None
        """
        self.model = Sequential([
            layers.BatchNormalization(),  # Normalize the input values passed into the first layer of the network
            layers.Dense(units=units, activation='relu', kernel_regularizer=regularizer, input_shape=[self.X_train.shape[1]]),  # First hidden layer
            layers.Dropout(rate=dropout_rate),  # Regularization to prevent overfitting
            layers.BatchNormalization(),
            layers.Dense(units=units//2, activation='relu', kernel_regularizer=regularizer),  # Second hidden layer
            layers.Dropout(rate=dropout_rate),
            layers.BatchNormalization(),
            layers.Dense(units=units//4, activation='relu', kernel_regularizer=regularizer),  # Third hidden layer
            layers.Dropout(rate=dropout_rate),
            layers.BatchNormalization(),
             layers.Dense(units=units//6, activation='relu', kernel_regularizer=regularizer),  # Third hidden layer
            layers.Dropout(rate=dropout_rate),
            layers.BatchNormalization(),
             layers.Dense(units=units//4, activation='relu', kernel_regularizer=regularizer),  # Third hidden layer
            layers.Dropout(rate=dropout_rate),
            layers.BatchNormalization(),
            layers.Dense(units=units//2, activation='relu', kernel_regularizer=regularizer),  # Fourth hidden layer
            layers.Dropout(rate=dropout_rate),
            layers.BatchNormalization(),
            layers.Dense(units=round(units), activation='relu', kernel_regularizer=regularizer),  # Fifth hidden layer
            layers.Dropout(rate=dropout_rate),
            layers.BatchNormalization(),
            layers.Dense(units=1, activation='sigmoid')  # Output layer
        ])

        # Compile the model
        self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['binary_accuracy'])

        # Fit the model
        self.model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), batch_size=512,
                       epochs=100, callbacks=[early_stopping], verbose=0)

    def predict(self, X_test=None):
        """
        Predicts the target values for the input data.

        Parameters:
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

    def best_fit(self, units_list = [128, 256, 512],
                 regularizers_list=[None, keras.regularizers.l1_l2(l1=0.01, l2=0.01), keras.regularizers.l1(0.01), keras.regularizers.l2(0.01)], dropout_rate=0.3):
        """
        Finds the best model based on accuracy score.

        Parameters:
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
                self.fit(units, regularizer, dropout_rate)
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

if __name__ == '__main__':
    from .. import data_processing
    from data_processing import DataProcessor

    dataset_df = pd.read_csv("./data/train.csv")
    dataset_df = DataProcessor(dataset_df)

    X_train, X_test, y_train, y_test = dataset_df.run_and_split()


    nn_classifier = NeuralNetworkClassifier(X_train, X_test, y_train, y_test)
    nn_classifier.best_fit(regularizers_list = [None])
    nn_classifier.print_best_params()
