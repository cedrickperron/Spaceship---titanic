import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class RandomForestClassifierCustom:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.model = None
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.accuracy = None
        self.params = None
        
    def fit(self, max_depth, min_samples_split, criterion):
        """
        Fits the model with the training dataset.

        Parameters:
        max_depth: int
            Maximum depth of the tree.
        min_samples_split: int
            Minimum number of samples required to split an internal node.
        criterion: str
            Function to measure the quality of a split.

        Returns:
        None
        """
        self.model = RandomForestClassifier(random_state=1, criterion=criterion, min_samples_split=min_samples_split, max_depth=max_depth)
        self.model.fit(self.X_train, self.y_train)

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
        return accuracy_score(y_true, y_pred)

    def best_fit(self, max_depths = [1, 5, 10, 15, 20, 25, 30, 50, 100],
                 min_samples_splits = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
                 criterions=["entropy", "gini"] ):
        """
        Finds the best model based on accuracy score.

        Parameters:
        max_depths: list of int
            Maximum depths of the trees to try.
        min_samples_splits: list of int
            Minimum number of samples required to split an internal node.
        criterions: list of str
            Functions to measure the quality of a split.

        Returns:
        sklearn.ensemble.RandomForestClassifier
            Best trained model.
        tuple
            Hyperparameters (max_depth, min_samples_split, criterion) for the best model.
        float
            Accuracy score of the best model.
        """
        max_accuracy = 0.0
        best_hyperparameters = None
        best_model = None
        for criterion in criterions:
            for max_depth in max_depths:
                for min_samples_split in min_samples_splits:
                    self.fit(max_depth, min_samples_split, criterion)
                    predictions = self.predict()
                    accuracy = self.score(self.y_test, predictions)
                    if accuracy > max_accuracy:
                        max_accuracy = accuracy
                        best_hyperparameters = (max_depth, min_samples_split, criterion)
                        best_model = self.model

        self.accuracy = max_accuracy
        self.params = best_hyperparameters
        self.model = best_model
        return self.model

    def parameter_plot(self):
        """
        Plots accuracy scores for different values of k.

        Parameters:
        None

        Returns:
        None
        """
        pass
    
    def print_best_params(self):
        """
        Prints the best hyperparameters and accuracy score.
        """
        print(f"Validation accuracy for Random Forest: {self.accuracy:.3f}")
        print(f"Best parameters: {self.params}")

if __name__ == '__main__':
    from .. import data_processing
    from data_processing import DataProcessor

    dataset_df = pd.read_csv("./data/train.csv")
    dataset_df = DataProcessor(dataset_df)

    X_train, X_test, y_train, y_test = dataset_df.run_and_split()
    
    rf_classifier = RandomForestClassifierCustom(X_train, X_test, y_train, y_test)
    rf_classifier.best_fit()
    
    rf_classifier.print_best_params()
