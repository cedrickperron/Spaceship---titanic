import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

class KNNClassifier:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.model = None
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test =y_test
        self.k_scores_test = []
        self.accuracy = None
        self.params = None
        
    def fit(self, n_neighbors, weight):
        """
        Fits the model with the training dataset.

        Parameters:
        X_train: array-like, shape (n_samples, n_features)
            Training data.
        y_train: array-like, shape (n_samples,)
            Target values.
        n_neighbors: int
            Number of neighbors to use.
        weights: str
            Weight function used in prediction.
        """
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weight)
        self.model.fit(self.X_train, self.y_train)

    def predict(self, X_test= None):
        """
        Predicts the target values for the input data.

        Parameters:
        X_test: array-like, shape (n_samples, n_features)
            Test data.

        Returns:
        array-like, shape (n_samples,)
            Predicted target values.
        """
        if self.model is None:
            raise ValueError("Model is not fitted yet. Call fit method first.")

        if X_test == None:
            X_test = self.X_test
        return self.model.predict(X_test)


    def score(self, y_true, y_pred):
        """
        Takes as input the true values of y and the predicted values of y and returns the accuracy score
        """
        
        return accuracy_score(y_true, y_pred)


    
    def best_fit(self, max_neighbors=75, weights=['uniform', 'distance']):
        """
        Finds the best model based on accuracy score.

        Parameters:
        max_neighbors: int, optional (default=100)
            Maximum number of neighbors to consider.
        weights: list of str, optional (default=['uniform', 'distance'])
            Weight function used in prediction.

        Returns:
        sklearn.neighbors.KNeighborsClassifier
            Best trained model.
        int
            Number of neighbors for the best model.
        float
            Accuracy score of the best model.
        """
        best_score = 0
        best_k = 0
        best_model = None
        for weight in weights:
            for k in range(1, max_neighbors + 1):
                self.fit(k, weight)
                knn_predictions_test = self.predict()
                score = accuracy_score(self.y_test, knn_predictions_test)
                self.k_scores_test.append(score)
                if score > best_score:
                    best_score = score
                    self.accuracy = best_score
                    self.params = [weight, k]
                    best_model = self.model

        return self.model

    def parameter_plot(self):
        """
        Plots accuracy scores for different values of k.

        Parameters:
        None

        Returns:
        None
        """
        k_list = list(range(1, len(self.k_scores_test) // 2 + 1))
        plt.plot(k_list, self.k_scores_test[:len(k_list)], label='Uniform Weight')
        plt.plot(k_list, self.k_scores_test[len(k_list):], label='Distance Weight')
        plt.xlabel("k")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()

    def print_best_params(self):
        print(f"Validation accuracy for KNN: {self.accuracy:.3f}")
        print(f"Parameters of KNN: {self.params}")

if __name__ == '__main__':
    from .. import data_processing
    from data_processing import DataProcessor

    dataset_df = pd.read_csv("../../data/train.csv")
    dataset_df  = DataProcessor(dataset_df)

    X_train, X_test, y_train, y_test = dataset_df.run_and_split() # Run the standard list of operations
    
    knn_classifier = KNNClassifier(X_train, X_test, y_train, y_test)
    knn_classifier.best_fit()
    knn_classifier.print_best_params()
    
    knn_classifier.parameter_plot()
    
