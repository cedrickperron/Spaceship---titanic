import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class LogisticRegressionClassifier:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.model = None
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.accuracy = None
        self.params = None
    
    def fit(self, C, solver):
        """
        Fits the model with the training dataset.

        Parameters:
        C: float
            Inverse of regularization strength; smaller values specify stronger regularization.
        solver: str
            Algorithm to use in the optimization problem.

        Returns:
        None
        """
        self.model = LogisticRegression(C=C, random_state=1, max_iter=150, solver=solver)
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
    def best_fit(self, hyper_params=np.arange(0.25, 5, 0.25), solver=["newton-cg", "lbfgs", "liblinear", 'sag', "saga"]):
        """
        Finds the best model based on accuracy score.

        Parameters:
        hyper_params: array-like, optional (default=None)
            Values for regularization strength.
        solver: list of str, optional (default=None)
            Algorithms to use in the optimization problem.

        Returns:
        sklearn.linear_model.LogisticRegression
            Best trained model.
        float
            Regularization strength for the best model.
        float
            Accuracy score of the best model.
        """
        max_accuracy = 0.0
        max_acc_params = None
        best_model =  None
        for s in solver:
            for C in hyper_params:
                self.fit(C, s)
                LR_Model_predictions_test = self.predict()
                LR_Model_accuracy_test = self.score(self.y_test, LR_Model_predictions_test)
                if LR_Model_accuracy_test > max_accuracy:
                    max_accuracy = LR_Model_accuracy_test
                    max_acc_params = (s, C)
                    best_model = self.model
        self.accuracy = max_accuracy
        self.params = max_acc_params
        self.model = best_model
        return self.model

    def parameter_plot(self):
        """
        Plots accuracy scores for different values of regularization strength.

        Parameters:
        None

        Returns:
        None
        """
        pass
    
    def print_best_params(self):
        print(f"Validation accuracy for Logistic Regression: {self.accuracy:.3f}")
        print(f"Parameters of Logistic Regression: {self.params}")

if __name__ == '__main__':
    from .. import data_processing
    from data_processing import DataProcessor

    dataset_df = pd.read_csv("./data/train.csv")
    dataset_df  = DataProcessor(dataset_df)

    X_train, X_test, y_train, y_test = dataset_df.run_and_split() # Run the standard list of operations

    
    lr_classifier = LogisticRegressionClassifier(X_train, X_test, y_train, y_test)
    lr_classifier.best_fit()

    lr_classifier.print_best_params()
