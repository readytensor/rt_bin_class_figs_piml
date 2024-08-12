import os
import warnings
from typing import Optional, List

import joblib
import numpy as np
import pandas as pd
from piml.models import FIGSClassifier
from sklearn.exceptions import NotFittedError

warnings.filterwarnings("ignore")

PREDICTOR_FILE_NAME = "predictor.joblib"


class Classifier:
    """A wrapper class for the Generalized Additive Model Classifier in Pygam."""

    model_name = "Generalized Additive Model Classifier in Pygam"

    def __init__(
        self,
        feature_names: List[str] = None,
        feature_types: List[str] = None,
        max_depth: int = 5,
        min_samples_leaf: int = 1,
        learning_rate: float = 1.0,
        max_iter: int = 20,
        **kwargs,
    ):
        """
        Initializes the model with specified configurations.

        Parameters:
            feature_names (Optional[List[str]]): The list of feature names. 
                                                 Default is None.
            feature_types (Optional[List[str]]): The list of feature types. 
                                                 Available types include “numerical” 
                                                 and “categorical”. Default is None.
            max_depth (int): The max tree depth, which means no constraint on max_depth.
            min_samples_leaf (int): The minimum number of samples required to be at a 
                                    leaf node. A split point at any depth will only be
                                    considered if it leaves at least min_samples_leaf
                                    training samples in each of the left and right
                                    branches. This may have the effect of smoothing the
                                    model, especially in regression.
            learning_rate (float): The learning rate of each tree.
            max_iter (int): The max number of iterations, each iteration is a splitting step.
            **kwargs: Additional keyword arguments for model configuration.

        Returns:
            None.
        """
        self.feature_names = feature_names
        self.feature_types = feature_types
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.learning_rate = float(learning_rate)
        self.max_iter = int(max_iter)
        self.kwargs = kwargs
        self.model = self.build_model()
        self._is_trained = False

    def build_model(self) -> FIGSClassifier:
        """Build a new binary classifier."""
        model = FIGSClassifier(
            feature_names=self.feature_names,
            feature_types=self.feature_types,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
        )
        return model

    def fit(self, train_inputs: pd.DataFrame, train_targets: pd.Series) -> None:
        """Fit the binary classifier to the training data.

        Args:
            train_inputs (pandas.DataFrame): The features of the training data.
            train_targets (pandas.Series): The labels of the training data.
        """
        self.model.fit(train_inputs, train_targets)
        self._is_trained = True

    def predict(self, inputs: pd.DataFrame) -> np.ndarray:
        """Predict class labels for the given data.

        Args:
            inputs (pandas.DataFrame): The input data.
        Returns:
            numpy.ndarray: The predicted class labels.
        """
        return self.model.predict(inputs)

    def predict_proba(self, inputs: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities for the given data.

        Args:
            inputs (pandas.DataFrame): The input data.
        Returns:
            numpy.ndarray: The predicted class probabilities.
        """
        return self.model.predict_proba(inputs)

    def evaluate(self, test_inputs: pd.DataFrame, test_targets: pd.Series) -> float:
        """Evaluate the binary classifier and return the accuracy.

        Args:
            test_inputs (pandas.DataFrame): The features of the test data.
            test_targets (pandas.Series): The labels of the test data.
        Returns:
            float: The accuracy of the binary classifier.
        """
        if self.model is not None:
            return self.model.score(test_inputs, test_targets)
        raise NotFittedError("Model is not fitted yet.")

    def save(self, model_dir_path: str) -> None:
        """Save the binary classifier to disk.

        Args:
            model_dir_path (str): Dir path to which to save the model.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")
        joblib.dump(self, os.path.join(model_dir_path, PREDICTOR_FILE_NAME))

    @classmethod
    def load(cls, model_dir_path: str) -> "Classifier":
        """Load the binary classifier from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            Classifier: A new instance of the loaded binary classifier.
        """
        model = joblib.load(os.path.join(model_dir_path, PREDICTOR_FILE_NAME))
        return model

    def __str__(self):
        return (
            f"Model name: {self.model_name} ("
            f"feature_types: {self.feature_types}, "
            f"max_depth: {self.max_depth}, "
            f"min_samples_leaf: {self.min_samples_leaf}, "
            f"learning_rate: {self.learning_rate}, "
            f"max_iter: {self.max_iter})"
        )


def train_predictor_model(
    train_inputs: pd.DataFrame, train_targets: pd.Series, hyperparameters: dict
) -> Classifier:
    """
    Instantiate and train the predictor model.

    Args:
        train_X (pd.DataFrame): The training data inputs.
        train_y (pd.Series): The training data labels.
        hyperparameters (dict): Hyperparameters for the classifier.

    Returns:
        'Classifier': The classifier model
    """
    classifier = Classifier(**hyperparameters)
    classifier.fit(train_inputs=train_inputs, train_targets=train_targets)
    return classifier


def predict_with_model(
    classifier: Classifier, data: pd.DataFrame, return_probs=False
) -> np.ndarray:
    """
    Predict class probabilities for the given data.

    Args:
        classifier (Classifier): The classifier model.
        data (pd.DataFrame): The input data.
        return_probs (bool): Whether to return class probabilities or labels.
            Defaults to True.

    Returns:
        np.ndarray: The predicted classes or class probabilities.
    """
    if return_probs:
        return classifier.predict_proba(data)
    return classifier.predict(data)


def save_predictor_model(model: Classifier, predictor_dir_path: str) -> None:
    """
    Save the classifier model to disk.

    Args:
        model (Classifier): The classifier model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> Classifier:
    """
    Load the classifier model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        Classifier: A new instance of the loaded classifier model.
    """
    return Classifier.load(predictor_dir_path)


def evaluate_predictor_model(
    model: Classifier, x_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    Evaluate the classifier model and return the accuracy.

    Args:
        model (Classifier): The classifier model.
        x_test (pd.DataFrame): The features of the test data.
        y_test (pd.Series): The labels of the test data.

    Returns:
        float: The accuracy of the classifier model.
    """
    return model.evaluate(x_test, y_test)
