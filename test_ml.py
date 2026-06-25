import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ml.data import apply_label
from ml.model import train_model, compute_model_metrics


def test_compute_model_metrics():
    """
    Test that perfect predictions produce perfect metrics.
    """
    y = [0, 1, 0, 1]
    preds = [0, 1, 0, 1]

    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert precision == 1.0
    assert recall == 1.0
    assert fbeta == 1.0
    



def test_apply_label():
    """
    Test that apply_label converts predictions correctly.
    """
    assert apply_label([1]) == ">50K"
    assert apply_label([0]) == "<=50K"
    



def test_train_model():
    """
    Test that train_model returns a RandomForestClassifier.
    """
    X_train = np.array([
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
    ])

    y_train = np.array([0, 0, 1, 1])

    model = train_model(X_train, y_train)

    assert isinstance(model, RandomForestClassifier)
    
