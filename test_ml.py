import pytest
from ml.data import apply_label
from ml.model import compute_model_metrics


def test_compute_model_metrics():
    """
    Test that perfect redictions produce perfect metrics.
    """
    y = [0, 1, 0, 1]
    preds = [0, 1, 0, 1]

    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert precision == 1.0
    assert recall == 1.0
    assert fbeta == 1.0
    pass



def test_apply_label_positive():
    """
    Test conversion of positive prediction label.
    """
    assert apply_label([1]) == ">50K"
    pass



def test_apply_label_negative():
    """
    Test conversion of negative prediction label.
    """
    assert apply_label([0]) == "<=50K"
    pass
