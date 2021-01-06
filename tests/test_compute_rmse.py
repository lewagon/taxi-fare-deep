from taxifare_deep.utils import compute_rmse
import numpy as np


def test_rmse_is_a_float():
    y_pred = np.array([0, 0, 0])
    y_true = np.array([1, 1, 1])
    assert compute_rmse(y_pred, y_true) == 1.0
