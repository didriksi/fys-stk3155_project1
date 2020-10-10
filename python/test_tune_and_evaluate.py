import pytest
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

import tune_and_evaluate

@pytest.mark.parametrize(
    "p, x, expected",
    [
        (1, np.array([[1, 1], [2, 1], [1, 2]], dtype=float), np.array([[1, 1, 1], [1, 1, 2], [1, 2, 1]], dtype=float)),
        (2, np.array([[2, 3], [1, 1]], dtype=float), np.array([[1, 3, 9, 2, 6, 4], [1, 1, 1, 1, 1, 1]], dtype=float)),
        (11, np.array([[2, 3], [1, 1]], dtype=float), np.array([[1, 3, 9, 2, 6, 4], [1, 1, 1, 9, 7, 1]], dtype=float))
    ],
)
def test_poly_design_matrix(p, x, expected):
    """Test that tune_and_evaluate.poly_design_matrix makes design matrixes with correct columns.
    """
    computed = tune_and_evaluate.poly_design_matrix(p, x)
    sklearn_poly = PolynomialFeatures(p)
    expected = np.array(sklearn_poly.fit_transform(x))
    expected = np.sort(expected.reshape(-1))
    computed = np.sort(computed.reshape(-1))
    assert np.all(np.isclose(computed, expected, atol=1e-1)), f"computed {computed} != expected {expected} for {p} and {x}"