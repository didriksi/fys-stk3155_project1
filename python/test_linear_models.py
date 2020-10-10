import numpy as np
import pytest
from sklearn.linear_model import Ridge

import linear_models

@pytest.mark.parametrize(
    "_lambda, X, y",
    [
        (1e-2, np.array([[1, 3.5, 6], [1, .5, 2], [1, 8, 6], [1, 4, 3]]), np.array([1, 2.8, 9, 5])),
        (1e-10, np.array([[1, 8, 9], [1, 0, 1.9], [1, 3.5, 6], [1, 1, 1]]), np.array([8, .8, 0, 1])),
        (1e+10, np.array([[1, 8, 9], [1, 0, 1.9], [1, 3.5, 6], [1, 1, 1]]), np.array([8, .8, 0, 1]))
    ],
)
def test_beta_ridge(_lambda, X, y):
    """Test that linear_models.beta_ridge gives similar results to sklearn.beta_ridge. 
    """
    computed = linear_models.beta_ridge(_lambda, X, y)

    ridge = Ridge(alpha=_lambda, fit_intercept=False)
    ridge.fit(X, y)
    expected = ridge.coef_

    assert np.all(np.isclose(expected, computed)), "Beta ridge function converged at different solution than Scikit-Learn"