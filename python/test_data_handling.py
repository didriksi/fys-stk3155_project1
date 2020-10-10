import numpy as np
import pytest

import data_handling

@pytest.mark.parametrize(
    "selection_parameter, random",
    [
        ((0.5, 0.2), True),
        ((2, ), False),
    ],
)
def test_make_data_dict(selection_parameter, random):
    """Test that data_handling.make_data_dict makes data of corrrect shape, and without disconnecting predictor and dependent variable.
    """
    n = 400
    y = np.random.rand(n,n)
    
    data = data_handling.make_data_dict(y, *selection_parameter, random=random)

    if random == False:
        x_sparsity = selection_parameter[0]
        assert data['x_train'].shape == ((n/x_sparsity)**2, 2), "Training set has wrong shape"
        assert data['x_validate'].shape == ((n/x_sparsity)**2, 2), "Validation set has wrong shape"
        assert data['x_test'].shape == (n**2 - 2*(n/x_sparsity)**2, 2), "Testing set has wrong shape"
        assert data['y_test'].shape == (n**2 - 2*(n/x_sparsity)**2, ), "y data has wrong shape"

    assert np.all(y[data['x_train'][:,0],data['x_train'][:,1]] == data['y_train']), "Data handling destroyed link between x and y data"