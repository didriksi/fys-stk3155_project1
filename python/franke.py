import numpy as np

import data_handling
import plotting

def franke_function(x1, x2):
    """Takes in two dimensional mesh, returns terrain-like height for the points these create.

    Parameters:
    -----------
    x1:         array of shape (n, n)
                Array typically derived from numpy.meshgrid, with values to evaluate the Franke function for.
    x2:         array of shape (n, n)
                Array typically derived from numpy.meshgrid, with values to evaluate the Franke function for.
    """
    term1 = 0.75*np.exp(-(0.25*(9*x1-2)**2) - 0.25*((9*x2-2)**2))
    term2 = 0.75*np.exp(-((9*x1+1)**2)/49.0 - 0.1*(9*x2+1))
    term3 = 0.5*np.exp(-(9*x1-7)**2/4.0 - 0.25*((9*x2-3)**2))
    term4 = -0.2*np.exp(-(9*x1-4)**2 - (9*x2-7)**2)
    return term1 + term2 + term3 + term4

def get_data(x_sparsity=20, noise_std=0.2):
    """Returns data dict with training, validation and testing data based on grid-size for training data and noise standard deviation.

    Parameters:
    -----------
    x_sparsity: int
                Grid size for selection of training and validation data. Must be at least 2.
    noise_std:  float
                Standard deviation of noise added on top of Franke function to make the dependent variable.
    
    Returns:
    --------
    data:       dict
                Big dictionary with training, testing and validation data.
    """
    x1, x2 = np.meshgrid(np.arange(0, 1, 0.0025), np.arange(0, 1, 0.0025))
    y = franke_function(x1, x2) + np.random.normal(0, noise_std, size=x1.shape)
    x = np.array([x1, x2]).T.reshape(-1, 2)
    data = data_handling.make_data_dict(y, x_sparsity)
    return data

def plot_data(**data_args):
    """Plots Franke function data, side by side with Franke function + noise data.

    Parameters:
    -----------
    *data_args:  
                Passed on to get_data()

    """
    data = get_data(**data_args)
    data_args['noise_std'] = 0
    data_wo_noise = get_data(**data_args)

    plotting.side_by_side(
        ["$f(x_1, x_2)$", data['x'], data_wo_noise['y'].reshape(-1)],
        [f"$f(x_1, x_2) + \\varepsilon$, where $\\varepsilon \\sim N(0,{0.2})$", data['x'], data['y'].reshape(-1)],
        title="Franke function ($f(x_1, x_2)$)",
        filename="franke")

if __name__ == '__main__':
    plot_data()