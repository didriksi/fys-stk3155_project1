import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import linear_models
import franke
import real_terrain
import resampling
import metrics
import plotting

def poly_design_matrix(p, x):
    """Make a design matrix where each column is a combination of the input x's data columns to powers up to p.

    Parameters:
    -----------
    p:          int
                Maximum polynomial degree of columns.
    x:          array of shape (n, f)
                Predictor variable, with each row being one datapoint.

    Returns:
    --------
    X:          2-dimensional array
                Design matrix with rows as data points and columns as features.
    """
    powers = np.arange(p+1)[np.newaxis,:].repeat(x.shape[1], axis=0)
    # Make a (p+1)**x.shape[0] x 2 matrix with pairs of power combinations
    pow_combs = np.array(np.meshgrid(*powers)).T.reshape(-1, x.shape[1])
    # Remove all combinations with combined powers of over p
    pow_combs = pow_combs[np.sum(pow_combs, axis=1) <= p]
    # Make third order tensor where last dimension is used for each input dimension to its appropriate power
    X_powers = x[:,:,np.newaxis].astype(float).repeat(p+1, axis=2)**powers
    # Multiply the powers of X together according to pow_combs
    X = X_powers[:,0,pow_combs[:,0]] * X_powers[:,1,pow_combs[:,1]]
    return X

class Tune:
    """Validates, finds optimal hyperparameters, and tests them.

    Parameters:
    -----------
    models:     list[object]
                Instances of classes with a scaler, a fit method and a predict method.
    data:       dict
                All training and testing data, as constructed by data_handling.make_data_dict,
    poly_iter:  iterable
                All maximum polynomials to validate for.
    para_iters: list[iterable]
                As many sets of hyperparameters as there are models.
    verbose:    int
                How much to print out during validation, with 0 being nothing, and 2 being evrything.
    """
    def __init__(self, models, data, poly_iter, para_iters, verbose=1):
        self.models = models
        self.poly_iter = poly_iter
        self.para_iters = para_iters
        self.verbose = verbose
        self.data = data

    def validate(self):
        """Validates all models for all polynomials and all parameters, and stores data in validation_errors.

        Creates and populates pandas.DataFrame validation_errors with MSE from bootstrap and kfold resampling techniques,
        as well as model bias and variance from bootstrap, for all combinations of hyperparameters.
        """
        # Make dataframe for validation data
        lambda_list = np.concatenate(self.para_iters) if len(self.para_iters) > 1 else self.para_iters[0]
        validation_index = pd.MultiIndex.from_product([
            ['Boot MSE', 'Boot Bias', 'Boot Var', 'Kfold MSE'],
            [model.name for model in self.models],
            [_lambda for _lambda in lambda_list]],
            names = ['Metric', 'Model', 'Lambda']
            )
        self.validation_errors = pd.DataFrame(dtype=float, index=validation_index, columns=self.poly_iter)
        self.validation_errors.sort_index(inplace=True)

        for i, p in enumerate(self.poly_iter):
            if self.verbose >= 1:
                print(f"{(i+1):2d}/{len(poly_iter)}: Polynomial of degree {p}")

            X_train_validate = poly_design_matrix(p, self.data['x_train_validate'])
            X_train, X_validate = np.split(X_train_validate, [self.data['x_train'].shape[0]])

            for j, (model, para_iter) in enumerate(zip(self.models, self.para_iters)):
                if self.verbose >= 1:
                    print(f"    {(j+1):2d}/{len(self.models)}: Model: {model.name}")

                for k, _lambda in enumerate(para_iter):
                    if self.verbose >= 2:
                        print(f"        {(k+1):2d}/{len(para_iter)}: Lambda = {_lambda}")

                    model.set_lambda(_lambda)
                    boot_mse, boot_bias, boot_var = resampling.bootstrap(model, X_train, X_validate, self.data['y_train'], self.data['y_validate'], R=50)
                    kfold_mse = resampling.kfold(model, X_train_validate, self.data['y_train_validate'], n_folds=10)

                    self.validation_errors.loc['Boot MSE', model.name, _lambda][p] = boot_mse
                    self.validation_errors.loc['Boot Bias', model.name, _lambda][p] = boot_bias
                    self.validation_errors.loc['Boot Var', model.name, _lambda][p] = boot_var
                    self.validation_errors.loc['Kfold MSE', model.name, _lambda][p] = kfold_mse

                    #if self.verbose >= 2:
                        #print(f"         error: {(boot_mse+kfold_mse)/2}")

        self.validation_errors.dropna(inplace=True)
        self.validation_errors.drop_duplicates(inplace=True)

    def optimal_model_search(self):
        """Uses validation_errors to find the best models.

        Assumes pandas.DataFrame validation_errors has alreadybeen created by validate method, and makes two new dataframes.
        optimal_per_poly stores the optimal parameter for each polynomial, and optimal_model stores the absolute best combination
        of polynnomial and lambda for each model.
        """
        # Make dataframe for optimal hyperparameters for each polynomial degree
        optimal_per_poly_index = pd.MultiIndex.from_product([
            [model.name for model in self.models],
            ['Parameter', 'Error']],
            names = ['Model', None]
            )
        self.optimal_per_poly = pd.DataFrame(dtype=float, index=optimal_per_poly_index, columns=self.poly_iter)

        # Parameters for optimal model

        optimal_model_index = pd.MultiIndex.from_product([
            [model.name for model in self.models],
            ['Bootstrap', 'Kfold', 'Combined']],
            names= ['Model', 'Resampling technique']
            )
        self.optimal_model = pd.DataFrame(index=optimal_model_index, columns=['Lambda', 'Polynomial', 'Test error'])

        average_MSE = (self.validation_errors.loc['Boot MSE'] + self.validation_errors.loc['Kfold MSE'])/2
        
        for model in self.models:
            self.optimal_model.loc[model.name, 'Bootstrap']['Lambda', 'Polynomial'] = self.validation_errors.loc['Boot MSE', model.name].stack().idxmin()
            self.optimal_model.loc[model.name, 'Kfold']['Lambda', 'Polynomial'] = self.validation_errors.loc['Kfold MSE', model.name].stack().idxmin()
            self.optimal_model.loc[model.name, 'Combined']['Lambda', 'Polynomial'] = average_MSE.loc[model.name].stack().idxmin()
            
            # Best parameter for each polynomial
            self.optimal_per_poly.loc[model.name, 'Parameter'] = average_MSE.loc[model.name].idxmin()
            # The respective error
            self.optimal_per_poly.loc[model.name, 'Error'] = average_MSE.loc[model.name].min()

    def test(self):
        """Tests out the models that validate and optimal_model_search has determined are best.

        Uses the pandas.DataFrame optimal_model created by optimal_model_search method, and tests each of the models
        on testing data. Plots the result side by side with the actual dependent variable.

        """
        plotting.side_by_side(
                self.data['x_train'],
                self.data['y_train'],
                self.data['y_test'],
                x2=self.data['x_test'],
                y1_title='Train',
                y2_title='Test',
                title=f"Training and testing data",
                filename=f"train_and_test")
        for (model_name, technique), (_lambda, poly, _) in self.optimal_model.iterrows():

            for _model in self.models:
                if _model.name == model_name:
                    model = _model

            X = poly_design_matrix(int(poly), self.data['x_train'])

            model.scaler.fit(X[:,1:])
            X[:,1:] = model.scaler.transform(X[:,1:])
            model.set_lambda(_lambda)
            model.fit(X, self.data['y_train'])

            X_test = poly_design_matrix(int(poly), self.data['x_test'])
            X_test[:,1:] = model.scaler.transform(X_test[:,1:])
            y_pred = model.predict(X_test)

            mse = metrics.MSE(data['y_test'], y_pred)
            self.optimal_model.loc[model_name, technique]['Test error'] = mse
            print(f"MSE: {mse} for poly = {poly} and lambda = {_lambda} with model {model_name}.")
            
            plotting.side_by_side(
                self.data['x_test'],
                y_pred,
                self.data['y_test'],
                title=f"Test: {model.name}, $p$={poly}, $\\lambda$={_lambda}, best according to {technique}",
                filename=f"test_{model.name}_p{poly}{[f'_lambda{_lambda}', ''][_lambda==0]}")
            plotting.side_by_side(
                self.data['x_test'],
                y_pred,
                self.data['y_test'],
                title=f"Test: {model.name}, $p$={poly}, $\\lambda$={_lambda}",
                filename=f"test_{model.name}_p{poly}{[f'_lambda{_lambda}', ''][_lambda==0]}",
                animation=True)

    def save(self):
        """Saves dataframes created by .validation and .optimal_model_search to csv files.
        """
        if 'validation_errors' in self.__dict__:
            self.validation_errors.to_csv('../dataframe/validation_errors.csv')
        if 'optimal_per_poly' in self.__dict__:
            self.optimal_per_poly.to_csv('../dataframe/optimal_per_poly.csv')
        if 'optimal_model' in self.__dict__:
            self.optimal_model.to_csv('../dataframe/optimal_model.csv')

if __name__ == '__main__':
    models = [linear_models.RegularisedLinearRegression("Ridge", linear_models.beta_ridge),
              linear_models.RegularisedLinearRegression("Lasso", linear_models.beta_lasso),
              linear_models.LinearRegression("OLS")]
    if sys.argv[1] == 'franke':
        poly_iter = range(1, 35, 1)
        N = 30
        para_iters = [np.logspace(-4, 8, N)]
        data = franke.get_data(x_sparsity=2)
    elif sys.argv[1] == 'real':
        poly_iter = range(3, 16, 1)
        N = 10
        para_iters = [np.logspace(-8, 1, N), np.logspace(-3/2, 1, N), np.array([0.])]
        data = real_terrain.get_data(20) #(0.5, 0.2, random=True)

    tune = Tune(models, data, poly_iter, para_iters, verbose=2)
    tune.validate()
    print(tune.validation_errors)
    tune.optimal_model_search()
    tune.test()
    tune.save()





