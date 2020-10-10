import numpy as np
import matplotlib.pyplot as plt

import franke
import data_handling
import tune_and_evaluate
import linear_models
import metrics
import resampling
import plotting

np.random.seed(10)

# Make small dataset of only 400 points, where 100 are training, 100 validation, and 200 testing
x1, x2 = np.meshgrid(np.linspace(-0.9, 1.1, 15), np.linspace(-0.9, 1.1, 15))
y = franke.franke_function(x1, x2) + np.random.normal(0, 0.2, size=x1.shape)
data = data_handling.make_data_dict(y, 2)

P = 6
poly_iter = range(1, P)

errors = {'training': np.empty((P-1,2)), 'testing': np.empty((P-1,2))}

for p in poly_iter:
    X_train = tune_and_evaluate.poly_design_matrix(p, data['x_train'])    
    X_test = tune_and_evaluate.poly_design_matrix(p, data['x_test'])    

    model = linear_models.LinearRegression("OLS")
    model.scaler.fit(X_train[:,1:])
    X_train[:,1:] = model.scaler.transform(X_train[:,1:])
    X_test[:,1:] = model.scaler.transform(X_test[:,1:])

    model.fit(X_train, data['y_train'])
    y_tilde = model.predict(X_train)
    y_pred = model.predict(X_test)

    conf_intervals, conf_deviation = model.conf_interval_beta(data['y_train'], y_tilde, X_train)

    errors['training'][p-1] = np.array([metrics.MSE(data['y_train'], y_tilde), metrics.R_2(data['y_train'], y_tilde)])
    errors['testing'][p-1] = np.array([metrics.MSE(data['y_test'], y_pred), metrics.R_2(data['y_test'], y_pred)])

# Training vs testing, demonstrating overfitting
fig = plt.figure(figsize=(7,7))
plt.plot(poly_iter, errors['training'][:,0], label="Training")
plt.plot(poly_iter, errors['testing'][:,0], label="Testing")
plt.legend()
plt.xlabel("Complexity (maximum degree)")
plt.ylabel("Mean squared error")
plt.savefig("../plots/overfitting_franke.png")
plt.close()

# Confidence interval of Beta
fig = plt.figure(figsize=(7,7))
plt.errorbar(model.beta, np.arange(model.beta.size, 0, -1), xerr=conf_deviation, ecolor='salmon', capsize=1.5, elinewidth=0.5)
plt.yticks([])
plt.xlabel("Values of parameters, with 99% confidence intervals")
plt.ylabel("Parameters of $\\beta$")
plt.savefig("../plots/conf_interval_beta_franke.png")
plt.close()


# Bias, variance, MSE
P = 5
poly_iter = range(1, P)
resampled_errors = {'OLS boot bias': np.empty((P-1,2)), 'OLS boot variance': np.empty((P-1,2)), 'OLS boot MSE': np.empty((P-1,2)),
                    'Ridge boot bias': np.empty((P-1,2)), 'Ridge boot variance': np.empty((P-1,2)), 'Ridge boot MSE': np.empty((P-1,2)),
                    'LASSO boot bias': np.empty((P-1,2)), 'LASSO boot variance': np.empty((P-1,2)), 'LASSO boot MSE': np.empty((P-1,2)),
                    'OLS kfold MSE': np.empty((P-1,2)), 'Ridge kfold MSE': np.empty((P-1,2)), 'LASSO kfold MSE': np.empty((P-1,2)),}

for i, p in enumerate(poly_iter):
    X_train = tune_and_evaluate.poly_design_matrix(p, data['x_train'])    
    X_test = tune_and_evaluate.poly_design_matrix(p, data['x_test'])

    X = np.append(X_train, X_test, axis=0)
    y = np.append(data['y_train'], data['y_test'], axis=0)

    models = [linear_models.RegularisedLinearRegression("Ridge", linear_models.beta_ridge),
              linear_models.RegularisedLinearRegression("LASSO", linear_models.beta_lasso),
              linear_models.LinearRegression("OLS")]
    for model, _lambda in zip(models, [0.001, 0.001, 0]):
        model.set_lambda(_lambda)
        boot_mse, bias, var = resampling.bootstrap(model, X_train, X_test, data['y_train'], data['y_test'], R=100)
        kfold_mse = resampling.kfold(model, X, y)
        resampled_errors[f'{model.name} boot bias'][i] = bias
        resampled_errors[f'{model.name} boot variance'][i] = var
        resampled_errors[f'{model.name} boot MSE'][i] = boot_mse
        resampled_errors[f'{model.name} kfold MSE'][i] = kfold_mse

plotting.side_by_side(
    ['OLS', poly_iter, [[resampled_errors['OLS boot MSE'], 'MSE'], [resampled_errors['OLS boot bias'], 'Bias'], [resampled_errors['OLS boot variance'], 'Variance']]],
    ['Ridge ($\\lambda=0.001$)', poly_iter, [[resampled_errors['Ridge boot MSE'], 'MSE'], [resampled_errors['Ridge boot bias'], 'Bias'], [resampled_errors['Ridge boot variance'], 'Variance']]],
    ['LASSO ($\\lambda=0.001$)', poly_iter, [[resampled_errors['LASSO boot MSE'], 'MSE'], [resampled_errors['LASSO boot bias'], 'Bias'], [resampled_errors['LASSO boot variance'], 'Variance']]],
    axis_labels=['Complexity  (maximum degree)', 'Error'],
    filename="bias_variance_franke",
    title="Bias variance tradeoff for different models",
    _3d=False)

plotting.side_by_side(
    ['OLS', poly_iter, [[resampled_errors['OLS boot MSE'], 'Bootstrap'], [resampled_errors['OLS kfold MSE'], 'Kfold']]],
    ['Ridge ($\\lambda=0.001$)', poly_iter, [[resampled_errors['Ridge boot MSE'], 'Bootstrap'], [resampled_errors['Ridge kfold MSE'], 'Kfold']]],
    ['LASSO ($\\lambda=0.001$)', poly_iter, [[resampled_errors['LASSO boot MSE'], 'Bootstrap'], [resampled_errors['LASSO kfold MSE'], 'Kfold']]],
    axis_labels=['Complexity  (maximum degree)', 'Mean squared error'],
    filename="boot_kfold_franke",
    title="Bootstrap and kfold estimates of MSE",
    _3d=False)


