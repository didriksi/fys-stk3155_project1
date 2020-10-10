import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def get_data():
    val_error_columns = pd.read_csv('../dataframe/real_validation_errors.csv', nrows = 1)
    val_errors = pd.read_csv('../dataframe/real_validation_errors.csv', names=val_error_columns.columns, index_col=[0,1,2], skiprows=1)

    boot_errors = val_errors.loc['Boot MSE'].values.reshape(-1)
    kfold_errors = val_errors.loc['Kfold MSE'].values.reshape(-1)
    test_errors = val_errors.loc['Test MSE'].values.reshape(-1)

    data = {}
    data['x'] = np.append(boot_errors[:,np.newaxis], kfold_errors[:,np.newaxis], axis=1)
    data['y'] = test_errors

    data['x_train_validate'], data['x_test'], data['y_train_validate'], data['y_test'] = train_test_split(data['x'], data['y'], test_size=0.2)
    data['x_train'], data['x_validate'], data['y_train'], data['y_validate'] = train_test_split(data['x_train_validate'], data['y_train_validate'], test_size=0.2/0.8)

    return data



