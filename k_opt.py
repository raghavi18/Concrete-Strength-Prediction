# import necessary libraries
import numpy as np
import pandas as pd
  
# import the KNNimputer class
from sklearn.impute import KNNImputer


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

rmse = lambda y, yhat: np.sqrt(mean_squared_error(y, yhat))

dataset=pd.read_csv('project_data.csv')


def optimize_k(data, target):
    errors = []
    for k in range(1, 20, 2):
        imputer = KNNImputer(n_neighbors=k)
        imputed = imputer.fit_transform(data)
        dataset_imputed = pd.DataFrame(imputed, columns=dataset.columns)
        
        X = dataset_imputed.drop(target, axis=1)
        y = dataset_imputed[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        error = rmse(y_test, preds)
        errors.append({'K': k, 'RMSE': error})
        
    return errors


k_errors1 = optimize_k(data=dataset, target='Compressive strength of cement fce(MPa)')
k_errors2 = optimize_k(data=dataset, target='Tensile strength of cement fct(MPa)')
k_errors3 = optimize_k(data=dataset, target='Dmax of Crushed stone (mm)')
k_errors4 = optimize_k(data=dataset, target='Stone powder content in Sand (%)')

print (k_errors1)
print("\n\n")
print (k_errors2)
print("\n\n")
print (k_errors3)
print("\n\n")
print (k_errors4)
print("\n\n")
