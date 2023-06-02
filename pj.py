# import necessary libraries
import numpy as np
import pandas as pd
  
# import the KNNimputer class
from sklearn.impute import KNNImputer

dataset=pd.read_csv('project_data.csv')
print (len(dataset))
print(dataset.head())
  

  
# creating a data frame from the list 
Before_imputation = pd.DataFrame(dataset)
#print dataset before imputaion
print("Data Before performing imputation\n",Before_imputation)
  
# create an object for KNNImputer
imputer = KNNImputer(n_neighbors=1)
After_imputation = imputer.fit_transform(Before_imputation)
# print dataset after performing the operation
print("\n\nAfter performing imputation\n",After_imputation)

dataset_imputed = pd.DataFrame(After_imputation, columns=dataset.columns)