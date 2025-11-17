# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Data Preparation: Generate or load house data with features (Area, Bedrooms, Bathrooms) and targets (Price, Occupants) and split into training and test sets.

2. Feature Scaling: Standardize features using StandardScaler to normalize ranges for better SGD convergence.

3. Model Training: Train a MultiOutputRegressor with SGDRegressor to predict both Price and Occupants simultaneously.

4. Prediction & Evaluation: Predict on test/new data and evaluate using MAE, RMSE, and R² for both targets.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and
number of occupants in the house with SGD regressor.
Developed by: BAUDHIGAN D
Register Number:  212223230028
*/
```
```py
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
data=fetch_california_housing()
X=data.data[:,:3]
Y=np.column_stack((data.target,data.data[:,6]))
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
scaler_X=StandardScaler()
scaler_Y=StandardScaler()
X_train =scaler_X.fit_transform(X_train)
X_test=scaler_X.transform(X_test)
Y_train=scaler_Y.fit_transform(Y_train)
Y_test=scaler_Y.transform(Y_test)
sgd=SGDRegressor(max_iter=1000, tol=1e-3)
multi_output_sgd=MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train,Y_train)
Y_pred=multi_output_sgd.predict(X_test)
Y_pred=scaler_Y.inverse_transform(Y_pred)
Y_test=scaler_Y.inverse_transform(Y_test)
mse=mean_squared_error(Y_test,Y_pred)
print("Mean Square Error:",mse)
print("\nPredictions:\n",Y_pred[:5])

```
## Output:
<img width="402" height="179" alt="image" src="https://github.com/user-attachments/assets/9e57eda1-3a2f-4f97-ab33-0862791f28b4" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
