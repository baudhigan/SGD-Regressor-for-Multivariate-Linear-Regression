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
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

np.random.seed(42)
n_samples = 200
area = np.random.randint(800, 4000, n_samples)
bedrooms = np.random.randint(1, 6, n_samples)
bathrooms = np.random.randint(1, 4, n_samples)

price = area * 120 + bedrooms * 50000 + bathrooms * 30000 + np.random.randint(-20000, 20000, n_samples)

occupants = bedrooms + np.random.randint(0, 3, n_samples)

data = pd.DataFrame({
    "Area": area,
    "Bedrooms": bedrooms,
    "Bathrooms": bathrooms,
    "Price": price,
    "Occupants": occupants
})

X = data[["Area", "Bedrooms", "Bathrooms"]]
y = data[["Price", "Occupants"]]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

sgd = MultiOutputRegressor(SGDRegressor(max_iter=1000, tol=1e-3, random_state=42))
sgd.fit(X_train_scaled, y_train)

y_pred = sgd.predict(X_test_scaled)

print("Price Prediction Metrics:")
print("MAE:", mean_absolute_error(y_test["Price"], y_pred[:, 0]))
print("RMSE:", np.sqrt(mean_squared_error(y_test["Price"], y_pred[:, 0])))
print("R²:", r2_score(y_test["Price"], y_pred[:, 0]))

print("\nOccupants Prediction Metrics:")
print("MAE:", mean_absolute_error(y_test["Occupants"], y_pred[:, 1]))
print("RMSE:", np.sqrt(mean_squared_error(y_test["Occupants"], y_pred[:, 1])))
print("R²:", r2_score(y_test["Occupants"], y_pred[:, 1]))

new_house = pd.DataFrame([[2500, 4, 3]], columns=["Area", "Bedrooms", "Bathrooms"])
new_house_scaled = scaler.transform(new_house)

predicted = sgd.predict(new_house_scaled)[0]

print("\nNew house prediction:")
print("Predicted Price:", int(predicted[0]))
print("Predicted Occupants:", round(predicted[1], 1))

```
## Output:
### Price Prediction
<img width="263" height="89" alt="image" src="https://github.com/user-attachments/assets/10cc5c2b-707a-4f1e-9855-52343c5e8428" />

### Occupants Prediction
<img width="262" height="95" alt="image" src="https://github.com/user-attachments/assets/1d78ebf4-abea-40f2-8048-b3b9e0046378" />

### New house prediction
<img width="317" height="68" alt="image" src="https://github.com/user-attachments/assets/0b50df09-7a8e-4cf1-a18a-b433ed42cae1" />

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
