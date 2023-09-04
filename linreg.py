# lin reg creates a line of best fit, predicting a continuous y-value (dep.var.) given an x-value (indep.var.)

import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv(r"Fun\Python\new\datasci\datasets\train_LR.csv")
df2 = pd.read_csv(r"Fun\Python\new\datasci\datasets\test_LR.csv")
df = df.dropna()
df2 = df2.dropna()

X_train = df['x'].to_numpy()
X_test = df2['x'].to_numpy()
y_train = df['y'].to_numpy()
y_test = df2['y'].to_numpy()
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)

linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_test)
# print(linreg.score(X_test, y_test))
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.8f}")
print(f"R-squared (R²): {r2:.8f}")
# print("Coefficients:", linreg.coef_)
# print("Intercept:", linreg.intercept_)

plt.scatter(X_train, y_train, color='blue', label='Training Data', s=5)
plt.scatter(X_test, y_test, color='orange', label='Testing Data', s=5)
plt.plot(X_test, y_pred, color='red', label='Trend Line')
plt.xlabel("Independent Variable (Feature)")
plt.ylabel("Dependent Variable (Target)")
plt.legend()
plt.show()

df3 = pd.read_csv(r"Fun\Python\new\datasci\datasets\kc_house_data.csv")
df3 = df3.dropna()
X = df3.drop(['id', 'date', 'price'], axis=1).to_numpy()
y = df3['price'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
linreg2 = LinearRegression()
linreg2.fit(X_train, y_train)
y_pred = linreg2.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.8f}")
print(f"R-squared (R²): {r2:.8f}")
# print("Coefficients:", linreg2.coef_)
# print("Intercept:", linreg2.intercept_)

# https://www.kaggle.com/code/hellbuoy/carprice-prediction-mlr-rfe-vif