# Polynomial Regression
import os
os.chdir('Part-2-Regression/Section-6-Polynomial-Regression')
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values

# Fitting a Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
linearRegressor = LinearRegression()
linearRegressor.fit(X, y)

# Fitting a Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
polynomialRegressor = PolynomialFeatures(degree=4)
X_poly = polynomialRegressor.fit_transform(X, y)
linearRegressor2 = LinearRegression()
linearRegressor2.fit(X_poly, y)

# Visualizing the Linear Regression results
plt.scatter(X, y, color='red')
plt.plot(X, linearRegressor.predict(X), color='blue')
plt.title('Linear Regression Salary model')
plt.show()

# Visualizing the Polynomial Regression results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid)), 1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, linearRegressor2.predict(polynomialRegressor.fit_transform(X_grid)), color='blue')
plt.title('Polynomial Regression Salary model')
plt.show()

# Predicting a new result with Linear Regression
linearRegressor.predict(6.5)

# Predicting a new result with Polynomial Regression
linearRegressor2.predict(polynomialRegressor.fit_transform(6.5))