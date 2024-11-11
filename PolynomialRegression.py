'''
This program fits a polynomial regression model to a synthetic dataset with a non-linear relationship.
It transforms the input features into polynomial terms, trains the model,
and evaluates its performance using Mean Squared Error (MSE) and R-squared (R2) score.
Finally, it visualizes the data points and the fitted polynomial curve,
showing how well the model captures the non-linear trend.
'''
# pip install scikit-learn pandas numpy

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Generate synthetic data
np.random.seed(0)
X = 6 * np.random.rand(100, 1) - 3  # Independent variable (random values between -3 and 3)
y = 0.5 * X**3 - X**2 + X + 2 + np.random.randn(100, 1) * 2  # Non-linear relationship with some noise

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the degree of the polynomial
degree = 3
poly_features = PolynomialFeatures(degree=degree)
X_poly_train = poly_features.fit_transform(X_train)
X_poly_test = poly_features.transform(X_test)

# Train the polynomial regression model
model = LinearRegression()
model.fit(X_poly_train, y_train)

# Predict on the test set
y_pred = model.predict(X_poly_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Polynomial Degree: {degree}")
print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)
print("Mean Squared Error (MSE):", mse)
print("R-squared (R2) Score:", r2)

# Plotting the results
plt.scatter(X, y, color='blue', label='Data Points')
X_range = np.linspace(X.min(), X.max(), 100).reshape(100, 1)
X_range_poly = poly_features.transform(X_range)
y_range_pred = model.predict(X_range_poly)
plt.plot(X_range, y_range_pred, color='red', label=f'Polynomial Regression (degree={degree})')
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
