'''
Steps:
  1. Import Libraries
  2. Generate or Load Data
  3. Split the Data
  4. Create and Train the Model
  5. Make Predictions
  6. Evaluate the Model
  7. Visualize the Results
  8. Interpret Model Parameters
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate synthetic data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)  # Independent variable (100 samples)
y = 4 + 3 * X + np.random.randn(100, 1)  # Dependent variable with some noise

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create a Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate Mean Squared Error
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

print(f"Training MSE: {mse_train}")
print(f"Test MSE: {mse_test}")

# Plot the data and the regression line
plt.scatter(X_train, y_train, color="blue", label="Training data")
plt.scatter(X_test, y_test, color="green", label="Test data")
plt.plot(X, model.predict(X), color="red", label="Regression line")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("Univariate Linear Regression")
plt.show()

print(f"Intercept (θ₀): {model.intercept_[0]}")
print(f"Slope (θ₁): {model.coef_[0][0]}")
