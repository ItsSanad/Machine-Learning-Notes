# My journey of learning Machine Learning with Scikit-learn 🔵🟠
![Sklearn1](https://github.com/user-attachments/assets/4d6cd2c6-475c-49fe-833a-b21df34f323f)

### template.py
This program predicts breast cancer data by scaling the data, testing multiple model settings to find the best one, and then displaying the results to show which settings work best for accurate predictions.

### templateWithPredict.py
This program loads the breast cancer dataset, splits it into training (80%) and testing (20%) sets, and builds a pipeline to scale the features and apply a K-Nearest Neighbors (KNN) regression model. Using `GridSearchCV`, it optimizes the number of neighbors in KNN through cross-validation. Finally, it predicts on the test set and calculates the Mean Squared Error (MSE) to evaluate the model's performance.

### LinearRegression.py
With this implementation, you have a complete workflow for applying Univariate Linear Regression using scikit-learn. You can extend this foundation to work with other types of regression models and evaluate different performance metrics as needed.

### PolynomialRegression.py
This example demonstrates how to fit a polynomial curve to a dataset, as well as how to select the degree of the polynomial.

### BackPropagation_Improved.ipynb
The program implements a fully customizable neural network using backpropagation to train and evaluate on a dataset. It allows the user to define the architecture with any number of layers and units, enabling flexible experimentation with hyperparameters like learning rate, number of iterations, and layer sizes. The backpropagation algorithm computes gradients for weights and biases to minimize the mean squared error through gradient descent. The program includes functions for forward propagation, backpropagation, parameter updates, and performance evaluation, with visualization of training loss and accuracy over iterations. It demonstrates its functionality by training on datasets like the Iris dataset or generated datasets (e.g., "moons") to achieve accurate predictions.