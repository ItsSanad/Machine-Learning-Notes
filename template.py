'''
This program builds a K-Nearest Neighbors (KNN) regressor pipeline to predict breast cancer data.
It scales the data, then uses GridSearchCV to find the best number of neighbors (1 to 10) with cross-validation.
Finally, it outputs the results, showing how different parameter values affect model performance.
'''
# pip install pandas
# pip install --upgrade scikit-learn

# import section
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import pandas as pd

# load the data
X, y = load_breast_cancer(return_X_y=True)

# create a pipeline
pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", KNeighborsRegressor(n_neighbors=1))
  ])

# create a GridSearchCV (model)
mod = GridSearchCV(estimator=pipe,
                   param_grid={'model__n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
                   cv=3)

# fit the model
mod.fit(X, y);

# display the model
print(pd.DataFrame(mod.cv_results_))
