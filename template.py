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
