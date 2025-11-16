import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def train_linear_regression(preprocessor, X_train, y_train):
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ])
    model.fit(X_train, y_train)
    return model

def train_random_forest(preprocessor, X_train, y_train):
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(random_state=42))
    ])

    param_grid = {
        "model__n_estimators": [100, 300],
        "model__max_depth": [5, 10, 20],
    }

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring="neg_root_mean_squared_error"
    )

    grid.fit(X_train, y_train)
    return grid

def train_xgboost(preprocessor, X_train, y_train):
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", XGBRegressor(objective="reg:squarederror", eval_metric="rmse"))
    ])

    param_grid = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [4, 6, 8],
        "model__learning_rate": [0.05, 0.1, 0.2]
    }

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring="neg_root_mean_squared_error"
    )

    grid.fit(X_train, y_train)
    return grid
