import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit


# chronological folds
def create_folds(matches, X_train_raw):
    dates = pd.to_datetime(matches.loc[X_train_raw.index, "date"])
    unique_dates = np.sort(dates.unique())
    cv = []
    for past, future in TimeSeriesSplit(n_splits=3).split(unique_dates):
        train_indices = np.flatnonzero(dates.isin(unique_dates[past]))
        valid_indices = np.flatnonzero(dates.isin(unique_dates[future]))
        cv.append((train_indices, valid_indices))

    return cv


# model dependent param grid
def create_param_grid(model_name):
    grids = {
        "forest": {
            "n_estimators": [100, 200],
            "max_depth": [10, 20, None],
            "min_samples_leaf": [1, 5],
        },
        "xgb": {
            "n_estimators": [100, 200],
            "max_depth": [2, 3],
            "learning_rate": [0.05, 0.1],
        },
    }

    return {
        f"{model_name}__{param}": values for param, values in grids[model_name].items()
    }


# run grid search
def run_gs(pipeline, matches, X_train_raw):
    cv = create_folds(matches, X_train_raw)

    # model name
    model_name = pipeline.steps[-1][0]
    param_grid = create_param_grid(model_name)

    pipeline.set_params(**{f"{model_name}__n_jobs": 1})

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"},
        refit="f1_macro",
        n_jobs=-1,
        verbose=2,
        error_score="raise",
    )

    return grid_search
