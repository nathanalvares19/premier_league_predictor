from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

# column names
categorical_cols = ["team", "opponent", "venue"]
numeric_cols = ["rolling_gf", "rolling_ga", "rolling_xg", "rolling_xga"]

# preprocessor for encoding and null value handling
preprocessor = ColumnTransformer(
    [
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical_cols,
        ),
        (
            "numeric",
            SimpleImputer(strategy="median"),
            numeric_cols,
        ),
    ]
)

# logistic regression pipeline
logreg_pipeline = Pipeline(
    [
        ("preprocessor", clone(preprocessor)),
        ("logreg", LogisticRegression(max_iter=1000)),
    ]
)

# random forest pipeline
forest_pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        (
            "forest",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)

# XGBoost pipeline
xgb_pipeline = Pipeline(
    [
        ("preprocessor", clone(preprocessor)),
        (
            "xgb",
            XGBClassifier(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                objective="multi:softprob",
                tree_method="hist",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)
