from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from target_encoder import TargetEncoder


def create_sklearn_pipeline(config):
    # Creating the preprocessing pipelines for both numerical and categorical data
    numerical_pipeline = Pipeline([("scaler", StandardScaler())])
    categorical_pipeline = Pipeline([("target_enc", TargetEncoder())])

    # Combining preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, config["dataset"]["numerical_columns"]),
            ("cat", categorical_pipeline, config["dataset"]["categorical_columns"]),
        ]
    )

    # Creating the final pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "rfc",
                RandomForestClassifier(**config["pipeline"]["rfc"]),
            ),
        ]
    )

    return pipeline
