import argparse
import json

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from pipelines import create_sklearn_pipeline
from utils import get_col_dtypes, load_config, load_data_and_labels


def load_train_test_data(train_file, test_file, config):
    col_dtypes = get_col_dtypes(config["dataset"])
    label_column = config["dataset"]["label_column"]

    train_data = load_data_and_labels(train_file, col_dtypes, label_column)
    test_data = load_data_and_labels(test_file, col_dtypes, label_column)

    return train_data, test_data


def train_model(
    train_data,
    config,
):
    # Define and fit model
    pipeline = create_sklearn_pipeline(config)
    pipeline.fit(train_data["X"], train_data["y"])

    return pipeline


def evaluate_model(pipeline, test_data):
    # Predictions
    y_pred = pipeline.predict(test_data["X"])
    y_test = test_data["y"]

    # Metrics
    metrics = {
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "Accuracy": accuracy_score(y_test, y_pred),
    }
    return json.loads(
        json.dumps(metrics), parse_float=lambda x: round(float(x), ndigits=4)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and Evaluate Hotel Booking Cancellation Model"
    )
    parser.add_argument(
        "config_file",
        type=str,
        help="Configuration file with parameters",
        default="params.json",
    )
    parser.add_argument(
        "train_file",
        type=str,
        help="Input CSV file path for the training set",
        default="train.csv",
    )
    parser.add_argument(
        "test_file",
        type=str,
        help="Input CSV file path for the test set",
        default="test.csv",
    )
    args = parser.parse_args()

    config = load_config(args.config_file)
    train_data, test_data = load_train_test_data(
        args.train_file, args.test_file, config
    )
    model = train_model(train_data, config)
    metrics = evaluate_model(model, test_data)
    print(metrics)
