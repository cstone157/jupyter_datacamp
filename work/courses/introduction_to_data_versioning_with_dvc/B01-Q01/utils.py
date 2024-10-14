import json

import pandas as pd


# Function to load the configuration
def load_config(filename):
    with open(filename, "r") as file:
        config = json.load(file)
    return config


def get_col_dtypes(dataset_config):
    col_dtypes = {}
    # Handle floats
    for c in dataset_config["numerical_columns"]:
        col_dtypes[c] = float
    # Handle floats
    for c in dataset_config["categorical_columns"]:
        col_dtypes[c] = str
    # Label as int
    col_dtypes[dataset_config["label_column"]] = int
    return col_dtypes


def load_data_and_labels(file_path, col_dtypes, label_column):
    data = pd.read_csv(file_path)

    # Select only the columns specified in column_types
    data = data[list(col_dtypes.keys())]

    # Convert the types of these columns as per column_types
    data = data.astype(col_dtypes)

    labels = data[label_column]

    # Drop label column
    data = data.drop(columns=[label_column], axis=1)

    return {"X": data, "y": labels}\
    
