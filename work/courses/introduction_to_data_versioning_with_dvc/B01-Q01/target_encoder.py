class TargetEncoder:
    def __init__(self):
        self.encodings = {}
        self.y_mean = None

    def fit(self, X, y):
        """
        Fit the target encoder to the dataset.

        X: DataFrame containing the categorical features.
        y: Series containing the target.
        """
        for col in X.columns:
            self.encodings[col] = y.groupby(X[col]).mean()
        self.y_mean = y.mean()
        return self

    def transform(self, X):
        """
        Transform the dataset with the fitted target encoder.

        X: DataFrame containing the categorical features to be transformed.
        """
        X_transformed = X.copy()
        for col in X.columns:
            X_transformed[col] = X[col].map(self.encodings[col])
            # Fill NaN values in case there are categories in the test set
            # that were not present in the training set
            X_transformed[col].fillna(self.y_mean, inplace=True)
        return X_transformed

    def fit_transform(self, X, y):
        """
        Fit and transform the dataset with the target encoder.

        X: DataFrame containing the categorical features.
        y: Series containing the target.
        """
        return self.fit(X, y).transform(X)
