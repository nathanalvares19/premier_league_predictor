from sklearn.preprocessing import OneHotEncoder

categorical_cols = ["team", "opponent", "venue"]


def encode_data(encoder, X_train, X_valid, X_test):
    X_train_encoded = encoder.fit_transform(X_train[categorical_cols])
    X_valid_encoded = encoder.transform(X_valid[categorical_cols])
    X_test_encoded = encoder.transform(X_test[categorical_cols])

    X_train_encoded = X_train_encoded.join(X_train.drop(columns=categorical_cols))
    X_valid_encoded = X_valid_encoded.join(X_valid.drop(columns=categorical_cols))
    X_test_encoded = X_test_encoded.join(X_test.drop(columns=categorical_cols))

    return X_train_encoded, X_valid_encoded, X_test_encoded
