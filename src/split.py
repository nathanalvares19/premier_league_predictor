# data splitting into TRAIN, VALIDATION, TEST
from src import encode


def split_data(matches):
    dates = matches["date"].sort_values()

    # split - 70% 15% 15%
    valid_start = dates.iloc[int(len(dates) * 0.70)]
    test_start = dates.iloc[int(len(dates) * 0.85)]

    train = matches.loc[matches["date"] < valid_start].copy()

    valid = matches.loc[
        (matches["date"] >= valid_start) & (matches["date"] < test_start)
    ].copy()

    test = matches.loc[matches["date"] >= test_start].copy()

    return train, valid, test


def split_label(data, features):
    X = data[features]
    y = data["result"]

    return X, y


def final_split(matches, features, encoder):
    train, valid, test = split_data(matches)

    X_train, y_train = split_label(train, features)
    X_valid, y_valid = split_label(valid, features)
    X_test, y_test = split_label(test, features)

    X_train_encoded, X_valid_encoded, X_test_encoded = encode.encode_data(
        encoder, X_train, X_valid, X_test
    )

    results_label = {"W": 0, "L": 1, "D": 2}

    y_train_encoded = y_train.map(results_label)
    y_valid_encoded = y_valid.map(results_label)
    y_test_encoded = y_test.map(results_label)

    return (
        X_train_encoded,
        X_valid_encoded,
        X_test_encoded,
        y_train_encoded,
        y_valid_encoded,
        y_test_encoded,
    )
