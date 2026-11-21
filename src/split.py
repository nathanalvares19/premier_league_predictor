# data splitting into TRAIN, VALIDATION, TEST


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
