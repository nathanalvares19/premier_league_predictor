# rolling averages

stats = ["gf", "ga", "xg", "xga"]


def compute_rolling(matches):
    for stat in stats:
        matches[f"rolling_{stat}"] = matches.groupby("team")[stat].transform(
            lambda values: values.shift(1).rolling(window=5, min_periods=1).mean()
        )

    return matches
