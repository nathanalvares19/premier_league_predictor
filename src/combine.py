# merge home + away rows into one row per match
from src import stats

rolling_cols = [f"rolling_{s}" for s in stats.stats]


def combine(matches):
    home = matches[matches["venue"] == "Home"]
    away = matches[matches["venue"] == "Away"]

    games = home[["date", "team", "opponent", "result", *rolling_cols]].merge(
        away[["date", "team", "opponent", *rolling_cols]],
        left_on=["date", "team", "opponent"],
        right_on=["date", "opponent", "team"],
        suffixes=("_home", "_away"),
    )

    # result stays from home pov: W = home win, L = away win, D = draw
    games = games.rename(columns={"team_home": "home", "team_away": "away"})
    games = games.rename(columns={f"{c}_home": f"home_{c}" for c in rolling_cols})
    games = games.rename(columns={f"{c}_away": f"away_{c}" for c in rolling_cols})

    return games.drop(columns=["opponent_home", "opponent_away"])
