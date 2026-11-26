import pandas as pd
from src import stats

# model classes: 0 = home win, 1 = away win, 2 = draw
labels = {0: "home_win", 1: "away_win", 2: "draw"}


def form(matches, team):
    # mean of the team's last 5 matches, same window as training
    return (
        matches[matches["team"] == team].sort_values("date").tail(5)[stats.stats].mean()
    )


def predict(model, matches, home, away):
    home_form, away_form = form(matches, home), form(matches, away)

    row = pd.DataFrame(
        [
            {
                "home": home,
                "away": away,
                **{f"home_rolling_{s}": home_form[s] for s in stats.stats},
                **{f"away_rolling_{s}": away_form[s] for s in stats.stats},
            }
        ]
    )

    probs = model.predict_proba(row)[0]
    return {labels[c]: float(p) for c, p in zip(model.classes_, probs)}
