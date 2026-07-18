"""Shared two-feature ridge regression for the memory modules.

Used by both `trade_memory.py` (A-vs-B edge forecast) and
`portfolio_memory.py` (pooled dip-signal forecast) to fit next-session
return on (dip size, news score). Implemented directly, no dependency, to
keep this Pi service dependency-free and auditable.
"""

import math


def fit_two_feature_ridge(
    rows: list[tuple[float, int | None, float]], dip: float, score: int | None
) -> tuple[float, float] | None:
    """Fit a ridge-stabilized 2-feature linear regression (dip, news score)
    -> next-session return over `rows`, then predict at (`dip`, `score`).

    Returns (predicted, correlation) with `predicted` clamped to [-10, 10],
    or None if there is not enough feature variation to solve for weights.
    """
    count = len(rows)
    xs = [(float(row[0]), float(row[1] if row[1] is not None else 0)) for row in rows]
    ys = [float(row[2]) for row in rows]
    mean_x = [sum(values[index] for values in xs) / count for index in range(2)]
    mean_y = sum(ys) / count
    centered = [[values[index] - mean_x[index] for index in range(2)] for values in xs]
    # Small ridge penalty makes a nearly constant news score harmless.
    a = sum(row[0] * row[0] for row in centered) + 1.0
    b = sum(row[0] * row[1] for row in centered)
    d = sum(row[1] * row[1] for row in centered) + 1.0
    determinant = a * d - b * b
    if determinant <= 1e-9:
        return None
    target = [sum(row[index] * (y - mean_y) for row, y in zip(centered, ys)) for index in range(2)]
    beta_dip = (d * target[0] - b * target[1]) / determinant
    beta_news = (a * target[1] - b * target[0]) / determinant
    predicted = mean_y + beta_dip * (dip - mean_x[0]) + beta_news * ((score or 0) - mean_x[1])
    predicted = max(-10.0, min(10.0, predicted))
    fitted = [mean_y + beta_dip * row[0] + beta_news * row[1] for row in centered]
    mean_fit = sum(fitted) / count
    variance_y = sum((value - mean_y) ** 2 for value in ys)
    variance_fit = sum((value - mean_fit) ** 2 for value in fitted)
    covariance = sum((y - mean_y) * (fit - mean_fit) for y, fit in zip(ys, fitted))
    correlation = covariance / math.sqrt(variance_y * variance_fit) if variance_y > 0 and variance_fit > 0 else 0.0
    return predicted, correlation
