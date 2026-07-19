"""Shared two-feature ridge regression for the memory modules.

Used by both `trade_memory.py` (A-vs-B edge forecast) and
`portfolio_memory.py` (pooled dip-signal forecast) to fit next-session
return on (dip size, news score). Implemented directly, no dependency, to
keep this Pi service dependency-free and auditable.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TwoFeatureRidgeModel:
    mean_dip: float
    mean_score: float
    mean_target: float
    beta_dip: float
    beta_news: float
    correlation: float

    def predict(self, dip: float, score: int | None) -> float:
        predicted = self.mean_target
        predicted += self.beta_dip * (dip - self.mean_dip)
        predicted += self.beta_news * ((score or 0) - self.mean_score)
        return max(-10.0, min(10.0, predicted))


def fit_two_feature_ridge_model(
    rows: list[tuple[float, int | None, float]],
) -> TwoFeatureRidgeModel | None:
    """Fit the fixed two-feature model from one pass of sufficient statistics."""
    if not rows:
        return None
    count = 0
    mean_dip = mean_score = mean_target = 0.0
    centered_dip2 = centered_score2 = centered_target2 = 0.0
    centered_dip_score = centered_dip_target = centered_score_target = 0.0
    for raw_dip, raw_score, raw_target in rows:
        dip = float(raw_dip)
        score = float(raw_score if raw_score is not None else 0)
        target = float(raw_target)
        count += 1
        delta_dip = dip - mean_dip
        delta_score = score - mean_score
        delta_target = target - mean_target
        mean_dip += delta_dip / count
        mean_score += delta_score / count
        mean_target += delta_target / count
        centered_dip2 += delta_dip * (dip - mean_dip)
        centered_score2 += delta_score * (score - mean_score)
        centered_target2 += delta_target * (target - mean_target)
        centered_dip_score += delta_dip * (score - mean_score)
        centered_dip_target += delta_dip * (target - mean_target)
        centered_score_target += delta_score * (target - mean_target)

    # Small ridge penalty makes a nearly constant news score harmless.
    a = centered_dip2 + 1.0
    b = centered_dip_score
    d = centered_score2 + 1.0
    determinant = a * d - b * b
    if determinant <= 1e-9:
        return None
    beta_dip = (d * centered_dip_target - b * centered_score_target) / determinant
    beta_news = (a * centered_score_target - b * centered_dip_target) / determinant

    # Centered fitted values are beta_dip*x + beta_news*score. Their mean is
    # zero by construction, allowing correlation to use the same statistics.
    variance_fit = (
        beta_dip * beta_dip * centered_dip2
        + 2.0 * beta_dip * beta_news * centered_dip_score
        + beta_news * beta_news * centered_score2
    )
    covariance = beta_dip * centered_dip_target + beta_news * centered_score_target
    raw_correlation = (
        covariance / math.sqrt(centered_target2 * variance_fit)
        if centered_target2 > 0 and variance_fit > 0
        else 0.0
    )
    correlation = max(-1.0, min(1.0, raw_correlation))
    return TwoFeatureRidgeModel(
        mean_dip,
        mean_score,
        mean_target,
        beta_dip,
        beta_news,
        correlation,
    )


def fit_two_feature_ridge(
    rows: list[tuple[float, int | None, float]], dip: float, score: int | None
) -> tuple[float, float] | None:
    """Fit a ridge-stabilized 2-feature linear regression (dip, news score)
    -> next-session return over `rows`, then predict at (`dip`, `score`).

    Returns (predicted, correlation) with `predicted` clamped to [-10, 10],
    or None if there is not enough feature variation to solve for weights.
    """
    model = fit_two_feature_ridge_model(rows)
    if model is None:
        return None
    return model.predict(dip, score), model.correlation
