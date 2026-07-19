"""Asset-class-agnostic decision math shared by the equity and crypto strategies.

Extracted from AssetRotationStrategy (strategy.py) so CryptoRotationStrategy
can reuse the same walk-forward validation, posture-adjusted ranking, and
position-count math without duplicating it or depending on the equity
strategy class. AssetRotationStrategy keeps its existing method names as thin
aliases onto these functions so its own call sites and tests are unaffected.
"""

import math

# How strongly the risky/conservative reasoning pattern reshapes a symbol's
# historical edge in posture_adjusted_edge: conservative leans on consistency
# (penalizing variance and bad-news days harder), risky leans on raw edge
# (barely discounting variance or negative news). These never change the
# expected-profit eligibility threshold itself, only which already-qualifying
# candidate looks best and which holding looks weakest.
POSTURE_VARIANCE_PENALTY = {"conservative": 0.6, "risky": 0.15}
POSTURE_CONSISTENCY_WEIGHT = {"conservative": 1.0, "risky": 0.25}
POSTURE_NEWS_DISCOUNT_PER_POINT = {"conservative": 0.15, "risky": 0.05}
# How much weight a ready learned-edge forecast gets when it disagrees with a
# symbol's raw historical expected_profit: risky leans into the pooled,
# cross-symbol learned edge more; conservative trusts the unadjusted
# historical backtest more until the learned edge has proven itself. Same
# never-changes-eligibility invariant as the other posture weights above.
POSTURE_LEARNED_EDGE_WEIGHT = {"conservative": 0.25, "risky": 0.6}
POSTURE_MAX_ADJUSTMENT_PERCENT = 3.0


def walk_forward_net_returns(
    returns: list[float],
    round_trip_cost_percent: float,
    minimum_observations: int,
    entry_threshold_percent: float,
) -> list[float]:
    """Evaluate only trades selected from information available beforehand.

    Each validation result uses a historical mean formed strictly before that
    event. This prevents a candidate's realised return from helping select
    itself, unlike an in-sample average.
    """
    outcomes: list[float] = []
    for index in range(minimum_observations, len(returns)):
        prior_net_mean = (
            sum(value - round_trip_cost_percent for value in returns[:index]) / index
        )
        if prior_net_mean >= entry_threshold_percent:
            outcomes.append(returns[index] - round_trip_cost_percent)
    return outcomes


def posture_adjusted_edge(
    signal: dict[str, float | int | str | None],
    posture: str,
    news_score: float | int | None,
) -> float:
    """Reshape a symbol's historical edge through a risky or conservative lens.

    Conservative leans on consistency: it penalizes return variance and a
    negative news day harder. Risky leans on raw edge: it barely discounts
    variance or bad news. This never changes the expected-profit eligibility
    threshold itself; it only reweights which already-qualifying candidate
    looks best and which current holding looks weakest.
    """
    posture = posture if posture in ("conservative", "risky") else "conservative"
    expected_profit = float(signal["expected_profit"])
    stdev = float(signal.get("return_stdev") or 0.0)
    win_probability = float(signal.get("win_probability") or 0.5)
    adjustment = -POSTURE_VARIANCE_PENALTY[posture] * stdev
    adjustment += (win_probability - 0.5) * 2.0 * POSTURE_CONSISTENCY_WEIGHT[posture]
    if news_score is not None:
        capped_score = max(-10.0, min(10.0, float(news_score)))
        adjustment -= max(0.0, -capped_score) * POSTURE_NEWS_DISCOUNT_PER_POINT[posture]
    if signal.get("learned_edge_ready") and signal.get("learned_edge") is not None:
        learned_edge = float(signal["learned_edge"])
        adjustment += (learned_edge - expected_profit) * POSTURE_LEARNED_EDGE_WEIGHT[posture]
    max_adjustment = POSTURE_MAX_ADJUSTMENT_PERCENT
    adjustment = max(-max_adjustment, min(max_adjustment, adjustment))
    return expected_profit + adjustment


def optimal_position_count(
    total_capital: float,
    min_order_dollars: float,
    candidate_edges: list[tuple[float, float]],
    configured_max_positions: int,
) -> int:
    """How many of today's ranked candidates are worth splitting capital across.

    `candidate_edges` is `(expected_profit_percent, return_stdev_percent)` per
    eligible candidate, in the caller's posture-adjusted ranking order -- the
    order buys actually happen in, so each prefix scored below is exactly the
    basket that many buys would create. Scores each feasible position count n
    by the Sharpe-like ratio of an equal-weighted, n-position basket -- mean
    edge divided by portfolio risk, where risk assumes zero correlation
    between candidates (equal-weighted variance of n independent bets falls
    off as 1/n). That independence assumption is an optimistic upper bound:
    symbols sharing a market factor (broad index ETFs moving together in a
    dip, for instance) diversify less than this in practice, so the result is
    a ceiling suggestion, not a promise. n never exceeds
    configured_max_positions -- this narrows that configured ceiling to what
    today's capital and candidate quality actually support; it never widens
    it.
    """
    if configured_max_positions < 1:
        return 1
    if not candidate_edges or total_capital <= 0 or min_order_dollars <= 0:
        return 1
    feasible_cap = max(1, int(total_capital // min_order_dollars))
    ceiling = max(1, min(configured_max_positions, feasible_cap, len(candidate_edges)))

    best_n = 1
    best_score = float("-inf")
    for n in range(1, ceiling + 1):
        top = candidate_edges[:n]
        mean_edge = sum(edge for edge, _ in top) / n
        mean_variance = sum(stdev * stdev for _, stdev in top) / n
        portfolio_stdev = math.sqrt(mean_variance / n)
        score = mean_edge / portfolio_stdev if portfolio_stdev > 0 else mean_edge * 1e6
        if score > best_score:
            best_score = score
            best_n = n
    return best_n
