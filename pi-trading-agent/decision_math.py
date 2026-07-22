"""Asset-class-agnostic decision math shared by the equity and crypto strategies.

Extracted from AssetRotationStrategy (strategy.py) so CryptoRotationStrategy
can reuse the same walk-forward validation, posture-adjusted ranking, and
position-count math without duplicating it or depending on the equity
strategy class. Both strategies call these functions directly.
"""

import math

import numpy as np

# How strongly the risky/conservative reasoning pattern reshapes a symbol's
# historical edge in posture_adjusted_edge: conservative leans on consistency
# (penalizing variance and bad-news days harder), risky leans on raw edge
# (barely discounting variance or negative news).
POSTURE_VARIANCE_PENALTY = {"conservative": 0.6, "risky": 0.15}
POSTURE_CONSISTENCY_WEIGHT = {"conservative": 1.0, "risky": 0.25}
# The LLM score is a signed purchase signal: constructive assessments improve
# ranking, while negative assessments reduce it.  The deliberately small
# weights and the shared clamp below keep it subordinate to the measured
# historical edge; callers still enforce their raw profit/OOS eligibility
# floors before any order can be submitted.
POSTURE_LLM_SCORE_WEIGHT = {"conservative": 0.10, "risky": 0.15}
# How much weight a ready learned-edge forecast gets when it disagrees with a
# symbol's raw historical expected_profit: risky leans into the pooled,
# cross-symbol learned edge more; conservative trusts the unadjusted
# historical backtest more until the learned edge has proven itself. Same
# never-changes-eligibility invariant as the other posture weights above.
POSTURE_LEARNED_EDGE_WEIGHT = {"conservative": 0.25, "risky": 0.6}
POSTURE_MAX_ADJUSTMENT_PERCENT = 3.0
SYMBOL_NEWS_SCORE_WEIGHT = 0.05


def effective_profit_floor(
    configured_floor_percent: float,
    posture: str,
    risky_multiplier: float = 0.5,
) -> float:
    """Return the posture-aware net-profit hurdle used by every entry gate.

    Conservative mode preserves the configured hurdle. Risky mode can accept
    a smaller *positive* measured edge, but never turns a positive configured
    floor into permission to buy a forecasted loser.
    """
    configured = max(0.0, float(configured_floor_percent))
    if str(posture).lower() != "risky":
        return configured
    multiplier = max(0.0, min(1.0, float(risky_multiplier)))
    return configured * multiplier


def llm_exposure_multiplier(llm_score: float | int | None) -> float:
    """Translate a non-vetoed LLM score into a bounded new-capital fraction.

    Neutral and constructive assessments leave the strategy fully deployed.
    Adverse scores progressively reserve cash, but never create a trade or
    override the caller's hard-veto threshold and quantitative entry gates.
    """
    if llm_score is None:
        return 1.0
    score = max(-10.0, min(10.0, float(llm_score)))
    return max(0.25, min(1.0, 1.0 + score * 0.10))


def learned_edge_allows_purchase(
    signal: dict[str, float | int | str | bool | None],
) -> bool:
    """Let validated pooled knowledge veto a forecasted losing entry."""
    if not signal.get("learned_edge_ready"):
        return True
    learned_edge = signal.get("learned_edge")
    return learned_edge is not None and float(learned_edge) >= 0.0


def portfolio_signal_rejection_reason(
    signal: dict[str, float | int | str | bool | None],
    *,
    dip_threshold_percent: float,
    minimum_observations: int,
    minimum_profit_percent: float,
    oos_minimum_observations: int,
    oos_minimum_profit_percent: float,
) -> str | None:
    """Return the first entry gate a portfolio signal fails, or ``None``.

    Keeping this classification beside the decision math prevents the email
    diagnostics from becoming a second, subtly different eligibility path.
    """
    if float(signal.get("dip") or 0.0) < float(dip_threshold_percent):
        return "below dip threshold"
    if not bool(signal.get("qualifies")):
        return "no comparable historical dips"
    if int(signal.get("observations") or 0) < int(minimum_observations):
        return "insufficient historical samples"
    expected_profit = signal.get("expected_profit")
    if expected_profit is None or float(expected_profit) < float(minimum_profit_percent):
        return "historical edge below floor"
    if int(signal.get("oos_observations") or 0) < int(oos_minimum_observations):
        return "insufficient walk-forward samples"
    oos_expected_profit = signal.get("oos_expected_profit")
    if oos_expected_profit is None or float(oos_expected_profit) < float(
        oos_minimum_profit_percent
    ):
        return "walk-forward edge below floor"
    if not learned_edge_allows_purchase(signal):
        return "learned edge veto"
    return None


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
    if minimum_observations >= len(returns):
        return []
    running_net_sum = sum(returns[:minimum_observations]) - (
        round_trip_cost_percent * minimum_observations
    )
    outcomes: list[float] = []
    for index in range(minimum_observations, len(returns)):
        prior_net_mean = running_net_sum / index
        if prior_net_mean >= entry_threshold_percent:
            outcomes.append(returns[index] - round_trip_cost_percent)
        running_net_sum += returns[index] - round_trip_cost_percent
    return outcomes


def historical_dip_returns(
    highs: np.ndarray,
    closes: np.ndarray,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned historical dip sizes and next-session returns.

    Each dip uses only the preceding ``lookback`` highs. The final bar has no
    known next-session return, so it is intentionally excluded.
    """
    count = min(highs.size, closes.size)
    if lookback < 1 or count <= lookback + 1:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty
    highs = np.asarray(highs[:count], dtype=np.float64)
    closes = np.asarray(closes[:count], dtype=np.float64)
    prior_windows = np.lib.stride_tricks.sliding_window_view(highs, lookback)
    prior_highs = prior_windows[: count - lookback - 1].max(axis=1)
    event_closes = closes[lookback:-1]
    next_closes = closes[lookback + 1 :]
    dips = ((prior_highs - event_closes) / prior_highs) * 100.0
    next_returns = ((next_closes - event_closes) / event_closes) * 100.0
    return dips, next_returns


def historical_dips(
    highs: np.ndarray,
    closes: np.ndarray,
    lookback: int,
) -> np.ndarray:
    """Return each bar's dip from the preceding rolling high.

    Unlike :func:`historical_dip_returns`, this includes the final bar because
    memory backfills retain that last observation for later settlement.
    """
    count = min(highs.size, closes.size)
    if lookback < 1 or count <= lookback:
        return np.empty(0, dtype=np.float64)
    highs = np.asarray(highs[:count], dtype=np.float64)
    closes = np.asarray(closes[:count], dtype=np.float64)
    prior_windows = np.lib.stride_tricks.sliding_window_view(highs, lookback)
    prior_highs = prior_windows[: count - lookback].max(axis=1)
    return ((prior_highs - closes[lookback:]) / prior_highs) * 100.0


def posture_adjusted_edge(
    signal: dict[str, float | int | str | None],
    posture: str,
    llm_score: float | int | None = None,
    symbol_news_score: float | int | None = None,
) -> float:
    """Reshape a symbol's historical edge through a risky or conservative lens.

    Conservative leans on consistency. A signed LLM assessment contributes a
    bounded purchase signal: positive scores improve ranking and negative
    scores reduce it. This never changes the expected-profit eligibility
    threshold itself; it only reweights which already-qualifying candidate
    looks best and which current holding looks weakest.
    """
    posture = posture if posture in ("conservative", "risky") else "conservative"
    expected_profit = float(signal["expected_profit"])
    stdev = float(signal.get("return_stdev") or 0.0)
    win_probability = float(signal.get("win_probability") or 0.5)
    adjustment = -POSTURE_VARIANCE_PENALTY[posture] * stdev
    adjustment += (win_probability - 0.5) * 2.0 * POSTURE_CONSISTENCY_WEIGHT[posture]
    if llm_score is not None:
        capped_llm_score = max(-10.0, min(10.0, float(llm_score)))
        adjustment += capped_llm_score * POSTURE_LLM_SCORE_WEIGHT[posture]
    if symbol_news_score is not None:
        # Company/pair-specific context stays distinct from the aggregate LLM
        # opinion. It is bounded and ranking-only: headlines cannot make an
        # otherwise ineligible price signal tradable.
        capped_symbol_score = max(-10.0, min(10.0, float(symbol_news_score)))
        adjustment += capped_symbol_score * SYMBOL_NEWS_SCORE_WEIGHT
    if signal.get("learned_edge_ready") and signal.get("learned_edge") is not None:
        learned_edge = float(signal["learned_edge"])
        adjustment += (learned_edge - expected_profit) * POSTURE_LEARNED_EDGE_WEIGHT[posture]
    max_adjustment = POSTURE_MAX_ADJUSTMENT_PERCENT
    adjustment = max(-max_adjustment, min(max_adjustment, adjustment))
    return expected_profit + adjustment


def qualified_position_count(
    total_capital: float,
    min_order_dollars: float,
    available_symbol_count: int,
    configured_max_positions: int,
) -> int:
    """Return every fundable qualified slot up to the configured ceiling."""
    if configured_max_positions < 1:
        return 0
    if available_symbol_count < 1 or total_capital <= 0 or min_order_dollars <= 0:
        return 0
    feasible_cap = int(total_capital // min_order_dollars)
    return max(0, min(configured_max_positions, feasible_cap, available_symbol_count))


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
    edge_sum = 0.0
    variance_sum = 0.0
    for n, (edge, stdev) in enumerate(candidate_edges[:ceiling], start=1):
        edge_sum += edge
        variance_sum += stdev * stdev
        mean_edge = edge_sum / n
        mean_variance = variance_sum / n
        portfolio_stdev = math.sqrt(mean_variance / n)
        score = mean_edge / portfolio_stdev if portfolio_stdev > 0 else mean_edge * 1e6
        if score > best_score:
            best_score = score
            best_n = n
    return best_n
