"""Persistent, lightweight learning from news scores and next-session returns."""

import json
import math
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class LearningResult:
    """Explain the current state and output of the adaptive model."""

    observations: int
    ready: bool
    predicted_return_percent: float | None
    slope: float | None
    correlation: float | None
    explanation: str


class AdaptiveNewsModel:
    """Fit a bounded one-variable regression using a rolling observation set."""

    def __init__(self, state_path: Path, minimum_observations: int, maximum_observations: int):
        self.state_path = state_path
        self.minimum_observations = minimum_observations
        self.maximum_observations = maximum_observations

    def update(
        self,
        evaluation_date: str,
        current_price: float,
        news_score: int | None,
    ) -> LearningResult:
        """Resolve yesterday's outcome, store today's input, and fit the model."""
        state = self._load_state()
        observations = state["observations"]
        pending = state.get("pending")

        if pending and pending.get("date") != evaluation_date:
            prior_price = float(pending.get("price", 0))
            prior_score = int(pending.get("news_score", 0))
            if prior_price > 0 and current_price > 0:
                return_percent = ((current_price - prior_price) / prior_price) * 100.0
                if math.isfinite(return_percent):
                    observations.append(
                        {
                            "news_score": prior_score,
                            "return_percent": max(-25.0, min(25.0, return_percent)),
                        }
                    )
                    state["observations"] = observations[-self.maximum_observations :]

        if news_score is not None:
            state["pending"] = {
                "date": evaluation_date,
                "price": current_price,
                "news_score": news_score,
            }
        elif pending and pending.get("date") != evaluation_date:
            state["pending"] = None

        self._save_state(state)
        return self._fit(state["observations"], news_score)

    def _fit(self, observations: list[dict[str, Any]], current_score: int | None) -> LearningResult:
        count = len(observations)
        if current_score is None:
            return LearningResult(
                observations=count,
                ready=False,
                predicted_return_percent=None,
                slope=None,
                correlation=None,
                explanation="No current news score is available for a forecast.",
            )
        if count < self.minimum_observations:
            return LearningResult(
                observations=count,
                ready=False,
                predicted_return_percent=None,
                slope=None,
                correlation=None,
                explanation=(
                    f"Learning safely: {count}/{self.minimum_observations} required "
                    "completed observations collected."
                ),
            )

        xs = [float(item["news_score"]) for item in observations]
        ys = [float(item["return_percent"]) for item in observations]
        mean_x = sum(xs) / count
        mean_y = sum(ys) / count
        sum_xx = sum((value - mean_x) ** 2 for value in xs)
        sum_yy = sum((value - mean_y) ** 2 for value in ys)
        sum_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))

        if sum_xx < 1.0 or len(set(xs)) < 2:
            return LearningResult(
                observations=count,
                ready=False,
                predicted_return_percent=None,
                slope=None,
                correlation=None,
                explanation=(
                    f"Collected {count} observations, but news scores do not vary "
                    "enough to fit a trustworthy relationship."
                ),
            )

        # Ridge stabilization prevents extreme coefficients when scores barely vary.
        slope = sum_xy / (sum_xx + 1.0)
        intercept = mean_y - (slope * mean_x)
        prediction = intercept + (slope * float(current_score))
        prediction = max(-10.0, min(10.0, prediction))
        correlation = (
            sum_xy / math.sqrt(sum_xx * sum_yy)
            if sum_xx > 0 and sum_yy > 0
            else 0.0
        )
        return LearningResult(
            observations=count,
            ready=True,
            predicted_return_percent=prediction,
            slope=slope,
            correlation=correlation,
            explanation=(
                f"Adaptive model used {count} observations; predicted next-session "
                f"return {prediction:+.2f}%, score sensitivity {slope:+.3f}, "
                f"correlation {correlation:+.2f}."
            ),
        )

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {"version": 1, "observations": [], "pending": None}
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            observations = state.get("observations", [])
            if not isinstance(observations, list):
                raise ValueError("observations must be a list")
            return {
                "version": 1,
                "observations": observations[-self.maximum_observations :],
                "pending": state.get("pending"),
            }
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            # Preserve a corrupt file for diagnosis instead of trusting its data.
            corrupt_path = self.state_path.with_suffix(self.state_path.suffix + ".corrupt")
            with suppress(OSError):
                self.state_path.replace(corrupt_path)
            return {"version": 1, "observations": [], "pending": None}

    def _save_state(self, state: dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(state, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(self.state_path)
