from pathlib import Path

from config_support import resolve_state_paths, select_parameters
from strategy_support import build_memory_inputs, update_memory_forecasts


def test_select_parameters_uses_lowercase_names_and_explicit_aliases() -> None:
    config = {"FIRST_VALUE": 1, "RENAMED_VALUE": 2}

    assert select_parameters(
        config,
        ("FIRST_VALUE", "RENAMED_VALUE"),
        aliases={"RENAMED_VALUE": "runtime_name"},
    ) == {"first_value": 1, "runtime_name": 2}


def test_resolve_state_paths_keeps_the_manifest_names(tmp_path: Path) -> None:
    assert resolve_state_paths(tmp_path, {"state_file": ".state.json"}) == {
        "state_file": str(tmp_path / ".state.json")
    }


def test_build_memory_inputs_preserves_signal_context() -> None:
    inputs = build_memory_inputs(
        [
            {
                "symbol": "AAPL",
                "price": 201.5,
                "dip": 3.2,
                "qualifies": True,
                "recent_avg_volume": 12_000_000,
                "expected_profit": 1.4,
                "win_probability": 0.63,
            }
        ],
        llm_score=2,
        include_recent_volume=True,
    )

    assert len(inputs) == 1
    assert inputs[0].symbol == "AAPL"
    assert inputs[0].signal_present is True
    assert inputs[0].llm_score == 2
    assert inputs[0].recent_avg_volume == 12_000_000
    assert inputs[0].historical_expected_profit == 1.4


def test_update_memory_forecasts_returns_per_symbol_disabled_context() -> None:
    forecasts = update_memory_forecasts(
        signals=[
            {"symbol": "AAPL", "price": 100, "dip": 2},
            {"symbol": "MSFT", "price": 200, "dip": 3},
        ],
        evaluation_date="2026-07-21",
        llm_score=None,
        enabled=False,
        memory_factory=lambda: (_ for _ in ()).throw(
            AssertionError("disabled memory must not be opened")
        ),
        disabled_explanation="memory disabled",
        failure_label="Portfolio memory",
        log_message=lambda *args, **kwargs: None,
    )

    assert set(forecasts) == {"AAPL", "MSFT"}
    assert all(not forecast.ready for forecast in forecasts.values())
    assert all(
        forecast.explanation == "memory disabled" for forecast in forecasts.values()
    )
