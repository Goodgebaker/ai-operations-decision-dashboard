"""Regression tests for realistic scenario-level synthetic variation."""

from __future__ import annotations

import unittest

import pandas as pd

from src.generate_synthetic_v2 import build_dataset


class SyntheticScenarioTests(unittest.TestCase):
    def test_different_seeds_change_daily_shape_and_latest_model_condition(self) -> None:
        first, _ = build_dataset(seed=101)
        second, _ = build_dataset(seed=202)

        self.assertEqual(
            set(first["simulation_scenario"]),
            {"逐步改善", "平稳运行", "持续承压"},
        )
        self.assertEqual(first.groupby("model_id")["simulation_scenario"].nunique().max(), 1)

        first_daily = first.groupby(first["timestamp"].dt.floor("D")).size()
        second_daily = second.groupby(second["timestamp"].dt.floor("D")).size()
        self.assertFalse(first_daily.equals(second_daily))

        def latest_model_summary(logs):
            latest_day = logs["timestamp"].dt.floor("D").max()
            latest = logs[logs["timestamp"].dt.floor("D").eq(latest_day)].copy()
            latest["is_success"] = latest["status_code"].eq(200)
            return latest.groupby("model_id").agg(
                success_rate=("is_success", "mean"),
                p95_latency_ms=("latency_ms", lambda values: values.quantile(0.95)),
            )

        first_latest = latest_model_summary(first)
        second_latest = latest_model_summary(second)
        self.assertFalse(first_latest.round(3).equals(second_latest.round(3)))
        relative_latency_change = (
            first_latest["p95_latency_ms"] / second_latest["p95_latency_ms"]
        ).apply(lambda value: max(value, 1 / value))
        self.assertGreater(relative_latency_change.max(), 1.35)

    def test_generated_scenario_stays_within_operationally_reasonable_bounds(self) -> None:
        logs, _ = build_dataset(seed=303)
        daily_calls = logs.groupby(logs["timestamp"].dt.floor("D")).size()
        success_rate = logs["status_code"].eq(200).mean()

        self.assertGreater(success_rate, 0.85)
        self.assertLess(success_rate, 0.995)
        self.assertLess(daily_calls.max() / daily_calls.min(), 5.0)
        self.assertGreater(daily_calls.max() / daily_calls.median(), 1.35)
        self.assertLess(daily_calls.min() / daily_calls.median(), 0.75)
        self.assertTrue(logs["latency_ms"].ge(logs["first_token_latency_ms"] + 30).all())

    def test_default_dataset_covers_90_days_and_spreads_anomalies(self) -> None:
        logs, truth = build_dataset(seed=404)
        observed_days = logs["timestamp"].dt.normalize()
        self.assertEqual(90, observed_days.nunique())
        self.assertEqual(89, (observed_days.max() - observed_days.min()).days)

        anomaly_days = pd.to_datetime(truth["start_time"]).dt.normalize()
        offsets = (anomaly_days - observed_days.min()).dt.days
        self.assertLess(offsets.min(), 15)
        self.assertGreater(offsets.max(), 75)


if __name__ == "__main__":
    unittest.main()
