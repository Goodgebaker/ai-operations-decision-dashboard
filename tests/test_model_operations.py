from __future__ import annotations

import unittest

import pandas as pd

from src.model_operations import (
    DEFAULT_CONFIG,
    build_daily_operating_metrics,
    build_latest_snapshot,
    score_model_operations,
)
from src.model_scoring import load_scoring_policy


def sample_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    log_rows: list[dict[str, object]] = []
    hourly_rows: list[dict[str, object]] = []
    request_number = 0
    for day_index, day in enumerate(pd.date_range("2026-06-01", periods=4, freq="D")):
        for model_id, latency_offset, cost_multiplier in [
            ("model-fast", 0, 1.0),
            ("model-slow", 2200, 1.3),
        ]:
            for hour in range(4):
                timestamp = day + pd.Timedelta(hours=hour)
                request_number += 1
                log_rows.append(
                    {
                        "request_id": f"REQ-{request_number}",
                        "timestamp": timestamp,
                        "model_id": model_id,
                        "total_tokens": 1000,
                        "estimated_cost": (0.001 + day_index * 0.0001) * cost_multiplier,
                        "latency_ms": 1000 + latency_offset + hour * 100,
                        "status_code": 500 if hour == 3 else 200,
                    }
                )
                hourly_rows.append(
                    {
                        "hour": timestamp,
                        "model_id": model_id,
                        "request_count": 1,
                        "success_rate": 0 if hour == 3 else 100,
                        "p95_latency_ms": 1000 + latency_offset + hour * 100,
                    }
                )
    return pd.DataFrame(log_rows), pd.DataFrame(hourly_rows)


class DailyOperatingMetricTests(unittest.TestCase):
    def test_raw_logs_drive_daily_percentiles_and_success_rate(self) -> None:
        logs, hourly = sample_inputs()
        daily = build_daily_operating_metrics(logs, hourly)
        first = daily[
            (daily["date"] == pd.Timestamp("2026-06-01"))
            & (daily["model_id"] == "model-fast")
        ].iloc[0]
        self.assertEqual(first["request_count"], 4)
        self.assertEqual(first["success_rate"], 75)
        self.assertEqual(first["p50_latency_ms"], 1150)
        self.assertEqual(first["p95_latency_ms"], 1285)
        self.assertGreater(first["latency_cv"], 0)

    def test_cost_trend_uses_only_prior_days_and_marks_readiness(self) -> None:
        logs, hourly = sample_inputs()
        daily = build_daily_operating_metrics(logs, hourly)
        fast = daily[daily["model_id"] == "model-fast"].sort_values("date")
        self.assertEqual(fast["cost_baseline_ready"].tolist(), [False, False, False, True])
        self.assertEqual(fast.iloc[0]["cost_trend_ratio"], 1.0)
        expected = fast.iloc[3]["cost_per_request"] / fast.iloc[:3][
            "cost_per_request"
        ].median()
        self.assertAlmostEqual(fast.iloc[3]["cost_trend_ratio"], expected, places=4)

    def test_invalid_baseline_window_is_rejected(self) -> None:
        logs, hourly = sample_inputs()
        with self.assertRaisesRegex(ValueError, "最小历史天数"):
            build_daily_operating_metrics(
                logs, hourly, baseline_days=2, minimum_baseline_days=3
            )

    def test_stability_is_missing_when_only_one_hour_is_observed(self) -> None:
        logs, hourly = sample_inputs()
        one_log = logs.iloc[[0]].copy()
        one_hour = hourly.iloc[[0]].copy()
        daily = build_daily_operating_metrics(one_log, one_hour)
        self.assertEqual(daily.iloc[0]["observed_hours"], 1)
        self.assertTrue(pd.isna(daily.iloc[0]["latency_cv"]))
        self.assertTrue(pd.isna(daily.iloc[0]["success_rate_std_pct"]))


class ModelOperatingScoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = load_scoring_policy(DEFAULT_CONFIG)

    def test_scores_are_config_driven_and_bounded(self) -> None:
        logs, hourly = sample_inputs()
        daily = build_daily_operating_metrics(logs, hourly)
        quality = pd.DataFrame(
            {
                "model_id": ["model-fast", "model-slow"],
                "quality_score": [92.0, 80.0],
                "capability_run_count": [100, 100],
                "capability_dimension_count": [4, 4],
            }
        )
        scored = score_model_operations(daily, self.policy, quality)
        score_columns = [
            "success_score", "latency_score", "stability_score",
            "performance_score", "cost_efficiency_score",
            "cost_performance_score", "health_score",
        ]
        self.assertTrue(scored[score_columns].notna().all().all())
        self.assertTrue(scored[score_columns].apply(lambda column: column.between(0, 100).all()).all())
        latest = build_latest_snapshot(scored)
        self.assertEqual(latest.iloc[0]["model_id"], "model-fast")
        self.assertEqual(latest["health_rank"].tolist(), [1, 2])
        self.assertTrue(set(latest["health_level"]).issubset({"高风险", "需关注", "健康", "优秀"}))

    def test_cost_performance_is_missing_without_capability_quality(self) -> None:
        logs, hourly = sample_inputs()
        daily = build_daily_operating_metrics(logs, hourly)
        scored = score_model_operations(daily, self.policy)
        self.assertTrue(scored["quality_score"].isna().all())
        self.assertTrue(scored["cost_performance_score"].isna().all())
        self.assertTrue(scored["health_score"].notna().all())


if __name__ == "__main__":
    unittest.main()
