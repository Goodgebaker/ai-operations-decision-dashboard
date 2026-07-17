from __future__ import annotations

import unittest

import pandas as pd

from src.model_health_risk import (
    DEFAULT_CONFIG,
    RiskPolicy,
    _ascending_risk,
    _descending_risk,
    build_diagnostic_evidence,
    build_health_risks,
    load_risk_policy,
)
from src.model_scoring import load_scoring_policy


POLICY_VALUES = {
    "baseline_window_days": 7,
    "minimum_baseline_days": 3,
    "performance_score_drop_warning_points": 8,
    "performance_score_drop_severe_points": 25,
    "absolute_performance_warning_score": 70,
    "absolute_performance_severe_score": 30,
    "p95_latency_increase_warning_ratio": 1.25,
    "p95_latency_increase_severe_ratio": 2,
    "success_drop_warning_points": 2,
    "success_drop_severe_points": 10,
    "absolute_success_warning_pct": 94,
    "absolute_success_severe_pct": 75,
    "cost_increase_warning_ratio": 1.15,
    "cost_increase_severe_ratio": 1.5,
    "cost_efficiency_drop_warning_points": 8,
    "cost_efficiency_drop_severe_points": 30,
    "single_component_floor_multiplier": 0.6,
    "model_side_risk_floor": 80,
    "capability_or_probe_risk_floor": 45,
    "platform_or_traffic_risk_floor": 35,
    "environment_latency_risk_floor": 20,
    "evidence_risk_threshold": 30,
    "route_downweight_risk_threshold": 30,
    "route_switch_risk_threshold": 60,
    "minimum_candidate_health_score": 70,
}


def _operating_rows() -> pd.DataFrame:
    dates = pd.date_range("2026-06-01", periods=5, freq="D")
    rows = []
    for model_id, performance, p95, success, cost, efficiency, health in (
        ("model-a", [90, 90, 90, 90, 40], [1000, 1000, 1000, 1000, 2400], [99, 99, 99, 99, 80], [1, 1, 1, 1, 1], [90] * 5, [90, 90, 90, 90, 40]),
        ("model-b", [92] * 5, [900] * 5, [99] * 5, [1] * 5, [92] * 5, [92] * 5),
    ):
        for index, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "model_id": model_id,
                    "performance_score": performance[index],
                    "p95_latency_ms": p95[index],
                    "success_rate": success[index],
                    "cost_per_request": cost[index],
                    "cost_per_1k_tokens": cost[index],
                    "cost_efficiency_score": efficiency[index],
                    "health_score": health[index],
                }
            )
    return pd.DataFrame(rows)


def _diagnosis_rows() -> pd.DataFrame:
    dates = pd.date_range("2026-06-01", periods=5, freq="D")
    rows = []
    for model_id, provider in (("model-a", "provider-a"), ("model-b", "provider-b")):
        for date in dates:
            degraded = model_id == "model-a" and date == dates[-1]
            rows.append(
                {
                    "date": date,
                    "model_id": model_id,
                    "provider": provider,
                    "diagnosis_type": "model_side_degradation" if degraded else "healthy",
                    "diagnosis_severity": "high" if degraded else "none",
                    "diagnosis_reason": "双侧同步下降" if degraded else "双侧健康",
                    "probe_http_success_rate": 80 if degraded else 100,
                    "probe_p95_latency_ms": 2200 if degraded else 800,
                    "probe_performance_score": 40 if degraded else 95,
                }
            )
    return pd.DataFrame(rows)


class RiskPolicyTests(unittest.TestCase):
    def test_policy_validates_threshold_direction(self) -> None:
        policy = RiskPolicy.from_mapping(POLICY_VALUES)
        self.assertEqual(policy.baseline_window_days, 7)
        invalid = dict(POLICY_VALUES, route_switch_risk_threshold=20)
        with self.assertRaisesRegex(ValueError, "切换风险阈值"):
            RiskPolicy.from_mapping(invalid)

    def test_piecewise_risk_starts_at_warning_band(self) -> None:
        self.assertEqual(_ascending_risk(8, 8, 25), 0)
        self.assertGreaterEqual(_ascending_risk(9, 8, 25), 30)
        self.assertEqual(_ascending_risk(25, 8, 25), 100)
        self.assertEqual(_descending_risk(94, 94, 75), 0)
        self.assertEqual(_descending_risk(75, 94, 75), 100)

    def test_workbook_policy_is_loadable(self) -> None:
        policy = load_risk_policy(DEFAULT_CONFIG)
        self.assertEqual(policy.model_side_risk_floor, 80)
        self.assertEqual(policy.minimum_baseline_days, 3)


class HealthRiskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scoring = load_scoring_policy(DEFAULT_CONFIG)
        cls.policy = RiskPolicy.from_mapping(POLICY_VALUES)

    def test_no_history_uses_absolute_signals_and_marks_baseline_not_ready(self) -> None:
        risks = build_health_risks(
            _operating_rows(), _diagnosis_rows(), self.scoring, self.policy
        )
        first = risks[(risks["model_id"] == "model-a")].iloc[0]
        self.assertFalse(first["risk_baseline_ready"])
        self.assertEqual(first["statistical_risk_score"], 0)

    def test_severe_performance_and_success_event_reaches_serious_risk(self) -> None:
        risks = build_health_risks(
            _operating_rows(), _diagnosis_rows(), self.scoring, self.policy
        )
        event = risks[(risks["model_id"] == "model-a")].iloc[-1]
        self.assertEqual(event["performance_risk"], 100)
        self.assertEqual(event["success_risk"], 100)
        self.assertEqual(event["risk_score"], 80)
        self.assertEqual(event["risk_level"], "严重")
        self.assertEqual(event["primary_risk_driver"], "fusion_diagnosis")

    def test_diagnostic_evidence_selects_healthy_cross_provider_candidate(self) -> None:
        risks = build_health_risks(
            _operating_rows(), _diagnosis_rows(), self.scoring, self.policy
        )
        profiles = pd.DataFrame(
            {
                "model_id": ["model-a", "model-b"],
                "routing_readiness_score": [80, 90],
                "confidence_score": [90, 95],
                "recommended_role": ["辅助路由候选", "主路由候选"],
                "dominant_capability": ["reasoning", "tool_call"],
            }
        )
        evidence = build_diagnostic_evidence(risks, profiles, self.policy)
        event = evidence[
            (evidence["model_id"] == "model-a")
            & evidence["date"].eq(pd.Timestamp("2026-06-05"))
        ].iloc[0]
        self.assertEqual(event["switch_recommendation"], "建议切换")
        self.assertEqual(event["target_model_id"], "model-b")
        self.assertIn("双侧同步下降", event["possible_cause"])
        self.assertIn("model-b", event["recommended_action"])


if __name__ == "__main__":
    unittest.main()
