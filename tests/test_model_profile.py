from __future__ import annotations

import unittest

import pandas as pd

from src.model_profile import (
    DEFAULT_CONFIG,
    DiagnosisPolicy,
    build_daily_probe_metrics,
    build_fusion_diagnosis,
    build_model_profiles,
    load_diagnosis_policy,
)
from src.model_scoring import load_scoring_policy


POLICY_VALUES = {
    "real_success_warning_pct": 95,
    "probe_success_warning_pct": 95,
    "real_performance_warning_score": 70,
    "probe_performance_warning_score": 70,
    "probe_p95_latency_warning_ms": 3000,
    "latency_gap_warning_ratio": 1.6,
    "primary_route_min_score": 85,
    "backup_route_min_score": 70,
    "freshness_decay_per_day": 10,
    "primary_route_min_stability_score": 70,
    "primary_route_min_confidence_score": 80,
}


class DiagnosisPolicyTests(unittest.TestCase):
    def test_policy_validates_threshold_relationships(self) -> None:
        policy = DiagnosisPolicy.from_mapping(POLICY_VALUES)
        self.assertEqual(policy.latency_gap_warning_ratio, 1.6)
        invalid = dict(POLICY_VALUES, primary_route_min_score=60)
        with self.assertRaisesRegex(ValueError, "主路由阈值"):
            DiagnosisPolicy.from_mapping(invalid)

    def test_workbook_policy_is_loadable(self) -> None:
        policy = load_diagnosis_policy(DEFAULT_CONFIG)
        self.assertEqual(policy.primary_route_min_score, 85)
        self.assertEqual(policy.backup_route_min_score, 70)
        self.assertEqual(policy.latency_gap_warning_ratio, 2.8)
        self.assertEqual(policy.real_success_warning_pct, 94)
        self.assertEqual(policy.primary_route_min_stability_score, 70)
        self.assertEqual(policy.primary_route_min_confidence_score, 80)


class ProbeMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scoring_policy = load_scoring_policy(DEFAULT_CONFIG)

    def test_daily_probe_metrics_use_weighted_quality_and_common_latency_policy(self) -> None:
        runs = pd.DataFrame(
            {
                "started_at": pd.to_datetime(["2026-06-01 00:00", "2026-06-01 06:00"]),
                "model_id": ["model-a", "model-a"],
                "provider": ["provider-a", "provider-a"],
                "task_id": ["T1", "T2"],
                "capability_dimension": ["reasoning", "tool_call"],
                "task_weight": [2.0, 1.0],
                "status_code": [200, 200],
                "latency_ms": [1000, 2000],
                "task_score": [100.0, 0.0],
                "passed": [True, False],
                "evaluator_confidence": [100.0, 100.0],
            }
        )
        daily = build_daily_probe_metrics(runs, self.scoring_policy)
        self.assertEqual(len(daily), 1)
        self.assertEqual(daily.iloc[0]["probe_run_count"], 2)
        self.assertAlmostEqual(daily.iloc[0]["probe_quality_score"], 66.6667, places=4)
        self.assertTrue(0 <= daily.iloc[0]["probe_latency_score"] <= 100)
        self.assertTrue(0 <= daily.iloc[0]["probe_performance_score"] <= 100)


class FusionDiagnosisTests(unittest.TestCase):
    def test_all_control_variable_diagnosis_branches(self) -> None:
        dates = pd.date_range("2026-06-01", periods=5, freq="D")
        operating = pd.DataFrame(
            {
                "date": dates,
                "model_id": ["model-a"] * 5,
                "success_rate": [99, 90, 99, 90, 99],
                "p95_latency_ms": [1200, 1200, 1200, 1200, 2000],
                "performance_score": [90, 90, 90, 60, 90],
                "stability_score": [90] * 5,
                "health_score": [90, 70, 90, 60, 90],
            }
        )
        probe = pd.DataFrame(
            {
                "date": dates,
                "model_id": ["model-a"] * 5,
                "provider": ["provider-a"] * 5,
                "probe_http_success_rate": [100, 100, 100, 90, 100],
                "probe_p95_latency_ms": [1000] * 5,
                "probe_performance_score": [90, 90, 60, 60, 90],
                "probe_consistency_score": [90] * 5,
                "probe_quality_score": [95] * 5,
            }
        )
        diagnosis = build_fusion_diagnosis(
            operating, probe, DiagnosisPolicy.from_mapping(POLICY_VALUES)
        )
        self.assertEqual(
            diagnosis["diagnosis_type"].tolist(),
            [
                "healthy",
                "platform_or_traffic_issue",
                "capability_or_probe_issue",
                "model_side_degradation",
                "environment_latency_gap",
            ],
        )
        self.assertEqual(
            diagnosis.loc[diagnosis["diagnosis_type"].eq("model_side_degradation"), "switch_recommendation"].iloc[0],
            "建议切换",
        )


class ProfileIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scoring_policy = load_scoring_policy(DEFAULT_CONFIG)
        cls.diagnosis_policy = load_diagnosis_policy(DEFAULT_CONFIG)

    def test_project_data_produces_three_complete_route_profiles(self) -> None:
        operating = pd.read_csv("outputs/model_operating_scores.csv", parse_dates=["date"])
        runs = pd.read_csv(
            "data/capability_probe_runs.csv",
            parse_dates=["started_at", "completed_at"],
        )
        availability = pd.read_csv(
            "data/probe_runs.csv", parse_dates=["started_at", "completed_at"]
        )
        capability = pd.read_csv(
            "outputs/model_capability_scores.csv", parse_dates=["latest_run_at"]
        )
        daily_probe = build_daily_probe_metrics(
            runs, self.scoring_policy, availability
        )
        diagnosis = build_fusion_diagnosis(
            operating, daily_probe, self.diagnosis_policy
        )
        profiles = build_model_profiles(
            operating,
            capability,
            daily_probe,
            diagnosis,
            self.scoring_policy,
            self.diagnosis_policy,
        )
        self.assertEqual(len(diagnosis), 90)
        self.assertEqual(len(profiles), 3)
        score_columns = [
            "capability_score", "profile_stability_score",
            "profile_performance_score", "confidence_score",
            "routing_readiness_score",
        ]
        self.assertTrue(profiles[score_columns].notna().all().all())
        self.assertTrue(
            profiles[score_columns]
            .apply(lambda column: column.between(0, 100).all())
            .all()
        )
        self.assertEqual(profiles["profile_rank"].tolist(), [1, 2, 3])
        qwen_role = profiles.loc[
            profiles["model_id"].eq("qwen-plus"), "recommended_role"
        ].iloc[0]
        self.assertEqual(qwen_role, "辅助路由候选")
        self.assertEqual(
            set(profiles["dominant_capability"])
            | set(profiles["weakest_capability"]),
            {"instruction_following", "reasoning", "tool_call"},
        )


if __name__ == "__main__":
    unittest.main()
