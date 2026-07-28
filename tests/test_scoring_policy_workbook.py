from __future__ import annotations

from pathlib import Path
import unittest

from src.model_scoring import calculate_family_score, classify_score, load_scoring_policy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "docs" / "ai_monitoring_metric_dictionary.xlsx"


class ScoringPolicyWorkbookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = load_scoring_policy(CONFIG_PATH)

    def test_all_planned_score_families_are_available(self) -> None:
        expected = {
            "latency",
            "stability",
            "performance",
            "success",
            "cost_efficiency",
            "cost_performance",
            "health",
            "risk",
            "confidence",
        }
        actual = {rule.score_family for rule in self.policy.component_rules}
        self.assertTrue(expected.issubset(actual))

    def test_configured_health_weights_produce_expected_score(self) -> None:
        score = calculate_family_score(
            "health",
            {
                "success_score": 100,
                "performance_score": 80,
                "cost_efficiency_score": 40,
            },
            self.policy,
        )
        self.assertEqual(score, 81)
        health_components = {
            rule.component for rule in self.policy.rules_for("health")
        }
        self.assertNotIn("stability_score", health_components)
        self.assertEqual(classify_score("health", score, self.policy), "健康")

    def test_risk_band_boundaries_match_the_dictionary(self) -> None:
        self.assertEqual(classify_score("risk", 29.999, self.policy), "低")
        self.assertEqual(classify_score("risk", 30, self.policy), "中")
        self.assertEqual(classify_score("risk", 60, self.policy), "高")
        self.assertEqual(classify_score("risk", 80, self.policy), "严重")

    def test_relative_baselines_center_normal_performance_and_cost_near_80(self) -> None:
        latency = calculate_family_score(
            "latency",
            {
                "p50_latency_ratio": 1.0,
                "p95_latency_ratio": 1.0,
                "p99_latency_ratio": 1.0,
            },
            self.policy,
        )
        cost_efficiency = calculate_family_score(
            "cost_efficiency",
            {
                "cost_per_request_ratio": 1.0,
                "cost_per_1k_tokens_ratio": 1.0,
                "tokens_per_request_ratio": 1.0,
            },
            self.policy,
        )
        cost_total = calculate_family_score(
            "cost_performance",
            {"quality_score": 95.0, "cost_efficiency_score": cost_efficiency},
            self.policy,
        )

        self.assertEqual(latency, 80)
        self.assertEqual(cost_efficiency, 80)
        self.assertEqual(cost_total, 84.5)
        cost_weights = {
            rule.component: rule.weight
            for rule in self.policy.rules_for("cost_performance")
        }
        self.assertEqual(cost_weights, {"quality_score": 0.3, "cost_efficiency_score": 0.7})
        self.assertTrue(
            all(
                rule.tolerance_value == 0.25
                for rule in self.policy.rules_for("cost_efficiency")
            )
        )


if __name__ == "__main__":
    unittest.main()
