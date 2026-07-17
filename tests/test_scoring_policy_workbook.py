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
                "stability_score": 60,
                "cost_efficiency_score": 40,
            },
            self.policy,
        )
        self.assertEqual(score, 76)
        self.assertEqual(classify_score("health", score, self.policy), "健康")

    def test_risk_band_boundaries_match_the_dictionary(self) -> None:
        self.assertEqual(classify_score("risk", 29.999, self.policy), "低")
        self.assertEqual(classify_score("risk", 30, self.policy), "中")
        self.assertEqual(classify_score("risk", 60, self.policy), "高")
        self.assertEqual(classify_score("risk", 80, self.policy), "严重")


if __name__ == "__main__":
    unittest.main()
