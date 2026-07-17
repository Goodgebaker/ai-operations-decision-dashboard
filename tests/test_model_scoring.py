from __future__ import annotations

import unittest

from src.model_scoring import (
    ScoringPolicy,
    calculate_family_score,
    classify_score,
    clamp_score,
    score_higher_is_better,
    score_lower_is_better,
    score_volatility,
)


def component(
    policy_id: str,
    family: str,
    name: str,
    weight: float,
    direction: str = "passthrough",
    target: float | None = None,
    tolerance: float | None = None,
) -> dict[str, object]:
    return {
        "policy_id": policy_id,
        "policy_type": "component",
        "score_family": family,
        "component": name,
        "direction": direction,
        "weight": weight,
        "target_value": target,
        "tolerance_value": tolerance,
        "status": "active",
    }


def band(
    policy_id: str,
    family: str,
    minimum: float,
    maximum: float,
    label: str,
) -> dict[str, object]:
    return {
        "policy_id": policy_id,
        "policy_type": "band",
        "score_family": family,
        "min_score": minimum,
        "max_score": maximum,
        "label_cn": label,
        "status": "active",
    }


class PrimitiveScoreTests(unittest.TestCase):
    def test_clamp_score_covers_both_boundaries(self) -> None:
        self.assertEqual(clamp_score(-1), 0)
        self.assertEqual(clamp_score(101), 100)

    def test_higher_is_better_reaches_target_and_caps_at_100(self) -> None:
        self.assertEqual(score_higher_is_better(99, 99), 100)
        self.assertEqual(score_higher_is_better(120, 99), 100)
        self.assertAlmostEqual(score_higher_is_better(49.5, 99), 50)

    def test_lower_is_better_is_full_score_at_or_below_target(self) -> None:
        self.assertEqual(score_lower_is_better(0, 100), 100)
        self.assertEqual(score_lower_is_better(100, 100), 100)
        self.assertEqual(score_lower_is_better(200, 100), 50)

    def test_volatility_declines_linearly_to_zero(self) -> None:
        self.assertEqual(score_volatility(0, 0.5), 100)
        self.assertEqual(score_volatility(0.25, 0.5), 50)
        self.assertEqual(score_volatility(0.5, 0.5), 0)
        self.assertEqual(score_volatility(0.8, 0.5), 0)

    def test_non_finite_and_invalid_values_are_rejected(self) -> None:
        for value in (float("nan"), float("inf"), True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    clamp_score(value)
        with self.assertRaises(ValueError):
            score_lower_is_better(-1, 100)
        with self.assertRaises(ValueError):
            score_higher_is_better(1, 0)


class ScoringPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = ScoringPolicy.from_rows(
            [
                component("LAT-50", "latency", "p50", 0.25, "lower_better", 100),
                component("LAT-95", "latency", "p95", 0.50, "lower_better", 200),
                component("LAT-99", "latency", "p99", 0.25, "lower_better", 400),
                component(
                    "STAB-LAT",
                    "stability",
                    "latency_cv",
                    0.60,
                    "volatility",
                    tolerance=0.5,
                ),
                component(
                    "STAB-SR",
                    "stability",
                    "success_std",
                    0.40,
                    "volatility",
                    tolerance=10,
                ),
                component("HEALTH-S", "health", "success_score", 0.35),
                component("HEALTH-P", "health", "performance_score", 0.25),
                component("HEALTH-ST", "health", "stability_score", 0.25),
                component("HEALTH-C", "health", "cost_efficiency_score", 0.15),
                band("HB-1", "health", 0, 50, "高风险"),
                band("HB-2", "health", 50, 70, "需关注"),
                band("HB-3", "health", 70, 85, "健康"),
                band("HB-4", "health", 85, 100, "优秀"),
            ]
        )

    def test_family_score_transforms_and_weights_components(self) -> None:
        score = calculate_family_score(
            "latency",
            {"p50": 100, "p95": 400, "p99": 800},
            self.policy,
        )
        self.assertEqual(score, 62.5)

    def test_passthrough_family_score_is_weighted_average(self) -> None:
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

    def test_missing_component_is_rejected(self) -> None:
        with self.assertRaises(KeyError):
            calculate_family_score("latency", {"p50": 100}, self.policy)

    def test_weight_sum_must_equal_one(self) -> None:
        rows = [
            component("A", "broken", "one", 0.6),
            component("B", "broken", "two", 0.3),
        ]
        with self.assertRaisesRegex(ValueError, "权重合计必须为1"):
            ScoringPolicy.from_rows(rows)

    def test_duplicate_component_is_rejected(self) -> None:
        rows = [
            component("A", "broken", "same", 0.5),
            component("B", "broken", "same", 0.5),
        ]
        with self.assertRaisesRegex(ValueError, "评分组件重复"):
            ScoringPolicy.from_rows(rows)

    def test_band_boundaries_are_unambiguous(self) -> None:
        self.assertEqual(classify_score("health", 49.999, self.policy), "高风险")
        self.assertEqual(classify_score("health", 50, self.policy), "需关注")
        self.assertEqual(classify_score("health", 70, self.policy), "健康")
        self.assertEqual(classify_score("health", 85, self.policy), "优秀")
        self.assertEqual(classify_score("health", 100, self.policy), "优秀")

    def test_bands_must_cover_zero_to_one_hundred_without_gaps(self) -> None:
        rows = [
            component("A", "health", "score", 1.0),
            band("B1", "health", 0, 40, "低"),
            band("B2", "health", 50, 100, "高"),
        ]
        with self.assertRaisesRegex(ValueError, "空档或重叠"):
            ScoringPolicy.from_rows(rows)


if __name__ == "__main__":
    unittest.main()
