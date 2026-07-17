"""智能检测会话策略和动态事件测试。"""

from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from src.interactive_risk_policy import (
    build_signal_events,
    build_unknown_pattern_events,
    merge_signal_rule_table,
    risk_policy_mapping,
    scoring_policy_with_risk_bands,
    signal_rule_table,
)
from src.model_health_risk import RiskPolicy
from src.model_scoring import classify_score, load_scoring_policy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "docs" / "ai_monitoring_metric_dictionary.xlsx"


class InteractiveRiskPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy_frame = pd.read_excel(CONFIG, sheet_name="Risk Policy")
        cls.default_values = risk_policy_mapping(cls.policy_frame)

    def test_signal_editor_updates_policy_without_changing_rule_identity(self) -> None:
        editor = signal_rule_table(self.default_values)
        editor.loc[
            editor["检测信号"].eq("成功率相对基线下降"), "预警阈值"
        ] = 3.5
        updated = merge_signal_rule_table(self.default_values, editor)
        policy = RiskPolicy.from_mapping(updated)
        self.assertEqual(3.5, policy.success_drop_warning_points)
        self.assertEqual(set(self.default_values), set(updated))

    def test_invalid_signal_threshold_relationship_is_rejected(self) -> None:
        editor = signal_rule_table(self.default_values)
        target = editor["检测信号"].eq("P95 延迟相对基线上升")
        editor.loc[target, "预警阈值"] = 3.0
        editor.loc[target, "严重阈值"] = 2.0
        with self.assertRaisesRegex(ValueError, "必须高于"):
            merge_signal_rule_table(self.default_values, editor)

    def test_custom_risk_bands_replace_only_risk_classification(self) -> None:
        base = load_scoring_policy(CONFIG)
        custom = scoring_policy_with_risk_bands(base, 20, 50, 75)
        self.assertEqual("中", classify_score("risk", 25, custom))
        self.assertEqual("严重", classify_score("risk", 80, custom))
        self.assertEqual(base.rules_for("health"), custom.rules_for("health"))

    def test_dynamic_events_cover_specific_signals_and_fusion_diagnosis(self) -> None:
        policy = RiskPolicy.from_mapping(self.default_values)
        risks = pd.DataFrame(
            {
                "date": [pd.Timestamp("2026-06-30")],
                "model_id": ["model-a"],
                "risk_score": [82.0],
                "performance_score_drop_points": [12.0],
                "p95_latency_increase_ratio": [1.6],
                "performance_score": [65.0],
                "success_drop_points": [4.0],
                "success_rate": [92.0],
                "cost_per_request_increase_ratio": [1.3],
                "token_cost_increase_ratio": [1.1],
                "cost_efficiency_drop_points": [10.0],
                "diagnosis_type": ["platform_or_traffic_issue"],
                "diagnosis_severity": ["medium"],
                "diagnosis_reason": ["真实调用下降但主动拨测正常"],
            }
        )
        events = build_signal_events(risks, policy)
        event_types = set(events["event_type"])
        self.assertIn("性能分下降", event_types)
        self.assertIn("P95 长尾延迟突增", event_types)
        self.assertIn("成功率下降", event_types)
        self.assertIn("单请求成本上升", event_types)
        self.assertIn("成本效率下降", event_types)
        self.assertIn("真实调用与拨测表现不一致", event_types)

    def test_unknown_event_requires_configured_algorithm_consensus(self) -> None:
        scores = pd.DataFrame(
            {
                "hour": pd.to_datetime(["2026-06-30 10:00", "2026-06-30 11:00"]),
                "pred_mad": [True, True],
                "pred_stl": [True, False],
                "pred_isolation_forest": [False, False],
                "top_metric_mad": ["error_rate", "total_tokens"],
                "top_metric_stl": ["p95_latency_ms", ""],
            }
        )
        events = build_unknown_pattern_events(scores, minimum_algorithm_votes=2)
        self.assertEqual(1, len(events))
        self.assertEqual("统计未知异常", events.iloc[0]["event_type"])


if __name__ == "__main__":
    unittest.main()
