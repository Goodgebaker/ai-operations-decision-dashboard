from __future__ import annotations

import unittest

import pandas as pd

from src.capability_calibration import (
    DEFAULT_CONFIG,
    CapabilityTask,
    ModelTarget,
    build_dimension_scores,
    evaluate_task_result,
    load_calibration_config,
    simulate_history,
    validate_symmetric_results,
)


def task(
    task_id: str,
    dimension: str,
    evaluator: str,
    expected: str,
) -> CapabilityTask:
    return CapabilityTask(
        task_id=task_id,
        task_name_cn=task_id,
        capability_dimension=dimension,
        difficulty="basic",
        prompt_template=f"prompt-{task_id}",
        evaluator_type=evaluator,
        expected_value=expected,
        pass_threshold=100,
        task_weight=1,
        interval_hours=12,
        repeat_count=2,
        max_tokens=32,
        timeout_ms=5000,
        expected_output_version="1.0",
        version="0.7.0",
    )


class RuleEvaluatorTests(unittest.TestCase):
    def test_exact_match_is_strict_but_ignores_outer_whitespace(self) -> None:
        current = task("T1", "instruction_following", "exact_match", "OK")
        self.assertEqual(evaluate_task_result(current, " OK ")[:2], (100, True))
        self.assertEqual(evaluate_task_result(current, "ok")[:2], (0, False))

    def test_json_evaluator_scores_partial_field_matches(self) -> None:
        current = task(
            "T2",
            "structured_output",
            "json_exact",
            '{"status":"ok","count":2}',
        )
        score, passed, evidence = evaluate_task_result(
            current, '{"status":"ok","count":1}'
        )
        self.assertEqual(score, 50)
        self.assertFalse(passed)
        self.assertIn("1/2", evidence)

    def test_numeric_evaluator_extracts_number_from_short_answer(self) -> None:
        current = task("T3", "reasoning", "numeric_exact", "391")
        self.assertEqual(
            evaluate_task_result(current, "答案是 391。")[:2], (100, True)
        )

    def test_tool_and_http_failures_are_explained(self) -> None:
        current = task("T4", "tool_call", "tool_name", "get_weather")
        self.assertEqual(
            evaluate_task_result(current, "", "get_weather")[:2], (100, True)
        )
        score, passed, evidence = evaluate_task_result(
            current, "", "get_weather", status_code=503
        )
        self.assertEqual(score, 0)
        self.assertFalse(passed)
        self.assertEqual(evidence, "HTTP 503")


class SymmetricMatrixTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tasks = [
            task("T1", "instruction_following", "exact_match", "OK"),
            task("T2", "structured_output", "json_exact", '{"ok":true}'),
            task("T3", "reasoning", "numeric_exact", "391"),
            task("T4", "tool_call", "tool_name", "get_weather"),
        ]
        self.targets = [
            ModelTarget("Minimax-M2.5", "MiniMax", "cn-east", "KEY1"),
            ModelTarget("Qwen3.6-35B-A3B", "Qwen", "cn-east", "KEY2"),
            ModelTarget("DeepSeek-V4", "DeepSeek", "cn-east", "KEY3"),
        ]

    def test_simulation_is_deterministic_and_balanced(self) -> None:
        first = simulate_history(
            self.tasks, self.targets, pd.Timestamp("2026-06-01"), 2, 42
        )
        second = simulate_history(
            self.tasks, self.targets, pd.Timestamp("2026-06-01"), 2, 42
        )
        pd.testing.assert_frame_equal(first, second)
        counts = first.groupby(["model_id", "task_id"]).size().unstack()
        self.assertTrue((counts == 8).all().all())
        self.assertEqual(set(first["traffic_type"]), {"capability_probe"})
        self.assertEqual(first["input_hash"].str.len().unique().tolist(), [16])

    def test_asymmetric_results_are_rejected(self) -> None:
        frame = simulate_history(
            self.tasks, self.targets, pd.Timestamp("2026-06-01"), 1, 7
        )
        broken = frame.drop(frame.index[0])
        with self.assertRaisesRegex(ValueError, "样本数不一致"):
            validate_symmetric_results(broken)

    def test_dimension_scores_have_one_row_per_model_and_dimension(self) -> None:
        frame = simulate_history(
            self.tasks, self.targets, pd.Timestamp("2026-06-01"), 2, 9
        )
        scores = build_dimension_scores(frame)
        self.assertEqual(len(scores), 12)
        self.assertTrue(scores["quality_score"].between(0, 100).all())
        self.assertTrue(scores["consistency_score"].between(0, 100).all())
        self.assertTrue((scores["run_count"] == 8).all())


class WorkbookConfigurationTests(unittest.TestCase):
    def test_workbook_defines_four_tasks_for_three_models(self) -> None:
        tasks, targets = load_calibration_config(DEFAULT_CONFIG)
        self.assertEqual(
            {task.capability_dimension for task in tasks},
            {
                "instruction_following",
                "structured_output",
                "reasoning",
                "tool_call",
            },
        )
        self.assertEqual(len(tasks), 4)
        self.assertEqual(len(targets), 3)
        self.assertEqual(
            {target.model_id for target in targets},
            {"DeepSeek-V4", "Minimax-M2.5", "Qwen3.6-35B-A3B"},
        )
        self.assertTrue(all(task.version == "0.7.0" for task in tasks))


if __name__ == "__main__":
    unittest.main()
