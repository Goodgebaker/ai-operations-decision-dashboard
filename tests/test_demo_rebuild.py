"""Tests for the cross-platform synthetic demo rebuild entry point."""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from src.demo_rebuild import DemoRebuildError, build_rebuild_steps, rebuild_demo_data


class DemoRebuildTests(unittest.TestCase):
    def test_pipeline_varies_all_simulated_sources_without_touching_real_data(self) -> None:
        steps = build_rebuild_steps(100, "python-test")
        commands = [step.command for step in steps]

        self.assertIn(("python-test", "src/generate_sample_data.py", "--seed", "100"), commands)
        self.assertIn(("python-test", "src/generate_synthetic_v2.py", "--days", "90", "--seed", "101"), commands)
        self.assertIn(("python-test", "src/probe_runner.py", "--days", "90", "--seed", "102"), commands)
        self.assertIn(("python-test", "src/capability_calibration.py", "--days", "90", "--seed", "103"), commands)
        flattened = " ".join(part for command in commands for part in command)
        self.assertNotIn("newdata", flattened)
        self.assertNotIn("resource_model_timeseries", flattened)
        self.assertNotIn("resource_instance_hourly", flattened)

    @patch("src.demo_rebuild.subprocess.run")
    def test_rebuild_runs_the_complete_pipeline_and_reports_progress(self, run_mock) -> None:
        run_mock.return_value = subprocess.CompletedProcess([], 0, "ok", "")
        progress: list[tuple[int, int, str]] = []

        seed = rebuild_demo_data(
            seed=321,
            project_root=Path("project"),
            python_executable="python-test",
            on_step=lambda index, total, label: progress.append((index, total, label)),
        )

        self.assertEqual(321, seed)
        self.assertEqual(len(build_rebuild_steps(321)), run_mock.call_count)
        self.assertEqual(1, progress[0][0])
        self.assertEqual(len(progress), progress[-1][0])

    @patch("src.demo_rebuild.subprocess.run")
    def test_rebuild_stops_and_surfaces_the_failed_step(self, run_mock) -> None:
        run_mock.return_value = subprocess.CompletedProcess([], 1, "", "boom")

        with self.assertRaisesRegex(DemoRebuildError, "生成基础模拟调用日志失败.*boom"):
            rebuild_demo_data(seed=1, project_root=Path("project"))


if __name__ == "__main__":
    unittest.main()
