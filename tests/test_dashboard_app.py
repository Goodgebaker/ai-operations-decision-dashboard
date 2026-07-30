"""Regression tests for the decision-focused five-module dashboard."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from dashboard.decision_center import build_six_dimension_profiles


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULES = ["运营总览", "性能诊断", "成本分析", "模型画像与路由适配", "容量诊断"]


class DashboardSmokeTests(unittest.TestCase):
    def _app(self) -> AppTest:
        return AppTest.from_file(str(PROJECT_ROOT / "dashboard" / "app.py"), default_timeout=45).run()

    def test_cloud_style_entrypoint_resolves_project_modules(self) -> None:
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
        result = subprocess.run(
            [sys.executable, "app.py"], cwd=PROJECT_ROOT / "dashboard", env=env,
            capture_output=True, text=True, timeout=60, check=False,
        )
        self.assertEqual(0, result.returncode, msg=(result.stderr or result.stdout)[-4000:])

    def test_all_five_modules_render_without_exception(self) -> None:
        app = self._app()
        for module in MODULES:
            with self.subTest(module=module):
                next(button for button in app.sidebar.button if button.label == module).click().run()
                self.assertEqual([], list(app.exception))
                self.assertIn(module, [item.value for item in app.subheader])

    def test_navigation_uses_model_profile_instead_of_capability_calibration(self) -> None:
        app = self._app()
        labels = [button.label for button in app.sidebar.button if str(button.key).startswith("nav_")]
        self.assertEqual(MODULES, labels)
        self.assertNotIn("能力校准", labels)

    def test_overview_places_conclusion_ranking_and_breakdown_first(self) -> None:
        app = self._app()
        self.assertEqual([], list(app.exception))
        metrics = [item.label for item in app.metric]
        for label in ("综合健康", "成功率评分", "性能评分", "成本效率评分"):
            self.assertIn(label, metrics)
        markdown = [item.value for item in app.markdown]
        self.assertIn("### 当前健康结论", markdown)
        self.assertIn("#### 最新模型健康排行", markdown)
        self.assertIn("#### 排名依据 · 三项评分横向对比", markdown)
        self.assertIn("### 四项评分趋势", markdown)
        self.assertIn("#### 观察窗口上下文 · 不参与健康评分", markdown)
        self.assertNotIn("调用量", metrics)
        self.assertNotIn("Token 消耗", metrics)
        self.assertEqual(0, len(app.dataframe))

    def test_performance_has_two_dominant_diagnostic_zones(self) -> None:
        app = self._app()
        next(button for button in app.sidebar.button if button.label == "性能诊断").click().run()
        self.assertEqual([], list(app.exception))
        markdown = [item.value for item in app.markdown]
        self.assertIn("### 01 · 响应速度", markdown)
        self.assertIn("### 02 · 稳定性", markdown)
        metrics = [item.label for item in app.metric]
        self.assertIn("响应速度得分", metrics)
        self.assertIn("稳定性得分", metrics)

    def test_cost_focuses_on_quality_gated_efficiency(self) -> None:
        app = self._app()
        next(button for button in app.sidebar.button if button.label == "成本分析").click().run()
        self.assertEqual([], list(app.exception))
        metrics = [item.label for item in app.metric]
        self.assertIn("最省达标模型", metrics)
        self.assertIn("成本效率评分", metrics)
        markdown = [item.value for item in app.markdown]
        self.assertIn("### 01 · 钱花在哪里", markdown)
        self.assertIn("### 02 · 是否值得花", markdown)

    def test_pdf_six_dimension_scores_are_reproduced(self) -> None:
        benchmarks = pd.read_csv(PROJECT_ROOT / "data" / "external_model_benchmarks.csv")
        providers, models = build_six_dimension_profiles(benchmarks)
        scores = providers.set_index("provider")
        self.assertEqual(15, len(models))
        self.assertAlmostEqual(72.0, scores.loc["Minimax", "overall"], places=1)
        self.assertAlmostEqual(58.9, scores.loc["腾讯", "overall"], places=1)
        self.assertAlmostEqual(44.4, scores.loc["阿里百炼", "overall"], places=1)
        self.assertAlmostEqual(20.6, scores.loc["阿里百炼", "请求成功率"], places=1)
        self.assertAlmostEqual(71.4, scores.loc["阿里百炼", "生成速度"], places=1)

    def test_model_profile_separates_operating_dimensions_from_capability(self) -> None:
        app = self._app()
        next(button for button in app.sidebar.button if button.label == "模型画像与路由适配").click().run()
        self.assertEqual([], list(app.exception))
        markdown = [item.value for item in app.markdown]
        self.assertIn("### 六维画像横向对比", markdown)
        self.assertIn("### 路由决策矩阵", markdown)
        self.assertIn("### 任务级能力评测 · 补充证据", markdown)
        self.assertGreaterEqual(len(app.dataframe), 1)

    def test_capacity_is_ordered_as_conclusion_evidence_action(self) -> None:
        app = self._app()
        next(button for button in app.sidebar.button if button.label == "容量诊断").click().run()
        self.assertEqual([], list(app.exception))
        markdown = [item.value for item in app.markdown]
        self.assertIn("### 当前容量结论", markdown)
        self.assertIn("### 01 · 风险证据", markdown)
        self.assertIn("### 02 · 路由动作", markdown)
        self.assertIn("查看容量指标", [item.label for item in app.segmented_control])


if __name__ == "__main__":
    unittest.main()
