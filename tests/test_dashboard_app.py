"""Streamlit 五个可见模块页面的最小回归测试。"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DashboardSmokeTests(unittest.TestCase):
    def test_cloud_style_entrypoint_resolves_project_modules(self) -> None:
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

        result = subprocess.run(
            [sys.executable, "app.py"],
            cwd=PROJECT_ROOT / "dashboard",
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

        self.assertEqual(
            0,
            result.returncode,
            msg=(result.stderr or result.stdout)[-4000:],
        )

    def test_all_five_visible_modules_render_without_exception(self) -> None:
        expected_headings = {
            "运营总览": "运营总览",
            "性能诊断": "性能诊断",
            "成本分析": "成本分析",
            "能力校准": "主动拨测与模型能力校准",
            "资源与容量诊断": "资源与容量诊断",
        }
        app = AppTest.from_file(
            str(PROJECT_ROOT / "dashboard" / "app.py"),
            default_timeout=30,
        ).run()

        for module, expected_heading in expected_headings.items():
            with self.subTest(module=module):
                navigation_button = next(
                    button for button in app.sidebar.button if button.label == module
                )
                navigation_button.click().run()
                self.assertEqual([], list(app.exception))
                self.assertIn(expected_heading, [item.value for item in app.subheader])
                if module == "能力校准":
                    metric_labels = [item.label for item in app.metric]
                    self.assertIn("综合路由评分", metric_labels)
                    self.assertIn("真实表现指数", metric_labels)
                    self.assertIn("主动拨测指数", metric_labels)
                    self.assertNotIn("可信度评分", metric_labels)
                    self.assertIn(
                        "查看原始诊断证据与完整数据",
                        [item.label for item in app.expander],
                    )
                if module == "资源与容量诊断":
                    self.assertIn("最新真实数据", [item.label for item in app.metric])
                    self.assertIn("资源趋势指标", [item.label for item in app.segmented_control])

    def test_hidden_modules_are_not_in_navigation(self) -> None:
        app = AppTest.from_file(
            str(PROJECT_ROOT / "dashboard" / "app.py"),
            default_timeout=30,
        ).run()
        self.assertEqual([], list(app.exception))
        labels = [button.label for button in app.sidebar.button]
        self.assertNotIn("智能检测", labels)
        self.assertNotIn("诊断解释", labels)
        self.assertEqual(5, len([label for label in labels if label in {
            "运营总览", "性能诊断", "成本分析", "能力校准", "资源与容量诊断"
        }]))

    def test_overview_exposes_decision_and_external_capacity_context(self) -> None:
        app = AppTest.from_file(
            str(PROJECT_ROOT / "dashboard" / "app.py"),
            default_timeout=30,
        ).run()

        self.assertEqual([], list(app.exception))
        self.assertIn("今日决策摘要", [item.value for item in app.subheader])
        self.assertIn("外部容量基准", [item.value for item in app.subheader])
        self.assertIn("观察窗口", [item.label for item in app.segmented_control])
        self.assertIn("趋势指标", [item.label for item in app.segmented_control])
        self.assertIn("最高稳定测试并发", [item.label for item in app.metric])


if __name__ == "__main__":
    unittest.main()
