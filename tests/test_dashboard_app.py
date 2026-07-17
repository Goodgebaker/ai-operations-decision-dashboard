"""Streamlit 六模块页面的最小回归测试。"""

from __future__ import annotations

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DashboardSmokeTests(unittest.TestCase):
    def test_all_six_modules_render_without_exception(self) -> None:
        expected_headings = {
            "运营总览": "运营总览",
            "性能诊断": "性能诊断",
            "成本分析": "成本分析",
            "能力校准": "主动拨测与模型能力校准",
            "智能检测": "智能检测",
            "诊断解释": "智能诊断解释中心",
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
                if module == "智能检测":
                    self.assertIn(
                        "检测策略配置",
                        [item.label for item in app.expander],
                    )
                    self.assertIn(
                        "动态风险事件",
                        [item.label for item in app.metric],
                    )

    def test_detection_policy_can_be_applied_in_the_dashboard(self) -> None:
        app = AppTest.from_file(
            str(PROJECT_ROOT / "dashboard" / "app.py"),
            default_timeout=30,
        ).run()
        next(
            button for button in app.sidebar.button if button.label == "智能检测"
        ).click().run()
        next(
            item for item in app.number_input if item.label == "中风险起点"
        ).set_value(25.0)
        next(
            button for button in app.button if button.label == "应用策略并回放"
        ).click().run()

        self.assertEqual([], list(app.exception))
        self.assertEqual(
            25.0,
            app.session_state["detection_risk_bands"]["medium"],
        )
        self.assertIn(
            "自定义策略已应用，风险、事件和诊断证据已重新计算。",
            [item.value for item in app.success],
        )


if __name__ == "__main__":
    unittest.main()
