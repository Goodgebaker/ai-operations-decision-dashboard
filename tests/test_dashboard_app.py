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


if __name__ == "__main__":
    unittest.main()
