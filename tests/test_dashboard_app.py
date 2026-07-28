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
            "能力校准": "模型能力校准",
            "容量诊断": "容量诊断",
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
                    self.assertIn("真实表现", metric_labels)
                    self.assertIn("主动检查", metric_labels)
                    self.assertNotIn("主动拨测指数", metric_labels)
                    self.assertNotIn("可信度评分", metric_labels)
                    self.assertNotIn("评分依据", [item.label for item in app.expander])
                    self.assertIn("查看标准化测试明细", [item.label for item in app.expander])
                    self.assertIn("查看完整模型画像", [item.label for item in app.expander])
                    self.assertIn(
                        "查看原始诊断证据与完整数据",
                        [item.label for item in app.expander],
                    )
                    self.assertIn(
                        "查看近 7 次诊断轨迹",
                        [item.label for item in app.expander],
                    )
                if module == "性能诊断":
                    self.assertIn("时间范围", [item.label for item in app.segmented_control])
                    time_range = next(
                        item for item in app.segmented_control if item.label == "时间范围"
                    )
                    self.assertEqual(
                        ["过去 7 天", "过去 30 天", "过去 90 天"],
                        time_range.options,
                    )
                    self.assertIn("添加对比模型", [item.label for item in app.selectbox])
                    self.assertIn("与上一周期对比", [item.label for item in app.toggle])
                    metric_labels = [item.label for item in app.metric]
                    self.assertIn("响应速度", metric_labels)
                    self.assertIn("稳定性", metric_labels)
                    self.assertIn("P50", metric_labels)
                    self.assertIn("P95", metric_labels)
                    self.assertIn("P99", metric_labels)
                    self.assertIn(
                        "下载当前窗口原始数据",
                        [item.label for item in app.get("download_button")],
                    )
                if module == "成本分析":
                    self.assertIn("时间范围", [item.label for item in app.segmented_control])
                    self.assertIn("添加对比模型", [item.label for item in app.selectbox])
                    self.assertIn("与上一周期对比", [item.label for item in app.toggle])
                    metric_labels = [item.label for item in app.metric]
                    self.assertIn("成本效率", metric_labels)
                    self.assertIn("质量保障", metric_labels)
                    self.assertIn("单请求成本", metric_labels)
                    self.assertIn("千 Token 成本", metric_labels)
                    self.assertIn("历史基线倍数", metric_labels)
                    self.assertIn(
                        "下载当前窗口原始数据",
                        [item.label for item in app.get("download_button")],
                    )
                if module == "容量诊断":
                    self.assertIn("资源趋势指标", [item.label for item in app.segmented_control])
                    self.assertIn(
                        "查看完整容量与资源明细",
                        [item.label for item in app.expander],
                    )
                    self.assertIn(
                        "下载容量诊断原始数据",
                        [item.label for item in app.get("download_button")],
                    )

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
            "运营总览", "性能诊断", "成本分析", "能力校准", "容量诊断"
        }]))

    def test_overview_exposes_simplified_decision_context(self) -> None:
        app = AppTest.from_file(
            str(PROJECT_ROOT / "dashboard" / "app.py"),
            default_timeout=30,
        ).run()

        self.assertEqual([], list(app.exception))
        self.assertNotIn("今日决策摘要", [item.value for item in app.subheader])
        self.assertNotIn("外部容量基准", [item.value for item in app.subheader])
        self.assertIn("观察窗口", [item.label for item in app.segmented_control])
        overview_window = next(
            item for item in app.segmented_control if item.label == "观察窗口"
        )
        self.assertIn("近 1 天", overview_window.options)
        self.assertIn("选择下方趋势图的指标", [item.label for item in app.segmented_control])
        self.assertNotIn("最高稳定测试并发", [item.label for item in app.metric])
        self.assertNotIn("健康趋势范围", [item.label for item in app.segmented_control])
        self.assertIn("模型总体健康指数", [item.value for item in app.caption])
        self.assertIn("成功率评分", [item.label for item in app.metric])
        self.assertIn("性能评分", [item.label for item in app.metric])
        self.assertIn("成本评分", [item.label for item in app.metric])
        self.assertNotIn("成本效率评分", [item.label for item in app.metric])
        self.assertIn("风险分析与推荐动作", [item.value for item in app.caption])
        self.assertNotIn("稳定性评分", [item.label for item in app.metric])
        ranking_styles = app.dataframe[0].proto.arrow_data.styler.styles
        self.assertIn("#dcfce7", ranking_styles)
        self.assertTrue(
            any(color in ranking_styles for color in ("#fef9c3", "#ffedd5", "#fee2e2", "#dcfce7"))
        )

    def test_overall_health_gauge_ignores_model_filter(self) -> None:
        app = AppTest.from_file(
            str(PROJECT_ROOT / "dashboard" / "app.py"),
            default_timeout=30,
        ).run()

        self.assertEqual([], list(app.exception))
        gauge_before = app.get("vega_lite_chart")[0].proto.spec
        app.multiselect[0].set_value(["DeepSeek-V4"])
        app.run()

        self.assertEqual([], list(app.exception))
        gauge_after = app.get("vega_lite_chart")[0].proto.spec
        self.assertEqual(gauge_before, gauge_after)

    def test_performance_comparison_uses_available_model_data(self) -> None:
        app = AppTest.from_file(
            str(PROJECT_ROOT / "dashboard" / "app.py"),
            default_timeout=30,
        ).run()
        next(
            button for button in app.sidebar.button if button.label == "性能诊断"
        ).click().run()

        comparison = next(
            item for item in app.selectbox if item.label == "添加对比模型"
        )
        comparison.select("Minimax-M2.5").run()

        self.assertEqual([], list(app.exception))
        self.assertIn(
            "#### 与 Minimax-M2.5 对比",
            [item.value for item in app.markdown],
        )

    def test_cost_comparison_uses_available_model_data(self) -> None:
        app = AppTest.from_file(
            str(PROJECT_ROOT / "dashboard" / "app.py"),
            default_timeout=30,
        ).run()
        next(
            button for button in app.sidebar.button if button.label == "成本分析"
        ).click().run()

        comparison = next(
            item for item in app.selectbox if item.label == "添加对比模型"
        )
        comparison.select("Minimax-M2.5").run()

        self.assertEqual([], list(app.exception))
        self.assertIn(
            "#### 与 Minimax-M2.5 对比",
            [item.value for item in app.markdown],
        )


if __name__ == "__main__":
    unittest.main()
