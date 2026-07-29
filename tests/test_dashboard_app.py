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
                    self.assertIn("查看完整能力数据", [item.label for item in app.expander])
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
                    metric_labels = [item.label for item in app.metric]
                    self.assertIn("高风险模型", metric_labels)
                    self.assertIn("接近容量上限", metric_labels)
                    self.assertIn("当前受影响请求", metric_labels)
                    markdown_values = [item.value for item in app.markdown]
                    self.assertIn("### 模型处理清单", markdown_values)
                    self.assertIn("### 选中模型详情", markdown_values)
                    self.assertTrue(any("当前业务影响" in value for value in markdown_values))
                    self.assertTrue(any("建议操作" in value for value in markdown_values))
                    button_groups = list(app.get("button_group"))
                    risk_detail = next(
                        item for item in button_groups if item.label == "风险详情"
                    )
                    self.assertEqual(["处理建议", "高负载实例"], risk_detail.options)
                    self.assertTrue(
                        any("涉及：DeepSeek-V4" in item.value for item in app.caption)
                    )
                    self.assertIn("查看模型", [item.label for item in app.selectbox])
                    for indicator in ("NPU 峰值", "HBM 余量", "忙时并发", "等待队列"):
                        self.assertIn(f"**{indicator}**", markdown_values)
                    app.session_state["capacity_technical_details"] = True
                    app.run()
                    self.assertEqual([], list(app.exception))
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

    def test_peer_comparisons_use_the_same_window_for_every_model(self) -> None:
        app = AppTest.from_file(
            str(PROJECT_ROOT / "dashboard" / "app.py"),
            default_timeout=30,
        ).run()

        for module, selector_label in (
            ("性能诊断", "诊断模型"),
            ("成本分析", "成本分析模型"),
        ):
            next(button for button in app.sidebar.button if button.label == module).click().run()
            time_range = next(
                item for item in app.segmented_control if item.label == "时间范围"
            )
            time_range.set_value("过去 7 天").run()
            comparisons: list[str] = []
            selector = next(item for item in app.selectbox if item.label == selector_label)
            for model in selector.options:
                selector = next(item for item in app.selectbox if item.label == selector_label)
                selector.set_value(model).run()
                comparisons.append(
                    next(
                        item.value
                        for item in app.caption
                        if "同窗口模型平均" in item.value
                    )
                )

            self.assertTrue(any(text.startswith("高于") for text in comparisons))
            self.assertTrue(any(text.startswith("低于") for text in comparisons))

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

    def test_capacity_action_selects_the_requested_model(self) -> None:
        app = AppTest.from_file(
            str(PROJECT_ROOT / "dashboard" / "app.py"),
            default_timeout=30,
        ).run()
        next(
            button for button in app.sidebar.button if button.label == "容量诊断"
        ).click().run()

        self.assertEqual("DeepSeek-V4", app.selectbox[0].value)
        next(
            button
            for button in app.button
            if button.key == "capacity_select_Qwen3.6-35B-A3B"
        ).click().run()

        self.assertEqual([], list(app.exception))
        self.assertEqual("Qwen3.6-35B-A3B", app.selectbox[0].value)
        button_groups = list(app.get("button_group"))
        time_range = next(item for item in button_groups if item.label == "趋势时间范围")
        self.assertEqual(["过去 7 天", "过去 30 天", "过去 90 天"], time_range.options)


if __name__ == "__main__":
    unittest.main()
