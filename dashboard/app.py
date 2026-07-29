"""面向多模型智能路由的 AI 中台运营决策实验台。"""

from __future__ import annotations

import sys
from html import escape
from pathlib import Path


# Streamlit Community Cloud executes this file from the dashboard directory,
# which does not automatically place the repository root on sys.path. Bootstrap
# the root before importing the project's src package so local and cloud startup
# use the same module resolution behavior.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

_CAPACITY_DETAIL_SCROLL = st.components.v2.component(
    "capacity_detail_scroll",
    html="""<span aria-hidden="true" style="display:block;height:1px"></span>""",
    js="""
export default function ({ data, parentElement }) {
  if (!data?.enabled) return
  const root = parentElement.getRootNode()
  const target = root?.host ?? parentElement
  requestAnimationFrame(() => {
    setTimeout(() => target.scrollIntoView({ behavior: "smooth", block: "start" }), 80)
  })
}
""",
)

from src.external_benchmarks import build_capacity_profiles
from src.demo_rebuild import DemoRebuildError, rebuild_demo_data
from src.interactive_risk_policy import (
    build_signal_events,
    build_unknown_pattern_events,
    merge_signal_rule_table,
    risk_policy_mapping,
    scoring_policy_with_risk_bands,
    signal_rule_table,
)
from src.model_health_risk import RiskPolicy, build_diagnostic_evidence, build_health_risks
from src.model_scoring import load_scoring_policy

PATHS = {
    "logs": PROJECT_ROOT / "data" / "synthetic_logs_v2.csv",
    "truth": PROJECT_ROOT / "data" / "ground_truth.csv",
    "key_features": PROJECT_ROOT / "outputs" / "features" / "key_minute_features.csv",
    "fusion_alerts": PROJECT_ROOT / "outputs" / "fusion_alerts.csv",
    "scores": PROJECT_ROOT / "outputs" / "benchmark" / "anomaly_scores.csv",
    "benchmark": PROJECT_ROOT / "outputs" / "benchmark" / "model_benchmark_results.csv",
    "fusion_benchmark": PROJECT_ROOT / "outputs" / "benchmark" / "fusion_strategy_results.csv",
    "probe_runs": PROJECT_ROOT / "data" / "probe_runs.csv",
    "probe_hourly": PROJECT_ROOT / "outputs" / "probe_hourly_metrics.csv",
    "probe_alerts": PROJECT_ROOT / "outputs" / "probe_alerts.csv",
    "operating": PROJECT_ROOT / "outputs" / "model_operating_scores.csv",
    "snapshot": PROJECT_ROOT / "outputs" / "model_operating_snapshot.csv",
    "capability": PROJECT_ROOT / "outputs" / "model_capability_scores.csv",
    "diagnosis": PROJECT_ROOT / "outputs" / "model_fusion_diagnosis.csv",
    "profiles": PROJECT_ROOT / "outputs" / "model_capability_profiles.csv",
    "risks": PROJECT_ROOT / "outputs" / "model_health_risks.csv",
    "evidence": PROJECT_ROOT / "outputs" / "model_diagnostic_evidence.csv",
    "external_benchmarks": PROJECT_ROOT / "data" / "external_model_benchmarks.csv",
    "resource_model": PROJECT_ROOT / "data" / "resource_model_timeseries.csv",
    "resource_instances": PROJECT_ROOT / "data" / "resource_instance_hourly.csv",
    "resource_capacity": PROJECT_ROOT / "outputs" / "resource_capacity_daily.csv",
    "config": PROJECT_ROOT / "docs" / "ai_monitoring_metric_dictionary.xlsx",
}

REQUIRED_KEYS = [
    "logs", "operating", "snapshot", "capability", "diagnosis", "profiles",
    "risks", "evidence", "resource_model", "resource_instances",
    "resource_capacity", "config",
]

MODULES = [
    "运营总览",
    "性能诊断",
    "成本分析",
    "能力校准",
    "容量诊断",
]

MODULE_NAVIGATION = [
    ("运营总览", ":material/space_dashboard:", "nav_overview"),
    ("性能诊断", ":material/speed:", "nav_performance"),
    ("成本分析", ":material/paid:", "nav_cost"),
    ("能力校准", ":material/model_training:", "nav_calibration"),
    ("容量诊断", ":material/memory:", "nav_capacity"),
]

DIMENSION_LABELS = {
    "instruction_following": "指令遵循",
    "structured_output": "结构化输出",
    "reasoning": "推理能力",
    "tool_call": "工具调用",
}

PERFORMANCE_SCORE_LEVELS = (
    (80.0, "优秀", "green"),
    (60.0, "良好", "yellow"),
    (40.0, "一般", "orange"),
    (0.0, "较差", "red"),
)

SIGNIFICANT_SCORE_DROP = 5.0

ALGORITHM_OPTIONS = {
    "复合规则": ("pred_composite_rules", "score_composite_rules"),
    "滚动 MAD": ("pred_mad", "score_mad"),
    "STL 周期残差": ("pred_stl", "score_stl"),
    "Isolation Forest": ("pred_isolation_forest", "score_isolation_forest"),
}

DEFAULT_RISK_BANDS = {"medium": 30.0, "high": 60.0, "critical": 80.0}
DEFAULT_UNKNOWN_ALGORITHM_VOTES = 2


st.set_page_config(
    page_title="AI 中台运营决策实验台",
    page_icon=":material/route:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 用户明确要求左侧模块入口更大。只通过固定 key 生成的 class 定向放大导航按钮，
# 不改变下载、刷新等其他操作按钮的尺寸。
st.html(
    """
    <style>
      :root {
        --ops-blue: #155eef;
        --ops-border: #e5e9f0;
        --ops-muted: #667085;
        --ops-surface: #ffffff;
      }
      .stApp {
        background: #ffffff;
      }
      [data-testid="stVerticalBlockBorderWrapper"],
      [data-testid="stMetric"] {
        border-radius: 16px !important;
        border-color: var(--ops-border) !important;
        background: var(--ops-surface);
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.035);
        transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
      }
      [data-testid="stVerticalBlockBorderWrapper"]:hover,
      [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: #cfd7e6 !important;
        box-shadow: 0 8px 22px rgba(16, 24, 40, 0.07);
      }
      [data-testid="stMetricLabel"] {
        color: #344054;
        font-weight: 600;
      }
      [data-testid="stMetricValue"] {
        color: #172b4d;
        letter-spacing: -0.025em;
      }
      [data-testid="stDataFrame"],
      [data-testid="stVegaLiteChart"] {
        border-radius: 16px;
        overflow: hidden;
      }
      .st-key-health_core [data-testid="stMetricValue"] {
        font-size: 3rem;
        line-height: 1;
        font-weight: 700;
      }
      .st-key-health_action h3 {
        font-size: 1.25rem;
        line-height: 1.35;
        margin-top: 0.4rem;
      }
      .st-key-health_breakdown [data-testid="stMetricValue"] {
        font-size: 1.5rem;
      }
      [class*="st-key-nav_"] button {
        min-height: 3.1rem;
        justify-content: flex-start;
        padding-inline: 1rem;
        font-size: 1rem;
        font-weight: 600;
      }
    </style>
    """
)


def _signature(paths: dict[str, Path]) -> tuple[int, ...]:
    return tuple(paths[key].stat().st_mtime_ns for key in sorted(paths) if paths[key].exists())


def _read_csv(key: str, parse_dates: list[str] | None = None) -> pd.DataFrame:
    path = PATHS[key]
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=parse_dates)


@st.cache_data(show_spinner="正在加载运营决策数据…")
def load_all(_signature_value: tuple[int, ...]) -> dict[str, pd.DataFrame]:
    del _signature_value
    logs = _read_csv("logs", ["timestamp"])
    if not logs.empty:
        logs["is_success"] = logs["status_code"].between(200, 299)
        logs["date"] = logs["timestamp"].dt.normalize()
        logs["hour"] = logs["timestamp"].dt.floor("h")

    data = {
        "logs": logs,
        "truth": _read_csv("truth", ["start_time", "end_time"]),
        "key_features": _read_csv("key_features", ["minute"]),
        "fusion_alerts": _read_csv("fusion_alerts", ["detected_at"]),
        "scores": _read_csv("scores", ["hour"]),
        "benchmark": _read_csv("benchmark"),
        "fusion_benchmark": _read_csv("fusion_benchmark"),
        "probe_runs": _read_csv("probe_runs", ["started_at", "completed_at"]),
        "probe_hourly": _read_csv("probe_hourly", ["hour"]),
        "probe_alerts": _read_csv("probe_alerts", ["detected_at"]),
        "operating": _read_csv("operating", ["date"]),
        "snapshot": _read_csv("snapshot", ["date"]),
        "capability": _read_csv("capability", ["latest_run_at"]),
        "diagnosis": _read_csv("diagnosis", ["date"]),
        "profiles": _read_csv("profiles", ["date", "latest_capability_run_at"]),
        "risks": _read_csv("risks", ["date"]),
        "evidence": _read_csv("evidence", ["date"]),
        "external_benchmarks": _read_csv("external_benchmarks"),
        "resource_model": _read_csv(
            "resource_model", ["timestamp", "date", "source_date"]
        ),
        "resource_instances": _read_csv("resource_instances", ["hour", "date"]),
        "resource_capacity": _read_csv("resource_capacity", ["date"]),
    }
    config = PATHS["config"]
    if config.exists():
        for key, sheet in {
            "scoring_policy": "Scoring Policy",
            "risk_policy": "Risk Policy",
            "composite_rules": "Composite Rules",
            "conditions": "Rule Conditions",
            "fusion_strategies": "Fusion Strategies",
            "fusion_grading": "Severity Policy",
        }.items():
            try:
                data[key] = pd.read_excel(config, sheet_name=sheet)
            except ValueError:
                data[key] = pd.DataFrame()
    return data


@st.cache_resource(show_spinner=False)
def load_runtime_scoring_policy(_config_signature: int):
    """加载风险重算所需的评分权重；配置文件变化时自动失效。"""
    del _config_signature
    return load_scoring_policy(PATHS["config"])


def _date_filter(frame: pd.DataFrame, column: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if frame.empty or column not in frame:
        return frame.copy()
    return frame[frame[column].between(start, end, inclusive="left")].copy()


def _model_filter(frame: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    if frame.empty or "model_id" not in frame:
        return frame.copy()
    return frame[frame["model_id"].isin(models)].copy()


def _latest_by_model(frame: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    return frame.sort_values(date_column).groupby("model_id", as_index=False).tail(1)


def _fmt_delta(current: float, previous: float, suffix: str = "") -> str | None:
    if pd.isna(previous):
        return None
    return f"{current - previous:+,.1f}{suffix}"


def _performance_score_level(score: float) -> tuple[str, str]:
    """将性能类评分统一转成自然语言等级与语义颜色。"""
    if pd.isna(score):
        return "数据不足", "gray"
    for threshold, label, color in PERFORMANCE_SCORE_LEVELS:
        if float(score) >= threshold:
            return label, color
    return "数据不足", "gray"


def _capacity_state_badge(label: str, color: str) -> None:
    """容量风险使用独立红色，避免被全局低分棕红色主题覆盖。"""
    if color != "red":
        st.badge(label, color=color)
        return
    st.html(
        f"""
        <span style="display:inline-flex;align-items:center;width:fit-content;
          padding:0.16rem 0.46rem;border-radius:0.45rem;background:#FEE4E2;
          color:#B42318;font-size:0.875rem;font-weight:500;line-height:1.35;">
          {escape(label)}
        </span>
        """
    )


def _capacity_plain_state(state: str) -> tuple[str, str, str]:
    """把后端容量状态映射为面向业务用户的名称、颜色和图标。"""
    return {
        "容量风险": ("高峰期可能处理不过来", "red", ":material/warning:"),
        "需要关注": ("接近容量上限", "orange", ":material/error:"),
        "容量充足": ("容量充足", "green", ":material/check_circle:"),
    }.get(str(state), ("状态待确认", "gray", ":material/help:"))


def _capacity_plain_reason(row: pd.Series) -> str:
    reasons: list[str] = []
    headroom = float(row["hbm_headroom_pct"])
    npu_peak = float(row["npu_max"])
    concurrency = int(row["running_max_busy"])
    instances = max(int(row["instance_count"]), 1)
    waiting = int(row["waiting_max_busy"])
    if waiting > 0:
        reasons.append(f"高峰时最多有 {waiting} 个请求等待处理")
    if headroom <= 5:
        reasons.append(f"显存最低只剩 {headroom:.1f}%")
    elif headroom <= 10:
        reasons.append(f"显存余量仅剩 {headroom:.1f}%")
    if npu_peak >= 95:
        reasons.append("算力曾达到满载")
    elif npu_peak >= 70:
        reasons.append(f"算力峰值达到 {npu_peak:.0f}%")
    if concurrency / instances >= 0.8:
        reasons.append(f"高峰时同时处理 {concurrency} 个请求，已接近 {instances} 个实例的承载上限")
    if not reasons:
        reasons.append("当前算力、显存和并发均留有余量")
    return "；".join(reasons[:3]) + "。"


def _capacity_plain_impact(row: pd.Series) -> str:
    if int(row["waiting_max_busy"]) > 0:
        return "已经出现请求排队，用户可能感到响应变慢。"
    if str(row["capacity_state"]) == "容量风险":
        return "继续增加流量后，可能出现响应变慢或请求排队。"
    if str(row["capacity_state"]) == "需要关注":
        return "流量明显增加时，响应速度可能下降。"
    return "当前预计不会影响请求处理。"


def _capacity_current_impact(row: pd.Series) -> str:
    waiting = int(row["waiting_max_busy"])
    return f"已发现最多 {waiting} 个请求排队" if waiting else "暂未发现请求排队"


def _capacity_key_signal(row: pd.Series) -> str:
    """处理清单只展示一个最值得关注的容量信号。"""
    if int(row["waiting_max_busy"]) > 0:
        return f"已出现 {int(row['waiting_max_busy'])} 个高峰等待请求"
    if float(row["hbm_headroom_pct"]) <= 10:
        return f"显存余量仅剩 {float(row['hbm_headroom_pct']):.1f}%"
    if float(row["concurrency_ratio"]) >= 0.8:
        return f"忙时并发已达到 {float(row['concurrency_ratio']) * 100:.0f}% 的实例比例"
    if float(row["npu_max"]) >= 70:
        return f"算力峰值达到 {float(row['npu_max']):.0f}%"
    return "当前主要容量指标均留有余量"


def _capacity_plain_action(row: pd.Series) -> str:
    waiting = int(row["waiting_max_busy"])
    headroom = float(row["hbm_headroom_pct"])
    if waiting > 0:
        return "先分流新增请求，并检查是否需要增加实例。"
    if str(row["capacity_state"]) == "容量风险" and headroom <= 5:
        return "优先检查实例配置和流量分配，确认是否需要释放显存、增加实例或降低路由权重。"
    if str(row["capacity_state"]) == "容量风险":
        return "检查高峰负载和实例配置，必要时增加实例或迁移部分流量。"
    if str(row["capacity_state"]) == "需要关注":
        return "保持当前实例规模，重点观察高峰变化，暂缓一次性增加大量流量。"
    return "维持当前配置，并继续进行日常观察。"


def _select_capacity_model(model_id: str) -> None:
    st.session_state["capacity_detail_model"] = model_id
    st.session_state["capacity_scroll_to_detail"] = True
    st.session_state["capacity_scroll_request"] = (
        st.session_state.get("capacity_scroll_request", 0) + 1
    )


def _score_level_badge(label: str, color: str, target=None, icon: str | None = None) -> None:
    """评分低档徽章固定使用深红色，绕开 Streamlit 对 red 的主题派生。"""
    renderer = st if target is None else target
    if color != "red":
        renderer.badge(label, color=color, icon=icon)
        return
    renderer.html(
        f"""
        <span style="display:inline-flex;align-items:center;width:fit-content;
          padding:0.16rem 0.46rem;border-radius:0.45rem;background:#F8D7DA;
          color:#6b080c;font-size:0.875rem;font-weight:500;line-height:1.35;">
          {escape(label)}
        </span>
        """,
        width="content",
    )


def _score_gauge_color(score: float) -> str:
    """与性能和成本评分等级一致的仪表盘颜色。"""
    if pd.isna(score):
        return "#98A2B3"
    if score >= 80:
        return "#12B76A"
    if score >= 60:
        return "#FACC15"
    if score >= 40:
        return "#F97316"
    return "#6b080c"


def _score_bar(score: float) -> alt.LayerChart:
    """紧凑彩色评分横条，颜色与全站评分等级保持一致。"""
    value = min(100.0, max(0.0, float(score))) if pd.notna(score) else 0.0
    bar_data = pd.DataFrame({"value": [value], "full": [100.0], "row": ["评分"]})
    background = alt.Chart(bar_data).mark_bar(
        color="#EAECF0", cornerRadius=6, size=10
    ).encode(
        x=alt.X("full:Q", scale=alt.Scale(domain=[0, 100]), axis=None),
        y=alt.Y("row:N", axis=None),
    )
    foreground = alt.Chart(bar_data).mark_bar(
        color=_score_gauge_color(value), cornerRadius=6, size=10
    ).encode(
        x=alt.X("value:Q", scale=alt.Scale(domain=[0, 100]), axis=None),
        y=alt.Y("row:N", axis=None),
    )
    return alt.layer(background, foreground).properties(height=16).configure_view(stroke=None)


def _weighted_performance_summary(frame: pd.DataFrame) -> dict[str, float]:
    """按日调用量汇总性能窗口，不生成额外估算数据。"""
    columns = [
        "performance_score",
        "latency_score",
        "stability_score",
        "p50_latency_ms",
        "p95_latency_ms",
        "p99_latency_ms",
    ]
    summary: dict[str, float] = {}
    for column in columns:
        valid = frame.dropna(subset=[column])
        if valid.empty:
            summary[column] = np.nan
            continue
        weights = valid["request_count"].clip(lower=1)
        summary[column] = float(np.average(valid[column], weights=weights))
    return summary


def _weighted_cost_summary(frame: pd.DataFrame) -> dict[str, float]:
    """按日调用量汇总成本窗口，保留现有价格与质量口径。"""
    columns = [
        "cost_performance_score",
        "cost_efficiency_score",
        "quality_score",
        "cost_per_request",
        "cost_per_1k_tokens",
        "cost_trend_ratio",
    ]
    summary: dict[str, float] = {}
    for column in columns:
        valid = frame.dropna(subset=[column])
        if valid.empty:
            summary[column] = np.nan
            continue
        weights = valid["request_count"].clip(lower=1)
        summary[column] = float(np.average(valid[column], weights=weights))
    return summary


def _peer_window_average(
    frame: pd.DataFrame,
    excluded_model: str,
    score_column: str,
    summary_builder,
) -> float:
    """使用相同窗口和汇总口径计算其他模型的等权平均分。"""
    peer_scores = [
        summary_builder(model_frame).get(score_column, np.nan)
        for model_id, model_frame in frame.groupby("model_id")
        if model_id != excluded_model
    ]
    valid_scores = [float(score) for score in peer_scores if not pd.isna(score)]
    return float(np.mean(valid_scores)) if valid_scores else np.nan


def _score_change_text(
    current: float,
    previous: float,
    compare_previous: bool,
) -> tuple[str, str]:
    if not compare_previous:
        return "未开启周期对比", "gray"
    if pd.isna(previous):
        return "上一周期数据不足", "gray"
    change = round(float(current) - float(previous), 1)
    if change > 0:
        color = "blue"
    elif change <= -SIGNIFICANT_SCORE_DROP:
        color = "orange"
    elif change < 0:
        color = "yellow"
    else:
        color = "gray"
    return f"较上一周期 {change:+.1f} 分", color


def _latency_comparison_text(
    value: float,
    reference: float,
    reference_model: str,
) -> str:
    gap = value - reference
    direction = "慢" if gap > 0 else "快" if gap < 0 else "相当"
    magnitude = abs(gap)
    amount = f"{magnitude / 1_000:.1f} 秒" if magnitude >= 1_000 else f"{magnitude:,.0f} 毫秒"
    if direction == "相当":
        return f"与 {reference_model} 基本相当"
    return f"比 {reference_model} {direction} {amount}"


def _confidence_band(score: float) -> str:
    """将画像可信度转成便于阅读的展示档位，不改变底层评分。"""
    if score >= 85:
        return "高"
    if score >= 70:
        return "中"
    return "低"


def _performance_gap_label(gap: float) -> str:
    """用自然语言说明系统主动检查分与真实表现分的差异方向。"""
    if pd.isna(gap) or abs(gap) < 0.05:
        return "基本持平"
    if gap > 0:
        return f"主动检查高 {gap:.1f} 分"
    return f"真实高 {abs(gap):.1f} 分"


def _line_chart(
    frame: pd.DataFrame,
    x: str,
    y: str,
    color: str,
    y_title: str,
    tooltip: list[alt.Tooltip],
    height: int = 330,
) -> alt.Chart:
    return (
        alt.Chart(frame)
        .mark_line(point=alt.OverlayMarkDef(size=34), strokeWidth=2)
        .encode(
            x=alt.X(f"{x}:T", title=None, axis=alt.Axis(format="%m-%d", labelAngle=0)),
            y=alt.Y(f"{y}:Q", title=y_title, scale=alt.Scale(zero=False)),
            color=alt.Color(f"{color}:N", title="模型"),
            tooltip=tooltip,
        )
        .properties(height=height)
        .interactive(bind_y=False)
    )


def _health_gauge(score: float, color: str) -> alt.LayerChart:
    """紧凑环形健康仪表盘；颜色表达状态，中心数字表达绝对分。"""
    gauge_data = pd.DataFrame(
        {
            "segment": ["当前健康", "剩余空间"],
            "value": [score, max(0.0, 100.0 - score)],
        }
    )
    arc = (
        alt.Chart(gauge_data)
        .mark_arc(innerRadius=55, outerRadius=68, cornerRadius=8)
        .encode(
            theta=alt.Theta("value:Q", stack=True),
            color=alt.Color(
                "segment:N",
                scale=alt.Scale(
                    domain=["当前健康", "剩余空间"],
                    range=[color, "#EEF2F6"],
                ),
                legend=None,
            ),
            order=alt.Order("segment:N", sort="ascending"),
            tooltip=[
                alt.Tooltip("segment:N", title="指标"),
                alt.Tooltip("value:Q", title="分值", format=".1f"),
            ],
        )
    )
    score_text = (
        alt.Chart(pd.DataFrame({"label": [f"{score:.1f}"]}))
        .mark_text(fontSize=42, fontWeight=700, color="#172B4D")
        .encode(text="label:N")
    )
    return (arc + score_text).properties(height=165)


def _health_trend_chart(frame: pd.DataFrame, color: str) -> alt.Chart:
    """低干扰健康趋势图，保留悬浮日期与分值。"""
    return (
        alt.Chart(frame)
        .mark_area(
            line={"color": color, "strokeWidth": 2},
            color=color,
            opacity=0.10,
        )
        .encode(
            x=alt.X("date:T", axis=None),
            y=alt.Y("health_score:Q", axis=None, scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip("date:T", title="日期", format="%Y-%m-%d"),
                alt.Tooltip("health_score:Q", title="健康指数", format=".1f"),
            ],
        )
        .properties(height=72)
    )


def _section(title: str, caption: str) -> None:
    st.subheader(title)
    st.caption(caption)


def _metric_row(items: list[dict[str, object]]) -> None:
    row = st.container(horizontal=True, horizontal_alignment="distribute", gap="small")
    for item in items:
        row.metric(
            str(item["label"]),
            item["value"],
            delta=item.get("delta"),
            delta_color=str(item.get("delta_color", "normal")),
            help=item.get("help"),
            border=True,
            chart_data=item.get("chart_data"),
            chart_type=str(item.get("chart_type", "line")),
        )


def _overview_window(
    logs: pd.DataFrame,
    operating: pd.DataFrame,
    window_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """在全局日期范围内应用总览页的快捷观察窗口。"""
    window_days = {"近 1 天": 1, "近 7 天": 7, "近 30 天": 30}
    days = window_days.get(window_label)
    if days is None:
        return logs.copy(), operating.copy()

    latest_candidates = []
    if not logs.empty:
        latest_candidates.append(pd.Timestamp(logs["timestamp"].max()).normalize())
    if not operating.empty:
        latest_candidates.append(pd.Timestamp(operating["date"].max()).normalize())
    if not latest_candidates:
        return logs.copy(), operating.copy()

    window_start = max(latest_candidates) - pd.Timedelta(days=days - 1)
    visible_logs = (
        logs[logs["timestamp"].ge(window_start)].copy()
        if "timestamp" in logs.columns
        else logs.copy()
    )
    visible_operating = (
        operating[operating["date"].ge(window_start)].copy()
        if "date" in operating.columns
        else operating.copy()
    )
    return visible_logs, visible_operating


def _render_decision_summary(
    operating: pd.DataFrame,
    profiles: pd.DataFrame,
) -> None:
    """把评分转成运营人员可以直接执行的当日结论。"""
    latest = _latest_by_model(operating)
    if latest.empty:
        return

    weights = latest["request_count"].clip(lower=1)
    overall_health = float(np.average(latest["health_score"], weights=weights))
    previous_rows = operating[operating["date"] < latest["date"].min()]
    previous = _latest_by_model(previous_rows)
    previous_health = (
        float(np.average(previous["health_score"], weights=previous["request_count"].clip(lower=1)))
        if not previous.empty else np.nan
    )
    health_daily = (
        operating.groupby("date", as_index=False)
        .apply(lambda group: pd.Series({
            "health_score": np.average(group["health_score"], weights=group["request_count"].clip(lower=1))
        }), include_groups=False)
        .reset_index(drop=True)
    )
    component_scores = {
        "成功率": float(np.average(latest["success_score"], weights=weights)),
        "性能": float(np.average(latest["performance_score"], weights=weights)),
        "成本": float(np.average(latest["cost_efficiency_score"], weights=weights)),
    }
    weakest = latest.sort_values("health_score").iloc[0]
    risk_components = {
        "success_score": "成功率",
        "performance_score": "性能",
        "stability_score": "稳定性",
        "cost_efficiency_score": "成本效率",
    }
    primary_risk_column = min(risk_components, key=lambda column: float(weakest[column]))
    routing_candidates = profiles.dropna(subset=["routing_readiness_score"]).copy()
    best_route = (
        routing_candidates.sort_values("routing_readiness_score", ascending=False).iloc[0]
        if not routing_candidates.empty
        else None
    )

    status, status_badge_color = _performance_score_level(overall_health)
    gauge_color = _score_gauge_color(overall_health)

    weakest_health = float(weakest["health_score"])
    if weakest_health < 75:
        action_title = f"降低 {weakest['model_id']} 的路由权重"
        action_detail = "优先把新增流量切换到健康度更高的候选模型。"
        badge_color = "orange"
    elif weakest_health < 85:
        action_title = f"重点观察 {weakest['model_id']}"
        action_detail = "继续使用但暂不扩大流量，重点跟踪延迟与成功率。"
        badge_color = "orange"
    else:
        action_title = "维持当前路由策略"
        action_detail = "所有模型均处于健康区间，继续观察延迟与成本趋势。"
        badge_color = "green"

    status_col, action_col, breakdown_col = st.columns(
        [25, 45, 30],
        gap="medium",
        vertical_alignment="top",
    )
    with status_col:
        with st.container(
            border=True,
            height="stretch",
            key="health_core",
            vertical_alignment="distribute",
            gap="xsmall",
        ):
            st.caption("模型总体健康指数")
            st.altair_chart(
                _health_gauge(overall_health, gauge_color),
                width="stretch",
                theme=None,
            )
            delta = overall_health - previous_health if pd.notna(previous_health) else np.nan
            if pd.notna(delta):
                delta_color = "green" if delta >= 0 else "red"
                delta_direction = "上升" if delta >= 0 else "下降"
                st.badge(
                    f"较上日{delta_direction} {abs(delta):.1f} 分",
                    icon=":material/trending_up:" if delta >= 0 else ":material/trending_down:",
                    color=delta_color,
                )
            st.altair_chart(
                _health_trend_chart(health_daily, gauge_color),
                width="stretch",
                theme=None,
            )
    with action_col:
        with st.container(border=True, height="stretch", key="health_action", vertical_alignment="center"):
            st.caption("风险分析与推荐动作")
            _score_level_badge(
                status,
                status_badge_color,
                icon=":material/check_circle:" if status_badge_color in {"green", "blue"} else ":material/warning:",
            )
            st.markdown(f"### {action_title}")
            st.write(action_detail)
            st.space("small")
            st.markdown("**判断依据**")
            evidence_tags = st.container(horizontal=True, gap="small")
            evidence_tags.badge(
                f"{risk_components[primary_risk_column]}评分 "
                f"{float(weakest[primary_risk_column]):.1f}",
                color="orange",
            )
            if best_route is not None:
                evidence_tags.badge(
                    f"候选模型 {best_route['model_id']}",
                    color="blue",
                )
                evidence_tags.badge(
                    f"路由评分 {float(best_route['routing_readiness_score']):.1f}",
                    color="blue",
                )
            st.caption(f"风险模型：{weakest['model_id']} · 建议结合实时流量与容量余量执行。")
    with breakdown_col:
        with st.container(border=True, height="stretch", key="health_breakdown"):
            score_help = {
                "成功率": "业务调用成功率相对 99% 目标的得分，健康权重 35%。",
                "性能": "相对自身基线的响应速度评分 90% + 稳定性评分 10%，健康权重 50%。",
                "成本": "相对自身基线的成本效率综合得分，健康权重 15%。",
            }
            for label, score in component_scores.items():
                with st.container(border=True, key=f"health_score_{label}", gap="xsmall"):
                    score_row = st.container(
                        horizontal=True,
                        horizontal_alignment="distribute",
                        vertical_alignment="center",
                    )
                    score_row.metric(
                        f"{label}评分",
                        f"{score:.1f}",
                        help=score_help[label],
                    )
                    component_level, component_color = _performance_score_level(score)
                    _score_level_badge(component_level, component_color, target=score_row)
                    st.altair_chart(
                        _score_bar(score), width="stretch", theme=None
                    )


def _render_capacity_reference(benchmarks: pd.DataFrame) -> None:
    """展示外部真实压测的稳定并发、吞吐和 429 观测边界。"""
    _section(
        "外部容量基准",
        "来自真实外部压测汇总，用于候选模型选型；它不是当前生产并发，也不参与现有健康评分。",
    )
    if benchmarks.empty:
        st.info("尚未接入外部压测标准表。")
        return

    capacity = build_capacity_profiles(benchmarks)
    if capacity.empty:
        st.info("外部压测数据中没有可用的容量组合。")
        return

    capacity["endpoint"] = capacity["provider"].astype(str) + " · " + capacity["model_id"].astype(str)
    endpoint_options = sorted(capacity["endpoint"].unique())
    endpoint_col, profile_col = st.columns([1.25, 1], gap="large", vertical_alignment="bottom")
    with endpoint_col:
        endpoint = st.selectbox(
            "候选模型端点",
            endpoint_options,
            key="overview_capacity_endpoint",
            help="同一模型在不同供应商上按不同端点展示。",
        )
    endpoint_rows = capacity[capacity["endpoint"].eq(endpoint)].sort_values("input_tokens_target")
    io_options = endpoint_rows["io_profile"].tolist()
    with profile_col:
        io_profile = st.segmented_control(
            "输入/输出档位",
            io_options,
            default=io_options[0],
            required=True,
            key="overview_capacity_io_profile",
        )

    selected = endpoint_rows[endpoint_rows["io_profile"].eq(io_profile)].iloc[0]
    max_stable = int(selected["max_stable_concurrency"])
    tested_max = int(selected["tested_max_concurrency"])
    observed_limit = selected["observed_rate_limit_concurrency"]
    limit_value = f"{int(observed_limit)} 并发" if pd.notna(observed_limit) else "测试内未触发"
    ttft_value = (
        f"{selected['ttft_ms_at_max_stable'] / 1_000:.2f} 秒"
        if pd.notna(selected["ttft_ms_at_max_stable"])
        else "无有效结果"
    )
    throughput_value = (
        f"{selected['total_output_tokens_per_second_at_max_stable']:,.1f} token/s"
        if pd.notna(selected["total_output_tokens_per_second_at_max_stable"])
        else "无有效结果"
    )
    _metric_row(
        [
            {"label": "最高稳定测试并发", "value": f"{max_stable}" if max_stable else "无"},
            {"label": "观测到的 429 边界", "value": limit_value, "delta_color": "inverse"},
            {"label": "稳定并发下首字延迟", "value": ttft_value, "delta_color": "inverse"},
            {"label": "稳定并发下总吞吐", "value": throughput_value},
        ]
    )
    coverage = int(round(float(selected["stability_coverage_pct"])))
    st.progress(
        coverage,
        text=f"稳定结果覆盖测试上限：{max_stable} / {tested_max} 并发（{coverage}%）",
    )
    if bool(selected["rate_limit_observed"]):
        st.warning(
            f"这个组合在 **{limit_value}** 观测到 429；上线时应在该并发之前预留分流余量。",
            icon=":material/speed:",
        )
    else:
        st.caption("测试范围内未观测到 429，但由于缺少重复试验，不能把测试上限直接当作生产配额。")

    with st.expander("查看全部外部容量画像", icon=":material/table_chart:"):
        capacity_table = capacity.sort_values(
            ["rate_limit_observed", "max_stable_concurrency", "provider", "model_id"],
            ascending=[True, False, True, True],
        )
        st.dataframe(
            capacity_table,
            column_order=[
                "provider",
                "model_id",
                "io_profile",
                "max_stable_concurrency",
                "tested_max_concurrency",
                "stability_coverage_pct",
                "observed_rate_limit_concurrency",
                "ttft_ms_at_max_stable",
                "total_output_tokens_per_second_at_max_stable",
                "capacity_state",
                "capacity_confidence",
            ],
            column_config={
                "provider": "供应商",
                "model_id": "模型",
                "io_profile": "输入/输出档位",
                "max_stable_concurrency": "最高稳定测试并发",
                "tested_max_concurrency": "测试上限",
                "stability_coverage_pct": st.column_config.ProgressColumn(
                    "稳定覆盖", min_value=0, max_value=100, format="%.1f%%"
                ),
                "observed_rate_limit_concurrency": "429 最低观测并发",
                "ttft_ms_at_max_stable": st.column_config.NumberColumn(
                    "首字延迟", format="%.0f ms"
                ),
                "total_output_tokens_per_second_at_max_stable": st.column_config.NumberColumn(
                    "总吞吐", format="%.1f token/s"
                ),
                "capacity_state": "容量状态",
                "capacity_confidence": "可信度",
            },
            hide_index=True,
            height=360,
        )


def render_overview(
    logs: pd.DataFrame,
    operating: pd.DataFrame,
    profiles: pd.DataFrame,
    overall_operating: pd.DataFrame,
    overall_profiles: pd.DataFrame,
) -> None:
    st.subheader("运营总览")
    if logs.empty or operating.empty:
        st.info("当前筛选范围没有运营数据。")
        return

    description_column, window_column = st.columns(
        [1, 1], vertical_alignment="center"
    )
    description_column.caption("快速掌握平台整体健康、关键风险与各模型的运营表现。")
    window_controls = window_column.container(horizontal_alignment="right")
    window_label = window_controls.segmented_control(
        "观察窗口",
        ["近 1 天", "近 7 天", "近 30 天", "全部"],
        default="近 7 天",
        required=True,
        key="overview_window",
        label_visibility="collapsed",
        width="content",
    )
    logs, operating = _overview_window(logs, operating, str(window_label))
    _, summary_operating = _overview_window(
        pd.DataFrame(), overall_operating, str(window_label)
    )
    if logs.empty or operating.empty:
        st.info("当前快捷观察窗口没有运营数据。")
        return

    daily = logs.groupby("date", as_index=False).agg(
        request_count=("request_id", "count"),
        total_tokens=("total_tokens", "sum"),
        estimated_cost=("estimated_cost", "sum"),
        success_rate=("is_success", "mean"),
        p95_latency_ms=("latency_ms", lambda values: values.quantile(0.95)),
    )
    daily["success_rate"] *= 100
    latest = _latest_by_model(operating)
    _render_decision_summary(summary_operating, overall_profiles)

    _metric_row([
        {"label": "调用量", "value": f"{len(logs):,}", "help": "当前观察窗口内的请求总数。"},
        {
            "label": "Token",
            "value": f"{int(logs['total_tokens'].sum()):,}",
            "help": "当前观察窗口内所有请求输入与输出 Token 的合计。",
        },
        {
            "label": "估算成本",
            "value": f"¥{logs['estimated_cost'].sum():,.2f}",
            "help": "按项目中的模拟单价估算，用于观察趋势和比较模型，不代表真实账单。",
        },
        {
            "label": "成功率",
            "value": f"{logs['is_success'].mean() * 100:.2f}%",
            "help": "成功完成的请求数占全部请求数的比例。",
        },
        {
            "label": "P95 延迟",
            "value": f"{logs['latency_ms'].quantile(.95):,.0f} ms",
            "help": "95% 的请求会在该时间内完成；数值越低，通常表示响应越快。",
        },
    ])

    left, right = st.columns([1.45, 1], gap="large")
    with left:
        trend_label = st.segmented_control(
            "选择下方趋势图的指标",
            ["健康指数", "成功率", "P95 延迟", "单次成本"],
            default="健康指数",
            required=True,
            key="overview_trend_metric",
        )
        trend_specs = {
            "健康指数": ("health_score", "健康指数", ".1f"),
            "成功率": ("success_rate", "成功率（%）", ".2f"),
            "P95 延迟": ("p95_latency_ms", "P95 延迟（ms）", ",.0f"),
            "单次成本": ("cost_per_request", "单次成本（元）", ".4f"),
        }
        trend_column, trend_title, trend_format = trend_specs[str(trend_label)]
        st.markdown(f"#### {trend_label}趋势")
        chart = _line_chart(
            operating,
            "date",
            trend_column,
            "model_id",
            trend_title,
            [
                alt.Tooltip("date:T", title="日期", format="%Y-%m-%d"),
                alt.Tooltip("model_id:N", title="模型"),
                alt.Tooltip(f"{trend_column}:Q", title=str(trend_label), format=trend_format),
                alt.Tooltip("health_level:N", title="等级"),
            ],
        )
        st.altair_chart(chart, width="stretch")
    with right:
        st.markdown("#### 最新健康排行")
        ranking = latest.sort_values("health_score", ascending=False).copy()
        ranking["健康排名"] = range(1, len(ranking) + 1)
        profile_cols = profiles[["model_id", "recommended_role", "routing_action"]] if not profiles.empty else pd.DataFrame()
        if not profile_cols.empty:
            ranking = ranking.merge(profile_cols, on="model_id", how="left")
        ranking_style = ranking.style.map(
            lambda value: (
                "background-color: #dcfce7; color: #166534; font-weight: 700;"
                if float(value) >= 80
                else "background-color: #fef9c3; color: #854d0e; font-weight: 700;"
                if float(value) >= 60
                else "background-color: #ffedd5; color: #9a3412; font-weight: 700;"
                if float(value) >= 40
                else "background-color: #f3e4df; color: #6b080c; font-weight: 700;"
            ),
            subset=["health_score"],
        )
        st.dataframe(
            ranking_style,
            column_order=["健康排名", "model_id", "health_score", "health_level", "recommended_role"],
            column_config={
                "健康排名": st.column_config.NumberColumn("排名", format="#%d"),
                "model_id": "模型",
                "health_score": st.column_config.NumberColumn("健康指数", format="%.1f"),
                "health_level": "健康等级",
                "recommended_role": "路由角色",
            },
            hide_index=True,
            height=290,
        )

    st.markdown("#### 健康评分构成")
    score_table = latest.sort_values("health_score", ascending=False)
    st.dataframe(
        score_table,
        column_order=[
            "model_id", "success_score", "performance_score", "stability_score",
            "cost_efficiency_score", "health_score", "request_count",
        ],
        column_config={
            "model_id": "模型",
            "success_score": st.column_config.NumberColumn("成功率评分", format="%.1f"),
            "performance_score": st.column_config.NumberColumn("性能评分", format="%.1f"),
            "stability_score": st.column_config.NumberColumn("稳定性评分", format="%.1f"),
            "cost_efficiency_score": st.column_config.NumberColumn("成本评分", format="%.1f"),
            "health_score": st.column_config.ProgressColumn("健康指数", min_value=0, max_value=100, format="%.1f"),
            "request_count": st.column_config.NumberColumn("当日调用量", format="%d"),
        },
        hide_index=True,
    )


def render_performance(operating: pd.DataFrame) -> None:
    _section(
        "性能诊断",
        "查看模型的响应速度、稳定性和异常情况。",
    )
    with st.expander("数据说明", icon=":material/info:"):
        st.caption(
            "模型名称与 TTFT 基线来自真实资源数据；"
            "请求级延迟历史为基于真实基线校准的模拟数据。"
        )
        st.caption("页面评分为日级聚合结果，不代表单次请求的保证值。")
    if operating.empty:
        st.info(
            "当前筛选范围没有性能数据，请调整全局日期或模型筛选。",
            icon=":material/info:",
        )
        return

    models = sorted(operating["model_id"].dropna().unique())
    control_main, control_time = st.columns([3, 2], gap="large", vertical_alignment="bottom")
    controls = control_main.container(horizontal=True, gap="medium", vertical_alignment="bottom")
    model = controls.selectbox("诊断模型", models, key="performance_model", width=260)
    compare_options: list[str | None] = [None] + [item for item in models if item != model]
    compare_model = controls.selectbox(
        "添加对比模型",
        compare_options,
        format_func=lambda value: "不对比" if value is None else str(value),
        key="performance_compare_model",
        width=260,
        disabled=len(compare_options) == 1,
        help=(
            "从当前全局筛选中选择第二个模型。"
            if len(compare_options) > 1
            else "当前只有一个模型，请先在全局筛选中加入其他模型。"
        ),
    )
    compare_previous = controls.toggle(
        "与上一周期对比",
        value=True,
        key="performance_compare_previous",
    )
    time_controls = control_time.container(horizontal_alignment="right")
    window_label = time_controls.segmented_control(
        "时间范围",
        ["过去 7 天", "过去 30 天", "过去 90 天"],
        default="过去 30 天",
        required=True,
        key="performance_window",
        width="content",
    )

    window_days = {"过去 7 天": 7, "过去 30 天": 30, "过去 90 天": 90}
    days = window_days[str(window_label)]
    model_history = operating[operating["model_id"].eq(model)].sort_values("date")
    latest_date = pd.Timestamp(model_history["date"].max()).normalize()
    window_start = latest_date - pd.Timedelta(days=days - 1)
    previous_end = window_start - pd.Timedelta(days=1)
    previous_start = previous_end - pd.Timedelta(days=days - 1)
    selected = model_history[model_history["date"].between(window_start, latest_date)].copy()
    previous = model_history[model_history["date"].between(previous_start, previous_end)].copy()
    if selected.empty:
        st.info("所选时间范围没有可用性能数据。", icon=":material/info:")
        return

    current_summary = _weighted_performance_summary(selected)
    previous_summary = _weighted_performance_summary(previous)
    peer_window = operating[operating["date"].between(window_start, latest_date)]
    peer_average = _peer_window_average(
        peer_window,
        model,
        "latency_score",
        _weighted_performance_summary,
    )
    observed_days = int(selected["date"].nunique())

    stability_level, _ = _performance_score_level(current_summary["stability_score"])
    weakest_label = (
        "响应速度"
        if current_summary["latency_score"] <= current_summary["stability_score"]
        else "稳定性"
    )
    experience_explanation = (
        "主要短板：响应速度较慢"
        if weakest_label == "响应速度"
        else "主要短板：稳定性需关注"
    )
    if pd.isna(peer_average):
        response_explanation = "当前筛选范围暂无其他模型基线"
    else:
        peer_gap = current_summary["latency_score"] - peer_average
        response_explanation = (
            f"高于同窗口模型平均 {peer_gap:.1f} 分"
            if peer_gap >= 0
            else f"低于同窗口模型平均 {abs(peer_gap):.1f} 分"
        )
    stability_explanation = (
        f"所选窗口内 {observed_days} 天整体表现稳定"
        if stability_level in {"优秀", "良好"}
        else f"所选窗口内 {observed_days} 天的波动偏大"
    )
    conclusions = [
        ("综合体验", current_summary["performance_score"], experience_explanation, "performance_score"),
        ("响应速度", current_summary["latency_score"], response_explanation, "latency_score"),
        ("稳定性", current_summary["stability_score"], stability_explanation, "stability_score"),
    ]
    conclusion_cols = st.columns(3, gap="medium")
    total_title, total_score, total_explanation, total_column = conclusions[0]
    total_level, total_badge_color = _performance_score_level(total_score)
    total_change, total_change_color = _score_change_text(
        total_score, previous_summary[total_column], compare_previous
    )
    with conclusion_cols[0].container(
        border=True, height=280, gap="xsmall", vertical_alignment="distribute"
    ):
        st.markdown(f"**{total_title}**")
        st.altair_chart(
            _health_gauge(total_score, _score_gauge_color(total_score)),
            width="stretch",
            theme=None,
        )
        total_tags = st.container(horizontal=True, gap="small")
        _score_level_badge(total_level, total_badge_color, target=total_tags)
        total_tags.badge(total_change, color=total_change_color)
    component_weights = {"响应速度": "90%", "稳定性": "10%"}
    component_help = {
        "响应速度": "由 P50、P95、P99 相对模型自身前 7 日基线计算，正常基线为 80 分，占综合体验 90%。",
        "稳定性": "由日内 P95 延迟波动和成功率波动计算；低流量模型自动扩大到3或6小时统计桶，占综合体验10%。",
    }
    for column, (title, score, explanation, score_column) in zip(conclusion_cols[1:], conclusions[1:]):
        level, color = _performance_score_level(score)
        change_text, change_color = _score_change_text(
            score,
            previous_summary[score_column],
            compare_previous,
        )
        with column.container(
            border=True, height=280, gap="xsmall", vertical_alignment="distribute"
        ):
            st.caption(f"总评分构成 · 权重 {component_weights[title]}")
            title_row = st.container(
                horizontal=True,
                horizontal_alignment="distribute",
                vertical_alignment="center",
            )
            title_row.metric(title, f"{score:.0f} / 100", help=component_help[title])
            _score_level_badge(level, color, target=title_row)
            st.altair_chart(_score_bar(score), width="stretch", theme=None)
            st.caption(explanation)
            st.badge(change_text, color=change_color)
    st.caption("综合体验由右侧两项计算：响应速度 × 90% + 稳定性 × 10%。")

    attention_days = selected[
        selected["latency_score"].lt(40) | selected["stability_score"].lt(40)
    ]
    if attention_days.empty:
        st.caption("所选窗口内未出现响应速度或稳定性评分低于 40 的性能日。")
    else:
        st.warning(
            f"所选窗口内有 {attention_days['date'].nunique()} 天需重点关注，"
            "请在趋势图和技术指标中查看详情。",
            icon=":material/warning:",
        )

    chart_models = [str(model)] + ([str(compare_model)] if compare_model is not None else [])
    chart_source = operating[
        operating["model_id"].isin(chart_models)
        & operating["date"].between(window_start, latest_date)
    ].copy()
    latency = chart_source.melt(
        id_vars=["date", "model_id"],
        value_vars=["p50_latency_ms", "p95_latency_ms", "p99_latency_ms"],
        var_name="percentile",
        value_name="latency_ms",
    )
    latency["percentile"] = latency["percentile"].map({
        "p50_latency_ms": "P50", "p95_latency_ms": "P95", "p99_latency_ms": "P99"
    })
    latency["序列"] = latency["model_id"] + " · " + latency["percentile"]
    left, right = st.columns([1.5, 1], gap="large")
    with left:
        st.markdown("#### 延迟分位数趋势")
        chart = _line_chart(
            latency,
            "date",
            "latency_ms",
            "序列",
            "延迟（ms）",
            [alt.Tooltip("date:T", title="日期", format="%Y-%m-%d"), alt.Tooltip("model_id:N", title="模型"), alt.Tooltip("percentile:N", title="分位数"), alt.Tooltip("latency_ms:Q", title="延迟", format=",.0f")],
        )
        st.altair_chart(chart, width="stretch")
    with right:
        st.markdown("#### 波动与稳定性")
        stability = chart_source[["date", "model_id", "stability_score", "performance_score"]].melt(
            ["date", "model_id"], var_name="metric", value_name="score"
        )
        stability["metric"] = stability["metric"].map({"stability_score": "稳定性", "performance_score": "综合体验"})
        stability["序列"] = stability["model_id"] + " · " + stability["metric"]
        chart = _line_chart(
            stability,
            "date",
            "score",
            "序列",
            "评分",
            [alt.Tooltip("date:T", title="日期", format="%Y-%m-%d"), alt.Tooltip("model_id:N", title="模型"), alt.Tooltip("metric:N", title="指标"), alt.Tooltip("score:Q", title="评分", format=".1f")],
        )
        st.altair_chart(chart, width="stretch")

    if compare_model is not None:
        comparison = operating[
            operating["model_id"].eq(compare_model)
            & operating["date"].between(window_start, latest_date)
        ]
        if comparison.empty:
            st.info("对比模型在所选窗口内没有数据。", icon=":material/info:")
        else:
            comparison_summary = _weighted_performance_summary(comparison)
            p99_change = (
                (current_summary["p99_latency_ms"] / comparison_summary["p99_latency_ms"] - 1) * 100
                if comparison_summary["p99_latency_ms"]
                else np.nan
            )
            stability_gap = current_summary["stability_score"] - comparison_summary["stability_score"]
            experience_gap = current_summary["performance_score"] - comparison_summary["performance_score"]
            with st.container(border=True):
                st.markdown(f"#### 与 {compare_model} 对比")
                comparison_cols = st.columns(2, gap="medium")
                with comparison_cols[0]:
                    st.markdown(
                        f"**典型响应时间**　"
                        f"{_latency_comparison_text(current_summary['p50_latency_ms'], comparison_summary['p50_latency_ms'], str(compare_model))}"
                    )
                    st.markdown(
                        f"**95% 请求响应时间**　"
                        f"{_latency_comparison_text(current_summary['p95_latency_ms'], comparison_summary['p95_latency_ms'], str(compare_model))}"
                    )
                    p99_direction = "降低" if p99_change < 0 else "增加" if p99_change > 0 else "持平"
                    st.markdown(f"**极端响应时间**　较 {compare_model} {p99_direction} {abs(p99_change):.1f}%")
                with comparison_cols[1]:
                    stability_direction = "高" if stability_gap > 0 else "低" if stability_gap < 0 else "持平"
                    experience_direction = "高" if experience_gap > 0 else "低" if experience_gap < 0 else "持平"
                    st.markdown(f"**稳定性**　较 {compare_model} {stability_direction} {abs(stability_gap):.1f} 分")
                    st.markdown(f"**综合体验**　较 {compare_model} {experience_direction} {abs(experience_gap):.1f} 分")
                    st.caption("对比值按所选窗口内日调用量加权，并非单次请求测试。")

    latest = selected.iloc[-1]
    with st.expander("展开技术指标", icon=":material/monitoring:"):
        st.markdown("**原始延迟值**")
        _metric_row([
            {"label": "P50", "value": f"{latest['p50_latency_ms']:,.0f} ms", "help": "50% 请求在该时间内完成。"},
            {"label": "P95", "value": f"{latest['p95_latency_ms']:,.0f} ms", "help": "95% 请求在该时间内完成。"},
            {"label": "P99", "value": f"{latest['p99_latency_ms']:,.0f} ms", "help": "99% 请求在该时间内完成，用于观察极端慢请求。"},
        ])
        st.markdown("**分数计算方式**")
        st.caption(
            "响应速度评分由 P50、P95、P99 相对模型自身前 7 日基线计算，等于基线约 80 分；"
            "综合体验评分 = 响应速度评分 × 90% + 稳定性评分 × 10%。"
        )
        detail_cols = st.columns(2, gap="large")
        with detail_cols[0]:
            st.markdown("**数据采样周期**")
            st.caption("请求级数据按日汇总；稳定性按日调用量采用1/3/6小时统计桶计算P95与成功率波动，降低低样本随机偏差。")
            st.markdown("**基线来源**")
            st.caption("模型名称与 TTFT 基线来自真实资源观测；请求延迟历史为基线校准后的模拟数据。")
        with detail_cols[1]:
            st.markdown("**异常判断方式**")
            st.caption("本页将响应速度或稳定性评分低于 40 的性能日标记为需重点关注；正式异常告警由检测模块独立生成。")
            st.markdown("**校准说明**")
            st.caption("校准用于使模拟请求的延迟量级贴近真实资源基线，不会将模拟请求标记为真实生产流量。")
        st.download_button(
            "下载当前窗口原始数据",
            chart_source.sort_values(["model_id", "date"]).to_csv(index=False).encode("utf-8-sig"),
            f"performance_raw_{model}_{window_start:%Y%m%d}_{latest_date:%Y%m%d}.csv",
            "text/csv",
            key="download_performance_raw",
            icon=":material/download:",
        )


def render_cost(operating: pd.DataFrame) -> None:
    _section(
        "成本分析",
        "查看模型的使用成本、质量表现和投入产出效率。",
    )
    with st.expander("数据说明", icon=":material/info:"):
        st.caption("当前没有真实账单数据，本页金额均为模拟价格假设，只用于相对成本趋势和模型间比较。")
        st.caption("质量评分来自同一套标准能力任务，不是根据价格反推的数据。")
    if operating.empty:
        st.info(
            "当前筛选范围没有成本数据，请调整全局日期或模型筛选。",
            icon=":material/info:",
        )
        return

    models = sorted(operating["model_id"].dropna().unique())
    control_main, control_time = st.columns([3, 2], gap="large", vertical_alignment="bottom")
    controls = control_main.container(horizontal=True, gap="medium", vertical_alignment="bottom")
    model = controls.selectbox("成本分析模型", models, key="cost_model", width=260)
    compare_options: list[str | None] = [None] + [item for item in models if item != model]
    compare_model = controls.selectbox(
        "添加对比模型",
        compare_options,
        format_func=lambda value: "不对比" if value is None else str(value),
        key="cost_compare_model",
        width=260,
        disabled=len(compare_options) == 1,
        help=(
            "从当前全局筛选中选择第二个模型。"
            if len(compare_options) > 1
            else "当前只有一个模型，请先在全局筛选中加入其他模型。"
        ),
    )
    compare_previous = controls.toggle(
        "与上一周期对比",
        value=True,
        key="cost_compare_previous",
    )
    time_controls = control_time.container(horizontal_alignment="right")
    window_label = time_controls.segmented_control(
        "时间范围",
        ["过去 7 天", "过去 30 天", "过去 90 天"],
        default="过去 30 天",
        required=True,
        key="cost_window",
        width="content",
    )

    window_days = {"过去 7 天": 7, "过去 30 天": 30, "过去 90 天": 90}
    days = window_days[str(window_label)]
    model_history = operating[operating["model_id"].eq(model)].sort_values("date")
    latest_date = pd.Timestamp(model_history["date"].max()).normalize()
    window_start = latest_date - pd.Timedelta(days=days - 1)
    previous_end = window_start - pd.Timedelta(days=1)
    previous_start = previous_end - pd.Timedelta(days=days - 1)
    selected = model_history[model_history["date"].between(window_start, latest_date)].copy()
    previous = model_history[model_history["date"].between(previous_start, previous_end)].copy()
    if selected.empty:
        st.info("所选时间范围没有可用成本数据。", icon=":material/info:")
        return

    current_summary = _weighted_cost_summary(selected)
    previous_summary = _weighted_cost_summary(previous)
    peer_window = operating[operating["date"].between(window_start, latest_date)]
    peer_average = _peer_window_average(
        peer_window,
        model,
        "cost_efficiency_score",
        _weighted_cost_summary,
    )
    weakest_label = (
        "成本效率"
        if current_summary["cost_efficiency_score"] <= current_summary["quality_score"]
        else "质量保障"
    )
    total_explanation = f"主要短板：{weakest_label}需关注"
    if pd.isna(peer_average):
        efficiency_explanation = "当前筛选范围暂无其他模型基线"
    else:
        peer_gap = current_summary["cost_efficiency_score"] - peer_average
        efficiency_explanation = (
            f"高于同窗口模型平均 {peer_gap:.1f} 分"
            if peer_gap >= 0
            else f"低于同窗口模型平均 {abs(peer_gap):.1f} 分"
        )
    quality_explanation = "来自统一标准能力任务的质量校准结果"
    conclusions = [
        ("成本总评分", current_summary["cost_performance_score"], total_explanation, "cost_performance_score"),
        ("成本效率", current_summary["cost_efficiency_score"], efficiency_explanation, "cost_efficiency_score"),
        ("质量保障", current_summary["quality_score"], quality_explanation, "quality_score"),
    ]
    conclusion_cols = st.columns(3, gap="medium")
    total_title, total_score, total_explanation, total_column = conclusions[0]
    total_level, total_badge_color = _performance_score_level(total_score)
    total_change, total_change_color = _score_change_text(
        total_score, previous_summary[total_column], compare_previous
    )
    with conclusion_cols[0].container(
        border=True, height=280, gap="xsmall", vertical_alignment="distribute"
    ):
        st.markdown(f"**{total_title}**")
        st.altair_chart(
            _health_gauge(total_score, _score_gauge_color(total_score)),
            width="stretch",
            theme=None,
        )
        total_tags = st.container(horizontal=True, gap="small")
        _score_level_badge(total_level, total_badge_color, target=total_tags)
        total_tags.badge(total_change, color=total_change_color)
    component_weights = {"成本效率": "70%", "质量保障": "30%"}
    component_help = {
        "成本效率": "综合单请求成本、千 Token 成本和单请求 Token 相对模型自身基线的变化，占成本总评分 70%。",
        "质量保障": "来自统一标准能力任务的质量评分，占成本总评分 30%。",
    }
    for column, (title, score, explanation, score_column) in zip(conclusion_cols[1:], conclusions[1:]):
        level, color = _performance_score_level(score)
        change_text, change_color = _score_change_text(
            score,
            previous_summary[score_column],
            compare_previous,
        )
        with column.container(
            border=True, height=280, gap="xsmall", vertical_alignment="distribute"
        ):
            st.caption(f"总评分构成 · 权重 {component_weights[title]}")
            title_row = st.container(
                horizontal=True,
                horizontal_alignment="distribute",
                vertical_alignment="center",
            )
            title_row.metric(title, f"{score:.0f} / 100", help=component_help[title])
            _score_level_badge(level, color, target=title_row)
            st.altair_chart(_score_bar(score), width="stretch", theme=None)
            st.caption(explanation)
            st.badge(change_text, color=change_color)
    st.caption("成本总评分由右侧两项计算：成本效率 × 70% + 质量保障 × 30%。")

    attention_days = selected[
        selected["cost_efficiency_score"].lt(40) | selected["quality_score"].lt(40)
    ]
    if attention_days.empty:
        st.caption("所选窗口内未出现成本效率或质量评分低于 40 的成本日。")
    else:
        st.warning(
            f"所选窗口内有 {attention_days['date'].nunique()} 天需重点关注，"
            "请在趋势图和技术指标中查看详情。",
            icon=":material/warning:",
        )

    chart_models = [str(model)] + ([str(compare_model)] if compare_model is not None else [])
    chart_source = operating[
        operating["model_id"].isin(chart_models)
        & operating["date"].between(window_start, latest_date)
    ].copy()
    chart_source["cost_per_1k_requests"] = chart_source["cost_per_request"] * 1000

    left, right = st.columns([1.35, 1], gap="large")
    with left:
        st.markdown("#### 单请求成本趋势")
        chart = _line_chart(
            chart_source,
            "date",
            "cost_per_1k_requests",
            "model_id",
            "成本（元 / 千次请求）",
            [alt.Tooltip("date:T", title="日期", format="%Y-%m-%d"), alt.Tooltip("model_id:N", title="模型"), alt.Tooltip("cost_per_1k_requests:Q", title="千次请求成本（元）", format=".3f"), alt.Tooltip("cost_per_request:Q", title="单请求成本（元）", format=".6f"), alt.Tooltip("cost_trend_ratio:Q", title="基线倍数", format=".3f")],
        )
        st.altair_chart(chart, width="stretch")
    with right:
        st.markdown("#### 质量—成本效率矩阵")
        latest_all = _latest_by_model(peer_window)
        chart = (
            alt.Chart(latest_all)
            .mark_circle(opacity=.86, stroke="white", strokeWidth=1.5)
            .encode(
                x=alt.X("cost_efficiency_score:Q", title="成本评分", scale=alt.Scale(domain=[0, 100])),
                y=alt.Y("quality_score:Q", title="质量评分", scale=alt.Scale(domain=[0, 100])),
                size=alt.Size("request_count:Q", title="调用量", scale=alt.Scale(range=[300, 1200])),
                color=alt.Color("model_id:N", title="模型"),
                tooltip=[alt.Tooltip("model_id:N", title="模型"), alt.Tooltip("quality_score:Q", title="质量", format=".1f"), alt.Tooltip("cost_efficiency_score:Q", title="成本效率", format=".1f"), alt.Tooltip("cost_performance_score:Q", title="成本总评分", format=".1f")],
            )
            .properties(height=330)
        )
        st.altair_chart(chart, width="stretch")

    if compare_model is not None:
        comparison = operating[
            operating["model_id"].eq(compare_model)
            & operating["date"].between(window_start, latest_date)
        ]
        if comparison.empty:
            st.info("对比模型在所选窗口内没有数据。", icon=":material/info:")
        else:
            comparison_summary = _weighted_cost_summary(comparison)
            request_pct = (
                (current_summary["cost_per_request"] / comparison_summary["cost_per_request"] - 1) * 100
                if comparison_summary["cost_per_request"]
                else np.nan
            )
            token_pct = (
                (current_summary["cost_per_1k_tokens"] / comparison_summary["cost_per_1k_tokens"] - 1) * 100
                if comparison_summary["cost_per_1k_tokens"]
                else np.nan
            )
            quality_gap = current_summary["quality_score"] - comparison_summary["quality_score"]
            efficiency_gap = current_summary["cost_efficiency_score"] - comparison_summary["cost_efficiency_score"]
            total_gap = current_summary["cost_performance_score"] - comparison_summary["cost_performance_score"]
            with st.container(border=True):
                st.markdown(f"#### 与 {compare_model} 对比")
                comparison_cols = st.columns(2, gap="medium")
                with comparison_cols[0]:
                    request_direction = "高" if request_pct > 0 else "低" if request_pct < 0 else "持平"
                    token_direction = "高" if token_pct > 0 else "低" if token_pct < 0 else "持平"
                    st.markdown(f"**单请求成本**　比 {compare_model} {request_direction} {abs(request_pct):.1f}%")
                    st.markdown(f"**千 Token 成本**　比 {compare_model} {token_direction} {abs(token_pct):.1f}%")
                with comparison_cols[1]:
                    quality_direction = "高" if quality_gap > 0 else "低" if quality_gap < 0 else "持平"
                    efficiency_direction = "高" if efficiency_gap > 0 else "低" if efficiency_gap < 0 else "持平"
                    total_direction = "高" if total_gap > 0 else "低" if total_gap < 0 else "持平"
                    st.markdown(f"**质量保障**　较 {compare_model} {quality_direction} {abs(quality_gap):.1f} 分")
                    st.markdown(f"**成本效率**　较 {compare_model} {efficiency_direction} {abs(efficiency_gap):.1f} 分")
                    st.markdown(f"**成本总评分**　较 {compare_model} {total_direction} {abs(total_gap):.1f} 分")

    latest = selected.iloc[-1]
    with st.expander("展开技术指标", icon=":material/monitoring:"):
        st.markdown("**原始成本值**")
        _metric_row([
            {"label": "单请求成本", "value": f"¥{latest['cost_per_request']:.6f}", "help": "当日估算总成本 ÷ 请求量。"},
            {"label": "千 Token 成本", "value": f"¥{latest['cost_per_1k_tokens']:.6f}", "help": "当日估算总成本 ÷ Token 总量 × 1,000。"},
            {"label": "历史基线倍数", "value": f"{latest['cost_trend_ratio']:.3f}×", "help": "当前单请求成本相对历史基线的倍数。"},
        ])
        st.markdown("**分数计算方式**")
        st.caption("成本效率按模型自身前 7 日基线评分，等于基线约 80 分；成本总评分 = 成本效率评分 × 70% + 质量评分 × 30%。")
        detail_cols = st.columns(2, gap="large")
        with detail_cols[0]:
            st.markdown("**数据采样周期**")
            st.caption("请求、Token 和估算成本按日汇总，页面窗口按日调用量加权。")
            st.markdown("**价格与基线来源**")
            st.caption("金额使用项目模型目录中的模拟价格；成本趋势基线为前 7 个历史日单请求成本中位数，至少 3 个历史日后启用。")
        with detail_cols[1]:
            st.markdown("**质量校准方式**")
            st.caption("同一能力维度内按标准任务权重汇总，再按有效运行次数加权成模型质量评分。")
            st.markdown("**异常判断方式**")
            st.caption("本页将成本效率或质量评分低于 40 的成本日标记为需重点关注，不将模拟金额解读为真实账单异常。")
        st.download_button(
            "下载当前窗口原始数据",
            chart_source.sort_values(["model_id", "date"]).to_csv(index=False).encode("utf-8-sig"),
            f"cost_raw_{model}_{window_start:%Y%m%d}_{latest_date:%Y%m%d}.csv",
            "text/csv",
            key="download_cost_raw",
            icon=":material/download:",
        )


def render_calibration(
    profiles: pd.DataFrame,
    capability: pd.DataFrame,
    diagnosis: pd.DataFrame,
    probe_runs: pd.DataFrame,
    probe_events: pd.DataFrame,
) -> None:
    _section(
        "模型能力校准",
        "综合模型能力、稳定性、性能与成本，判断它更适合承担哪类请求。",
    )
    with st.expander("数据说明", icon=":material/info:"):
        st.caption("当前能力得分和历史主动检查为模拟假设；模型名称与延迟基线已按真实资源数据校准。主动检查是系统定时发送标准请求，以确认模型能否正常响应，不代表真实业务流量。")
    if profiles.empty:
        st.info("缺少模型能力画像数据，请先运行 capability_calibration.py、model_operations.py 和 model_profile.py。")
        return
    model = st.selectbox("画像模型", sorted(profiles["model_id"].unique()), key="profile_model")
    profile = profiles[profiles["model_id"].eq(model)].sort_values("date").iloc[-1]

    route_score = float(profile["routing_readiness_score"])
    route_color = _score_gauge_color(route_score)
    if route_score >= 80:
        route_state, route_badge_color = "优先候选", "green"
    elif route_score >= 60:
        route_state, route_badge_color = "可用但需观察", "blue"
    elif route_score >= 40:
        route_state, route_badge_color = "谨慎分配流量", "orange"
    else:
        route_state, route_badge_color = "暂不建议扩大流量", "red"
    decision_score, decision_detail, score_evidence = st.columns(
        [25, 45, 30], gap="medium", vertical_alignment="top"
    )
    with decision_score:
        with st.container(
            border=True,
            height=330,
            key="routing_score_core",
            vertical_alignment="distribute",
            gap="xsmall",
        ):
            st.caption("综合路由评分")
            st.altair_chart(_health_gauge(route_score, route_color), width="stretch", theme=None)
            st.caption("分数越高，越适合承接新增请求。")
    with decision_detail:
        with st.container(
            border=True,
            height=330,
            key="routing_score_action",
            vertical_alignment="distribute",
            gap="xsmall",
        ):
            st.caption("路由建议")
            st.badge(route_state, color=route_badge_color)
            st.markdown(f"### {profile['recommended_role']}")
            st.write(profile["routing_action"])
            dominant = DIMENSION_LABELS.get(profile["dominant_capability"], profile["dominant_capability"])
            weakest = DIMENSION_LABELS.get(profile["weakest_capability"], profile["weakest_capability"])
            scene_cols = st.columns(2)
            scene_cols[0].markdown(f"**适合承接**  \n{dominant}类任务")
            scene_cols[1].markdown(f"**需要谨慎**  \n{weakest}类任务")
            st.caption(f"执行提示：{profile['recommended_action']}")
    with score_evidence:
        with st.container(
            border=True,
            height=330,
            key="routing_score_evidence",
            vertical_alignment="distribute",
            gap="xsmall",
        ):
            st.caption("评分依据")
            with st.container(gap="small"):
                st.markdown(f"**能力 35%**　{profile['capability_score']:.1f}")
                st.markdown(f"**稳定性 20%**　{profile['profile_stability_score']:.1f}")
                st.markdown(f"**性能 25%**　{profile['profile_performance_score']:.1f}")
                st.markdown(f"**成本 20%**　{profile['cost_performance_score']:.1f}")
            confidence = float(profile["confidence_score"])
            confidence_level, confidence_color = _performance_score_level(confidence)
            st.divider()
            confidence_row = st.container(
                horizontal=True,
                horizontal_alignment="distribute",
                vertical_alignment="center",
            )
            confidence_row.markdown(f"**证据可信度 · {confidence:.1f}**")
            _score_level_badge(confidence_level, confidence_color, target=confidence_row)
            st.altair_chart(_score_bar(confidence), width="stretch", theme=None)
            st.caption("表示本次画像的数据证据是否充分，不代表模型能力高低。")

    filtered_capability = capability[capability["model_id"].eq(model)].copy()
    model_diagnosis = diagnosis[diagnosis["model_id"].eq(model)].sort_values("date", ascending=False)
    diagnosis_summary = pd.DataFrame()
    task_col, diagnosis_col = st.columns([1, 1.15], gap="large", vertical_alignment="top")
    with task_col:
        st.markdown("#### 标准化测试任务")
        st.caption("在相同环境下比较模型各类任务的完成质量。")
        if filtered_capability.empty:
            st.info("当前模型没有标准化测试记录。")
        else:
            filtered_capability["能力维度"] = filtered_capability["capability_dimension"].map(DIMENSION_LABELS)
            chart = (
                alt.Chart(filtered_capability)
                .mark_bar(cornerRadiusEnd=4)
                .encode(
                    y=alt.Y("能力维度:N", title=None, sort="-x"),
                    x=alt.X("quality_score:Q", title="质量评分", scale=alt.Scale(domain=[0, 100])),
                    color=alt.Color("能力维度:N", legend=None),
                    tooltip=[alt.Tooltip("能力维度:N"), alt.Tooltip("quality_score:Q", title="质量", format=".1f"), alt.Tooltip("consistency_score:Q", title="一致性", format=".1f"), alt.Tooltip("p95_latency_ms:Q", title="P95 延迟", format=",.0f")],
                )
                .properties(height=300)
            )
            st.altair_chart(chart, width="stretch")
            with st.expander("查看标准化测试明细"):
                st.dataframe(
                    filtered_capability,
                    column_order=["能力维度", "run_count", "pass_rate", "quality_score", "consistency_score", "p50_latency_ms", "p95_latency_ms"],
                    column_config={
                        "能力维度": "标准任务维度",
                        "run_count": "样本数",
                        "pass_rate": st.column_config.NumberColumn("通过率", format="%.1f%%"),
                        "quality_score": st.column_config.NumberColumn("能力/质量", format="%.1f"),
                        "consistency_score": st.column_config.NumberColumn("稳定性", format="%.1f"),
                        "p50_latency_ms": st.column_config.NumberColumn("P50（ms）", format="%,.0f"),
                        "p95_latency_ms": st.column_config.NumberColumn("P95（ms）", format="%,.0f"),
                    },
                    hide_index=True,
                    height=285,
                )
    with diagnosis_col:
        st.markdown("#### 真实使用与主动检查")
        st.caption("对比真实使用和系统标准检查，帮助判断问题来自模型还是运行环境。")
        if model_diagnosis.empty:
            st.info("当前筛选范围没有该模型的诊断记录。")
        else:
            latest_diagnosis = model_diagnosis.iloc[0]
            with st.container(border=True):
                st.markdown(f"**最新诊断 · {latest_diagnosis['date']:%Y-%m-%d}**")
                real_score, probe_score = st.columns(2)
                real_score.metric("真实表现", f"{latest_diagnosis['performance_score']:.1f}")
                probe_score.metric("主动检查", f"{latest_diagnosis['probe_performance_score']:.1f}", help="系统在固定环境下发送标准请求后得到的表现评分。技术上称为主动拨测。")
                st.caption(f"表现差异：{_performance_gap_label(latest_diagnosis['performance_gap_score'])}")
                st.markdown(f"**原因判断：** {latest_diagnosis['diagnosis_reason']}")
                st.markdown(f"**切换判断：** {latest_diagnosis['switch_recommendation']}")
                st.info(latest_diagnosis["recommended_action"], icon=":material/recommend:")

    st.markdown("#### 多模型路由画像")
    st.caption("横向比较各模型承接新增请求的适合程度。")
    st.dataframe(
        profiles.sort_values("profile_rank"),
        column_order=["profile_rank", "model_id", "routing_readiness_score", "capability_score", "profile_stability_score", "profile_performance_score", "cost_performance_score", "confidence_score", "recommended_role"],
        column_config={
            "profile_rank": st.column_config.NumberColumn("排名", format="#%d", width="small", pinned=True),
            "model_id": st.column_config.TextColumn("模型", width="medium", pinned=True),
            "routing_readiness_score": st.column_config.ProgressColumn("综合路由评分", min_value=0, max_value=100, format="%.1f", width="medium"),
            "capability_score": st.column_config.NumberColumn("能力", format="%.1f", width="small"),
            "profile_stability_score": st.column_config.NumberColumn("稳定性", format="%.1f", width="small"),
            "profile_performance_score": st.column_config.NumberColumn("性能", format="%.1f", width="small"),
            "cost_performance_score": st.column_config.NumberColumn("成本", format="%.1f", width="small"),
            "confidence_score": st.column_config.NumberColumn("可信度", format="%.1f", width="small"),
            "recommended_role": st.column_config.TextColumn("建议角色", width="large"),
        },
        hide_index=True,
        height=250,
    )
    full_capability_data = st.expander("查看完整能力数据")
    with full_capability_data.expander("查看完整模型画像"):
        st.dataframe(
            profiles.sort_values("profile_rank"),
            column_order=["profile_rank", "model_id", "capability_score", "profile_stability_score", "profile_performance_score", "confidence_score", "routing_readiness_score", "dominant_capability", "weakest_capability", "recommended_role", "routing_action"],
            column_config={
                "profile_rank": st.column_config.NumberColumn("排名", format="#%d"),
                "model_id": "模型",
                "capability_score": st.column_config.NumberColumn("能力", format="%.1f"),
                "profile_stability_score": st.column_config.NumberColumn("稳定性", format="%.1f"),
                "profile_performance_score": st.column_config.NumberColumn("性能", format="%.1f"),
                "confidence_score": st.column_config.NumberColumn("可信度", format="%.1f"),
                "routing_readiness_score": st.column_config.ProgressColumn("综合路由评分", min_value=0, max_value=100, format="%.1f"),
                "dominant_capability": "优势能力",
                "weakest_capability": "相对弱项",
                "recommended_role": "建议角色",
                "routing_action": "路由动作",
            },
            hide_index=True,
        )

    if not model_diagnosis.empty:
        diagnosis_summary = model_diagnosis.head(7).copy()
        diagnosis_summary["表现差异"] = diagnosis_summary["performance_gap_score"].map(_performance_gap_label)

        with full_capability_data.expander("查看原始诊断证据与完整数据"):
            st.caption("以下字段用于复核诊断计算，默认折叠以避免干扰路由决策。")
            st.dataframe(
                model_diagnosis,
                column_order=["date", "success_rate", "probe_http_success_rate", "p95_latency_ms", "probe_p95_latency_ms", "performance_score", "probe_performance_score", "performance_gap_score", "stability_score", "probe_consistency_score", "diagnosis_reason", "switch_recommendation", "recommended_action"],
                column_config={
                    "date": st.column_config.DateColumn("日期", format="YYYY-MM-DD"),
                    "success_rate": st.column_config.NumberColumn("真实成功率", format="%.2f%%"),
                    "probe_http_success_rate": st.column_config.NumberColumn("主动检查成功率", format="%.2f%%"),
                    "p95_latency_ms": st.column_config.NumberColumn("真实 P95", format="%,.0f ms"),
                    "probe_p95_latency_ms": st.column_config.NumberColumn("主动检查 P95", format="%,.0f ms"),
                    "performance_score": st.column_config.NumberColumn("真实性能分", format="%.1f"),
                    "probe_performance_score": st.column_config.NumberColumn("主动检查性能分", format="%.1f"),
                    "performance_gap_score": st.column_config.NumberColumn("性能分差", format="%.1f"),
                    "stability_score": st.column_config.NumberColumn("真实稳定性", format="%.1f"),
                    "probe_consistency_score": st.column_config.NumberColumn("主动检查一致性", format="%.1f"),
                    "diagnosis_reason": "原因判断",
                    "switch_recommendation": "切换判断",
                    "recommended_action": "建议动作",
                },
                hide_index=True,
                height=330,
            )
            st.download_button(
                "下载完整融合诊断 CSV",
                model_diagnosis.to_csv(index=False).encode("utf-8-sig"),
                f"{model}_fusion_diagnosis.csv",
                "text/csv",
                icon=":material/download:",
            )

    with full_capability_data.expander("查看主动可用性检查（技术详情与导出）"):
        if probe_runs.empty:
            st.info("当前范围没有主动可用性检查记录。")
        else:
            st.caption("系统定时发送标准请求，检查模型能否连接、成功响应以及响应是否过慢。以下为技术明细，普通用户通常无需查看。")
            availability = probe_runs["success"].astype(bool).mean() * 100
            cols = st.columns(3)
            cols[0].metric("检查成功率", f"{availability:.2f}%", border=True)
            cols[1].metric("较慢请求的等待时间", f"{probe_runs['ttft_ms'].quantile(.95):,.0f} ms", help="95% 的检查请求会在这个时间内开始返回内容。", border=True)
            cols[2].metric("异常检查次数", len(probe_events), border=True)
            latest_probe = probe_runs.sort_values("started_at").groupby("probe_id", as_index=False).tail(1)
            st.dataframe(
                latest_probe,
                column_order=["probe_name_cn", "provider", "model_id", "region", "success", "latency_ms", "ttft_ms", "failed_assertions"],
                column_config={"probe_name_cn": "检查项目", "provider": "供应商", "model_id": "模型", "region": "区域", "success": "是否成功", "latency_ms": "总等待时间（ms）", "ttft_ms": "开始返回时间（ms）", "failed_assertions": "未通过项目"},
                hide_index=True,
            )
            c1, c2 = st.columns(2)
            c1.download_button("下载检查记录 CSV", probe_runs.to_csv(index=False).encode("utf-8-sig"), "probe_runs.csv", "text/csv", icon=":material/download:")
            c2.download_button("下载异常记录 CSV", probe_events.to_csv(index=False).encode("utf-8-sig"), "probe_alerts.csv", "text/csv", icon=":material/download:")

    if not model_diagnosis.empty:
        with full_capability_data.expander("查看近 7 次诊断轨迹"):
            st.caption("按日期回看真实使用表现、系统主动检查结果和当时的路由判断。")
            st.dataframe(
                diagnosis_summary,
                column_order=["date", "performance_score", "probe_performance_score", "表现差异", "diagnosis_reason", "switch_recommendation"],
                column_config={
                    "date": st.column_config.DateColumn("日期", format="YYYY-MM-DD"),
                    "performance_score": st.column_config.ProgressColumn("真实表现", min_value=0, max_value=100, format="%.1f"),
                    "probe_performance_score": st.column_config.ProgressColumn("主动检查", min_value=0, max_value=100, format="%.1f"),
                    "表现差异": "差异",
                    "diagnosis_reason": "原因判断",
                    "switch_recommendation": "切换判断",
                },
                hide_index=True,
                height=290,
            )


def _render_resource_capacity_legacy(
    model_series: pd.DataFrame,
    instance_hourly: pd.DataFrame,
    capacity: pd.DataFrame,
) -> None:
    _section(
        "容量诊断",
        "判断当前资源是否足以承接流量，并给出扩容、迁移或继续观察的建议。",
    )
    with st.expander("数据说明", icon=":material/info:"):
        st.caption("最新一天来自真实资源工作簿；此前 89 天是依据真实日内曲线校准生成的模拟历史，仅用于趋势和基线演示。实例标识已经匿名化，中台模型不在监控范围内。当前没有独立的容量评分规则，因此本页直接展示风险状态和判断依据。")
    if model_series.empty or instance_hourly.empty or capacity.empty:
        st.info("尚未导入完整的真实资源数据。请把每日三份 Excel 放入指定目录后运行“更新每日数据.bat”。")
        return

    latest_date = pd.Timestamp(capacity["date"].max())
    latest = capacity[capacity["date"].eq(latest_date)].copy()
    high_npu_samples = int(latest["high_npu_samples"].sum())
    minimum_headroom = float(latest["hbm_headroom_pct"].min())
    risk_count = int(latest["capacity_state"].eq("容量风险").sum())
    attention_count = int(latest["capacity_state"].eq("需要关注").sum())
    state_priority = latest["capacity_state"].map({"容量风险": 0, "需要关注": 1}).fillna(2)
    primary = latest.assign(_priority=state_priority).sort_values(
        ["_priority", "hbm_headroom_pct", "npu_max"]
    ).iloc[0]
    if risk_count:
        overall_state, state_color = "存在容量风险", "red"
        action_title = f"优先处理 {primary['model_id']}"
        action_detail = "先核查显存驻留和高负载尖峰，再评估增加实例、迁移流量或降低路由权重。"
    elif attention_count:
        overall_state, state_color = "容量需要关注", "orange"
        action_title = "暂不扩大新增流量"
        action_detail = "保持当前实例规模，观察忙时并发和显存余量，达到趋势基线后再决定是否扩容。"
    else:
        overall_state, state_color = "容量充足", "green"
        action_title = "维持当前容量配置"
        action_detail = "当前没有明显资源瓶颈，可继续承接流量并保持日常观察。"

    status_col, action_col, evidence_col = st.columns(
        [25, 45, 30], gap="medium", vertical_alignment="top"
    )
    with status_col:
        with st.container(border=True, height=350, key="capacity_status", vertical_alignment="distribute", gap="xsmall"):
            st.caption("整体容量状态")
            _capacity_state_badge(overall_state, state_color)
            st.markdown(f"## {risk_count} 个高风险模型")
            st.markdown(f"**{attention_count} 个需要关注**")
            with st.popover("查看涉及模型", icon=":material/visibility:"):
                for _, row in latest.assign(_priority=state_priority).sort_values(["_priority", "model_id"]).iterrows():
                    color = "red" if row["capacity_state"] == "容量风险" else "orange"
                    _capacity_state_badge(f"{row['model_id']} · {row['capacity_state']}", color)
                    st.caption(row["diagnosis"])
            st.caption(f"监控 {latest['model_id'].nunique()} 个模型 · {int(latest['instance_count'].sum())} 个实例")
            st.caption(f"数据日期：{latest_date:%Y-%m-%d}")
    with action_col:
        with st.container(border=True, height=350, key="capacity_action", vertical_alignment="distribute", gap="xsmall"):
            st.caption("容量建议")
            st.markdown(f"### {action_title}")
            st.write(action_detail)
            st.markdown(f"**主要判断：** {primary['diagnosis']}")
            st.caption(f"优先检查模型：{primary['model_id']} · 当前状态：{primary['capacity_state']}")
    with evidence_col:
        with st.container(border=True, height=350, key="capacity_evidence", vertical_alignment="distribute", gap="xsmall"):
            st.caption("判断依据")
            with st.container(gap="small"):
                st.markdown(f"**最大等待**　{int(latest['waiting_max_busy'].max())}")
                st.markdown(f"**NPU 高负载样本**　{high_npu_samples}")
                st.markdown(f"**最小 HBM 余量**　{minimum_headroom:.1f}%")
                st.markdown(f"**NPU 最高峰值**　{float(latest['npu_max'].max()):.0f}%")
            st.caption("等待反映排队压力；HBM 余量反映显存空间；NPU 峰值用于识别短时算力拥堵。")

    if not bool(latest["baseline_ready"].all()):
        observed_days = int(latest.get("observed_days", pd.Series([1])).max())
        st.info(
            f"当前仅有 {observed_days} 天真实记录；达到 7 天后再启用容量趋势基线，当前结论只反映当日状态。",
            icon=":material/calendar_clock:",
        )

    trend_title_col, trend_window_col = st.columns([1, 1], vertical_alignment="center")
    trend_title_col.markdown("#### 容量趋势")
    capacity_window = trend_window_col.container(horizontal_alignment="right").segmented_control(
        "时间范围",
        ["过去 7 天", "过去 30 天", "过去 90 天"],
        default="过去 30 天",
        required=True,
        key="capacity_window",
        label_visibility="collapsed",
        width="content",
    )
    capacity_days = {"过去 7 天": 7, "过去 30 天": 30, "过去 90 天": 90}[str(capacity_window)]
    capacity_window_start = latest_date - pd.Timedelta(days=capacity_days - 1)
    trend_col, diagnosis_col = st.columns([1.35, 1], gap="large", vertical_alignment="top")
    with trend_col:
        metric_label = st.segmented_control(
            "资源趋势指标",
            ["运行并发", "等待队列", "首字延迟", "服务吞吐"],
            default="运行并发",
            required=True,
            key="resource_metric",
        )
    metric_specs = {
        "运行并发": ("running", "同时处理的请求数", "越高表示同时处理的请求越多；持续接近实例数时需要关注。"),
        "等待队列": ("waiting", "等待处理的请求数", "大于 0 表示请求开始排队；持续排队通常需要扩容或分流。"),
        "首字延迟": ("ttft_ms", "开始返回内容的等待时间（ms）", "越低越好；突然升高可能来自排队、网络或模型负载。"),
        "服务吞吐": ("tokens_per_second", "每秒输出 Token", "表示服务输出速度；需结合并发和等待一起判断容量。"),
    }
    metric_column, axis_title, metric_help = metric_specs[str(metric_label)]
    chart_data = model_series[
        pd.to_datetime(model_series["timestamp"]).between(
            capacity_window_start, latest_date + pd.Timedelta(days=1)
        )
    ].dropna(subset=[metric_column]).copy()
    with trend_col:
        if chart_data.empty:
            st.info("当前指标没有可用趋势数据。")
        else:
            peak_row = chart_data.loc[chart_data[metric_column].idxmax()]
            st.caption(
                f"{metric_help} 当前最高值为 {peak_row[metric_column]:,.1f}，"
                f"来自 {peak_row['model_id']}。"
            )
        chart = (
            alt.Chart(chart_data)
            .mark_line(interpolate="step-after", strokeWidth=2)
            .encode(
                x=alt.X("timestamp:T", title="时间"),
                y=alt.Y(f"{metric_column}:Q", title=axis_title),
                color=alt.Color("model_id:N", title="模型"),
                tooltip=[
                    alt.Tooltip("timestamp:T", title="时间", format="%Y-%m-%d %H:%M"),
                    alt.Tooltip("model_id:N", title="模型"),
                    alt.Tooltip(f"{metric_column}:Q", title=str(metric_label), format=",.2f"),
                ],
            )
            .properties(height=340)
        )
        st.altair_chart(chart, width="stretch")
    with diagnosis_col:
        st.markdown("#### 最新容量判断")
        st.caption("按风险优先级查看每个模型当前最需要关注的问题。")
        for _, row in latest.assign(_priority=state_priority).sort_values(["_priority", "model_id"]).iterrows():
            badge_color = "red" if row["capacity_state"] == "容量风险" else "orange"
            with st.container(border=True):
                heading, badge = st.columns([3, 1], vertical_alignment="center")
                heading.markdown(f"**{row['model_id']}**")
                with badge:
                    _capacity_state_badge(row["capacity_state"], badge_color)
                st.write(row["diagnosis"])
                st.caption(
                    f"实例 {int(row['instance_count'])} · 忙时并发 {int(row['running_max_busy'])} · "
                    f"NPU 峰值 {row['npu_max']:.0f}% · HBM 余量 {row['hbm_headroom_pct']:.1f}%"
                )

    left, right = st.columns([1.2, 1.15], gap="large", vertical_alignment="top")
    with left:
        st.markdown("#### 实例负载分布")
        st.caption("颜色越深表示该匿名实例在对应时段的 NPU 峰值越高。")
        heatmap_model = st.selectbox(
            "查看模型",
            sorted(instance_hourly["model_id"].unique()),
            key="resource_heatmap_model",
        )
        heatmap_data = instance_hourly[
            instance_hourly["model_id"].eq(heatmap_model)
        ].copy()
        heatmap = (
            alt.Chart(heatmap_data)
            .mark_rect()
            .encode(
                x=alt.X("hour:T", title="小时"),
                y=alt.Y("instance_id:N", title="匿名实例"),
                color=alt.Color(
                    "npu_max:Q",
                    title="NPU 峰值（%）",
                    scale=alt.Scale(domain=[0, 100], scheme="yelloworangered"),
                ),
                tooltip=[
                    alt.Tooltip("hour:T", title="时间", format="%Y-%m-%d %H:%M"),
                    alt.Tooltip("instance_id:N", title="匿名实例"),
                    alt.Tooltip("npu_mean:Q", title="平均 NPU", format=".1f"),
                    alt.Tooltip("npu_max:Q", title="峰值 NPU", format=".1f"),
                ],
            )
            .properties(height=330)
        )
        st.altair_chart(heatmap, width="stretch")
    with right:
        st.markdown("#### 模型容量概览")
        st.caption("先看状态和余量，需要复核时再展开完整技术数据。")
        st.dataframe(
            latest.sort_values(["hbm_headroom_pct", "model_id"]),
            column_order=["model_id", "capacity_state", "instance_count", "hbm_headroom_pct", "npu_max"],
            column_config={
                "model_id": st.column_config.TextColumn("模型", pinned=True, width="medium"),
                "capacity_state": st.column_config.TextColumn("状态", width="small"),
                "instance_count": st.column_config.NumberColumn("实例", width="small"),
                "hbm_headroom_pct": st.column_config.NumberColumn("显存余量", format="%.1f%%", width="small"),
                "npu_max": st.column_config.NumberColumn("NPU峰值", format="%.0f%%", width="small"),
            },
            hide_index=True,
            height=330,
        )

    with st.expander("查看完整容量与资源明细"):
        st.caption("以下为容量判断使用的技术字段，供排障和数据复核。")
        st.dataframe(
            latest.sort_values(["capacity_state", "model_id"]),
            column_order=[
                "model_id", "capacity_state", "instance_count", "running_max_busy",
                "waiting_max_busy", "concurrency_ratio", "ttft_mean_ms",
                "tokens_per_second_mean", "npu_mean", "npu_p95", "npu_max",
                "cache_pct", "hbm_pct", "hbm_headroom_pct", "diagnosis",
            ],
            column_config={
                "model_id": "模型",
                "capacity_state": "容量状态",
                "instance_count": "实例数",
                "running_max_busy": "忙时并发",
                "waiting_max_busy": "最大等待",
                "concurrency_ratio": st.column_config.ProgressColumn("并发/实例", min_value=0, max_value=1, format="%.1f"),
                "ttft_mean_ms": st.column_config.NumberColumn("平均 TTFT", format="%,.0f ms"),
                "tokens_per_second_mean": st.column_config.NumberColumn("服务吞吐", format="%.2f token/s"),
                "npu_mean": st.column_config.NumberColumn("平均 NPU", format="%.1f%%"),
                "npu_p95": st.column_config.NumberColumn("NPU P95", format="%.1f%%"),
                "npu_max": st.column_config.NumberColumn("NPU 峰值", format="%.0f%%"),
                "cache_pct": st.column_config.NumberColumn("Cache", format="%.2f%%"),
                "hbm_pct": st.column_config.NumberColumn("HBM", format="%.2f%%"),
                "hbm_headroom_pct": st.column_config.NumberColumn("HBM 余量", format="%.2f%%"),
                "diagnosis": "诊断说明",
            },
            hide_index=True,
            height=300,
        )
        st.download_button(
            "下载容量诊断原始数据",
            latest.to_csv(index=False).encode("utf-8-sig"),
            f"capacity_diagnosis_{latest_date:%Y%m%d}.csv",
            "text/csv",
            icon=":material/download:",
        )


def render_resource_capacity(
    model_series: pd.DataFrame,
    instance_hourly: pd.DataFrame,
    capacity: pd.DataFrame,
) -> None:
    _section(
        "容量诊断",
        "快速判断是否存在容量风险、哪个模型需要优先处理，以及下一步应采取什么行动。",
    )
    if model_series.empty or instance_hourly.empty or capacity.empty:
        st.info(
            "尚未导入完整的资源数据。请把每日三份 Excel 放入指定目录后运行“更新每日数据.bat”。",
            icon=":material/database_off:",
        )
        return

    latest_date = pd.Timestamp(capacity["date"].max()).normalize()
    latest = capacity[pd.to_datetime(capacity["date"]).dt.normalize().eq(latest_date)].copy()
    risk_priority = {"容量风险": 0, "需要关注": 1, "容量充足": 2}
    latest["_priority"] = latest["capacity_state"].map(risk_priority).fillna(3)
    latest = latest.sort_values(["_priority", "hbm_headroom_pct", "npu_max"], ascending=[True, True, False])
    primary = latest.iloc[0]
    risk_count = int(latest["capacity_state"].eq("容量风险").sum())
    attention_count = int(latest["capacity_state"].eq("需要关注").sum())
    affected_requests = int(latest["waiting_max_busy"].clip(lower=0).sum())

    header_context, header_window = st.columns([1.2, 1], vertical_alignment="center")
    header_context.caption(
        f"当前资源池 · {latest['model_id'].nunique()} 个模型 · "
        f"{int(latest['instance_count'].sum())} 个实例 · 数据日期 {latest_date:%Y-%m-%d}"
    )
    capacity_window = header_window.container(horizontal_alignment="right").segmented_control(
        "趋势时间范围",
        ["过去 7 天", "过去 30 天", "过去 90 天"],
        default="过去 30 天",
        required=True,
        key="capacity_window_decision",
        width="content",
    )
    capacity_days = {"过去 7 天": 7, "过去 30 天": 30, "过去 90 天": 90}[str(capacity_window)]
    capacity_window_start = latest_date - pd.Timedelta(days=capacity_days - 1)

    with st.expander("数据说明", icon=":material/info:"):
        st.caption(
            "最新一天来自真实资源工作簿；此前 89 天是依据真实日内曲线校准生成的模拟历史，"
            "只用于趋势和基线演示。页面保留原始容量判断逻辑，并将技术结论翻译为通俗说明。"
        )

    primary_state, primary_color, primary_icon = _capacity_plain_state(str(primary["capacity_state"]))
    if risk_count:
        overview_title = f"当前有 {risk_count} 个模型存在高峰容量风险"
    elif attention_count:
        overview_title = f"当前有 {attention_count} 个模型接近容量上限"
    else:
        overview_title = "当前模型容量充足"
    impact_summary = (
        f"已发现 {affected_requests} 个高峰等待请求"
        if affected_requests
        else "尚未发现请求排队，当前没有明确影响线上请求"
    )

    with st.container(border=True, key="capacity_risk_overview"):
        overview_main, overview_metrics = st.columns([1.65, 1], gap="large", vertical_alignment="center")
        with overview_main:
            st.caption("风险总览")
            st.markdown(f"## {primary_icon} {overview_title}")
            st.markdown(
                f"**{primary['model_id']}** 最需要优先处理。"
                f"{_capacity_plain_reason(primary)}{_capacity_plain_impact(primary)}"
            )
            st.markdown(f"**当前业务影响：** {impact_summary}")
            st.markdown(f"**建议操作：** {_capacity_plain_action(primary)}")
            overview_focus = st.segmented_control(
                "风险详情",
                ["处理建议", "高负载实例"],
                default="处理建议",
                required=True,
                key="capacity_overview_focus",
                help="选择后，下方会显示对应内容；蓝色选项表示当前正在查看的内容。",
            )
            if overview_focus == "处理建议":
                st.info(
                    f"{primary['model_id']}：{_capacity_plain_action(primary)}",
                    icon=":material/recommend:",
                )
            else:
                st.caption(
                    "这里列出最新资源日中出现高负载的匿名实例。高负载表示算力峰值达到 70% 以上，"
                    "用于定位需要技术人员进一步检查的实例。"
                )
                primary_instances = instance_hourly[
                    instance_hourly["model_id"].eq(primary["model_id"])
                    & pd.to_datetime(instance_hourly["date"]).dt.normalize().eq(latest_date)
                ]
                high_load_instances = (
                    primary_instances.groupby("instance_id", as_index=False)
                    .agg(
                        npu_max=("npu_max", "max"),
                        npu_mean=("npu_mean", "mean"),
                        high_npu_samples=("high_npu_samples", "sum"),
                    )
                    .query("npu_max >= 70 or high_npu_samples > 0")
                    .sort_values(["npu_max", "high_npu_samples"], ascending=False)
                )
                if high_load_instances.empty:
                    st.success("最新资源日没有发现高负载实例。", icon=":material/check_circle:")
                else:
                    st.dataframe(
                        high_load_instances,
                        column_order=["instance_id", "npu_max", "npu_mean", "high_npu_samples"],
                        column_config={
                            "instance_id": "匿名实例",
                            "npu_max": st.column_config.NumberColumn("算力峰值", format="%.0f%%"),
                            "npu_mean": st.column_config.NumberColumn("平均算力", format="%.1f%%"),
                            "high_npu_samples": "高负载记录数",
                        },
                        hide_index=True,
                        height=155,
                    )
        with overview_metrics:
            risk_models = latest.loc[latest["capacity_state"].eq("容量风险"), "model_id"].astype(str).tolist()
            attention_models = latest.loc[latest["capacity_state"].eq("需要关注"), "model_id"].astype(str).tolist()
            affected_models = latest.loc[latest["waiting_max_busy"].gt(0), "model_id"].astype(str).tolist()
            metric_specs = [
                (
                    "高风险模型",
                    risk_count,
                    risk_models,
                    "高峰期可能处理不过来，需要优先采取行动的模型数量。",
                ),
                (
                    "接近容量上限",
                    attention_count,
                    attention_models,
                    "目前还能工作，但继续增加流量可能出现压力的模型数量。",
                ),
                (
                    "当前受影响请求",
                    affected_requests,
                    affected_models,
                    "最新资源日中，各模型高峰等待队列最大值的合计。0 表示尚未发现排队。",
                ),
            ]
            metric_columns = st.columns(3, gap="small", vertical_alignment="top")
            for metric_column, (label, value, models, help_text) in zip(
                metric_columns, metric_specs, strict=True
            ):
                with metric_column.container(
                    border=True,
                    height=220,
                    vertical_alignment="distribute",
                    key=f"capacity_summary_{label}",
                ):
                    with st.container(gap=None):
                        st.metric(label, value, help=help_text)
                    with st.container(gap=None):
                        st.caption(f"涉及：{'、'.join(models) if models else '暂无'}")

    st.markdown("### 模型处理清单")
    st.caption("已按风险优先级排序；清单只保留关键状态，完整影响与建议在下方查看。")
    for _, row in latest.iterrows():
        state_label, state_color, state_icon = _capacity_plain_state(str(row["capacity_state"]))
        with st.container(border=True, key=f"capacity_action_item_{row['model_id']}"):
            identity, status, signal, action = st.columns(
                [1.35, 1.45, 2.1, 1.25],
                gap="medium",
                vertical_alignment="center",
            )
            with identity:
                st.markdown(f"**{row['model_id']}**")
                st.caption(f"{int(row['instance_count'])} 个实例")
            with status:
                _capacity_state_badge(state_label, state_color)
            with signal:
                st.caption("关键状态")
                st.write(_capacity_key_signal(row))
            with action:
                st.button(
                    "查看处理建议",
                    icon=":material/link:",
                    key=f"capacity_select_{row['model_id']}",
                    on_click=_select_capacity_model,
                    args=(str(row["model_id"]),),
                    width="stretch",
                )

    model_options = latest["model_id"].astype(str).tolist()
    if st.session_state.get("capacity_detail_model") not in model_options:
        st.session_state["capacity_detail_model"] = str(primary["model_id"])
    if st.session_state.pop("capacity_scroll_to_detail", False):
        _CAPACITY_DETAIL_SCROLL(
            data={"enabled": True},
            key=f"capacity-scroll-{st.session_state.get('capacity_scroll_request', 0)}",
            height=1,
            width="stretch",
        )
    detail_heading, detail_selector = st.columns([1.2, 1], vertical_alignment="bottom")
    detail_heading.markdown("### 选中模型详情")
    selected_model = detail_selector.selectbox(
        "查看模型",
        model_options,
        key="capacity_detail_model",
        help="可从处理清单点击模型，也可以在这里直接切换。",
    )
    selected = latest[latest["model_id"].eq(selected_model)].iloc[0]
    selected_state, selected_color, selected_icon = _capacity_plain_state(str(selected["capacity_state"]))

    with st.container(border=True, key="capacity_selected_summary"):
        summary_left, summary_right = st.columns([1.35, 1], gap="large", vertical_alignment="top")
        with summary_left:
            st.caption("当前发生了什么")
            st.markdown(f"## {selected_icon} {selected_model}")
            _capacity_state_badge(selected_state, selected_color)
            st.write(_capacity_plain_reason(selected))
            st.markdown(f"**可能造成的影响：** {_capacity_plain_impact(selected)}")
            st.markdown(f"**是否已经影响线上用户：** {_capacity_current_impact(selected)}")
        with summary_right:
            st.caption("建议如何处理")
            st.markdown(f"### {_capacity_plain_action(selected)}")
            st.write("处理后重点观察等待队列是否持续为 0、显存余量是否回升，以及高峰响应时间是否改善。")
            st.caption("下方四项指标直接展示本次判断所依据的容量信号。")

        indicator_columns = st.columns(4, gap="medium")
        indicator_specs = [
            (
                "NPU 峰值",
                f"{selected['npu_max']:.0f}%",
                "表示算力在最忙时使用了多少。",
                "算力曾经完全占满" if selected["npu_max"] >= 95 else f"算力最高使用到 {selected['npu_max']:.0f}%",
                "已满载" if selected["npu_max"] >= 95 else "仍有算力余量",
                "red" if selected["npu_max"] >= 95 else "green",
            ),
            (
                "HBM 余量",
                f"{selected['hbm_headroom_pct']:.1f}%",
                "表示模型占用后还剩多少显存空间。",
                f"显存最低只剩 {selected['hbm_headroom_pct']:.1f}%",
                "显存接近用完" if selected["hbm_headroom_pct"] <= 5 else "显存余量需关注" if selected["hbm_headroom_pct"] <= 10 else "显存余量充足",
                "red" if selected["hbm_headroom_pct"] <= 5 else "orange" if selected["hbm_headroom_pct"] <= 10 else "green",
            ),
            (
                "忙时并发",
                f"{int(selected['running_max_busy'])}",
                "表示高峰时模型同时处理多少个请求。",
                f"高峰时同时处理 {int(selected['running_max_busy'])} 个请求",
                "已达到实例上限" if selected["concurrency_ratio"] >= 1 else "接近实例上限" if selected["concurrency_ratio"] >= 0.8 else "并发仍有空间",
                "red" if selected["concurrency_ratio"] >= 1 else "orange" if selected["concurrency_ratio"] >= 0.8 else "green",
            ),
            (
                "等待队列",
                f"{int(selected['waiting_max_busy'])}",
                "表示高峰时有多少请求还没开始处理。",
                "当前还没有请求排队" if selected["waiting_max_busy"] == 0 else f"最多有 {int(selected['waiting_max_busy'])} 个请求等待处理",
                "暂未产生排队" if selected["waiting_max_busy"] == 0 else "已出现排队",
                "green" if selected["waiting_max_busy"] == 0 else "red",
            ),
        ]
        for column, (label, value, tooltip, explanation, status, color) in zip(indicator_columns, indicator_specs, strict=True):
            with column.container(border=True, key=f"capacity_indicator_{label}"):
                st.markdown(f"**{label}**", help=tooltip)
                st.markdown(f"## {value}")
                st.caption(explanation)
                _capacity_state_badge(status, color)

    technical_expander = st.expander(
        "查看技术依据、趋势与实例数据",
        icon=":material/monitoring:",
        key="capacity_technical_details",
        on_change="rerun",
    )
    if not technical_expander.open:
        return
    with technical_expander:
        st.caption(
            "以下内容供技术人员排障。NPU 表示算力使用率；HBM 表示显存；"
            "忙时并发表示高峰同时处理的请求数；等待队列表示尚未开始处理的请求。"
        )
        metric_label = st.segmented_control(
            "资源趋势指标",
            ["运行并发", "等待队列", "首字延迟", "服务吞吐"],
            default="运行并发",
            required=True,
            key="resource_metric_decision",
        )
        metric_specs = {
            "运行并发": ("running", "同时处理的请求数", "同时处理的请求越多，越接近实例承载上限。"),
            "等待队列": ("waiting", "等待处理的请求数", "大于 0 表示请求已经开始排队。"),
            "首字延迟": ("ttft_ms", "开始返回内容的等待时间（ms）", "越低越好；突然升高可能意味着处理压力增大。"),
            "服务吞吐": ("tokens_per_second", "每秒输出 Token", "表示模型生成内容的速度。"),
        }
        metric_column, axis_title, metric_help = metric_specs[str(metric_label)]
        chart_data = model_series[
            pd.to_datetime(model_series["timestamp"]).between(
                capacity_window_start,
                latest_date + pd.Timedelta(days=1),
                inclusive="left",
            )
        ].dropna(subset=[metric_column]).copy()
        trend_col, heatmap_col = st.columns([1.25, 1], gap="large", vertical_alignment="top")
        with trend_col:
            st.markdown("#### 容量趋势")
            st.caption(metric_help)
            if chart_data.empty:
                st.info("当前指标没有可用趋势数据。", icon=":material/info:")
            else:
                chart = (
                    alt.Chart(chart_data)
                    .mark_line(interpolate="step-after", strokeWidth=2)
                    .encode(
                        x=alt.X("timestamp:T", title="时间"),
                        y=alt.Y(f"{metric_column}:Q", title=axis_title),
                        color=alt.Color("model_id:N", title="模型"),
                        tooltip=[
                            alt.Tooltip("timestamp:T", title="时间", format="%Y-%m-%d %H:%M"),
                            alt.Tooltip("model_id:N", title="模型"),
                            alt.Tooltip(f"{metric_column}:Q", title=str(metric_label), format=",.2f"),
                        ],
                    )
                    .properties(height=330)
                )
                st.altair_chart(chart, width="stretch")
        with heatmap_col:
            st.markdown("#### 实例负载")
            st.caption(f"显示 {selected_model} 各匿名实例的算力峰值；颜色越深表示负载越高。")
            heatmap_data = instance_hourly[
                instance_hourly["model_id"].eq(selected_model)
                & pd.to_datetime(instance_hourly["hour"]).between(
                    capacity_window_start,
                    latest_date + pd.Timedelta(days=1),
                    inclusive="left",
                )
            ].copy()
            heatmap = (
                alt.Chart(heatmap_data)
                .mark_rect()
                .encode(
                    x=alt.X("hour:T", title="小时"),
                    y=alt.Y("instance_id:N", title="匿名实例"),
                    color=alt.Color("npu_max:Q", title="NPU 峰值（%）", scale=alt.Scale(domain=[0, 100], scheme="yelloworangered")),
                    tooltip=[
                        alt.Tooltip("hour:T", title="时间", format="%Y-%m-%d %H:%M"),
                        alt.Tooltip("instance_id:N", title="匿名实例"),
                        alt.Tooltip("npu_mean:Q", title="平均算力使用率", format=".1f"),
                        alt.Tooltip("npu_max:Q", title="算力峰值", format=".1f"),
                    ],
                )
                .properties(height=330)
            )
            st.altair_chart(heatmap, width="stretch")

        st.markdown("#### 完整技术字段")
        st.dataframe(
            latest.drop(columns="_priority").sort_values(["capacity_state", "model_id"]),
            column_order=[
                "model_id", "capacity_state", "instance_count", "running_max_busy",
                "waiting_max_busy", "concurrency_ratio", "ttft_mean_ms",
                "tokens_per_second_mean", "npu_mean", "npu_p95", "npu_max",
                "cache_pct", "hbm_pct", "hbm_headroom_pct", "diagnosis",
            ],
            column_config={
                "model_id": "模型",
                "capacity_state": "原始容量状态",
                "instance_count": "实例数",
                "running_max_busy": "忙时并发",
                "waiting_max_busy": "最大等待",
                "concurrency_ratio": st.column_config.ProgressColumn("并发/实例", min_value=0, max_value=1, format="%.1f"),
                "ttft_mean_ms": st.column_config.NumberColumn("平均首字延迟", format="%,.0f ms"),
                "tokens_per_second_mean": st.column_config.NumberColumn("服务吞吐", format="%.2f token/s"),
                "npu_mean": st.column_config.NumberColumn("平均 NPU", format="%.1f%%"),
                "npu_p95": st.column_config.NumberColumn("NPU P95", format="%.1f%%"),
                "npu_max": st.column_config.NumberColumn("NPU 峰值", format="%.0f%%"),
                "cache_pct": st.column_config.NumberColumn("Cache", format="%.2f%%"),
                "hbm_pct": st.column_config.NumberColumn("HBM", format="%.2f%%"),
                "hbm_headroom_pct": st.column_config.NumberColumn("HBM 余量", format="%.2f%%"),
                "diagnosis": "原始技术结论",
            },
            hide_index=True,
            height=260,
        )
        st.download_button(
            "下载容量诊断原始数据",
            latest.drop(columns="_priority").to_csv(index=False).encode("utf-8-sig"),
            f"capacity_diagnosis_{latest_date:%Y%m%d}.csv",
            "text/csv",
            icon=":material/download:",
        )


def _active_detection_settings(
    default_policy_values: dict[str, float],
) -> tuple[dict[str, float], dict[str, float], int, bool]:
    custom_policy = st.session_state.get("detection_policy_values")
    policy_values = (
        {key: float(value) for key, value in custom_policy.items()}
        if custom_policy is not None
        else default_policy_values.copy()
    )
    risk_bands = {
        key: float(value)
        for key, value in st.session_state.get("detection_risk_bands", DEFAULT_RISK_BANDS).items()
    }
    algorithm_votes = int(
        st.session_state.get("detection_unknown_algorithm_votes", DEFAULT_UNKNOWN_ALGORITHM_VOTES)
    )
    return policy_values, risk_bands, algorithm_votes, custom_policy is not None


def _render_detection_policy_editor(
    default_policy_values: dict[str, float],
    active_policy_values: dict[str, float],
    risk_bands: dict[str, float],
    algorithm_votes: int,
    is_custom: bool,
) -> None:
    notice = st.session_state.pop("detection_policy_notice", None)
    if notice:
        st.success(str(notice), icon=":material/check_circle:")

    with st.expander("检测策略配置", expanded=False):
        st.caption("修改只作用于当前浏览器会话；指标字典仍是默认策略。应用后会用当前筛选区间回放风险和事件，不会改写仓库文件。")
        if is_custom:
            st.info("当前使用：会话自定义策略", icon=":material/edit_note:")
        else:
            st.info("当前使用：指标字典默认策略", icon=":material/verified:")

        revision = int(st.session_state.setdefault("detection_policy_revision", 0))
        with st.form(f"detection_policy_form_{revision}"):
            st.markdown("##### 信号判断阈值")
            edited_rules = st.data_editor(
                signal_rule_table(active_policy_values),
                key=f"detection_rule_editor_{revision}",
                disabled=["检测信号", "风险维度", "触发方向", "单位"],
                num_rows="fixed",
                column_config={
                    "检测信号": st.column_config.TextColumn("检测信号", pinned=True),
                    "风险维度": "风险维度",
                    "触发方向": "触发方向",
                    "预警阈值": st.column_config.NumberColumn("预警阈值", min_value=0.0, format="%.2f"),
                    "严重阈值": st.column_config.NumberColumn("严重阈值", min_value=0.0, format="%.2f"),
                    "单位": "单位",
                },
                hide_index=True,
            )

            st.markdown("##### 基线与统计异常")
            baseline_window_col, minimum_baseline_col, votes_col = st.columns(3)
            baseline_window = baseline_window_col.number_input(
                "基线窗口（天）",
                min_value=1,
                max_value=90,
                value=int(active_policy_values["baseline_window_days"]),
            )
            minimum_baseline = minimum_baseline_col.number_input(
                "最少历史天数",
                min_value=1,
                max_value=90,
                value=int(active_policy_values["minimum_baseline_days"]),
            )
            unknown_votes = votes_col.number_input(
                "未知异常最少一致算法数",
                min_value=1,
                max_value=3,
                value=algorithm_votes,
                help="MAD、STL、Isolation Forest 中至少多少个算法同时命中，才生成未知模式事件。",
            )

            st.markdown("##### 风险等级与路由动作")
            medium_col, high_col, critical_col = st.columns(3)
            medium_threshold = medium_col.number_input(
                "中风险起点", min_value=1.0, max_value=97.0, value=risk_bands["medium"], step=1.0
            )
            high_threshold = high_col.number_input(
                "高风险起点", min_value=2.0, max_value=98.0, value=risk_bands["high"], step=1.0
            )
            critical_threshold = critical_col.number_input(
                "严重风险起点", min_value=3.0, max_value=99.0, value=risk_bands["critical"], step=1.0
            )
            evidence_col, downweight_col, switch_col, candidate_col = st.columns(4)
            evidence_threshold = evidence_col.number_input(
                "进入诊断中心", min_value=0.0, max_value=100.0,
                value=float(active_policy_values["evidence_risk_threshold"]), step=1.0,
            )
            downweight_threshold = downweight_col.number_input(
                "建议降权", min_value=0.0, max_value=100.0,
                value=float(active_policy_values["route_downweight_risk_threshold"]), step=1.0,
            )
            switch_threshold = switch_col.number_input(
                "建议切换", min_value=0.0, max_value=100.0,
                value=float(active_policy_values["route_switch_risk_threshold"]), step=1.0,
            )
            candidate_health = candidate_col.number_input(
                "候选模型最低健康分", min_value=0.0, max_value=100.0,
                value=float(active_policy_values["minimum_candidate_health_score"]), step=1.0,
            )

            st.markdown("##### 高级：融合诊断风险下限")
            model_floor_col, capability_floor_col, platform_floor_col, environment_floor_col = st.columns(4)
            model_floor = model_floor_col.number_input(
                "模型侧同步下降", min_value=0.0, max_value=100.0,
                value=float(active_policy_values["model_side_risk_floor"]), step=1.0,
            )
            capability_floor = capability_floor_col.number_input(
                "能力或拨测异常", min_value=0.0, max_value=100.0,
                value=float(active_policy_values["capability_or_probe_risk_floor"]), step=1.0,
            )
            platform_floor = platform_floor_col.number_input(
                "平台或流量异常", min_value=0.0, max_value=100.0,
                value=float(active_policy_values["platform_or_traffic_risk_floor"]), step=1.0,
            )
            environment_floor = environment_floor_col.number_input(
                "业务环境延迟差", min_value=0.0, max_value=100.0,
                value=float(active_policy_values["environment_latency_risk_floor"]), step=1.0,
            )
            signal_floor_multiplier = st.number_input(
                "单项严重信号保护系数",
                min_value=0.0,
                max_value=1.0,
                value=float(active_policy_values["single_component_floor_multiplier"]),
                step=0.05,
                help="最终风险不低于最高单项风险乘以该系数。",
            )

            submitted = st.form_submit_button(
                "应用策略并回放", type="primary", icon=":material/play_arrow:"
            )

        if submitted:
            try:
                candidate = merge_signal_rule_table(active_policy_values, edited_rules)
                candidate.update(
                    {
                        "baseline_window_days": float(baseline_window),
                        "minimum_baseline_days": float(minimum_baseline),
                        "evidence_risk_threshold": float(evidence_threshold),
                        "route_downweight_risk_threshold": float(downweight_threshold),
                        "route_switch_risk_threshold": float(switch_threshold),
                        "minimum_candidate_health_score": float(candidate_health),
                        "model_side_risk_floor": float(model_floor),
                        "capability_or_probe_risk_floor": float(capability_floor),
                        "platform_or_traffic_risk_floor": float(platform_floor),
                        "environment_latency_risk_floor": float(environment_floor),
                        "single_component_floor_multiplier": float(signal_floor_multiplier),
                    }
                )
                RiskPolicy.from_mapping(candidate)
                scoring_policy_with_risk_bands(
                    load_runtime_scoring_policy(PATHS["config"].stat().st_mtime_ns),
                    medium_threshold,
                    high_threshold,
                    critical_threshold,
                )
            except (TypeError, ValueError) as exc:
                st.error(f"策略未应用：{exc}")
            else:
                st.session_state["detection_policy_values"] = candidate
                st.session_state["detection_risk_bands"] = {
                    "medium": float(medium_threshold),
                    "high": float(high_threshold),
                    "critical": float(critical_threshold),
                }
                st.session_state["detection_unknown_algorithm_votes"] = int(unknown_votes)
                st.session_state["detection_policy_revision"] = revision + 1
                st.session_state["detection_policy_notice"] = "自定义策略已应用，风险、事件和诊断证据已重新计算。"
                st.rerun()

        if st.button(
            "恢复指标字典默认值",
            key=f"reset_detection_policy_{revision}",
            icon=":material/restore:",
        ):
            st.session_state.pop("detection_policy_values", None)
            st.session_state.pop("detection_risk_bands", None)
            st.session_state.pop("detection_unknown_algorithm_votes", None)
            st.session_state["detection_policy_revision"] = revision + 1
            st.session_state["detection_policy_notice"] = "已恢复指标字典默认策略。"
            st.rerun()


def render_detection(
    risks: pd.DataFrame,
    evidence: pd.DataFrame,
    default_risks: pd.DataFrame,
    scores: pd.DataFrame,
    benchmark: pd.DataFrame,
    fusion_benchmark: pd.DataFrame,
    truth: pd.DataFrame,
    default_policy_values: dict[str, float],
    active_policy_values: dict[str, float],
    risk_bands: dict[str, float],
    algorithm_votes: int,
    is_custom_policy: bool,
) -> None:
    _section(
        "智能检测",
        "从固定事件报警升级为可配置的模型健康风险识别：确定性阈值负责已知风险，统计模型补充发现未知模式。",
    )
    _render_detection_policy_editor(
        default_policy_values,
        active_policy_values,
        risk_bands,
        algorithm_votes,
        is_custom_policy,
    )

    active_policy = RiskPolicy.from_mapping(active_policy_values)
    default_policy = RiskPolicy.from_mapping(default_policy_values)
    signal_events = build_signal_events(risks, active_policy)
    unknown_events = build_unknown_pattern_events(scores, algorithm_votes)
    dynamic_events = pd.concat([signal_events, unknown_events], ignore_index=True)
    if not dynamic_events.empty:
        dynamic_events = dynamic_events.sort_values(
            ["event_time", "risk_score"], ascending=[False, False]
        ).reset_index(drop=True)

    default_events = pd.concat(
        [
            build_signal_events(default_risks, default_policy),
            build_unknown_pattern_events(scores, DEFAULT_UNKNOWN_ALGORITHM_VOTES),
        ],
        ignore_index=True,
    )
    if risks.empty:
        st.info("当前筛选范围没有健康风险数据。")
        return
    highest = risks.sort_values("risk_score", ascending=False).iloc[0]
    medium_plus = int(risks["risk_score"].ge(risk_bands["medium"]).sum())
    switch_count = int(evidence["switch_recommendation"].astype(str).str.contains("建议切换").sum()) if not evidence.empty else 0
    _metric_row([
        {"label": "最高风险分", "value": f"{highest['risk_score']:.1f}", "delta": f"{highest['risk_level']} · {highest['model_id']}", "delta_color": "inverse"},
        {"label": "中风险及以上", "value": medium_plus},
        {"label": "动态风险事件", "value": len(dynamic_events)},
        {"label": "建议切换", "value": switch_count},
    ])
    event_delta = len(dynamic_events) - len(default_events)
    strategy_name = "会话自定义策略" if is_custom_policy else "指标字典默认策略"
    st.caption(
        f"当前使用：{strategy_name} · 相比默认策略事件数 {event_delta:+d} · "
        f"未知模式事件 {len(unknown_events)} 条 · 诊断证据 {len(evidence)} 条"
    )

    st.markdown("#### 动态风险事件")
    st.caption("风险维度保持稳定，具体事件由当前阈值和统计模型动态产生；同一天同一模型可以触发多个信号。")
    if dynamic_events.empty:
        st.success("当前策略下没有触发风险事件。", icon=":material/check_circle:")
    else:
        st.dataframe(
            dynamic_events.head(50),
            column_order=["event_time", "scope", "risk_dimension", "event_type", "severity", "detection_method", "risk_score", "observed_value", "threshold", "evidence"],
            column_config={
                "event_time": st.column_config.DatetimeColumn("时间", format="YYYY-MM-DD HH:mm"),
                "scope": "影响范围",
                "risk_dimension": "风险维度",
                "event_type": "事件",
                "severity": "等级",
                "detection_method": "检测方式",
                "risk_score": st.column_config.ProgressColumn("风险分", min_value=0, max_value=100, format="%.1f"),
                "observed_value": "观测值",
                "threshold": "当前阈值",
                "evidence": "判断依据",
            },
            hide_index=True,
            height=340,
        )

    st.markdown("#### 模型健康风险趋势")
    chart = _line_chart(
        risks,
        "date",
        "risk_score",
        "model_id",
        "风险评分",
        [alt.Tooltip("date:T", title="日期", format="%Y-%m-%d"), alt.Tooltip("model_id:N", title="模型"), alt.Tooltip("risk_score:Q", title="风险分", format=".1f"), alt.Tooltip("risk_level:N", title="等级"), alt.Tooltip("primary_risk_driver_cn:N", title="主要驱动")],
        height=350,
    )
    thresholds = pd.DataFrame(
        {
            "risk_score": [risk_bands["medium"], risk_bands["high"], risk_bands["critical"]],
            "label": ["中风险", "高风险", "严重"],
        }
    )
    rules = alt.Chart(thresholds).mark_rule(strokeDash=[5, 4], opacity=.55).encode(y="risk_score:Q", color=alt.Color("label:N", title="风险阈值"))
    st.altair_chart(chart + rules, width="stretch")

    latest = _latest_by_model(risks)
    period_max = risks.loc[risks.groupby("model_id")["risk_score"].idxmax(), ["model_id", "risk_score", "risk_level", "date"]].rename(columns={"risk_score": "period_max_risk", "risk_level": "period_max_level", "date": "max_risk_date"})
    summary = latest.merge(period_max, on="model_id", how="left").sort_values("period_max_risk", ascending=False)
    st.markdown("#### 风险构成与决策优先级")
    st.dataframe(
        summary,
        column_order=["model_id", "risk_score", "risk_level", "period_max_risk", "max_risk_date", "performance_risk", "success_risk", "cost_risk", "primary_risk_driver_cn", "diagnosis_reason"],
        column_config={
            "model_id": "模型",
            "risk_score": st.column_config.ProgressColumn("当前风险", min_value=0, max_value=100, format="%.1f"),
            "risk_level": "当前等级",
            "period_max_risk": st.column_config.NumberColumn("区间最高", format="%.1f"),
            "max_risk_date": st.column_config.DateColumn("最高风险日", format="YYYY-MM-DD"),
            "performance_risk": st.column_config.NumberColumn("性能风险", format="%.1f"),
            "success_risk": st.column_config.NumberColumn("成功率风险", format="%.1f"),
            "cost_risk": st.column_config.NumberColumn("成本风险", format="%.1f"),
            "primary_risk_driver_cn": "主要驱动",
            "diagnosis_reason": "融合判断",
        },
        hide_index=True,
    )

    with st.expander("检测算法实验对比（保留原实验能力）"):
        if benchmark.empty:
            st.info("缺少算法基准结果。")
        else:
            display = benchmark.copy()
            for column in ["precision", "recall", "f1", "event_recall"]:
                display[column] = display[column] * 100
            st.dataframe(
                display,
                column_order=["algorithm", "precision", "recall", "f1", "event_recall", "false_alarms_per_day", "mean_detection_delay_minutes"],
                column_config={
                    "algorithm": "算法",
                    "precision": st.column_config.NumberColumn("准确率", format="%.1f%%"),
                    "recall": st.column_config.NumberColumn("小时召回", format="%.1f%%"),
                    "f1": st.column_config.NumberColumn("F1", format="%.1f%%"),
                    "event_recall": st.column_config.NumberColumn("事件召回", format="%.1f%%"),
                    "false_alarms_per_day": st.column_config.NumberColumn("日均误报", format="%.2f"),
                    "mean_detection_delay_minutes": "平均延迟（分钟）",
                },
                hide_index=True,
            )
            if not fusion_benchmark.empty:
                st.markdown("##### 分层融合策略")
                st.dataframe(fusion_benchmark, hide_index=True)
            if not scores.empty:
                algorithm = st.selectbox("查看算法时序证据", list(ALGORITHM_OPTIONS), key="algorithm_evidence")
                pred_col, score_col = ALGORITHM_OPTIONS[algorithm]
                plot_data = scores[["hour", score_col, pred_col, "truth_types"]].copy()
                base = alt.Chart(plot_data).mark_line().encode(
                    x=alt.X("hour:T", title=None), y=alt.Y(f"{score_col}:Q", title="异常分数"),
                    tooltip=[alt.Tooltip("hour:T", title="时间"), alt.Tooltip(f"{score_col}:Q", title="分数", format=".2f")],
                )
                points = alt.Chart(plot_data[plot_data[pred_col].astype(bool)]).mark_point(size=80, filled=True, color="#D92D20").encode(x="hour:T", y=f"{score_col}:Q", tooltip=["hour:T", "truth_types:N"])
                layers: alt.Chart | alt.LayerChart = base + points
                if not truth.empty:
                    event_ranges = alt.Chart(truth).mark_rect(color="#F79009", opacity=.12).encode(
                        x=alt.X("start_time:T"), x2=alt.X2("end_time:T")
                    )
                    layers = event_ranges + base + points
                st.altair_chart(layers.properties(height=300), width="stretch")
                st.caption("红点为算法判定，橙色区域为独立标注的真实异常事件。")


def _text_or_dash(value: object) -> str:
    return "—" if pd.isna(value) or str(value).strip() in {"", "nan", "None"} else str(value)


def render_diagnosis_center(
    evidence: pd.DataFrame,
    fusion_alerts: pd.DataFrame,
    probe_events: pd.DataFrame,
    config_data: dict[str, pd.DataFrame],
) -> None:
    _section(
        "智能诊断解释中心",
        "把风险信号转化为可执行决策：说明异常是什么、可能原因、是否需要切换模型，以及下一步动作。",
    )
    if evidence.empty:
        st.success("当前筛选范围没有进入解释中心的风险事件。")
    else:
        levels = evidence["risk_level"].value_counts()
        switch_count = int(evidence["switch_recommendation"].astype(str).str.contains("建议切换").sum())
        _metric_row([
            {"label": "待解释事件", "value": len(evidence)},
            {"label": "严重 / 高风险", "value": int(levels.get("严重", 0) + levels.get("高", 0))},
            {"label": "建议切换", "value": switch_count},
            {"label": "平均证据可信度", "value": f"{evidence['evidence_confidence_score'].mean():.1f}"},
        ])

        st.markdown("#### 诊断事件队列")
        queue = evidence.sort_values(["risk_score", "date"], ascending=[False, False]).copy()
        queue["event_label"] = queue.apply(
            lambda row: f"{row['evidence_id']}｜{pd.Timestamp(row['date']).strftime('%m-%d')}｜{row['model_id']}｜风险 {row['risk_score']:.0f}",
            axis=1,
        )
        selected_label = st.selectbox("选择事件", queue["event_label"].tolist(), key="diagnostic_event")
        selected = queue[queue["event_label"].eq(selected_label)].iloc[0]

        summary_cols = st.columns([1, 1, 1, 1])
        summary_cols[0].metric("风险评分", f"{selected['risk_score']:.1f}", selected["risk_level"], delta_color="inverse", border=True)
        summary_cols[1].metric("证据可信度", f"{selected['evidence_confidence_score']:.1f}", border=True)
        summary_cols[2].metric("画像可信度", f"{selected['model_profile_confidence_score']:.1f}", border=True)
        summary_cols[3].metric("路由就绪度", f"{selected['routing_readiness_score']:.1f}", border=True)

        left, right = st.columns([1.15, 1], gap="large")
        with left:
            with st.container(border=True):
                st.markdown("##### 异常是什么")
                st.write(selected["what_happened"])
                st.caption(f"{selected['model_id']} · {selected['provider']} · {pd.Timestamp(selected['date']).strftime('%Y-%m-%d')} · 主要驱动：{selected['primary_risk_driver_cn']}")
            with st.container(border=True):
                st.markdown("##### 可能原因")
                st.write(selected["possible_cause"])
                st.caption(selected["diagnosis_reason"])
        with right:
            switch_text = _text_or_dash(selected["switch_recommendation"])
            with st.container(border=True):
                st.markdown("##### 是否需要切换模型")
                if "建议切换" in switch_text:
                    st.error(f"{switch_text} → {_text_or_dash(selected['target_model_id'])}")
                    st.caption(_text_or_dash(selected["target_reason"]))
                elif "降低" in switch_text or "灰度" in switch_text:
                    st.warning(switch_text)
                else:
                    st.info(switch_text)
            with st.container(border=True):
                st.markdown("##### 推荐动作")
                st.write(selected["recommended_action"])
                st.caption(f"决策状态：{selected['decision_state']}")

        with st.expander("查看完整风险证据"):
            st.text(selected["risk_evidence"])

        st.markdown("#### 全部诊断证据")
        st.dataframe(
            queue,
            column_order=["date", "model_id", "risk_score", "risk_level", "what_happened", "possible_cause", "switch_recommendation", "target_model_id", "recommended_action", "decision_state"],
            column_config={
                "date": st.column_config.DateColumn("日期", format="YYYY-MM-DD"),
                "model_id": "模型",
                "risk_score": st.column_config.ProgressColumn("风险", min_value=0, max_value=100, format="%.1f"),
                "risk_level": "等级",
                "what_happened": "异常是什么",
                "possible_cause": "可能原因",
                "switch_recommendation": "切换判断",
                "target_model_id": "候选模型",
                "recommended_action": "推荐动作",
                "decision_state": "状态",
            },
            hide_index=True,
            height=350,
        )

    with st.expander("原始检测告警与规则配置（保留原有能力）"):
        alert_tabs = st.tabs(["融合告警", "拨测事件", "融合策略", "告警分级", "复合规则", "规则条件"])
        frames = [
            fusion_alerts,
            probe_events,
            config_data.get("fusion_strategies", pd.DataFrame()),
            config_data.get("fusion_grading", pd.DataFrame()),
            config_data.get("composite_rules", pd.DataFrame()),
            config_data.get("conditions", pd.DataFrame()),
        ]
        for tab, frame in zip(alert_tabs, frames):
            with tab:
                if frame.empty:
                    st.info("暂无数据。")
                else:
                    st.dataframe(frame, hide_index=True)
        if not fusion_alerts.empty:
            st.download_button("下载融合告警 CSV", fusion_alerts.to_csv(index=False).encode("utf-8-sig"), "fusion_alerts.csv", "text/csv", icon=":material/download:")


def _select_module(module: str) -> None:
    st.session_state.active_module = module


def sidebar_filters(data: dict[str, pd.DataFrame]) -> tuple[str, list[str], pd.Timestamp, pd.Timestamp, list[str]]:
    logs = data["logs"]
    st.sidebar.title("运营决策中心")
    st.sidebar.caption("业务模块")
    st.session_state.setdefault("active_module", MODULES[0])
    if st.session_state.active_module not in MODULES:
        st.session_state.active_module = MODULES[0]
    for label, icon, key in MODULE_NAVIGATION:
        st.sidebar.button(
            label,
            key=key,
            icon=icon,
            type="primary" if st.session_state.active_module == label else "secondary",
            width="stretch",
            on_click=_select_module,
            args=(label,),
        )
    module = st.session_state.active_module
    st.sidebar.divider()
    st.sidebar.subheader("全局筛选")
    minimum = logs["timestamp"].min().date()
    maximum = logs["timestamp"].max().date()
    all_models = sorted(logs["model_id"].dropna().unique())
    models = st.sidebar.multiselect("模型", all_models, default=all_models)
    with st.sidebar.expander("高级筛选", icon=":material/tune:"):
        selected_dates = st.date_input(
            "全局日期范围",
            value=(minimum, maximum),
            min_value=minimum,
            max_value=maximum,
            help="运营总览还可以在页面内使用近 1 天、近 7 天、近 30 天快捷窗口。",
        )
        all_customers = sorted(logs["customer_id"].dropna().unique())
        customers = st.multiselect("客户", all_customers, default=all_customers)
        st.caption("客户筛选只影响基于原始调用明细计算的驾驶舱 KPI；模型评分产物按全量模型流量生成。")
    if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
    else:
        start_date = end_date = selected_dates
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    st.sidebar.divider()
    rebuild_notice = st.session_state.pop("demo_rebuild_notice", None)
    if rebuild_notice:
        st.sidebar.success(rebuild_notice, icon=":material/check_circle:")
    action_reload, action_rebuild = st.sidebar.columns(2, gap="small")
    if action_reload.button(
        "重新加载",
        icon=":material/refresh:",
        width="stretch",
        help="清除页面缓存并重新读取现有数据文件。",
    ):
        st.cache_data.clear()
        st.rerun()
    with action_rebuild.popover(
        "重新生成",
        icon=":material/autorenew:",
        width="stretch",
        help="重新生成90天模拟数据及其分析产物，包含业务起伏和短时异常。",
    ):
        st.caption("将覆盖模拟日志和衍生分析产物；newdata 与真实资源数据不会被修改。")
        if st.button(
            "确认重新生成",
            type="primary",
            width="stretch",
            key="confirm_demo_rebuild",
        ):
            progress = st.progress(0, text="准备重新生成模拟数据…")

            def update_rebuild_progress(index: int, total: int, label: str) -> None:
                progress.progress(index / total, text=f"{label}（{index}/{total}）")

            try:
                seed = rebuild_demo_data(on_step=update_rebuild_progress)
            except DemoRebuildError as exc:
                st.error("重新生成失败，现有页面暂未重新加载。")
                with st.expander("查看错误详情"):
                    st.code(str(exc))
            except OSError as exc:
                st.error(f"当前部署环境无法写入模拟数据：{exc}")
            else:
                st.session_state.demo_rebuild_notice = f"模拟数据已重新生成（种子 {seed}）"
                st.cache_data.clear()
                st.rerun()
    st.sidebar.success("数据链路已就绪", icon=":material/check_circle:")
    policy_versions = data.get("risk_policy", pd.DataFrame()).get("version", pd.Series(dtype=str))
    policy_version = str(policy_versions.dropna().iloc[-1]) if not policy_versions.dropna().empty else "未知"
    latest_timestamp = logs["timestamp"].max()
    latest_data = latest_timestamp.strftime("%Y-%m-%d") if pd.notna(latest_timestamp) else "未知"
    st.sidebar.caption(f"评分配置 v{policy_version} · 最新数据 {latest_data}")
    return module, models, start, end, customers


def main() -> None:
    missing = [PATHS[key] for key in REQUIRED_KEYS if not PATHS[key].exists()]
    if missing:
        st.error("缺少运营决策产物：" + "、".join(path.name for path in missing))
        st.code(
            "python src/capability_calibration.py\n"
            "python src/model_operations.py\n"
            "python src/model_profile.py\n"
            "python src/model_health_risk.py"
        )
        st.stop()

    data = load_all(_signature(PATHS))
    module, models, start, end, customers = sidebar_filters(data)
    if not models:
        st.warning("请至少选择一个模型。")
        st.stop()

    logs = _date_filter(data["logs"], "timestamp", start, end)
    logs = logs[logs["model_id"].isin(models) & logs["customer_id"].isin(customers)].copy()
    operating = _model_filter(_date_filter(data["operating"], "date", start, end), models)
    diagnosis = _model_filter(_date_filter(data["diagnosis"], "date", start, end), models)
    default_risks = _model_filter(_date_filter(data["risks"], "date", start, end), models)
    profiles = _model_filter(data["profiles"], models)
    capability = _model_filter(data["capability"], models)
    resource_model = _model_filter(
        _date_filter(data["resource_model"], "timestamp", start, end), models
    )
    resource_instances = _model_filter(
        _date_filter(data["resource_instances"], "hour", start, end), models
    )
    resource_capacity = _model_filter(
        _date_filter(data["resource_capacity"], "date", start, end), models
    )
    probe_runs = _model_filter(_date_filter(data["probe_runs"], "started_at", start, end), models)
    probe_events = _model_filter(_date_filter(data["probe_alerts"], "detected_at", start, end), models)
    fusion_alerts = _date_filter(data["fusion_alerts"], "detected_at", start, end)
    scores = _date_filter(data["scores"], "hour", start, end)
    truth = data["truth"]
    if not truth.empty:
        truth = truth[(truth["start_time"] < end) & (truth["end_time"] >= start)].copy()

    default_policy_values = risk_policy_mapping(data["risk_policy"])
    active_policy_values, risk_bands, algorithm_votes, is_custom_policy = _active_detection_settings(
        default_policy_values
    )
    active_risk_policy = RiskPolicy.from_mapping(active_policy_values)
    scoring_policy = scoring_policy_with_risk_bands(
        load_runtime_scoring_policy(PATHS["config"].stat().st_mtime_ns),
        risk_bands["medium"],
        risk_bands["high"],
        risk_bands["critical"],
    )
    runtime_risks = build_health_risks(
        _model_filter(data["operating"], models),
        _model_filter(data["diagnosis"], models),
        scoring_policy,
        active_risk_policy,
    )
    runtime_evidence = build_diagnostic_evidence(
        runtime_risks,
        profiles,
        active_risk_policy,
    )
    risks = _date_filter(runtime_risks, "date", start, end)
    evidence = _date_filter(runtime_evidence, "date", start, end)

    st.title("AI 中台运营决策实验台")
    st.caption("真实资源 + 校准模拟调用 + 主动拨测 → 智能运营分析 → 模型能力画像 → 动态智能路由")
    st.caption(
        f":material/filter_alt: 当前视图：{module} · {start.strftime('%Y-%m-%d')} 至 "
        f"{(end - pd.Timedelta(days=1)).strftime('%Y-%m-%d')} · {len(models)} 个模型"
    )

    if module == "运营总览":
        render_overview(
            logs,
            operating,
            profiles,
            _date_filter(data["operating"], "date", start, end),
            data["profiles"],
        )
    elif module == "性能诊断":
        render_performance(operating)
    elif module == "成本分析":
        render_cost(operating)
    elif module == "能力校准":
        render_calibration(profiles, capability, diagnosis, probe_runs, probe_events)
    elif module == "容量诊断":
        render_resource_capacity(
            resource_model,
            resource_instances,
            resource_capacity,
        )
    elif module == "智能检测":
        render_detection(
            risks,
            evidence,
            default_risks,
            scores,
            data["benchmark"],
            data["fusion_benchmark"],
            truth,
            default_policy_values,
            active_policy_values,
            risk_bands,
            algorithm_votes,
            is_custom_policy,
        )
    else:
        render_diagnosis_center(evidence, fusion_alerts, probe_events, data)


main()
