"""面向多模型智能路由的 AI 中台运营决策实验台。"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]

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
    "config": PROJECT_ROOT / "docs" / "ai_monitoring_metric_dictionary.xlsx",
}

REQUIRED_KEYS = [
    "logs", "operating", "snapshot", "capability", "diagnosis", "profiles",
    "risks", "evidence", "config",
]

MODULES = [
    "运营总览",
    "性能诊断",
    "成本分析",
    "能力校准",
    "智能检测",
    "诊断解释",
]

MODULE_NAVIGATION = [
    ("运营总览", ":material/space_dashboard:", "nav_overview"),
    ("性能诊断", ":material/speed:", "nav_performance"),
    ("成本分析", ":material/paid:", "nav_cost"),
    ("能力校准", ":material/model_training:", "nav_calibration"),
    ("智能检测", ":material/health_and_safety:", "nav_detection"),
    ("诊断解释", ":material/troubleshoot:", "nav_diagnosis"),
]

DIMENSION_LABELS = {
    "instruction_following": "指令遵循",
    "structured_output": "结构化输出",
    "reasoning": "推理能力",
    "tool_call": "工具调用",
}

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


def _confidence_band(score: float) -> str:
    """将画像可信度转成便于阅读的展示档位，不改变底层评分。"""
    if score >= 85:
        return "高"
    if score >= 70:
        return "中"
    return "低"


def _performance_gap_label(gap: float) -> str:
    """用自然语言说明主动拨测分与真实表现分的差异方向。"""
    if pd.isna(gap) or abs(gap) < 0.05:
        return "基本持平"
    if gap > 0:
        return f"拨测高 {gap:.1f} 分"
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


def render_overview(
    logs: pd.DataFrame,
    operating: pd.DataFrame,
    profiles: pd.DataFrame,
) -> None:
    _section(
        "运营总览",
        "AI 中台运营驾驶舱：用真实调用指标与模型健康指数统一观察规模、质量、性能和成本。",
    )
    if logs.empty or operating.empty:
        st.info("当前筛选范围没有运营数据。")
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
    weights = latest["request_count"].clip(lower=1)
    health = float(np.average(latest["health_score"], weights=weights))
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
    _metric_row([
        {"label": "调用量", "value": f"{len(logs):,}", "chart_data": daily["request_count"].tolist()},
        {"label": "Token", "value": f"{int(logs['total_tokens'].sum()):,}", "chart_data": daily["total_tokens"].tolist()},
        {"label": "估算成本", "value": f"¥{logs['estimated_cost'].sum():,.2f}", "chart_data": daily["estimated_cost"].tolist(), "delta_color": "inverse"},
        {"label": "成功率", "value": f"{logs['is_success'].mean() * 100:.2f}%", "chart_data": daily["success_rate"].tolist()},
        {"label": "P95 延迟", "value": f"{logs['latency_ms'].quantile(.95):,.0f} ms", "chart_data": daily["p95_latency_ms"].tolist(), "delta_color": "inverse"},
        {
            "label": "模型健康指数",
            "value": f"{health:.1f}",
            "delta": _fmt_delta(health, previous_health, " 分"),
            "chart_data": health_daily["health_score"].tolist(),
            "help": "成功率 35% + 性能 25% + 稳定性 25% + 成本效率 15%，权重来自指标字典。",
        },
    ])

    left, right = st.columns([1.45, 1], gap="large")
    with left:
        st.markdown("#### 健康指数趋势")
        chart = _line_chart(
            operating,
            "date",
            "health_score",
            "model_id",
            "健康指数",
            [
                alt.Tooltip("date:T", title="日期", format="%Y-%m-%d"),
                alt.Tooltip("model_id:N", title="模型"),
                alt.Tooltip("health_score:Q", title="健康指数", format=".1f"),
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
        st.dataframe(
            ranking,
            column_order=["健康排名", "model_id", "health_score", "health_level", "recommended_role"],
            column_config={
                "健康排名": st.column_config.NumberColumn("排名", format="#%d"),
                "model_id": "模型",
                "health_score": st.column_config.ProgressColumn("健康指数", min_value=0, max_value=100, format="%.1f"),
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
            "cost_efficiency_score": st.column_config.NumberColumn("成本效率评分", format="%.1f"),
            "health_score": st.column_config.ProgressColumn("健康指数", min_value=0, max_value=100, format="%.1f"),
            "request_count": st.column_config.NumberColumn("当日调用量", format="%d"),
        },
        hide_index=True,
    )


def render_performance(operating: pd.DataFrame) -> None:
    _section(
        "性能诊断",
        "模型性能画像：同时观察典型延迟、尾部延迟、日内波动和稳定性，输出可比较的性能评分。",
    )
    if operating.empty:
        st.info("当前筛选范围没有性能数据。")
        return
    model = st.selectbox("诊断模型", sorted(operating["model_id"].unique()), key="performance_model")
    selected = operating[operating["model_id"].eq(model)].sort_values("date")
    latest = selected.iloc[-1]
    prior = selected.iloc[-2] if len(selected) > 1 else None
    _metric_row([
        {"label": "P50 延迟", "value": f"{latest['p50_latency_ms']:,.0f} ms", "delta": _fmt_delta(latest["p50_latency_ms"], prior["p50_latency_ms"] if prior is not None else np.nan, " ms"), "delta_color": "inverse", "chart_data": selected["p50_latency_ms"].tolist()},
        {"label": "P95 延迟", "value": f"{latest['p95_latency_ms']:,.0f} ms", "delta": _fmt_delta(latest["p95_latency_ms"], prior["p95_latency_ms"] if prior is not None else np.nan, " ms"), "delta_color": "inverse", "chart_data": selected["p95_latency_ms"].tolist()},
        {"label": "P99 延迟", "value": f"{latest['p99_latency_ms']:,.0f} ms", "delta": _fmt_delta(latest["p99_latency_ms"], prior["p99_latency_ms"] if prior is not None else np.nan, " ms"), "delta_color": "inverse", "chart_data": selected["p99_latency_ms"].tolist()},
        {"label": "延迟变异系数", "value": f"{latest['latency_cv'] * 100:.1f}%", "help": "P95 日内标准差 ÷ P95 日内均值；越低越稳定。", "chart_data": (selected["latency_cv"] * 100).tolist(), "delta_color": "inverse"},
        {"label": "稳定性评分", "value": f"{latest['stability_score']:.1f}", "chart_data": selected["stability_score"].tolist()},
        {"label": "模型性能评分", "value": f"{latest['performance_score']:.1f}", "help": "延迟评分 70% + 稳定性评分 30%。", "chart_data": selected["performance_score"].tolist()},
    ])

    latency = selected.melt(
        id_vars="date",
        value_vars=["p50_latency_ms", "p95_latency_ms", "p99_latency_ms"],
        var_name="percentile",
        value_name="latency_ms",
    )
    latency["percentile"] = latency["percentile"].map({
        "p50_latency_ms": "P50", "p95_latency_ms": "P95", "p99_latency_ms": "P99"
    })
    left, right = st.columns([1.5, 1], gap="large")
    with left:
        st.markdown("#### 延迟分位数趋势")
        chart = _line_chart(
            latency,
            "date",
            "latency_ms",
            "percentile",
            "延迟（ms）",
            [alt.Tooltip("date:T", title="日期", format="%Y-%m-%d"), alt.Tooltip("percentile:N", title="分位数"), alt.Tooltip("latency_ms:Q", title="延迟", format=",.0f")],
        )
        st.altair_chart(chart, width="stretch")
    with right:
        st.markdown("#### 波动与稳定性")
        stability = selected[["date", "stability_score", "performance_score"]].melt(
            "date", var_name="metric", value_name="score"
        )
        stability["metric"] = stability["metric"].map({"stability_score": "稳定性", "performance_score": "性能"})
        chart = _line_chart(
            stability,
            "date",
            "score",
            "metric",
            "评分",
            [alt.Tooltip("date:T", title="日期", format="%Y-%m-%d"), alt.Tooltip("metric:N", title="指标"), alt.Tooltip("score:Q", title="评分", format=".1f")],
        )
        st.altair_chart(chart, width="stretch")

    st.markdown("#### 模型性能横向评分")
    latest_all = _latest_by_model(operating).sort_values("performance_score", ascending=False)
    st.dataframe(
        latest_all,
        column_order=["model_id", "p50_latency_ms", "p95_latency_ms", "p99_latency_ms", "latency_cv", "stability_score", "performance_score"],
        column_config={
            "model_id": "模型",
            "p50_latency_ms": st.column_config.NumberColumn("P50（ms）", format="%,.0f"),
            "p95_latency_ms": st.column_config.NumberColumn("P95（ms）", format="%,.0f"),
            "p99_latency_ms": st.column_config.NumberColumn("P99（ms）", format="%,.0f"),
            "latency_cv": st.column_config.NumberColumn("延迟 CV", format="%.3f"),
            "stability_score": st.column_config.ProgressColumn("稳定性评分", min_value=0, max_value=100, format="%.1f"),
            "performance_score": st.column_config.ProgressColumn("性能评分", min_value=0, max_value=100, format="%.1f"),
        },
        hide_index=True,
    )


def render_cost(operating: pd.DataFrame) -> None:
    _section(
        "成本分析",
        "模型成本效率分析：比较单请求成本、Token 成本、趋势偏移和质量/成本综合表现。",
    )
    if operating.empty:
        st.info("当前筛选范围没有成本数据。")
        return
    model = st.selectbox("成本分析模型", sorted(operating["model_id"].unique()), key="cost_model")
    selected = operating[operating["model_id"].eq(model)].sort_values("date")
    latest = selected.iloc[-1]
    trend_pct = (latest["cost_trend_ratio"] - 1) * 100
    _metric_row([
        {"label": "单请求成本", "value": f"¥{latest['cost_per_request']:.6f}", "chart_data": selected["cost_per_request"].tolist(), "delta_color": "inverse"},
        {"label": "千 Token 成本", "value": f"¥{latest['cost_per_1k_tokens']:.6f}", "chart_data": selected["cost_per_1k_tokens"].tolist(), "delta_color": "inverse"},
        {"label": "成本趋势", "value": f"{trend_pct:+.1f}%", "help": "当前单请求成本 ÷ 前 7 个历史日中位数 - 1；至少 3 个历史日后启用。", "chart_data": ((selected["cost_trend_ratio"] - 1) * 100).tolist(), "delta_color": "inverse"},
        {"label": "质量评分", "value": f"{latest['quality_score']:.1f}", "chart_data": selected["quality_score"].tolist()},
        {"label": "成本效率评分", "value": f"{latest['cost_efficiency_score']:.1f}", "chart_data": selected["cost_efficiency_score"].tolist()},
        {"label": "成本性能评分", "value": f"{latest['cost_performance_score']:.1f}", "help": "质量评分 60% + 成本效率评分 40%，避免低价低质模型获得过高排名。", "chart_data": selected["cost_performance_score"].tolist()},
    ])

    left, right = st.columns([1.35, 1], gap="large")
    with left:
        st.markdown("#### 单请求成本趋势")
        chart = _line_chart(
            operating,
            "date",
            "cost_per_request",
            "model_id",
            "成本 / 请求",
            [alt.Tooltip("date:T", title="日期", format="%Y-%m-%d"), alt.Tooltip("model_id:N", title="模型"), alt.Tooltip("cost_per_request:Q", title="单请求成本", format=".6f"), alt.Tooltip("cost_trend_ratio:Q", title="基线倍数", format=".3f")],
        )
        st.altair_chart(chart, width="stretch")
    with right:
        st.markdown("#### 质量—成本效率矩阵")
        latest_all = _latest_by_model(operating)
        chart = (
            alt.Chart(latest_all)
            .mark_circle(opacity=.86, stroke="white", strokeWidth=1.5)
            .encode(
                x=alt.X("cost_efficiency_score:Q", title="成本效率评分", scale=alt.Scale(domain=[0, 100])),
                y=alt.Y("quality_score:Q", title="质量评分", scale=alt.Scale(domain=[0, 100])),
                size=alt.Size("request_count:Q", title="调用量", scale=alt.Scale(range=[300, 1200])),
                color=alt.Color("model_id:N", title="模型"),
                tooltip=[alt.Tooltip("model_id:N", title="模型"), alt.Tooltip("quality_score:Q", title="质量", format=".1f"), alt.Tooltip("cost_efficiency_score:Q", title="成本效率", format=".1f"), alt.Tooltip("cost_performance_score:Q", title="成本性能评分", format=".1f")],
            )
            .properties(height=330)
        )
        st.altair_chart(chart, width="stretch")

    st.markdown("#### 模型成本效率排行")
    st.dataframe(
        _latest_by_model(operating).sort_values("cost_performance_score", ascending=False),
        column_order=["model_id", "cost_per_request", "cost_per_1k_tokens", "cost_trend_ratio", "quality_score", "cost_efficiency_score", "cost_performance_score"],
        column_config={
            "model_id": "模型",
            "cost_per_request": st.column_config.NumberColumn("单请求成本", format="¥%.6f"),
            "cost_per_1k_tokens": st.column_config.NumberColumn("千 Token 成本", format="¥%.6f"),
            "cost_trend_ratio": st.column_config.NumberColumn("历史基线倍数", format="%.3f×"),
            "quality_score": st.column_config.NumberColumn("质量评分", format="%.1f"),
            "cost_efficiency_score": st.column_config.NumberColumn("成本效率", format="%.1f"),
            "cost_performance_score": st.column_config.ProgressColumn("成本性能评分", min_value=0, max_value=100, format="%.1f"),
        },
        hide_index=True,
    )


def render_calibration(
    profiles: pd.DataFrame,
    capability: pd.DataFrame,
    diagnosis: pd.DataFrame,
    probe_runs: pd.DataFrame,
    probe_events: pd.DataFrame,
) -> None:
    _section(
        "主动拨测与模型能力校准",
        "在固定输入和标准环境下校准能力、稳定性与响应速度，再与真实调用对照，定位异常来源并形成路由画像。",
    )
    if profiles.empty:
        st.info("缺少模型能力画像数据，请先运行 capability_calibration.py、model_operations.py 和 model_profile.py。")
        return
    model = st.selectbox("画像模型", sorted(profiles["model_id"].unique()), key="profile_model")
    profile = profiles[profiles["model_id"].eq(model)].sort_values("date").iloc[-1]

    st.markdown("#### 路由决策摘要")
    decision_score, decision_detail = st.columns([1, 2.2], gap="large", vertical_alignment="center")
    with decision_score:
        st.metric(
            "综合路由评分",
            f"{profile['routing_readiness_score']:.1f}",
            help="能力 35% + 稳定性 20% + 性能 25% + 成本性能 20%。这是本页用于路由决策的主分。",
            border=True,
        )
    with decision_detail:
        with st.container(border=True):
            st.markdown(f"**推荐角色：{profile['recommended_role']}**")
            st.write(profile["routing_action"])
            dominant = DIMENSION_LABELS.get(profile["dominant_capability"], profile["dominant_capability"])
            weakest = DIMENSION_LABELS.get(profile["weakest_capability"], profile["weakest_capability"])
            st.caption(f"优势能力：{dominant} · 相对弱项：{weakest}")

    st.markdown("#### 评分构成")
    _metric_row([
        {"label": "能力评分", "value": f"{profile['capability_score']:.1f}", "help": "四类标准任务按指标字典权重汇总。"},
        {"label": "稳定性评分", "value": f"{profile['profile_stability_score']:.1f}", "help": "真实调用稳定性 60% + 标准任务重复一致性 40%。"},
        {"label": "性能评分", "value": f"{profile['profile_performance_score']:.1f}", "help": "真实调用性能 60% + 标准环境性能 40%。"},
    ])
    confidence = float(profile["confidence_score"])
    with st.container(border=True):
        confidence_level = _confidence_band(confidence)
        st.markdown(f"**证据可信度：{confidence_level}（{confidence:.1f}/100）**")
        st.progress(confidence / 100)
        st.caption("可信度衡量任务覆盖、样本充分度、数据新鲜度和评测一致性，只表示这份画像有多可靠，不代表模型能力高低。")

    st.markdown("#### 标准化测试任务")
    filtered_capability = capability[capability["model_id"].eq(model)].copy()
    if not filtered_capability.empty:
        filtered_capability["能力维度"] = filtered_capability["capability_dimension"].map(DIMENSION_LABELS)
        left, right = st.columns([1.05, 1.4], gap="large")
        with left:
            chart = (
                alt.Chart(filtered_capability)
                .mark_bar(cornerRadiusEnd=4)
                .encode(
                    y=alt.Y("能力维度:N", title=None, sort="-x"),
                    x=alt.X("quality_score:Q", title="质量评分", scale=alt.Scale(domain=[0, 100])),
                    color=alt.Color("能力维度:N", legend=None),
                    tooltip=[alt.Tooltip("能力维度:N"), alt.Tooltip("quality_score:Q", title="质量", format=".1f"), alt.Tooltip("consistency_score:Q", title="一致性", format=".1f"), alt.Tooltip("p95_latency_ms:Q", title="P95 延迟", format=",.0f")],
                )
                .properties(height=285)
            )
            st.altair_chart(chart, width="stretch")
        with right:
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

    st.markdown("#### 真实调用 vs 主动拨测融合诊断")
    st.caption("真实调用 = 用户行为 + 网络环境 + 平台状态 + 模型能力；主动拨测 = 固定输入 + 标准环境 + 模型能力。差异用于判断异常来源。")
    model_diagnosis = diagnosis[diagnosis["model_id"].eq(model)].sort_values("date", ascending=False)
    if model_diagnosis.empty:
        st.info("当前筛选范围没有该模型的融合诊断记录。")
    else:
        latest_diagnosis = model_diagnosis.iloc[0]
        with st.container(border=True):
            st.markdown(f"**最新诊断 · {latest_diagnosis['date']:%Y-%m-%d}**")
            real_score, probe_score, gap_score = st.columns(3)
            real_score.metric("真实表现指数", f"{latest_diagnosis['performance_score']:.1f}")
            probe_score.metric("主动拨测指数", f"{latest_diagnosis['probe_performance_score']:.1f}")
            gap_score.metric("表现差异", _performance_gap_label(latest_diagnosis["performance_gap_score"]))
            st.markdown(f"**原因判断：** {latest_diagnosis['diagnosis_reason']}")
            st.markdown(f"**切换判断：** {latest_diagnosis['switch_recommendation']}")
            st.info(latest_diagnosis["recommended_action"], icon=":material/recommend:")

        st.markdown("##### 近 7 次诊断轨迹")
        diagnosis_summary = model_diagnosis.head(7).copy()
        diagnosis_summary["表现差异"] = diagnosis_summary["performance_gap_score"].map(_performance_gap_label)
        st.dataframe(
            diagnosis_summary,
            column_order=["date", "performance_score", "probe_performance_score", "表现差异", "diagnosis_reason", "switch_recommendation"],
            column_config={
                "date": st.column_config.DateColumn("日期", format="YYYY-MM-DD"),
                "performance_score": st.column_config.ProgressColumn("真实表现", min_value=0, max_value=100, format="%.1f"),
                "probe_performance_score": st.column_config.ProgressColumn("主动拨测", min_value=0, max_value=100, format="%.1f"),
                "表现差异": "差异",
                "diagnosis_reason": "原因判断",
                "switch_recommendation": "切换判断",
            },
            hide_index=True,
            height=290,
        )

        with st.expander("查看原始诊断证据与完整数据"):
            st.caption("以下字段用于复核诊断计算，默认折叠以避免干扰路由决策。")
            st.dataframe(
                model_diagnosis,
                column_order=["date", "success_rate", "probe_http_success_rate", "p95_latency_ms", "probe_p95_latency_ms", "performance_score", "probe_performance_score", "performance_gap_score", "stability_score", "probe_consistency_score", "diagnosis_reason", "switch_recommendation", "recommended_action"],
                column_config={
                    "date": st.column_config.DateColumn("日期", format="YYYY-MM-DD"),
                    "success_rate": st.column_config.NumberColumn("真实成功率", format="%.2f%%"),
                    "probe_http_success_rate": st.column_config.NumberColumn("拨测成功率", format="%.2f%%"),
                    "p95_latency_ms": st.column_config.NumberColumn("真实 P95", format="%,.0f ms"),
                    "probe_p95_latency_ms": st.column_config.NumberColumn("拨测 P95", format="%,.0f ms"),
                    "performance_score": st.column_config.NumberColumn("真实性能分", format="%.1f"),
                    "probe_performance_score": st.column_config.NumberColumn("拨测性能分", format="%.1f"),
                    "performance_gap_score": st.column_config.NumberColumn("性能分差", format="%.1f"),
                    "stability_score": st.column_config.NumberColumn("真实稳定性", format="%.1f"),
                    "probe_consistency_score": st.column_config.NumberColumn("拨测一致性", format="%.1f"),
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

    st.markdown("#### 多模型路由输入画像")
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
            "routing_readiness_score": st.column_config.ProgressColumn("路由就绪度", min_value=0, max_value=100, format="%.1f"),
            "dominant_capability": "优势能力",
            "weakest_capability": "相对弱项",
            "recommended_role": "建议角色",
            "routing_action": "路由动作",
        },
        hide_index=True,
    )

    with st.expander("查看原有可用性拨测与导出"):
        if probe_runs.empty:
            st.info("当前范围没有可用性拨测记录。")
        else:
            availability = probe_runs["success"].astype(bool).mean() * 100
            cols = st.columns(3)
            cols[0].metric("拨测可用率", f"{availability:.2f}%", border=True)
            cols[1].metric("P95 首 Token", f"{probe_runs['ttft_ms'].quantile(.95):,.0f} ms", border=True)
            cols[2].metric("拨测事件", len(probe_events), border=True)
            latest_probe = probe_runs.sort_values("started_at").groupby("probe_id", as_index=False).tail(1)
            st.dataframe(
                latest_probe,
                column_order=["probe_name_cn", "provider", "model_id", "region", "success", "latency_ms", "ttft_ms", "failed_assertions"],
                column_config={"probe_name_cn": "探针", "provider": "供应商", "model_id": "模型", "region": "区域", "success": "成功", "latency_ms": "延迟（ms）", "ttft_ms": "首 Token（ms）", "failed_assertions": "失败断言"},
                hide_index=True,
            )
            c1, c2 = st.columns(2)
            c1.download_button("下载拨测运行 CSV", probe_runs.to_csv(index=False).encode("utf-8-sig"), "probe_runs.csv", "text/csv", icon=":material/download:")
            c2.download_button("下载拨测事件 CSV", probe_events.to_csv(index=False).encode("utf-8-sig"), "probe_alerts.csv", "text/csv", icon=":material/download:")


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
    selected_dates = st.sidebar.date_input(
        "日期范围",
        value=(minimum, maximum),
        min_value=minimum,
        max_value=maximum,
    )
    if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
    else:
        start_date = end_date = selected_dates
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) + pd.Timedelta(days=1)

    all_models = sorted(logs["model_id"].dropna().unique())
    models = st.sidebar.multiselect("模型", all_models, default=all_models)
    with st.sidebar.expander("调用明细筛选"):
        all_customers = sorted(logs["customer_id"].dropna().unique())
        customers = st.multiselect("客户", all_customers, default=all_customers)
        st.caption("客户筛选只影响基于原始调用明细计算的驾驶舱 KPI；模型评分产物按全量模型流量生成。")
    st.sidebar.divider()
    if st.sidebar.button("重新加载数据", icon=":material/refresh:", width="stretch"):
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
    st.caption("真实调用 + 主动拨测 → 智能运营分析 → 模型能力画像 → 动态智能路由")
    st.info(
        f"当前视图：{module} · {start.strftime('%Y-%m-%d')} 至 {(end - pd.Timedelta(days=1)).strftime('%Y-%m-%d')} · {len(models)} 个模型",
        icon=":material/filter_alt:",
    )

    if module == "运营总览":
        render_overview(logs, operating, profiles)
    elif module == "性能诊断":
        render_performance(operating)
    elif module == "成本分析":
        render_cost(operating)
    elif module == "能力校准":
        render_calibration(profiles, capability, diagnosis, probe_runs, probe_events)
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
