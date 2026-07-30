"""Decision-focused views for the five dashboard modules.

This module only owns presentation and transparent view-level aggregation.
The operating score policy remains configuration-driven in ``src``.
"""

from __future__ import annotations

from dataclasses import dataclass

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


BLUE = "#155EEF"
GREEN = "#16A34A"
AMBER = "#F59E0B"
RED = "#D92D20"
GRAY = "#667085"
SERIES = [BLUE, "#12B76A", "#F79009", "#7A5AF8", "#06AED4", RED]

SCORE_COLUMNS = {
    "综合健康": "health_score",
    "成功率评分": "success_score",
    "性能评分": "performance_score",
    "成本效率评分": "cost_efficiency_score",
}

SIX_DIMENSIONS = [
    "生成速度",
    "首响时延",
    "性能波动系数",
    "并发劣化率",
    "长文本劣化率",
    "请求成功率",
]
SIX_WEIGHTS = {
    "生成速度": 0.16,
    "首响时延": 0.18,
    "性能波动系数": 0.12,
    "并发劣化率": 0.14,
    "长文本劣化率": 0.12,
    "请求成功率": 0.28,
}


def _section(title: str, question: str) -> None:
    st.subheader(title)
    st.caption(question)


def _latest(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    return frame.sort_values("date").groupby("model_id", as_index=False).tail(1)


def _weighted(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return float("nan")
    weights = frame["request_count"].clip(lower=1)
    return float(np.average(frame[column], weights=weights))


def _polish(chart: alt.Chart) -> alt.Chart:
    return (
        chart.configure_view(stroke="#E5E9F0")
        .configure_axis(
            labelColor="#667085",
            titleColor="#344054",
            gridColor="#EEF2F6",
            labelFontSize=12,
            titleFontSize=13,
        )
        .configure_legend(orient="top", title=None, labelColor="#475467")
    )


def _trend_chart(frame: pd.DataFrame, column: str, y_title: str, *, height: int = 230) -> alt.Chart:
    chart = (
        alt.Chart(frame)
        .mark_line(point=alt.OverlayMarkDef(size=34), strokeWidth=2.2)
        .encode(
            x=alt.X("date:T", title=None, axis=alt.Axis(format="%m-%d", labelOverlap=True)),
            y=alt.Y(f"{column}:Q", title=y_title, scale=alt.Scale(zero=False)),
            color=alt.Color("model_id:N", scale=alt.Scale(range=SERIES)),
            tooltip=[
                alt.Tooltip("date:T", title="日期", format="%Y-%m-%d"),
                alt.Tooltip("model_id:N", title="模型"),
                alt.Tooltip(f"{column}:Q", title=y_title, format=".2f"),
            ],
        )
        .properties(height=height)
    )
    return _polish(chart)


def _insight(title: str, body: str, *, tone: str = "blue") -> None:
    color = {"blue": "blue", "warning": "orange", "danger": "red", "success": "green"}[tone]
    st.markdown(f":{color}-badge[{title}] **{body}**")


def render_overview(logs: pd.DataFrame, operating: pd.DataFrame) -> None:
    _section("运营总览", "先看平台是否健康、谁最健康，再看四项评分如何变化。")
    latest = _latest(operating)
    if latest.empty:
        st.info("当前筛选范围没有运营评分数据。", icon=":material/info:")
        return

    previous = _latest(operating[operating["date"] < latest["date"].min()])
    overall = _weighted(latest, "health_score")
    previous_overall = _weighted(previous, "health_score")
    delta = overall - previous_overall if pd.notna(previous_overall) else None
    ranking = latest.sort_values("health_score", ascending=False).reset_index(drop=True)

    with st.container(border=True, key="overview_decision_core"):
        st.markdown("### 当前健康结论")
        health_col, score_col = st.columns([4, 8], gap="large", vertical_alignment="center")
        with health_col:
            st.caption("模型总体健康指数")
            st.metric(
                "综合健康",
                f"{overall:.1f}",
                None if delta is None else f"{delta:+.1f} 较上一日",
                border=False,
            )
            st.progress(int(np.clip(overall, 0, 100)), text="由成功率、性能、成本效率三项评分构成")
        with score_col:
            score_cards = st.columns(3, gap="small")
            for card, (label, column) in zip(
                score_cards,
                [
                    ("成功率评分", "success_score"),
                    ("性能评分", "performance_score"),
                    ("成本效率评分", "cost_efficiency_score"),
                ],
                strict=True,
            ):
                card.metric(label, f"{_weighted(latest, column):.1f}", border=True)

        st.markdown("#### 最新模型健康排行")
        rank_cols = st.columns(len(ranking), gap="medium")
        for index, (column, (_, row)) in enumerate(zip(rank_cols, ranking.iterrows(), strict=True)):
            with column.container(border=True, height="stretch", key=f"health_rank_{index + 1}"):
                st.caption(f"第 {index + 1} 名")
                st.markdown(f"### {row['model_id']}")
                st.metric("综合健康", f"{float(row['health_score']):.1f}", border=False)
                level = str(row.get("health_level", ""))
                st.badge(
                    "当前最健康" if index == 0 else level,
                    color="blue" if index == 0 else "green" if float(row["health_score"]) >= 85 else "orange",
                )

        st.markdown("#### 排名依据 · 三项评分横向对比")
        breakdown = latest.melt(
            id_vars="model_id",
            value_vars=["success_score", "performance_score", "cost_efficiency_score"],
            var_name="score_type",
            value_name="score",
        )
        breakdown["score_type"] = breakdown["score_type"].map(
            {"success_score": "成功率评分", "performance_score": "性能评分", "cost_efficiency_score": "成本效率评分"}
        )
        bars = (
            alt.Chart(breakdown)
            .mark_bar(cornerRadiusEnd=4)
            .encode(
                y=alt.Y("model_id:N", title=None, sort=ranking["model_id"].tolist()),
                x=alt.X("score:Q", title="得分", scale=alt.Scale(domain=[0, 100])),
                color=alt.Color("score_type:N", scale=alt.Scale(range=[GREEN, BLUE, AMBER])),
                yOffset="score_type:N",
                tooltip=["model_id:N", "score_type:N", alt.Tooltip("score:Q", format=".1f")],
            )
            .properties(height=180)
        )
        st.altair_chart(_polish(bars), width="stretch", theme=None)

    st.markdown("### 四项评分趋势")
    st.caption("每张图都使用相同时间窗口横向比较全部模型；它们直接解释上方当前结论。")
    trend_items = [
        ("综合健康", "health_score", "得分"),
        ("成功率评分", "success_score", "得分"),
        ("性能评分", "performance_score", "得分"),
        ("成本效率评分", "cost_efficiency_score", "得分"),
    ]
    for row_items in (trend_items[:2], trend_items[2:]):
        columns = st.columns(2, gap="large")
        for column, (title, field, unit) in zip(columns, row_items, strict=True):
            with column.container(border=True):
                st.markdown(f"**{title}趋势**")
                st.altair_chart(_trend_chart(operating, field, unit), width="stretch", theme=None)

    with st.container(border=True):
        st.markdown("#### 观察窗口上下文 · 不参与健康评分")
        active_days = max(1, logs["timestamp"].dt.normalize().nunique())
        context = st.container(horizontal=True, horizontal_alignment="distribute", gap="small")
        context.metric("平均日请求量", f"{len(logs) / active_days:,.0f}", help="当前筛选窗口请求总量 ÷ 有数据天数", border=True)
        context.metric("单请求平均 Token", f"{logs['total_tokens'].mean():,.0f}", help="当前筛选窗口总 Token ÷ 请求数", border=True)
        context.metric("监控模型数", f"{logs['model_id'].nunique()}", help="当前筛选窗口实际出现的模型数量", border=True)
        st.caption("已删除容易混淆的“调用量总数”和“Token 总量”卡片；规模数据只用于解释负载背景，不参与健康排名。")


def render_performance(operating: pd.DataFrame) -> None:
    _section("性能诊断", "把性能评分拆成两条因果链：响应速度是否够快，表现是否稳定。")
    latest = _latest(operating)
    if latest.empty:
        st.info("当前筛选范围没有性能数据。", icon=":material/info:")
        return
    slowest = latest.loc[latest["p95_latency_ms"].idxmax()]
    unstable = latest.loc[latest["stability_score"].idxmin()]

    speed, stability = st.columns(2, gap="large", vertical_alignment="top")
    with speed.container(border=True, key="performance_speed_zone"):
        st.markdown("### 01 · 响应速度")
        st.caption("性能评分中的速度部分，和延迟分位数放在同一区域。")
        st.metric("响应速度得分", f"{_weighted(latest, 'latency_score'):.1f}", border=False)
        metrics = st.columns(3, gap="small")
        metrics[0].metric("P50", f"{_weighted(latest, 'p50_latency_ms'):,.0f} ms", border=True)
        metrics[1].metric("P95", f"{_weighted(latest, 'p95_latency_ms'):,.0f} ms", border=True)
        metrics[2].metric("P99", f"{_weighted(latest, 'p99_latency_ms'):,.0f} ms", border=True)
        percentile_label = st.segmented_control(
            "延迟分位数",
            ["P50", "P95", "P99"],
            default="P95",
            key="performance_percentile",
        )
        percentile_column = {"P50": "p50_latency_ms", "P95": "p95_latency_ms", "P99": "p99_latency_ms"}[str(percentile_label)]
        chart = (
            alt.Chart(operating)
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X("date:T", title=None),
                y=alt.Y(f"{percentile_column}:Q", title=f"{percentile_label} 延迟（ms）", scale=alt.Scale(zero=False), axis=alt.Axis(format=",")),
                color=alt.Color("model_id:N", scale=alt.Scale(range=SERIES)),
                tooltip=["date:T", "model_id:N", alt.Tooltip(f"{percentile_column}:Q", title=str(percentile_label), format=",.0f")],
            )
            .properties(height=340)
        )
        st.altair_chart(_polish(chart), width="stretch", theme=None)
        _insight("主要瓶颈", f"{slowest['model_id']} 当前 P95 最高（{slowest['p95_latency_ms']:,.0f} ms），优先检查长请求与并发峰值。", tone="warning")

    with stability.container(border=True, key="performance_stability_zone"):
        st.markdown("### 02 · 稳定性")
        st.caption("稳定性得分和波动系数趋势放在同一区域；CV 越低越稳定。")
        st.metric("稳定性得分", f"{_weighted(latest, 'stability_score'):.1f}", border=False)
        support = st.columns(2, gap="small")
        support[0].metric("平均波动系数 CV", f"{_weighted(latest, 'latency_cv'):.3f}", border=True)
        support[1].metric("最不稳定模型", str(unstable["model_id"]), border=True)
        cv_chart = _trend_chart(operating, "latency_cv", "波动系数 CV", height=340)
        st.altair_chart(cv_chart, width="stretch", theme=None)
        _insight("稳定性结论", f"{unstable['model_id']} 的稳定性得分最低（{unstable['stability_score']:.1f}），先限流观察，不直接把偶发尖峰归因于模型。", tone="warning")

    st.markdown("### 模型横向诊断")
    comparison = latest[["model_id", "latency_score", "stability_score", "p95_latency_ms", "latency_cv"]].copy()
    comparison = comparison.sort_values(["latency_score", "stability_score"], ascending=False)
    st.dataframe(
        comparison,
        column_config={
            "model_id": "模型",
            "latency_score": st.column_config.ProgressColumn("速度得分", min_value=0, max_value=100, format="%.1f"),
            "stability_score": st.column_config.ProgressColumn("稳定性得分", min_value=0, max_value=100, format="%.1f"),
            "p95_latency_ms": st.column_config.NumberColumn("P95", format="%,.0f ms"),
            "latency_cv": st.column_config.NumberColumn("CV", format="%.3f"),
        },
        hide_index=True,
        width="stretch",
    )


def render_cost(operating: pd.DataFrame) -> None:
    _section("成本分析", "不是单纯找最便宜，而是在质量达标的前提下找到更高成本效率。")
    latest = _latest(operating)
    if latest.empty:
        st.info("当前筛选范围没有成本数据。", icon=":material/info:")
        return
    eligible = latest[latest["quality_score"] >= 90].copy()
    pool = eligible if not eligible.empty else latest
    best = pool.sort_values("cost_performance_score", ascending=False).iloc[0]

    with st.container(border=True):
        st.markdown("### 当前成本结论")
        hero, support = st.columns([5, 7], gap="large", vertical_alignment="center")
        with hero:
            st.caption("质量达标前提下的最优选择")
            st.metric("最省达标模型", str(best["model_id"]), f"性价比 {best['cost_performance_score']:.1f}", border=False)
            st.badge("质量评分 ≥ 90", color="green")
        with support:
            cards = st.columns(3, gap="small")
            cards[0].metric("成本效率评分", f"{_weighted(latest, 'cost_efficiency_score'):.1f}", border=True)
            cards[1].metric("窗口总成本", f"¥{operating['estimated_cost'].sum():,.2f}", help="当前筛选时间窗内全部模型估算成本", border=True)
            cards[2].metric("平均单请求成本", f"¥{operating['estimated_cost'].sum() / operating['request_count'].sum():.5f}", border=True)

    left, right = st.columns(2, gap="large")
    with left.container(border=True):
        st.markdown("### 01 · 钱花在哪里")
        st.caption("同一时间窗口下比较各模型单请求成本，避免只看平台总额。")
        st.altair_chart(_trend_chart(operating, "cost_per_request", "单请求成本（¥）", height=340), width="stretch", theme=None)
    with right.container(border=True):
        st.markdown("### 02 · 是否值得花")
        st.caption("越靠左上越好：成本低、质量高；气泡大小表示调用量。")
        scatter = (
            alt.Chart(latest)
            .mark_circle(opacity=0.85, stroke="white", strokeWidth=2)
            .encode(
                x=alt.X("cost_per_request:Q", title="单请求成本（¥）"),
                y=alt.Y("quality_score:Q", title="质量评分", scale=alt.Scale(zero=False)),
                size=alt.Size("request_count:Q", title="调用量", scale=alt.Scale(range=[250, 1600])),
                color=alt.condition(alt.datum.model_id == str(best["model_id"]), alt.value(BLUE), alt.value(GRAY)),
                tooltip=["model_id:N", alt.Tooltip("cost_per_request:Q", format=".6f"), alt.Tooltip("quality_score:Q", format=".1f"), "request_count:Q"],
            )
            .properties(height=340)
        )
        st.altair_chart(_polish(scatter), width="stretch", theme=None)
    _insight("建议", f"均衡路由优先 {best['model_id']}；若任务质量门槛高于当前 90 分，应先提高质量阈值再重新选型。")

    with st.expander("查看模型成本明细", icon=":material/table_chart:"):
        detail = latest[["model_id", "request_count", "cost_per_request", "cost_per_1k_tokens", "estimated_cost", "quality_score", "cost_performance_score"]].sort_values("cost_performance_score", ascending=False)
        st.dataframe(detail, hide_index=True, width="stretch")


def _higher_score(value: float, good: float, floor: float) -> float:
    return float(np.clip((value - floor) / (good - floor) * 100, 0, 100))


def _lower_score(value: float, good: float, floor: float) -> float:
    return float(np.clip((floor - value) / (floor - good) * 100, 0, 100))


@dataclass(frozen=True)
class ProviderProfile:
    provider: str
    overall: float
    ci_low: float
    ci_high: float
    rate_limit_pct: float


def build_six_dimension_profiles(benchmarks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reproduce the PDF's provider-first, model-equal six-dimension anchors."""
    if benchmarks.empty:
        return pd.DataFrame(), pd.DataFrame()
    model_rows: list[dict[str, object]] = []
    for (provider, model_id), group in benchmarks.groupby(["provider", "model_id"], sort=True):
        valid_speed = group.loc[~group["rate_limited"].astype(bool), "per_request_output_tokens_per_second"].dropna()
        valid_ttft = group.loc[~group["rate_limited"].astype(bool), "ttft_ms"].dropna() / 1000
        speed_mean = float(valid_speed.mean()) if not valid_speed.empty else np.nan
        ttft_mean = float(valid_ttft.mean()) if not valid_ttft.empty else np.nan
        speed_cv = float(valid_speed.std(ddof=0) / valid_speed.mean()) if len(valid_speed) > 1 and valid_speed.mean() else np.nan
        by_concurrency = group[~group["rate_limited"].astype(bool)].groupby("concurrency")["per_request_output_tokens_per_second"].mean()
        degradation = float(1 - by_concurrency.get(50, np.nan) / by_concurrency.get(10, np.nan))
        by_length = group[~group["rate_limited"].astype(bool)].groupby("input_tokens_target")["ttft_ms"].mean()
        long_ratio = float(by_length.get(32000, np.nan) / by_length.get(6000, np.nan))
        success_rate = float(1 - group["rate_limited"].astype(bool).mean())
        raw_values = {
            "生成速度": speed_mean,
            "首响时延": ttft_mean,
            "性能波动系数": speed_cv,
            "并发劣化率": degradation,
            "长文本劣化率": long_ratio,
            "请求成功率": success_rate,
        }
        scores = {
            "生成速度": _higher_score(speed_mean, 30.0, 5.0) if pd.notna(speed_mean) else np.nan,
            "首响时延": _lower_score(ttft_mean, 2.5, 12.0) if pd.notna(ttft_mean) else np.nan,
            "性能波动系数": _lower_score(speed_cv, 0.3, 1.2) if pd.notna(speed_cv) else np.nan,
            "并发劣化率": _lower_score(degradation, 0.2, 0.85) if pd.notna(degradation) else np.nan,
            "长文本劣化率": _lower_score(long_ratio, 1.5, 5.0) if pd.notna(long_ratio) else np.nan,
            "请求成功率": _higher_score(success_rate, 1.0, 0.8),
        }
        available_weight = sum(SIX_WEIGHTS[key] for key, value in scores.items() if pd.notna(value))
        overall = sum(SIX_WEIGHTS[key] * value for key, value in scores.items() if pd.notna(value)) / available_weight
        model_rows.append({
            "provider": provider,
            "model_id": model_id,
            "overall": overall,
            "rate_limit_pct": (1 - success_rate) * 100,
            **scores,
            **{f"raw_{key}": value for key, value in raw_values.items()},
        })

    models = pd.DataFrame(model_rows)
    provider_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(20260730)
    for provider, group in models.groupby("provider", sort=True):
        def score_provider(sample: pd.DataFrame) -> dict[str, float]:
            raw_means = {dimension: float(sample[f"raw_{dimension}"].mean()) for dimension in SIX_DIMENSIONS}
            return {
                "生成速度": _higher_score(raw_means["生成速度"], 30.0, 5.0),
                "首响时延": _lower_score(raw_means["首响时延"], 2.5, 12.0),
                "性能波动系数": _lower_score(raw_means["性能波动系数"], 0.3, 1.2),
                "并发劣化率": _lower_score(raw_means["并发劣化率"], 0.2, 0.85),
                "长文本劣化率": _lower_score(raw_means["长文本劣化率"], 1.5, 5.0),
                "请求成功率": _higher_score(raw_means["请求成功率"], 1.0, 0.8),
            }

        dimension_values = score_provider(group)
        overall = sum(SIX_WEIGHTS[key] * value for key, value in dimension_values.items())
        bootstrap = []
        for _ in range(4000):
            sample = group.iloc[rng.integers(0, len(group), len(group))]
            sampled_scores = score_provider(sample)
            bootstrap.append(sum(SIX_WEIGHTS[key] * value for key, value in sampled_scores.items()))
        provider_rows.append({
            "provider": provider,
            "overall": overall,
            "ci_low": float(np.quantile(bootstrap, 0.025)),
            "ci_high": float(np.quantile(bootstrap, 0.975)),
            "rate_limit_pct": float(group["rate_limit_pct"].mean()),
            **dimension_values,
        })
    return pd.DataFrame(provider_rows).sort_values("overall", ascending=False), models


def render_model_profile(benchmarks: pd.DataFrame, capability: pd.DataFrame) -> None:
    _section("模型画像与路由适配", "用六维真实压测画像回答“派谁”，能力评测只作为任务级补充，不混入六维运营分。")
    providers, model_details = build_six_dimension_profiles(benchmarks)
    if providers.empty:
        st.info("当前没有可用的外部压测画像数据。", icon=":material/info:")
        return

    st.caption("口径：138 条真实压测 · 六维绝对锚点 · 先按子模型计算再按供应商等权 · Bootstrap 4000 次 95%CI")
    cards = st.columns(len(providers), gap="medium")
    for rank, (card, (_, row)) in enumerate(zip(cards, providers.iterrows(), strict=True), start=1):
        with card.container(border=True, height="stretch", key=f"provider_profile_{rank}"):
            st.caption(f"综合运营分第 {rank} 名")
            st.markdown(f"### {row['provider']}")
            st.metric("六维综合运营分", f"{row['overall']:.1f}", border=False)
            st.caption(f"95%CI [{row['ci_low']:.1f}, {row['ci_high']:.1f}]")
            st.badge("主路由候选" if rank == 1 else "辅助候选", color="blue" if rank == 1 else "gray")
            if row["rate_limit_pct"] > 0:
                st.badge(f"限流率 {row['rate_limit_pct']:.1f}%", color="red" if row["rate_limit_pct"] >= 20 else "orange")

    st.markdown("### 六维画像横向对比")
    long = providers.melt(id_vars=["provider", "overall"], value_vars=SIX_DIMENSIONS, var_name="维度", value_name="得分")
    chart = (
        alt.Chart(long)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            y=alt.Y("维度:N", title=None, sort=SIX_DIMENSIONS),
            x=alt.X("得分:Q", title="锚点得分", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("provider:N", scale=alt.Scale(range=SERIES)),
            yOffset="provider:N",
            tooltip=["provider:N", "维度:N", alt.Tooltip("得分:Q", format=".1f")],
        )
        .properties(height=360)
    )
    st.altair_chart(_polish(chart), width="stretch", theme=None)
    st.caption("六维仅描述运营健康。限流率是独立门控，不计入综合分；成本、合规、多模态也不混入六维。")

    st.markdown("### 路由决策矩阵")
    ranking = providers["provider"].tolist()
    best, second, third = ranking[0], ranking[min(1, len(ranking) - 1)], ranking[-1]
    matrix = pd.DataFrame({
        "请求类型": ["短文本对话", "长文档 / RAG", "高并发客服", "代码 / 技术", "合规敏感"],
        "低并发 ≤10": [f"{third} / {best}", best, f"{best} / {second}", "待能力评测确认", "待合规门控"],
        "中并发 30": [f"{third} / {best}", best, f"{best} / {second}", "待能力评测确认", "待合规门控"],
        "高并发 50+": [f"{best} / {second}", best, f"{best} / {second}", "待能力评测确认", "待合规门控"],
        "门控说明": ["成功率 <80% 降级", "请求长度不超过已测 32k", "限流率 >20% 降级", "当前仅补充能力数据", "当前不可据此路由"],
    })
    st.dataframe(matrix, hide_index=True, width="stretch")

    with st.container(border=True):
        st.markdown("### 任务级能力评测 · 补充证据")
        st.caption("推理、指令遵循、工具调用、结构化输出来自独立能力评测；不参与六维运营分。")
        labels = {"reasoning": "推理能力", "instruction_following": "指令遵循", "tool_call": "工具调用", "structured_output": "结构化输出"}
        ability = capability.copy()
        ability["能力"] = ability["capability_dimension"].map(labels)
        ability_chart = (
            alt.Chart(ability)
            .mark_bar(cornerRadiusEnd=4)
            .encode(
                y=alt.Y("能力:N", title=None),
                x=alt.X("quality_score:Q", title="质量评分", scale=alt.Scale(domain=[0, 100])),
                color=alt.Color("model_id:N", scale=alt.Scale(range=SERIES)),
                yOffset="model_id:N",
                tooltip=["model_id:N", "能力:N", alt.Tooltip("quality_score:Q", format=".1f")],
            )
            .properties(height=260)
        )
        st.altair_chart(_polish(ability_chart), width="stretch", theme=None)

    with st.expander("查看 15 个压测模型明细", icon=":material/table_chart:"):
        st.dataframe(model_details.sort_values("overall", ascending=False), hide_index=True, width="stretch")


def render_capacity(resource_capacity: pd.DataFrame) -> None:
    _section("容量诊断", "只回答三件事：谁最危险、证据是什么、路由现在该怎么调整。")
    latest = _latest(resource_capacity)
    if latest.empty:
        st.info("当前筛选范围没有容量数据。", icon=":material/info:")
        return
    latest = latest.sort_values(["hbm_headroom_pct", "waiting_max_busy"])
    danger = latest.iloc[0]
    fallback = latest.sort_values(["hbm_headroom_pct", "waiting_max_busy"], ascending=[False, True]).iloc[0]
    high = latest[(latest["hbm_headroom_pct"] < 10) | (latest["waiting_max_busy"] > 0)]

    with st.container(border=True):
        st.markdown("### 当前容量结论")
        hero, support = st.columns([5, 7], gap="large", vertical_alignment="center")
        with hero:
            st.metric("最高风险模型", str(danger["model_id"]), border=False)
            st.badge("需要分流" if len(high) else "容量健康", color="red" if len(high) else "green")
            st.caption(f"主要证据：HBM 余量 {danger['hbm_headroom_pct']:.1f}% · 等待队列 {int(danger['waiting_max_busy'])}")
        with support:
            cards = st.columns(3, gap="small")
            cards[0].metric("高风险模型", len(high), border=True)
            cards[1].metric("最大排队", f"{int(latest['waiting_max_busy'].max())} 请求", border=True)
            cards[2].metric("最低 HBM 余量", f"{latest['hbm_headroom_pct'].min():.1f}%", border=True)

    evidence, actions = st.columns([7, 5], gap="large")
    with evidence.container(border=True):
        st.markdown("### 01 · 风险证据")
        metric = st.segmented_control("查看容量指标", ["HBM 余量", "NPU P95", "等待队列"], default="HBM 余量")
        mapping = {"HBM 余量": ("hbm_headroom_pct", "余量（%）"), "NPU P95": ("npu_p95", "利用率（%）"), "等待队列": ("waiting_max_busy", "请求数")}
        field, title = mapping[str(metric)]
        st.altair_chart(_trend_chart(resource_capacity, field, title, height=360), width="stretch", theme=None)
        st.caption("NPU：算力利用率；HBM：加速卡显存余量。平均利用率低但 HBM 很高，通常意味着模型驻留内存压力。")
    with actions.container(border=True):
        st.markdown("### 02 · 路由动作")
        for _, row in latest.iterrows():
            is_risk = row["hbm_headroom_pct"] < 10 or row["waiting_max_busy"] > 0
            st.markdown(f"**{row['model_id']}**")
            st.caption(f"NPU P95 {row['npu_p95']:.1f}% · HBM 余量 {row['hbm_headroom_pct']:.1f}% · 队列 {int(row['waiting_max_busy'])}")
            st.badge("降低路由权重 / 准备扩容" if is_risk else "维持并作为兜底", color="red" if is_risk else "green")
        st.space("small")
        _insight("切换规则", f"当 {danger['model_id']} HBM 余量低于 10% 或持续排队，将 30% 新流量切至 {fallback['model_id']}；余量恢复至 20% 后再逐步回切。", tone="danger")

    with st.expander("查看容量技术明细", icon=":material/table_chart:"):
        st.dataframe(latest, hide_index=True, width="stretch")
