"""基于真实调用数据生成模型运营评分与健康排行。

原始调用日志负责准确计算调用量、Token、成本、成功率和延迟分位数；
小时模型特征负责计算日内延迟与成功率波动。所有评分目标和权重均来自
指标字典的 ``Scoring Policy``，本模块只实现可复用的数据聚合与评分流程。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

try:
    from .model_scoring import (
        ScoringPolicy,
        calculate_family_score,
        classify_score,
        load_scoring_policy,
    )
except ImportError:  # 支持 ``python src/model_operations.py`` 直接运行。
    from model_scoring import (
        ScoringPolicy,
        calculate_family_score,
        classify_score,
        load_scoring_policy,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_INPUT = PROJECT_ROOT / "data" / "synthetic_logs_v2.csv"
DEFAULT_HOURLY_INPUT = (
    PROJECT_ROOT / "outputs" / "features" / "model_hourly_features.csv"
)
DEFAULT_CAPABILITY_INPUT = PROJECT_ROOT / "outputs" / "model_capability_scores.csv"
DEFAULT_CONFIG = PROJECT_ROOT / "docs" / "ai_monitoring_metric_dictionary.xlsx"
DEFAULT_SCORE_OUTPUT = PROJECT_ROOT / "outputs" / "model_operating_scores.csv"
DEFAULT_SNAPSHOT_OUTPUT = PROJECT_ROOT / "outputs" / "model_operating_snapshot.csv"

LOG_REQUIRED_COLUMNS = {
    "request_id",
    "timestamp",
    "model_id",
    "total_tokens",
    "estimated_cost",
    "latency_ms",
    "status_code",
}
HOURLY_REQUIRED_COLUMNS = {
    "hour",
    "model_id",
    "request_count",
    "success_rate",
    "p95_latency_ms",
}


def load_inputs(
    log_path: Path = DEFAULT_LOG_INPUT,
    hourly_path: Path = DEFAULT_HOURLY_INPUT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取真实调用日志和模型小时特征，并校验评分所需字段。"""

    if not log_path.exists():
        raise FileNotFoundError(f"找不到调用日志：{log_path}")
    if not hourly_path.exists():
        raise FileNotFoundError(f"找不到模型小时特征：{hourly_path}")
    logs = pd.read_csv(log_path, parse_dates=["timestamp"])
    hourly = pd.read_csv(hourly_path, parse_dates=["hour"])
    _require_columns(logs, LOG_REQUIRED_COLUMNS, "调用日志")
    _require_columns(hourly, HOURLY_REQUIRED_COLUMNS, "模型小时特征")
    if logs.empty or hourly.empty:
        raise ValueError("调用日志和模型小时特征不能为空")
    return logs, hourly


def build_daily_operating_metrics(
    logs: pd.DataFrame,
    hourly_features: pd.DataFrame,
    baseline_days: int = 7,
    minimum_baseline_days: int = 3,
) -> pd.DataFrame:
    """构建模型日级运营指标。

    ``cost_trend_ratio`` 使用当日单请求成本除以前序窗口的单请求成本中位数，
    从而避免调用量变化被误判为价格变化。历史不足时以1.0作为中性输入，并由
    ``cost_baseline_ready`` 明确标记，避免把默认值误认为已有稳定基线。
    """

    _require_columns(logs, LOG_REQUIRED_COLUMNS, "调用日志")
    _require_columns(hourly_features, HOURLY_REQUIRED_COLUMNS, "模型小时特征")
    if baseline_days <= 0 or minimum_baseline_days <= 0:
        raise ValueError("成本基线窗口和最小历史天数必须大于0")
    if minimum_baseline_days > baseline_days:
        raise ValueError("最小历史天数不能大于成本基线窗口")

    data = logs.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["date"] = data["timestamp"].dt.floor("D")
    data["is_success"] = data["status_code"].between(200, 299)
    daily = (
        data.groupby(["date", "model_id"], as_index=False)
        .agg(
            request_count=("request_id", "count"),
            total_tokens=("total_tokens", "sum"),
            estimated_cost=("estimated_cost", "sum"),
            success_rate=("is_success", lambda values: values.mean() * 100),
            p50_latency_ms=("latency_ms", lambda values: values.quantile(0.50)),
            p95_latency_ms=("latency_ms", lambda values: values.quantile(0.95)),
            p99_latency_ms=("latency_ms", lambda values: values.quantile(0.99)),
        )
        .sort_values(["model_id", "date"])
    )
    daily["cost_per_request"] = daily["estimated_cost"] / daily["request_count"]
    daily["cost_per_1k_tokens"] = (
        daily["estimated_cost"]
        / daily["total_tokens"].replace(0, np.nan)
        * 1000
    )

    hourly = hourly_features.copy()
    hourly["hour"] = pd.to_datetime(hourly["hour"])
    hourly["date"] = hourly["hour"].dt.floor("D")
    stability = (
        hourly.groupby(["date", "model_id"], as_index=False)
        .agg(
            observed_hours=("hour", "nunique"),
            p95_latency_mean_ms=("p95_latency_ms", "mean"),
            p95_latency_std_ms=("p95_latency_ms", lambda values: values.std(ddof=0)),
            success_rate_std_pct=("success_rate", lambda values: values.std(ddof=0)),
        )
    )
    stability["latency_cv"] = (
        stability["p95_latency_std_ms"]
        / stability["p95_latency_mean_ms"].replace(0, np.nan)
    )
    insufficient_hours = stability["observed_hours"].lt(2)
    stability.loc[
        insufficient_hours, ["latency_cv", "success_rate_std_pct"]
    ] = np.nan
    daily = daily.merge(stability, on=["date", "model_id"], how="left", validate="one_to_one")

    baseline = daily.groupby("model_id", group_keys=False)["cost_per_request"].transform(
        lambda values: values.shift(1).rolling(
            baseline_days, min_periods=minimum_baseline_days
        ).median()
    )
    daily["cost_baseline_per_request"] = baseline
    daily["cost_baseline_ready"] = baseline.notna()
    daily["cost_trend_ratio"] = (
        daily["cost_per_request"] / baseline.replace(0, np.nan)
    ).where(daily["cost_baseline_ready"], 1.0)
    return _round_numeric(daily.sort_values(["date", "model_id"]).reset_index(drop=True))


def load_capability_quality(path: Path = DEFAULT_CAPABILITY_INPUT) -> pd.DataFrame:
    """将能力维度结果按有效样本量加权汇总为模型质量分。"""

    if not path.exists():
        return pd.DataFrame(columns=["model_id", "quality_score"])
    scores = pd.read_csv(path)
    required = {"model_id", "quality_score", "run_count"}
    _require_columns(scores, required, "能力维度评分")
    scores["weighted_quality"] = scores["quality_score"] * scores["run_count"]
    quality = (
        scores.groupby("model_id", as_index=False)
        .agg(
            weighted_quality=("weighted_quality", "sum"),
            capability_run_count=("run_count", "sum"),
            capability_dimension_count=("capability_dimension", "nunique"),
        )
    )
    quality["quality_score"] = (
        quality["weighted_quality"] / quality["capability_run_count"]
    )
    return _round_numeric(
        quality.drop(columns="weighted_quality").sort_values("model_id")
    )


def score_model_operations(
    daily_metrics: pd.DataFrame,
    policy: ScoringPolicy,
    capability_quality: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """按配置生成成功、性能、成本和健康评分。"""

    required = {
        "date", "model_id", "success_rate", "p50_latency_ms", "p95_latency_ms",
        "p99_latency_ms", "latency_cv", "success_rate_std_pct",
        "cost_per_request", "cost_per_1k_tokens", "cost_trend_ratio",
    }
    _require_columns(daily_metrics, required, "模型日级运营指标")
    scored = daily_metrics.copy()
    scored["success_score"] = scored.apply(
        lambda row: _family_or_nan("success", row, policy), axis=1
    )
    scored["latency_score"] = scored.apply(
        lambda row: _family_or_nan("latency", row, policy), axis=1
    )
    scored["stability_score"] = scored.apply(
        lambda row: _family_or_nan("stability", row, policy), axis=1
    )
    scored["performance_score"] = scored.apply(
        lambda row: _family_or_nan("performance", row, policy), axis=1
    )
    scored["cost_efficiency_score"] = scored.apply(
        lambda row: _family_or_nan("cost_efficiency", row, policy), axis=1
    )

    if capability_quality is not None and not capability_quality.empty:
        quality_columns = [
            column for column in (
                "model_id", "quality_score", "capability_run_count",
                "capability_dimension_count",
            ) if column in capability_quality.columns
        ]
        scored = scored.merge(
            capability_quality[quality_columns],
            on="model_id",
            how="left",
            validate="many_to_one",
        )
    else:
        scored["quality_score"] = np.nan

    scored["cost_performance_score"] = scored.apply(
        lambda row: _family_or_nan("cost_performance", row, policy), axis=1
    )
    scored["health_score"] = scored.apply(
        lambda row: _family_or_nan("health", row, policy), axis=1
    )
    scored["health_level"] = scored["health_score"].map(
        lambda value: classify_score("health", value, policy)
        if pd.notna(value)
        else "数据不足"
    )
    return _round_numeric(scored.sort_values(["date", "model_id"]).reset_index(drop=True))


def build_latest_snapshot(scored: pd.DataFrame) -> pd.DataFrame:
    """提取每个模型最新有效记录并生成健康评分排行。"""

    _require_columns(scored, {"date", "model_id", "health_score"}, "运营评分")
    latest = (
        scored.sort_values(["model_id", "date"])
        .groupby("model_id", as_index=False)
        .tail(1)
        .copy()
    )
    latest["health_rank"] = (
        latest["health_score"].rank(method="min", ascending=False, na_option="bottom").astype(int)
    )
    preferred = [
        "health_rank", "date", "model_id", "health_score", "health_level",
        "success_score", "performance_score", "stability_score",
        "cost_efficiency_score", "cost_performance_score", "quality_score",
        "request_count", "total_tokens", "estimated_cost", "success_rate",
        "p50_latency_ms", "p95_latency_ms", "p99_latency_ms", "latency_cv",
        "cost_per_request", "cost_per_1k_tokens", "cost_trend_ratio",
        "cost_baseline_ready", "observed_hours",
    ]
    columns = [column for column in preferred if column in latest.columns]
    return latest[columns].sort_values(["health_rank", "model_id"]).reset_index(drop=True)


def _family_or_nan(
    family: str,
    row: Mapping[str, object],
    policy: ScoringPolicy,
) -> float:
    components = {
        rule.component: row.get(rule.component) for rule in policy.rules_for(family)
    }
    if any(value is None or not np.isfinite(float(value)) for value in components.values()):
        return float("nan")
    return calculate_family_score(family, components, policy)


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{label}缺少字段：{', '.join(sorted(missing))}")


def _round_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    numeric = result.select_dtypes(include="number").columns
    result[numeric] = result[numeric].round(4)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成模型运营评分和健康排行")
    parser.add_argument("--logs", type=Path, default=DEFAULT_LOG_INPUT)
    parser.add_argument("--hourly", type=Path, default=DEFAULT_HOURLY_INPUT)
    parser.add_argument("--capability", type=Path, default=DEFAULT_CAPABILITY_INPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--score-output", type=Path, default=DEFAULT_SCORE_OUTPUT)
    parser.add_argument("--snapshot-output", type=Path, default=DEFAULT_SNAPSHOT_OUTPUT)
    parser.add_argument("--baseline-days", type=int, default=7)
    parser.add_argument("--minimum-baseline-days", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logs, hourly = load_inputs(args.logs, args.hourly)
    policy = load_scoring_policy(args.config)
    daily = build_daily_operating_metrics(
        logs,
        hourly,
        baseline_days=args.baseline_days,
        minimum_baseline_days=args.minimum_baseline_days,
    )
    capability_quality = load_capability_quality(args.capability)
    scored = score_model_operations(daily, policy, capability_quality)
    snapshot = build_latest_snapshot(scored)
    args.score_output.parent.mkdir(parents=True, exist_ok=True)
    args.snapshot_output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.score_output, index=False, encoding="utf-8-sig")
    snapshot.to_csv(args.snapshot_output, index=False, encoding="utf-8-sig")
    print(f"模型日级运营评分：{args.score_output}（{len(scored):,} 行）")
    print(f"最新模型健康排行：{args.snapshot_output}（{len(snapshot):,} 行）")
    print(
        snapshot[
            [
                "health_rank", "model_id", "health_score", "health_level",
                "performance_score", "cost_efficiency_score",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
