"""融合真实调用与标准能力拨测，生成诊断结果和模型能力画像。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

try:
    from .model_scoring import (
        ScoringPolicy,
        calculate_family_score,
        clamp_score,
        load_scoring_policy,
    )
except ImportError:  # 支持 ``python src/model_profile.py`` 直接运行。
    from model_scoring import (
        ScoringPolicy,
        calculate_family_score,
        clamp_score,
        load_scoring_policy,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPERATING_INPUT = PROJECT_ROOT / "outputs" / "model_operating_scores.csv"
DEFAULT_PROBE_RUN_INPUT = PROJECT_ROOT / "data" / "capability_probe_runs.csv"
DEFAULT_AVAILABILITY_RUN_INPUT = PROJECT_ROOT / "data" / "probe_runs.csv"
DEFAULT_CAPABILITY_INPUT = PROJECT_ROOT / "outputs" / "model_capability_scores.csv"
DEFAULT_CONFIG = PROJECT_ROOT / "docs" / "ai_monitoring_metric_dictionary.xlsx"
DEFAULT_DIAGNOSIS_OUTPUT = PROJECT_ROOT / "outputs" / "model_fusion_diagnosis.csv"
DEFAULT_PROFILE_OUTPUT = PROJECT_ROOT / "outputs" / "model_capability_profiles.csv"

DIMENSIONS = (
    "instruction_following",
    "structured_output",
    "reasoning",
    "tool_call",
)
DIAGNOSIS_POLICY_KEYS = {
    "real_success_warning_pct",
    "probe_success_warning_pct",
    "real_performance_warning_score",
    "probe_performance_warning_score",
    "probe_p95_latency_warning_ms",
    "latency_gap_warning_ratio",
    "primary_route_min_score",
    "backup_route_min_score",
    "freshness_decay_per_day",
    "primary_route_min_stability_score",
    "primary_route_min_confidence_score",
}


@dataclass(frozen=True)
class DiagnosisPolicy:
    real_success_warning_pct: float
    probe_success_warning_pct: float
    real_performance_warning_score: float
    probe_performance_warning_score: float
    probe_p95_latency_warning_ms: float
    latency_gap_warning_ratio: float
    primary_route_min_score: float
    backup_route_min_score: float
    freshness_decay_per_day: float
    primary_route_min_stability_score: float
    primary_route_min_confidence_score: float

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "DiagnosisPolicy":
        missing = DIAGNOSIS_POLICY_KEYS.difference(values)
        if missing:
            raise ValueError(f"诊断策略缺少阈值：{', '.join(sorted(missing))}")
        numbers = {key: _finite_number(values[key], key) for key in DIAGNOSIS_POLICY_KEYS}
        for key in (
            "real_success_warning_pct", "probe_success_warning_pct",
            "real_performance_warning_score", "probe_performance_warning_score",
            "primary_route_min_score", "backup_route_min_score",
            "primary_route_min_stability_score",
            "primary_route_min_confidence_score",
        ):
            if not 0 <= numbers[key] <= 100:
                raise ValueError(f"{key} 必须处于0到100")
        if numbers["latency_gap_warning_ratio"] <= 0:
            raise ValueError("latency_gap_warning_ratio 必须大于0")
        if numbers["probe_p95_latency_warning_ms"] <= 0:
            raise ValueError("probe_p95_latency_warning_ms 必须大于0")
        if numbers["freshness_decay_per_day"] <= 0:
            raise ValueError("freshness_decay_per_day 必须大于0")
        if numbers["primary_route_min_score"] <= numbers["backup_route_min_score"]:
            raise ValueError("主路由阈值必须高于辅助路由阈值")
        return cls(**numbers)


def load_diagnosis_policy(path: Path = DEFAULT_CONFIG) -> DiagnosisPolicy:
    frame = pd.read_excel(path, sheet_name="Diagnosis Policy")
    active = frame[frame["status"].astype(str).str.lower().eq("active")]
    if active["metric"].duplicated().any():
        raise ValueError("Diagnosis Policy 存在重复 metric")
    values = dict(zip(active["metric"], active["threshold"], strict=False))
    return DiagnosisPolicy.from_mapping(values)


def build_daily_probe_metrics(
    probe_runs: pd.DataFrame,
    policy: ScoringPolicy,
    availability_runs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """聚合主动拨测；能力任务评质量，高频可用性探针评服务状态。"""

    required = {
        "started_at", "model_id", "provider", "task_id", "capability_dimension",
        "task_weight", "status_code", "latency_ms", "task_score", "passed",
        "evaluator_confidence",
    }
    _require_columns(probe_runs, required, "能力拨测记录")
    data = probe_runs.copy()
    data["started_at"] = pd.to_datetime(data["started_at"])
    data["date"] = data["started_at"].dt.floor("D")
    data["http_success"] = data["status_code"].between(200, 299)
    data["weighted_task_score"] = data["task_score"] * data["task_weight"]

    daily = (
        data.groupby(["date", "model_id"], as_index=False)
        .agg(
            provider=("provider", "first"),
            probe_run_count=("task_id", "size"),
            probe_task_count=("task_id", "nunique"),
            probe_dimension_count=("capability_dimension", "nunique"),
            task_weight_sum=("task_weight", "sum"),
            weighted_task_score=("weighted_task_score", "sum"),
            probe_http_success_rate=("http_success", lambda values: values.mean() * 100),
            probe_pass_rate=("passed", lambda values: values.astype(bool).mean() * 100),
            probe_score_std=("task_score", lambda values: values.std(ddof=0)),
            probe_p50_latency_ms=("latency_ms", lambda values: values.quantile(0.50)),
            probe_p95_latency_ms=("latency_ms", lambda values: values.quantile(0.95)),
            probe_p99_latency_ms=("latency_ms", lambda values: values.quantile(0.99)),
            evaluator_confidence_score=("evaluator_confidence", "mean"),
        )
    )
    daily["probe_quality_score"] = (
        daily["weighted_task_score"] / daily["task_weight_sum"]
    )
    daily["probe_consistency_score"] = (
        100 - daily["probe_score_std"]
    ).clip(lower=0, upper=100)

    if availability_runs is not None and not availability_runs.empty:
        availability_required = {
            "started_at", "model_id", "provider", "probe_type", "status_code",
            "latency_ms",
        }
        _require_columns(availability_runs, availability_required, "可用性拨测记录")
        availability = availability_runs.copy()
        availability = availability[
            availability["probe_type"].astype(str).eq("availability")
        ].copy()
        if availability.empty:
            raise ValueError("可用性拨测记录中没有 probe_type=availability 数据")
        availability["started_at"] = pd.to_datetime(availability["started_at"])
        availability["date"] = availability["started_at"].dt.floor("D")
        availability["http_success"] = availability["status_code"].between(200, 299)
        availability_daily = (
            availability.groupby(["date", "model_id"], as_index=False)
            .agg(
                availability_probe_run_count=("probe_type", "size"),
                availability_http_success_rate=(
                    "http_success", lambda values: values.mean() * 100
                ),
                availability_p50_latency_ms=(
                    "latency_ms", lambda values: values.quantile(0.50)
                ),
                availability_p95_latency_ms=(
                    "latency_ms", lambda values: values.quantile(0.95)
                ),
                availability_p99_latency_ms=(
                    "latency_ms", lambda values: values.quantile(0.99)
                ),
            )
        )
        daily = daily.merge(
            availability_daily,
            on=["date", "model_id"],
            how="left",
            validate="one_to_one",
        )
        if daily["availability_http_success_rate"].isna().any():
            raise ValueError("部分模型日期缺少可用性探针，无法完成对称融合诊断")
        daily["probe_http_success_rate"] = daily["availability_http_success_rate"]
        daily["probe_p50_latency_ms"] = daily["availability_p50_latency_ms"]
        daily["probe_p95_latency_ms"] = daily["availability_p95_latency_ms"]
        daily["probe_p99_latency_ms"] = daily["availability_p99_latency_ms"]
    daily["probe_latency_score"] = daily.apply(
        lambda row: calculate_family_score(
            "latency",
            {
                "p50_latency_ms": row["probe_p50_latency_ms"],
                "p95_latency_ms": row["probe_p95_latency_ms"],
                "p99_latency_ms": row["probe_p99_latency_ms"],
            },
            policy,
        ),
        axis=1,
    )
    daily["probe_performance_score"] = daily.apply(
        lambda row: calculate_family_score(
            "performance",
            {
                "latency_score": row["probe_latency_score"],
                "stability_score": row["probe_consistency_score"],
            },
            policy,
        ),
        axis=1,
    )
    return _round_numeric(
        daily.drop(columns=["task_weight_sum", "weighted_task_score"])
        .sort_values(["date", "model_id"])
        .reset_index(drop=True)
    )


def build_fusion_diagnosis(
    operating_scores: pd.DataFrame,
    daily_probe_metrics: pd.DataFrame,
    policy: DiagnosisPolicy,
) -> pd.DataFrame:
    """比较真实调用和控制变量拨测，输出异常来源判断与建议动作。"""

    operating_required = {
        "date", "model_id", "success_rate", "p95_latency_ms",
        "performance_score", "stability_score", "health_score",
    }
    probe_required = {
        "date", "model_id", "probe_http_success_rate", "probe_p95_latency_ms",
        "probe_performance_score", "probe_consistency_score", "probe_quality_score",
    }
    _require_columns(operating_scores, operating_required, "模型运营评分")
    _require_columns(daily_probe_metrics, probe_required, "能力拨测日指标")
    operating = operating_scores.copy()
    probe = daily_probe_metrics.copy()
    operating["date"] = pd.to_datetime(operating["date"])
    probe["date"] = pd.to_datetime(probe["date"])
    merged = operating.merge(
        probe,
        on=["date", "model_id"],
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("真实调用与主动拨测没有可对齐的模型日期")
    merged["success_gap_pct"] = (
        merged["probe_http_success_rate"] - merged["success_rate"]
    )
    merged["latency_gap_ratio"] = (
        merged["p95_latency_ms"]
        / merged["probe_p95_latency_ms"].replace(0, np.nan)
    )
    merged["performance_gap_score"] = (
        merged["probe_performance_score"] - merged["performance_score"]
    )

    diagnoses = [
        _diagnose_row(row, policy) for _, row in merged.iterrows()
    ]
    diagnosis_frame = pd.DataFrame(diagnoses, index=merged.index)
    merged = pd.concat([merged, diagnosis_frame], axis=1)
    columns = [
        "date", "model_id", "provider", "success_rate",
        "probe_http_success_rate", "success_gap_pct", "p95_latency_ms",
        "probe_p95_latency_ms", "latency_gap_ratio", "performance_score",
        "probe_performance_score", "performance_gap_score", "stability_score",
        "probe_consistency_score", "health_score", "probe_quality_score",
        "diagnosis_type", "diagnosis_severity", "diagnosis_reason",
        "switch_recommendation", "recommended_action",
    ]
    return _round_numeric(
        merged[[column for column in columns if column in merged.columns]]
        .sort_values(["date", "model_id"])
        .reset_index(drop=True)
    )


def _diagnose_row(row: pd.Series, policy: DiagnosisPolicy) -> dict[str, str]:
    real_issue = (
        row["success_rate"] < policy.real_success_warning_pct
        or row["performance_score"] < policy.real_performance_warning_score
    )
    probe_issue = (
        row["probe_http_success_rate"] < policy.probe_success_warning_pct
        or row["probe_performance_score"] < policy.probe_performance_warning_score
        or row["probe_p95_latency_ms"] > policy.probe_p95_latency_warning_ms
    )
    if real_issue and probe_issue:
        return {
            "diagnosis_type": "model_side_degradation",
            "diagnosis_severity": "high",
            "diagnosis_reason": "真实调用与标准环境同步下降，模型服务或模型能力更可能异常",
            "switch_recommendation": "建议切换",
            "recommended_action": "降低当前模型路由权重，启用候选模型并复核供应商状态与能力维度",
        }
    if real_issue:
        return {
            "diagnosis_type": "platform_or_traffic_issue",
            "diagnosis_severity": "medium",
            "diagnosis_reason": "真实调用下降但标准拨测正常，优先排查用户输入、网络和平台链路",
            "switch_recommendation": "暂不切换",
            "recommended_action": "检查网关、区域网络、队列和异常客户流量，再决定是否调整模型",
        }
    if probe_issue:
        return {
            "diagnosis_type": "capability_or_probe_issue",
            "diagnosis_severity": "medium",
            "diagnosis_reason": "真实调用正常但标准任务下降，可能存在特定能力回退或拨测环境问题",
            "switch_recommendation": "谨慎降权",
            "recommended_action": "复跑失败任务并核对评测器；确认能力回退后限制相关场景流量",
        }
    if row["latency_gap_ratio"] >= policy.latency_gap_warning_ratio:
        return {
            "diagnosis_type": "environment_latency_gap",
            "diagnosis_severity": "low",
            "diagnosis_reason": "标准拨测健康但真实调用延迟明显更高，差异更可能来自业务环境",
            "switch_recommendation": "暂不切换",
            "recommended_action": "检查请求长度、并发、区域网络和平台排队，不直接归因于模型",
        }
    return {
        "diagnosis_type": "healthy",
        "diagnosis_severity": "none",
        "diagnosis_reason": "真实调用与标准拨测均处于健康范围",
        "switch_recommendation": "无需切换",
        "recommended_action": "维持当前路由并持续观察",
    }


def build_model_profiles(
    operating_scores: pd.DataFrame,
    capability_scores: pd.DataFrame,
    daily_probe_metrics: pd.DataFrame,
    diagnosis: pd.DataFrame,
    scoring_policy: ScoringPolicy,
    diagnosis_policy: DiagnosisPolicy,
) -> pd.DataFrame:
    """生成能力、稳定性、性能、可信度和路由就绪度模型画像。"""

    _require_columns(
        capability_scores,
        {
            "model_id", "capability_dimension", "quality_score",
            "consistency_score", "run_count", "task_count", "latest_run_at",
        },
        "能力维度评分",
    )
    latest_operating = (
        operating_scores.assign(date=pd.to_datetime(operating_scores["date"]))
        .sort_values(["model_id", "date"])
        .groupby("model_id", as_index=False)
        .tail(1)
        [[
            "date", "model_id", "stability_score", "performance_score",
            "cost_performance_score", "health_score",
        ]]
    )
    latest_probe = (
        daily_probe_metrics.assign(date=pd.to_datetime(daily_probe_metrics["date"]))
        .sort_values(["model_id", "date"])
        .groupby("model_id", as_index=False)
        .tail(1)
    )
    latest_diagnosis = (
        diagnosis.assign(date=pd.to_datetime(diagnosis["date"]))
        .sort_values(["model_id", "date"])
        .groupby("model_id", as_index=False)
        .tail(1)
    )

    capability = capability_scores.copy()
    capability["latest_run_at"] = pd.to_datetime(capability["latest_run_at"])
    dimension_pivot = capability.pivot(
        index="model_id", columns="capability_dimension", values="quality_score"
    ).reset_index()
    dimension_pivot = dimension_pivot.rename(
        columns={dimension: f"{dimension}_score" for dimension in DIMENSIONS}
    )
    summary = (
        capability.assign(
            weighted_consistency=lambda frame: frame["consistency_score"] * frame["run_count"]
        )
        .groupby("model_id", as_index=False)
        .agg(
            capability_run_count=("run_count", "sum"),
            capability_task_count=("task_count", "sum"),
            weighted_consistency=("weighted_consistency", "sum"),
            latest_capability_run_at=("latest_run_at", "max"),
        )
    )
    summary["probe_consistency_score"] = (
        summary["weighted_consistency"] / summary["capability_run_count"]
    )
    summary = summary.drop(columns="weighted_consistency")

    profiles = latest_operating.merge(dimension_pivot, on="model_id", validate="one_to_one")
    profiles = profiles.merge(summary, on="model_id", validate="one_to_one")
    profiles = profiles.merge(
        latest_probe[
            ["model_id", "probe_performance_score", "probe_quality_score"]
        ],
        on="model_id",
        validate="one_to_one",
    )
    profiles = profiles.merge(
        latest_diagnosis[
            [
                "model_id", "diagnosis_type", "diagnosis_severity",
                "switch_recommendation", "recommended_action",
            ]
        ],
        on="model_id",
        validate="one_to_one",
    )
    profiles["capability_score"] = profiles.apply(
        lambda row: calculate_family_score(
            "profile_capability",
            {f"{dimension}_score": row[f"{dimension}_score"] for dimension in DIMENSIONS},
            scoring_policy,
        ),
        axis=1,
    )
    profiles["profile_stability_score"] = profiles.apply(
        lambda row: calculate_family_score(
            "profile_stability",
            {
                "real_stability_score": row["stability_score"],
                "probe_consistency_score": row["probe_consistency_score"],
            },
            scoring_policy,
        ),
        axis=1,
    )
    profiles["profile_performance_score"] = profiles.apply(
        lambda row: calculate_family_score(
            "profile_performance",
            {
                "real_performance_score": row["performance_score"],
                "probe_performance_score": row["probe_performance_score"],
            },
            scoring_policy,
        ),
        axis=1,
    )

    expected_tasks = profiles["capability_task_count"].max()
    expected_runs = profiles["capability_run_count"].max()
    profiles["task_coverage_score"] = (
        profiles["capability_task_count"] / expected_tasks * 100
    ).clip(upper=100)
    profiles["sample_sufficiency_score"] = (
        profiles["capability_run_count"] / expected_runs * 100
    ).clip(upper=100)
    as_of = profiles["latest_capability_run_at"].max()
    profiles["freshness_score"] = profiles["latest_capability_run_at"].map(
        lambda value: clamp_score(
            100 - max(0.0, (as_of - value).total_seconds() / 86400)
            * diagnosis_policy.freshness_decay_per_day
        )
    )
    profiles["evaluator_consistency_score"] = profiles["probe_consistency_score"]
    profiles["confidence_score"] = profiles.apply(
        lambda row: calculate_family_score(
            "confidence",
            {
                "task_coverage_score": row["task_coverage_score"],
                "sample_sufficiency_score": row["sample_sufficiency_score"],
                "freshness_score": row["freshness_score"],
                "evaluator_consistency_score": row["evaluator_consistency_score"],
            },
            scoring_policy,
        ),
        axis=1,
    )
    profiles["routing_readiness_score"] = profiles.apply(
        lambda row: calculate_family_score(
            "routing_readiness",
            {
                "capability_score": row["capability_score"],
                "stability_score": row["profile_stability_score"],
                "performance_score": row["profile_performance_score"],
                "cost_performance_score": row["cost_performance_score"],
            },
            scoring_policy,
        ),
        axis=1,
    )
    profiles["dominant_capability"] = profiles.apply(
        lambda row: max(DIMENSIONS, key=lambda dimension: row[f"{dimension}_score"]),
        axis=1,
    )
    profiles["weakest_capability"] = profiles.apply(
        lambda row: min(DIMENSIONS, key=lambda dimension: row[f"{dimension}_score"]),
        axis=1,
    )
    profiles["recommended_role"] = profiles.apply(
        lambda row: _recommended_role(row, diagnosis_policy), axis=1
    )
    profiles["routing_action"] = profiles.apply(_routing_action, axis=1)
    profiles["profile_rank"] = profiles["routing_readiness_score"].rank(
        method="min", ascending=False
    ).astype(int)

    columns = [
        "profile_rank", "date", "model_id", "capability_score",
        "profile_stability_score", "profile_performance_score", "confidence_score",
        "routing_readiness_score", "health_score", "cost_performance_score",
        *[f"{dimension}_score" for dimension in DIMENSIONS],
        "dominant_capability", "weakest_capability", "task_coverage_score",
        "sample_sufficiency_score", "freshness_score",
        "evaluator_consistency_score", "capability_run_count",
        "capability_task_count", "latest_capability_run_at", "diagnosis_type",
        "diagnosis_severity", "switch_recommendation", "recommended_role",
        "routing_action", "recommended_action",
    ]
    return _round_numeric(
        profiles[columns].sort_values(["profile_rank", "model_id"]).reset_index(drop=True)
    )


def _recommended_role(row: pd.Series, policy: DiagnosisPolicy) -> str:
    score = row["routing_readiness_score"]
    if (
        score >= policy.primary_route_min_score
        and row["profile_stability_score"]
        >= policy.primary_route_min_stability_score
        and row["confidence_score"] >= policy.primary_route_min_confidence_score
    ):
        return "主路由候选"
    if score >= policy.backup_route_min_score:
        return "辅助路由候选"
    return "兜底或专项模型"


def _routing_action(row: pd.Series) -> str:
    if row["switch_recommendation"] == "建议切换":
        return "降低当前模型权重，并切换到更高路由就绪度模型"
    if row["switch_recommendation"] == "谨慎降权":
        return "限制弱能力场景流量，复测确认后再调整全局权重"
    if row["recommended_role"] == "主路由候选":
        return "可作为主路由，按场景能力维度分配权重"
    if row["recommended_role"] == "辅助路由候选":
        return "作为辅助路由或特定能力场景候选"
    return "仅用于兜底或经过验证的专项场景"


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{label}缺少字段：{', '.join(sorted(missing))}")


def _round_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    numeric = result.select_dtypes(include="number").columns
    result[numeric] = result[numeric].round(4)
    return result


def _finite_number(value: object, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是有限数值") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} 必须是有限数值")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成真实调用与能力拨测融合诊断和模型画像")
    parser.add_argument("--operating", type=Path, default=DEFAULT_OPERATING_INPUT)
    parser.add_argument("--probe-runs", type=Path, default=DEFAULT_PROBE_RUN_INPUT)
    parser.add_argument(
        "--availability-runs", type=Path, default=DEFAULT_AVAILABILITY_RUN_INPUT
    )
    parser.add_argument("--capability", type=Path, default=DEFAULT_CAPABILITY_INPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--diagnosis-output", type=Path, default=DEFAULT_DIAGNOSIS_OUTPUT)
    parser.add_argument("--profile-output", type=Path, default=DEFAULT_PROFILE_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scoring_policy = load_scoring_policy(args.config)
    diagnosis_policy = load_diagnosis_policy(args.config)
    operating = pd.read_csv(args.operating, parse_dates=["date"])
    probe_runs = pd.read_csv(args.probe_runs, parse_dates=["started_at", "completed_at"])
    availability_runs = pd.read_csv(
        args.availability_runs, parse_dates=["started_at", "completed_at"]
    )
    capability = pd.read_csv(args.capability, parse_dates=["latest_run_at"])
    daily_probe = build_daily_probe_metrics(
        probe_runs, scoring_policy, availability_runs
    )
    diagnosis = build_fusion_diagnosis(operating, daily_probe, diagnosis_policy)
    profiles = build_model_profiles(
        operating,
        capability,
        daily_probe,
        diagnosis,
        scoring_policy,
        diagnosis_policy,
    )
    args.diagnosis_output.parent.mkdir(parents=True, exist_ok=True)
    args.profile_output.parent.mkdir(parents=True, exist_ok=True)
    diagnosis.to_csv(args.diagnosis_output, index=False, encoding="utf-8-sig")
    profiles.to_csv(args.profile_output, index=False, encoding="utf-8-sig")
    print(f"融合诊断：{args.diagnosis_output}（{len(diagnosis):,} 行）")
    print(f"模型能力画像：{args.profile_output}（{len(profiles):,} 行）")
    print(
        profiles[
            [
                "profile_rank", "model_id", "capability_score",
                "profile_stability_score", "profile_performance_score",
                "confidence_score", "routing_readiness_score", "recommended_role",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
