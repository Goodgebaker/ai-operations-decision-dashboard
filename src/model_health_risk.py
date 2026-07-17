"""识别模型健康风险并生成可直接用于决策的诊断证据。

风险由性能下降、成功率异常和成本异常三类统计信号组成。加权风险之外，
单项严重信号保护和真实调用/主动拨测融合诊断下限用于避免关键事件被均值稀释。
本模块只新增派生输出，不修改已有告警、运营评分和模型画像文件。
"""

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
        classify_score,
        clamp_score,
        load_scoring_policy,
    )
except ImportError:  # 支持 ``python src/model_health_risk.py`` 直接运行。
    from model_scoring import (
        ScoringPolicy,
        calculate_family_score,
        classify_score,
        clamp_score,
        load_scoring_policy,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPERATING_INPUT = PROJECT_ROOT / "outputs" / "model_operating_scores.csv"
DEFAULT_DIAGNOSIS_INPUT = PROJECT_ROOT / "outputs" / "model_fusion_diagnosis.csv"
DEFAULT_PROFILE_INPUT = PROJECT_ROOT / "outputs" / "model_capability_profiles.csv"
DEFAULT_CONFIG = PROJECT_ROOT / "docs" / "ai_monitoring_metric_dictionary.xlsx"
DEFAULT_RISK_OUTPUT = PROJECT_ROOT / "outputs" / "model_health_risks.csv"
DEFAULT_EVIDENCE_OUTPUT = PROJECT_ROOT / "outputs" / "model_diagnostic_evidence.csv"


RISK_POLICY_KEYS = {
    "baseline_window_days",
    "minimum_baseline_days",
    "performance_score_drop_warning_points",
    "performance_score_drop_severe_points",
    "absolute_performance_warning_score",
    "absolute_performance_severe_score",
    "p95_latency_increase_warning_ratio",
    "p95_latency_increase_severe_ratio",
    "success_drop_warning_points",
    "success_drop_severe_points",
    "absolute_success_warning_pct",
    "absolute_success_severe_pct",
    "cost_increase_warning_ratio",
    "cost_increase_severe_ratio",
    "cost_efficiency_drop_warning_points",
    "cost_efficiency_drop_severe_points",
    "single_component_floor_multiplier",
    "model_side_risk_floor",
    "capability_or_probe_risk_floor",
    "platform_or_traffic_risk_floor",
    "environment_latency_risk_floor",
    "evidence_risk_threshold",
    "route_downweight_risk_threshold",
    "route_switch_risk_threshold",
    "minimum_candidate_health_score",
}

DIAGNOSIS_RISK_FLOOR_KEYS = {
    "model_side_degradation": "model_side_risk_floor",
    "capability_or_probe_issue": "capability_or_probe_risk_floor",
    "platform_or_traffic_issue": "platform_or_traffic_risk_floor",
    "environment_latency_gap": "environment_latency_risk_floor",
}

RISK_DRIVER_NAMES = {
    "performance_risk": "性能下降",
    "success_risk": "成功率异常",
    "cost_risk": "成本异常",
    "fusion_diagnosis": "融合诊断",
}


@dataclass(frozen=True)
class RiskPolicy:
    baseline_window_days: int
    minimum_baseline_days: int
    performance_score_drop_warning_points: float
    performance_score_drop_severe_points: float
    absolute_performance_warning_score: float
    absolute_performance_severe_score: float
    p95_latency_increase_warning_ratio: float
    p95_latency_increase_severe_ratio: float
    success_drop_warning_points: float
    success_drop_severe_points: float
    absolute_success_warning_pct: float
    absolute_success_severe_pct: float
    cost_increase_warning_ratio: float
    cost_increase_severe_ratio: float
    cost_efficiency_drop_warning_points: float
    cost_efficiency_drop_severe_points: float
    single_component_floor_multiplier: float
    model_side_risk_floor: float
    capability_or_probe_risk_floor: float
    platform_or_traffic_risk_floor: float
    environment_latency_risk_floor: float
    evidence_risk_threshold: float
    route_downweight_risk_threshold: float
    route_switch_risk_threshold: float
    minimum_candidate_health_score: float

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "RiskPolicy":
        missing = RISK_POLICY_KEYS.difference(values)
        if missing:
            raise ValueError(f"风险策略缺少阈值：{', '.join(sorted(missing))}")
        numbers = {key: _finite_number(values[key], key) for key in RISK_POLICY_KEYS}
        baseline_window = _whole_number(numbers["baseline_window_days"], "baseline_window_days")
        minimum_baseline = _whole_number(numbers["minimum_baseline_days"], "minimum_baseline_days")
        if baseline_window < 1 or not 1 <= minimum_baseline <= baseline_window:
            raise ValueError("风险基线最小历史天数必须处于1到基线窗口之间")

        _require_ascending(
            numbers,
            "performance_score_drop_warning_points",
            "performance_score_drop_severe_points",
        )
        _require_descending(
            numbers,
            "absolute_performance_warning_score",
            "absolute_performance_severe_score",
        )
        _require_ascending(
            numbers,
            "p95_latency_increase_warning_ratio",
            "p95_latency_increase_severe_ratio",
        )
        _require_ascending(
            numbers, "success_drop_warning_points", "success_drop_severe_points"
        )
        _require_descending(
            numbers, "absolute_success_warning_pct", "absolute_success_severe_pct"
        )
        _require_ascending(
            numbers, "cost_increase_warning_ratio", "cost_increase_severe_ratio"
        )
        _require_ascending(
            numbers,
            "cost_efficiency_drop_warning_points",
            "cost_efficiency_drop_severe_points",
        )
        for key in (
            "absolute_performance_warning_score",
            "absolute_performance_severe_score",
            "absolute_success_warning_pct",
            "absolute_success_severe_pct",
            "model_side_risk_floor",
            "capability_or_probe_risk_floor",
            "platform_or_traffic_risk_floor",
            "environment_latency_risk_floor",
            "evidence_risk_threshold",
            "route_downweight_risk_threshold",
            "route_switch_risk_threshold",
            "minimum_candidate_health_score",
        ):
            if not 0 <= numbers[key] <= 100:
                raise ValueError(f"{key} 必须处于0到100")
        if not 0 <= numbers["single_component_floor_multiplier"] <= 1:
            raise ValueError("single_component_floor_multiplier 必须处于0到1")
        if not (
            numbers["route_downweight_risk_threshold"]
            < numbers["route_switch_risk_threshold"]
        ):
            raise ValueError("切换风险阈值必须高于降权风险阈值")
        numbers["baseline_window_days"] = baseline_window
        numbers["minimum_baseline_days"] = minimum_baseline
        return cls(**numbers)

    def diagnosis_floor(self, diagnosis_type: str) -> float:
        key = DIAGNOSIS_RISK_FLOOR_KEYS.get(str(diagnosis_type))
        return float(getattr(self, key)) if key else 0.0


def load_risk_policy(path: Path = DEFAULT_CONFIG) -> RiskPolicy:
    frame = pd.read_excel(path, sheet_name="Risk Policy")
    active = frame[frame["status"].astype(str).str.strip().str.lower().eq("active")]
    if active["metric"].duplicated().any():
        raise ValueError("Risk Policy 存在重复 metric")
    values = dict(zip(active["metric"], active["threshold"], strict=False))
    return RiskPolicy.from_mapping(values)


def build_health_risks(
    operating_scores: pd.DataFrame,
    fusion_diagnosis: pd.DataFrame,
    scoring_policy: ScoringPolicy,
    risk_policy: RiskPolicy,
) -> pd.DataFrame:
    """按模型日生成三类风险、统计风险和融合后的最终风险。"""

    operating_required = {
        "date", "model_id", "performance_score", "p95_latency_ms",
        "success_rate", "cost_per_request", "cost_per_1k_tokens",
        "cost_efficiency_score", "health_score",
    }
    diagnosis_required = {
        "date", "model_id", "provider", "diagnosis_type",
        "diagnosis_severity", "diagnosis_reason", "probe_http_success_rate",
        "probe_p95_latency_ms", "probe_performance_score",
    }
    _require_columns(operating_scores, operating_required, "模型运营评分")
    _require_columns(fusion_diagnosis, diagnosis_required, "融合诊断")

    data = operating_scores.copy()
    data["date"] = pd.to_datetime(data["date"]).dt.floor("D")
    data = data.sort_values(["model_id", "date"]).reset_index(drop=True)
    baseline_columns = (
        "performance_score", "p95_latency_ms", "success_rate",
        "cost_per_request", "cost_per_1k_tokens", "cost_efficiency_score",
    )
    for column in baseline_columns:
        data[f"{column}_baseline"] = data.groupby("model_id")[column].transform(
            lambda values: values.shift(1).rolling(
                risk_policy.baseline_window_days,
                min_periods=risk_policy.minimum_baseline_days,
            ).median()
        )
    data["risk_baseline_days"] = data.groupby("model_id")["performance_score"].transform(
        lambda values: values.shift(1).rolling(
            risk_policy.baseline_window_days, min_periods=1
        ).count()
    )
    data["risk_baseline_ready"] = data["risk_baseline_days"].ge(
        risk_policy.minimum_baseline_days
    )

    data["performance_score_drop_points"] = (
        data["performance_score_baseline"] - data["performance_score"]
    ).where(data["risk_baseline_ready"], 0.0).clip(lower=0.0)
    data["p95_latency_increase_ratio"] = _safe_ratio(
        data["p95_latency_ms"], data["p95_latency_ms_baseline"]
    ).where(data["risk_baseline_ready"], 1.0).fillna(1.0)
    data["success_drop_points"] = (
        data["success_rate_baseline"] - data["success_rate"]
    ).where(data["risk_baseline_ready"], 0.0).clip(lower=0.0)
    data["cost_per_request_increase_ratio"] = _safe_ratio(
        data["cost_per_request"], data["cost_per_request_baseline"]
    ).where(data["risk_baseline_ready"], 1.0).fillna(1.0)
    data["token_cost_increase_ratio"] = _safe_ratio(
        data["cost_per_1k_tokens"], data["cost_per_1k_tokens_baseline"]
    ).where(data["risk_baseline_ready"], 1.0).fillna(1.0)
    data["cost_efficiency_drop_points"] = (
        data["cost_efficiency_score_baseline"] - data["cost_efficiency_score"]
    ).where(data["risk_baseline_ready"], 0.0).clip(lower=0.0)

    data["performance_risk"] = data.apply(
        lambda row: max(
            _ascending_risk(
                row["performance_score_drop_points"],
                risk_policy.performance_score_drop_warning_points,
                risk_policy.performance_score_drop_severe_points,
            ),
            _ascending_risk(
                row["p95_latency_increase_ratio"],
                risk_policy.p95_latency_increase_warning_ratio,
                risk_policy.p95_latency_increase_severe_ratio,
            ),
            _descending_risk(
                row["performance_score"],
                risk_policy.absolute_performance_warning_score,
                risk_policy.absolute_performance_severe_score,
            ),
        ),
        axis=1,
    )
    data["success_risk"] = data.apply(
        lambda row: max(
            _ascending_risk(
                row["success_drop_points"],
                risk_policy.success_drop_warning_points,
                risk_policy.success_drop_severe_points,
            ),
            _descending_risk(
                row["success_rate"],
                risk_policy.absolute_success_warning_pct,
                risk_policy.absolute_success_severe_pct,
            ),
        ),
        axis=1,
    )
    data["cost_risk"] = data.apply(
        lambda row: max(
            _ascending_risk(
                row["cost_per_request_increase_ratio"],
                risk_policy.cost_increase_warning_ratio,
                risk_policy.cost_increase_severe_ratio,
            ),
            _ascending_risk(
                row["token_cost_increase_ratio"],
                risk_policy.cost_increase_warning_ratio,
                risk_policy.cost_increase_severe_ratio,
            ),
            _ascending_risk(
                row["cost_efficiency_drop_points"],
                risk_policy.cost_efficiency_drop_warning_points,
                risk_policy.cost_efficiency_drop_severe_points,
            ),
        ),
        axis=1,
    )
    data["statistical_risk_score"] = data.apply(
        lambda row: calculate_family_score(
            "risk",
            {
                "performance_risk": row["performance_risk"],
                "success_risk": row["success_risk"],
                "cost_risk": row["cost_risk"],
            },
            scoring_policy,
        ),
        axis=1,
    )

    diagnosis = fusion_diagnosis.copy()
    diagnosis["date"] = pd.to_datetime(diagnosis["date"]).dt.floor("D")
    diagnosis_columns = [
        "date", "model_id", "provider", "diagnosis_type",
        "diagnosis_severity", "diagnosis_reason", "probe_http_success_rate",
        "probe_p95_latency_ms", "probe_performance_score",
    ]
    data = data.merge(
        diagnosis[diagnosis_columns],
        on=["date", "model_id"],
        how="left",
        validate="one_to_one",
    )
    if data["diagnosis_type"].isna().any():
        raise ValueError("运营评分与融合诊断未能按模型和日期完整对齐")

    data["single_signal_floor_score"] = data[
        ["performance_risk", "success_risk", "cost_risk"]
    ].max(axis=1) * risk_policy.single_component_floor_multiplier
    data["diagnosis_floor_score"] = data["diagnosis_type"].map(
        risk_policy.diagnosis_floor
    )
    data["risk_score"] = data[
        ["statistical_risk_score", "single_signal_floor_score", "diagnosis_floor_score"]
    ].max(axis=1).map(clamp_score)
    data["risk_level"] = data["risk_score"].map(
        lambda value: classify_score("risk", value, scoring_policy)
    )
    data["primary_risk_driver"] = data.apply(_primary_risk_driver, axis=1)
    data["primary_risk_driver_cn"] = data["primary_risk_driver"].map(
        RISK_DRIVER_NAMES
    )
    data["risk_evidence"] = data.apply(_risk_evidence, axis=1)

    columns = [
        "date", "model_id", "provider", "health_score", "risk_score", "risk_level",
        "statistical_risk_score", "single_signal_floor_score",
        "diagnosis_floor_score", "performance_risk", "success_risk", "cost_risk",
        "primary_risk_driver", "primary_risk_driver_cn", "risk_baseline_ready",
        "risk_baseline_days", "performance_score", "performance_score_baseline",
        "performance_score_drop_points", "p95_latency_ms",
        "p95_latency_ms_baseline", "p95_latency_increase_ratio", "success_rate",
        "success_rate_baseline", "success_drop_points", "cost_per_request",
        "cost_per_request_baseline", "cost_per_request_increase_ratio",
        "cost_per_1k_tokens", "cost_per_1k_tokens_baseline",
        "token_cost_increase_ratio", "cost_efficiency_score",
        "cost_efficiency_score_baseline", "cost_efficiency_drop_points",
        "probe_http_success_rate", "probe_p95_latency_ms",
        "probe_performance_score", "diagnosis_type", "diagnosis_severity",
        "diagnosis_reason", "risk_evidence",
    ]
    return _round_numeric(data[columns].sort_values(["date", "model_id"]).reset_index(drop=True))


def build_diagnostic_evidence(
    risks: pd.DataFrame,
    model_profiles: pd.DataFrame,
    risk_policy: RiskPolicy,
) -> pd.DataFrame:
    """将风险转成异常解释、原因、切换建议、替代模型和推荐动作。"""

    risk_required = {
        "date", "model_id", "provider", "risk_score", "risk_level",
        "performance_risk", "success_risk", "cost_risk", "primary_risk_driver",
        "diagnosis_type", "diagnosis_reason", "risk_evidence", "health_score",
        "risk_baseline_ready", "probe_http_success_rate",
    }
    profile_required = {
        "model_id", "routing_readiness_score", "confidence_score",
        "recommended_role", "dominant_capability",
    }
    _require_columns(risks, risk_required, "模型健康风险")
    _require_columns(model_profiles, profile_required, "模型能力画像")

    actionable = risks[
        risks["risk_score"].ge(risk_policy.evidence_risk_threshold)
        | risks["diagnosis_type"].ne("healthy")
    ].copy()
    if actionable.empty:
        return pd.DataFrame(columns=_evidence_columns())

    profiles = model_profiles[
        [
            "model_id", "routing_readiness_score", "confidence_score",
            "recommended_role", "dominant_capability",
        ]
    ].drop_duplicates("model_id")
    actionable = actionable.merge(
        profiles,
        on="model_id",
        how="left",
        validate="many_to_one",
    )
    if actionable["routing_readiness_score"].isna().any():
        raise ValueError("风险模型缺少对应的模型能力画像")

    rows: list[dict[str, object]] = []
    for _, row in actionable.sort_values(["date", "risk_score"], ascending=[True, False]).iterrows():
        switch_recommendation = _switch_recommendation(row, risk_policy)
        target = ""
        target_reason = ""
        if switch_recommendation == "建议切换":
            target, target_reason = _select_route_candidate(
                row, risks, profiles, risk_policy
            )
        evidence_confidence = _evidence_confidence(row)
        rows.append(
            {
                "date": row["date"],
                "model_id": row["model_id"],
                "provider": row["provider"],
                "risk_score": row["risk_score"],
                "risk_level": row["risk_level"],
                "primary_risk_driver": row["primary_risk_driver"],
                "primary_risk_driver_cn": RISK_DRIVER_NAMES[row["primary_risk_driver"]],
                "what_happened": _what_happened(row),
                "possible_cause": _possible_cause(row),
                "switch_recommendation": switch_recommendation,
                "target_model_id": target,
                "target_reason": target_reason,
                "recommended_action": _recommended_action(
                    row, switch_recommendation, target
                ),
                "evidence_confidence_score": evidence_confidence,
                "routing_readiness_score": row["routing_readiness_score"],
                "model_profile_confidence_score": row["confidence_score"],
                "diagnosis_type": row["diagnosis_type"],
                "diagnosis_reason": row["diagnosis_reason"],
                "risk_evidence": row["risk_evidence"],
                "decision_state": "待处置" if switch_recommendation != "无需切换" else "观察",
            }
        )
    result = pd.DataFrame(rows)
    result.insert(
        0,
        "evidence_id",
        [f"EVID-{number:05d}" for number in range(1, len(result) + 1)],
    )
    return _round_numeric(result[_evidence_columns()])


def _select_route_candidate(
    current: pd.Series,
    risks: pd.DataFrame,
    profiles: pd.DataFrame,
    policy: RiskPolicy,
) -> tuple[str, str]:
    same_date = risks[pd.to_datetime(risks["date"]).eq(pd.Timestamp(current["date"]))]
    candidates = same_date[
        same_date["model_id"].ne(current["model_id"])
        & same_date["provider"].ne(current["provider"])
        & same_date["diagnosis_type"].eq("healthy")
        & same_date["health_score"].ge(policy.minimum_candidate_health_score)
        & same_date["risk_score"].lt(policy.route_downweight_risk_threshold)
    ][["model_id", "health_score", "risk_score"]]
    candidates = candidates.merge(profiles, on="model_id", how="inner")
    candidates = candidates[candidates["recommended_role"].ne("兜底或专项模型")]
    if candidates.empty:
        return "", "无满足跨供应商、健康和路由角色约束的候选模型"
    winner = candidates.sort_values(
        ["routing_readiness_score", "health_score", "risk_score", "model_id"],
        ascending=[False, False, True, True],
    ).iloc[0]
    return (
        str(winner["model_id"]),
        f"跨供应商且当日健康；路由就绪度{winner['routing_readiness_score']:.1f}，"
        f"健康指数{winner['health_score']:.1f}",
    )


def _switch_recommendation(row: pd.Series, policy: RiskPolicy) -> str:
    diagnosis_type = str(row["diagnosis_type"])
    if diagnosis_type == "model_side_degradation":
        return "建议切换"
    if diagnosis_type in {"platform_or_traffic_issue", "environment_latency_gap"}:
        return "暂不切换"
    if diagnosis_type == "capability_or_probe_issue":
        return "谨慎降权"
    if (
        row["risk_score"] >= policy.route_switch_risk_threshold
        and row["primary_risk_driver"] != "cost_risk"
    ):
        return "建议切换"
    if row["risk_score"] >= policy.route_downweight_risk_threshold:
        return "谨慎降权"
    return "无需切换"


def _what_happened(row: pd.Series) -> str:
    diagnosis_type = str(row["diagnosis_type"])
    if diagnosis_type == "model_side_degradation":
        return "真实调用与标准拨测同步下降，模型侧健康风险达到切换条件"
    if diagnosis_type == "platform_or_traffic_issue":
        return "真实调用表现下降，但固定环境主动拨测仍正常"
    if diagnosis_type == "capability_or_probe_issue":
        return "真实调用正常，但标准能力任务或拨测表现下降"
    if diagnosis_type == "environment_latency_gap":
        return "真实调用延迟显著高于标准环境，存在业务环境时延差"
    driver = str(row["primary_risk_driver"])
    if driver == "cost_risk":
        return "单请求成本或成本效率相对历史基线异常"
    if driver == "performance_risk":
        return "性能评分下降或P95延迟相对历史基线上升"
    if driver == "success_risk":
        return "成功率低于绝对阈值或历史基线"
    return "模型风险进入观察区间"


def _possible_cause(row: pd.Series) -> str:
    reason = str(row["diagnosis_reason"])
    driver = str(row["primary_risk_driver"])
    if driver == "cost_risk":
        return f"{reason}；同时检查价格版本、请求Token结构和异常长输出"
    if driver == "performance_risk":
        return f"{reason}；同时检查模型容量、队列、供应商限流和长请求占比"
    if driver == "success_risk":
        return f"{reason}；同时检查5xx、限流、超时和上游依赖"
    return reason


def _recommended_action(row: pd.Series, switch: str, target: str) -> str:
    driver = str(row["primary_risk_driver"])
    if switch == "建议切换":
        if target:
            return f"降低{row['model_id']}路由权重，灰度切换至{target}并持续复测"
        return "暂停自动扩量，启用兜底策略并人工确认可用替代模型"
    if str(row["diagnosis_type"]) == "platform_or_traffic_issue":
        return "保持模型路由，优先排查网关、区域网络、队列和异常业务流量"
    if str(row["diagnosis_type"]) == "environment_latency_gap":
        return "保持模型路由，检查请求长度、并发、区域网络和平台排队"
    if driver == "cost_risk":
        return "保持可用性路由，核对价格与Token结构并启用预算保护"
    if switch == "谨慎降权":
        return "降低当前模型权重，复跑标准任务并观察至少两个评估周期"
    return "维持当前路由并持续观察"


def _primary_risk_driver(row: pd.Series) -> str:
    component_values = {
        "performance_risk": float(row["performance_risk"]),
        "success_risk": float(row["success_risk"]),
        "cost_risk": float(row["cost_risk"]),
    }
    component = max(component_values, key=component_values.get)
    if (
        row["diagnosis_floor_score"] >= row["statistical_risk_score"]
        and row["diagnosis_floor_score"] >= row["single_signal_floor_score"]
        and row["diagnosis_floor_score"] > 0
    ):
        return "fusion_diagnosis"
    return component


def _risk_evidence(row: pd.Series) -> str:
    parts = [
        f"性能分{row['performance_score']:.1f}/基线{_fmt(row['performance_score_baseline'])}",
        f"P95 {row['p95_latency_ms']:.0f}ms/基线{_fmt(row['p95_latency_ms_baseline'], 0)}ms",
        f"成功率{row['success_rate']:.2f}%/基线{_fmt(row['success_rate_baseline'])}%",
        f"单请求成本{row['cost_per_request']:.6f}/基线{_fmt(row['cost_per_request_baseline'], 6)}",
        f"主动拨测成功率{row['probe_http_success_rate']:.2f}%",
        f"诊断={row['diagnosis_type']}",
    ]
    if not bool(row["risk_baseline_ready"]):
        parts.append("历史基线未就绪，趋势信号不参与")
    return "；".join(parts)


def _evidence_confidence(row: pd.Series) -> float:
    score = 40.0
    if bool(row["risk_baseline_ready"]):
        score += 30.0
    if pd.notna(row.get("probe_http_success_rate")):
        score += 20.0
    if str(row.get("diagnosis_type", "")):
        score += 10.0
    return clamp_score(score)


def _ascending_risk(value: object, warning: float, severe: float) -> float:
    number = _finite_or_default(value, warning)
    if number <= warning:
        return 0.0
    if number >= severe:
        return 100.0
    return clamp_score(30.0 + 70.0 * (number - warning) / (severe - warning))


def _descending_risk(value: object, warning: float, severe: float) -> float:
    number = _finite_or_default(value, warning)
    if number >= warning:
        return 0.0
    if number <= severe:
        return 100.0
    return clamp_score(30.0 + 70.0 * (warning - number) / (warning - severe))


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def _require_ascending(values: Mapping[str, float], warning: str, severe: str) -> None:
    if values[warning] < 0 or values[severe] <= values[warning]:
        raise ValueError(f"{severe} 必须高于 {warning}，且阈值不能为负")


def _require_descending(values: Mapping[str, float], warning: str, severe: str) -> None:
    if not 0 <= values[severe] < values[warning] <= 100:
        raise ValueError(f"{severe} 必须低于 {warning}，且均处于0到100")


def _whole_number(value: float, name: str) -> int:
    if not float(value).is_integer():
        raise ValueError(f"{name} 必须是整数")
    return int(value)


def _finite_or_default(value: object, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是有限数值")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是有限数值") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} 必须是有限数值")
    return number


def _fmt(value: object, digits: int = 1) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "无"
    if not math.isfinite(number):
        return "无"
    return f"{number:.{digits}f}"


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{label}缺少字段：{', '.join(sorted(missing))}")


def _round_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    numeric = result.select_dtypes(include="number").columns
    result[numeric] = result[numeric].round(4)
    return result


def _evidence_columns() -> list[str]:
    return [
        "evidence_id", "date", "model_id", "provider", "risk_score",
        "risk_level", "primary_risk_driver", "primary_risk_driver_cn",
        "what_happened", "possible_cause", "switch_recommendation",
        "target_model_id", "target_reason", "recommended_action",
        "evidence_confidence_score", "routing_readiness_score",
        "model_profile_confidence_score", "diagnosis_type", "diagnosis_reason",
        "risk_evidence", "decision_state",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成模型健康风险和智能诊断解释")
    parser.add_argument("--operating", type=Path, default=DEFAULT_OPERATING_INPUT)
    parser.add_argument("--diagnosis", type=Path, default=DEFAULT_DIAGNOSIS_INPUT)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILE_INPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--risk-output", type=Path, default=DEFAULT_RISK_OUTPUT)
    parser.add_argument("--evidence-output", type=Path, default=DEFAULT_EVIDENCE_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scoring_policy = load_scoring_policy(args.config)
    risk_policy = load_risk_policy(args.config)
    operating = pd.read_csv(args.operating, parse_dates=["date"])
    diagnosis = pd.read_csv(args.diagnosis, parse_dates=["date"])
    profiles = pd.read_csv(args.profiles, parse_dates=["date", "latest_capability_run_at"])
    risks = build_health_risks(operating, diagnosis, scoring_policy, risk_policy)
    evidence = build_diagnostic_evidence(risks, profiles, risk_policy)
    args.risk_output.parent.mkdir(parents=True, exist_ok=True)
    args.evidence_output.parent.mkdir(parents=True, exist_ok=True)
    risks.to_csv(args.risk_output, index=False, encoding="utf-8-sig")
    evidence.to_csv(args.evidence_output, index=False, encoding="utf-8-sig")
    print(f"模型健康风险：{args.risk_output}（{len(risks):,} 行）")
    print(f"诊断解释证据：{args.evidence_output}（{len(evidence):,} 行）")
    if not evidence.empty:
        print(
            evidence[
                [
                    "date", "model_id", "risk_score", "risk_level",
                    "what_happened", "switch_recommendation", "target_model_id",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
