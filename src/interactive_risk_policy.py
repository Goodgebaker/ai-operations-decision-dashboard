"""智能检测看板的会话策略、风险分级和动态信号事件。

本模块不依赖 Streamlit。用户在看板中调整的阈值只作用于当前浏览器会话，
底层离线指标字典仍作为可恢复、可审计的默认策略。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from .model_health_risk import RiskPolicy
from .model_scoring import ScoreBand, ScoringPolicy


@dataclass(frozen=True)
class EditableSignalRule:
    signal: str
    dimension: str
    warning_key: str
    severe_key: str
    unit: str
    direction: str


EDITABLE_SIGNAL_RULES = (
    EditableSignalRule("性能分相对基线下降", "性能", "performance_score_drop_warning_points", "performance_score_drop_severe_points", "分", "达到或超过"),
    EditableSignalRule("P95 延迟相对基线上升", "性能", "p95_latency_increase_warning_ratio", "p95_latency_increase_severe_ratio", "倍", "达到或超过"),
    EditableSignalRule("性能绝对分过低", "性能", "absolute_performance_warning_score", "absolute_performance_severe_score", "分", "达到或低于"),
    EditableSignalRule("成功率相对基线下降", "成功率", "success_drop_warning_points", "success_drop_severe_points", "百分点", "达到或超过"),
    EditableSignalRule("成功率绝对值过低", "成功率", "absolute_success_warning_pct", "absolute_success_severe_pct", "%", "达到或低于"),
    EditableSignalRule("单位成本相对基线上升", "成本", "cost_increase_warning_ratio", "cost_increase_severe_ratio", "倍", "达到或超过"),
    EditableSignalRule("成本效率分相对基线下降", "成本", "cost_efficiency_drop_warning_points", "cost_efficiency_drop_severe_points", "分", "达到或超过"),
)


EVENT_COLUMNS = [
    "event_time",
    "scope",
    "risk_dimension",
    "event_type",
    "severity",
    "detection_method",
    "risk_score",
    "observed_value",
    "threshold",
    "evidence",
]


def risk_policy_mapping(frame: pd.DataFrame) -> dict[str, float]:
    """从 Risk Policy 工作表数据中读取生效阈值。"""
    required = {"metric", "threshold", "status"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"风险策略表缺少字段：{', '.join(sorted(missing))}")
    active = frame[frame["status"].astype(str).str.strip().str.lower().eq("active")]
    if active["metric"].duplicated().any():
        raise ValueError("风险策略表存在重复生效指标")
    values = {
        str(metric): float(threshold)
        for metric, threshold in zip(active["metric"], active["threshold"], strict=False)
    }
    RiskPolicy.from_mapping(values)
    return values


def signal_rule_table(values: Mapping[str, object]) -> pd.DataFrame:
    """生成可直接交给 ``st.data_editor`` 的七类信号阈值表。"""
    rows = []
    for rule in EDITABLE_SIGNAL_RULES:
        rows.append(
            {
                "检测信号": rule.signal,
                "风险维度": rule.dimension,
                "触发方向": rule.direction,
                "预警阈值": float(values[rule.warning_key]),
                "严重阈值": float(values[rule.severe_key]),
                "单位": rule.unit,
            }
        )
    return pd.DataFrame(rows)


def merge_signal_rule_table(
    base_values: Mapping[str, object],
    edited: pd.DataFrame,
) -> dict[str, float]:
    """把用户编辑后的信号表合并回完整 RiskPolicy 映射并校验。"""
    required = {"检测信号", "预警阈值", "严重阈值"}
    missing = required.difference(edited.columns)
    if missing:
        raise ValueError(f"检测信号表缺少字段：{', '.join(sorted(missing))}")
    indexed = edited.set_index("检测信号")
    expected = {rule.signal for rule in EDITABLE_SIGNAL_RULES}
    if set(indexed.index) != expected or indexed.index.duplicated().any():
        raise ValueError("检测信号不允许新增、删除或重命名")
    values = {key: float(value) for key, value in base_values.items()}
    for rule in EDITABLE_SIGNAL_RULES:
        values[rule.warning_key] = float(indexed.at[rule.signal, "预警阈值"])
        values[rule.severe_key] = float(indexed.at[rule.signal, "严重阈值"])
    RiskPolicy.from_mapping(values)
    return values


def scoring_policy_with_risk_bands(
    policy: ScoringPolicy,
    medium_threshold: float,
    high_threshold: float,
    critical_threshold: float,
) -> ScoringPolicy:
    """保留评分权重，仅用会话阈值替换风险等级区间。"""
    medium = float(medium_threshold)
    high = float(high_threshold)
    critical = float(critical_threshold)
    if not 0 < medium < high < critical < 100:
        raise ValueError("风险等级阈值必须满足 0 < 中风险 < 高风险 < 严重风险 < 100")
    other_bands = tuple(band for band in policy.score_bands if band.score_family != "risk")
    risk_bands = (
        ScoreBand("RUNTIME-RISK-LOW", "risk", 0.0, medium, "低"),
        ScoreBand("RUNTIME-RISK-MEDIUM", "risk", medium, high, "中"),
        ScoreBand("RUNTIME-RISK-HIGH", "risk", high, critical, "高"),
        ScoreBand("RUNTIME-RISK-CRITICAL", "risk", critical, 100.0, "严重"),
    )
    result = ScoringPolicy(policy.component_rules, other_bands + risk_bands)
    result.validate()
    return result


def build_signal_events(risks: pd.DataFrame, policy: RiskPolicy) -> pd.DataFrame:
    """把三个风险维度拆成具体信号事件，并保留融合诊断事件。"""
    if risks.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    events: list[dict[str, object]] = []
    for _, row in risks.iterrows():
        common = {
            "event_time": pd.Timestamp(row["date"]),
            "scope": str(row["model_id"]),
            "risk_score": float(row["risk_score"]),
        }
        _append_ascending_event(
            events, common, row, "性能", "性能分下降", "performance_score_drop_points",
            policy.performance_score_drop_warning_points,
            policy.performance_score_drop_severe_points, "分",
            "当前性能分相对历史中位数下降",
        )
        _append_ascending_event(
            events, common, row, "性能", "P95 长尾延迟突增", "p95_latency_increase_ratio",
            policy.p95_latency_increase_warning_ratio,
            policy.p95_latency_increase_severe_ratio, "倍",
            "当前 P95 延迟相对历史中位数上升",
        )
        _append_descending_event(
            events, common, row, "性能", "性能绝对低位", "performance_score",
            policy.absolute_performance_warning_score,
            policy.absolute_performance_severe_score, "分",
            "当前性能绝对分低于允许范围",
        )
        _append_ascending_event(
            events, common, row, "成功率", "成功率下降", "success_drop_points",
            policy.success_drop_warning_points,
            policy.success_drop_severe_points, "百分点",
            "当前成功率相对历史中位数下降",
        )
        _append_descending_event(
            events, common, row, "成功率", "成功率绝对低位", "success_rate",
            policy.absolute_success_warning_pct,
            policy.absolute_success_severe_pct, "%",
            "当前成功率低于允许范围",
        )
        _append_ascending_event(
            events, common, row, "成本", "单请求成本上升", "cost_per_request_increase_ratio",
            policy.cost_increase_warning_ratio,
            policy.cost_increase_severe_ratio, "倍",
            "单请求成本相对历史中位数上升",
        )
        _append_ascending_event(
            events, common, row, "成本", "Token 成本上升", "token_cost_increase_ratio",
            policy.cost_increase_warning_ratio,
            policy.cost_increase_severe_ratio, "倍",
            "千 Token 成本相对历史中位数上升",
        )
        _append_ascending_event(
            events, common, row, "成本", "成本效率下降", "cost_efficiency_drop_points",
            policy.cost_efficiency_drop_warning_points,
            policy.cost_efficiency_drop_severe_points, "分",
            "成本效率分相对历史中位数下降",
        )

        diagnosis_type = str(row.get("diagnosis_type", "healthy"))
        if diagnosis_type != "healthy":
            severity = {
                "high": "严重",
                "medium": "预警",
                "low": "关注",
            }.get(str(row.get("diagnosis_severity", "")), "关注")
            events.append(
                {
                    **common,
                    "risk_dimension": "融合诊断",
                    "event_type": "真实调用与拨测表现不一致",
                    "severity": severity,
                    "detection_method": "控制变量融合",
                    "observed_value": diagnosis_type,
                    "threshold": "诊断类型非 healthy",
                    "evidence": str(row.get("diagnosis_reason", "")),
                }
            )

    return _event_frame(events)


def build_unknown_pattern_events(
    scores: pd.DataFrame,
    minimum_algorithm_votes: int,
) -> pd.DataFrame:
    """把 MAD、STL 和 Isolation Forest 的共识命中转成未知模式事件。"""
    if scores.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    votes_required = int(minimum_algorithm_votes)
    if votes_required not in {1, 2, 3}:
        raise ValueError("统计异常最少一致算法数必须为1、2或3")
    prediction_columns = ["pred_mad", "pred_stl", "pred_isolation_forest"]
    missing = set(prediction_columns + ["hour"]).difference(scores.columns)
    if missing:
        raise ValueError(f"统计异常数据缺少字段：{', '.join(sorted(missing))}")

    data = scores.copy()
    votes = sum(_boolean_series(data[column]) for column in prediction_columns)
    data = data.assign(algorithm_votes=votes)
    triggered = data[data["algorithm_votes"].ge(votes_required)]
    events = []
    for _, row in triggered.iterrows():
        metrics = [
            str(row.get("top_metric_mad", "")).strip(),
            str(row.get("top_metric_stl", "")).strip(),
        ]
        metrics = [value for value in metrics if value and value.lower() != "nan"]
        vote_count = int(row["algorithm_votes"])
        events.append(
            {
                "event_time": pd.Timestamp(row["hour"]),
                "scope": "全局调用链",
                "risk_dimension": "未知模式",
                "event_type": "统计未知异常",
                "severity": "严重" if vote_count == 3 else "预警",
                "detection_method": "MAD + STL + Isolation Forest",
                "risk_score": vote_count / 3 * 100,
                "observed_value": f"{vote_count}/3 个算法命中",
                "threshold": f"至少 {votes_required}/3 个算法命中",
                "evidence": "主要异常指标：" + ("、".join(dict.fromkeys(metrics)) or "多变量联合偏移"),
            }
        )
    return _event_frame(events)


def _append_ascending_event(
    events: list[dict[str, object]],
    common: dict[str, object],
    row: pd.Series,
    dimension: str,
    event_type: str,
    column: str,
    warning: float,
    severe: float,
    unit: str,
    evidence: str,
) -> None:
    value = row.get(column)
    if pd.isna(value) or float(value) < warning:
        return
    events.append(
        {
            **common,
            "risk_dimension": dimension,
            "event_type": event_type,
            "severity": "严重" if float(value) >= severe else "预警",
            "detection_method": "可配置阈值",
            "observed_value": f"{float(value):.2f}{unit}",
            "threshold": f"预警≥{warning:g}{unit}；严重≥{severe:g}{unit}",
            "evidence": evidence,
        }
    )


def _append_descending_event(
    events: list[dict[str, object]],
    common: dict[str, object],
    row: pd.Series,
    dimension: str,
    event_type: str,
    column: str,
    warning: float,
    severe: float,
    unit: str,
    evidence: str,
) -> None:
    value = row.get(column)
    if pd.isna(value) or float(value) > warning:
        return
    events.append(
        {
            **common,
            "risk_dimension": dimension,
            "event_type": event_type,
            "severity": "严重" if float(value) <= severe else "预警",
            "detection_method": "可配置阈值",
            "observed_value": f"{float(value):.2f}{unit}",
            "threshold": f"预警≤{warning:g}{unit}；严重≤{severe:g}{unit}",
            "evidence": evidence,
        }
    )


def _event_frame(events: list[dict[str, object]]) -> pd.DataFrame:
    if not events:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    return (
        pd.DataFrame(events)[EVENT_COLUMNS]
        .sort_values(["event_time", "risk_score"], ascending=[False, False])
        .reset_index(drop=True)
    )


def _boolean_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(int)
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"}).astype(int)
