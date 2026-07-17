"""从指标字典读取复合规则，在特征表上生成带证据的告警。"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from severity_policy import SeverityBand, breach_ratio, grade_alert, load_severity_policy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "docs" / "ai_monitoring_metric_dictionary.xlsx"
DEFAULT_FEATURE_DIR = PROJECT_ROOT / "outputs" / "features"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "composite_alerts.csv"

SUPPORTED_OPERATORS: dict[str, Callable[[pd.Series, pd.Series], pd.Series]] = {
    "gt": lambda left, right: left > right,
    "gte": lambda left, right: left >= right,
    "lt": lambda left, right: left < right,
    "lte": lambda left, right: left <= right,
    "eq": lambda left, right: left == right,
    "neq": lambda left, right: left != right,
}
SUPPORTED_DATASETS = {
    "hourly_features",
    "customer_hourly_features",
    "model_hourly_features",
    "provider_hourly_features",
    "key_minute_features",
}


def _text(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


@dataclass(frozen=True)
class CompositeRule:
    rule_id: str
    rule_name: str
    rule_name_cn: str
    dataset: str
    time_field: str
    entity_field: str
    logical_operator: str
    severity: str
    cooldown_minutes: int
    version: str
    description: str
    recommended_action: str


@dataclass(frozen=True)
class Condition:
    condition_id: str
    rule_id: str
    metric_name: str
    comparison_operator: str
    threshold_type: str
    threshold_value: float
    threshold_unit: str
    baseline_window: str
    minimum_sample_size: int
    condition_order: int


def load_composite_config(
    path: Path = DEFAULT_CONFIG,
    *,
    as_of: date | None = None,
) -> tuple[list[CompositeRule], dict[str, list[Condition]]]:
    """读取并校验当前生效的复合规则及其条件。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到指标字典配置：{path}")

    try:
        rule_frame = pd.read_excel(path, sheet_name="Composite Rules")
        condition_frame = pd.read_excel(path, sheet_name="Rule Conditions")
    except ValueError as exc:
        raise ValueError(
            "Metric dictionary must contain Composite Rules and Rule Conditions sheets"
        ) from exc

    required_rule_columns = {
        "rule_id", "rule_name", "rule_name_cn", "dataset", "time_field",
        "entity_field", "logical_operator", "severity", "cooldown_minutes",
        "status", "version", "valid_from", "description", "recommended_action",
    }
    required_condition_columns = {
        "condition_id", "rule_id", "metric_name", "comparison_operator",
        "threshold_type", "threshold_value", "threshold_unit", "baseline_window",
        "minimum_sample_size", "condition_order",
    }
    missing_rules = required_rule_columns.difference(rule_frame.columns)
    missing_conditions = required_condition_columns.difference(condition_frame.columns)
    if missing_rules:
        raise ValueError("复合规则工作表缺少列：" + ", ".join(sorted(missing_rules)))
    if missing_conditions:
        raise ValueError("规则条件工作表缺少列：" + ", ".join(sorted(missing_conditions)))

    evaluation_date = pd.Timestamp(as_of or date.today()).normalize()
    rule_frame["valid_from"] = pd.to_datetime(rule_frame["valid_from"], errors="coerce")
    active = rule_frame[
        rule_frame["status"].astype(str).str.strip().str.lower().eq("active")
        & (rule_frame["valid_from"].isna() | (rule_frame["valid_from"] <= evaluation_date))
    ].copy()
    if active.empty:
        raise ValueError("指标字典中没有当前生效的复合规则")
    if active["rule_id"].duplicated().any():
        raise ValueError("复合规则存在重复的活动 rule_id")

    active_ids = set(active["rule_id"].astype(str))
    orphan_conditions = set(condition_frame["rule_id"].astype(str)) - set(
        rule_frame["rule_id"].astype(str)
    )
    if orphan_conditions:
        raise ValueError(f"规则条件引用了不存在的规则：{sorted(orphan_conditions)}")
    selected_conditions = condition_frame[
        condition_frame["rule_id"].astype(str).isin(active_ids)
    ].copy()
    if selected_conditions["condition_id"].duplicated().any():
        raise ValueError("规则条件存在重复的 condition_id")

    rules: list[CompositeRule] = []
    conditions_by_rule: dict[str, list[Condition]] = {}
    for _, row in active.iterrows():
        dataset = _text(row["dataset"])
        logical_operator = _text(row["logical_operator"]).lower()
        if dataset not in SUPPORTED_DATASETS:
            raise ValueError(f"不支持的特征表：{dataset}")
        if logical_operator not in {"all", "any"}:
            raise ValueError("logical_operator 只支持 all 或 any")
        cooldown = pd.to_numeric(row["cooldown_minutes"], errors="coerce")
        if pd.isna(cooldown) or int(cooldown) < 0:
            raise ValueError(f"规则 {row['rule_id']} 的 cooldown_minutes 必须不小于0")

        rule = CompositeRule(
            rule_id=_text(row["rule_id"]),
            rule_name=_text(row["rule_name"]),
            rule_name_cn=_text(row["rule_name_cn"]),
            dataset=dataset,
            time_field=_text(row["time_field"]),
            entity_field=_text(row["entity_field"]),
            logical_operator=logical_operator,
            severity=_text(row["severity"]).lower(),
            cooldown_minutes=int(cooldown),
            version=_text(row["version"]),
            description=_text(row["description"]),
            recommended_action=_text(row["recommended_action"]),
        )
        rules.append(rule)

        rows = selected_conditions[selected_conditions["rule_id"].astype(str) == rule.rule_id]
        if rows.empty:
            raise ValueError(f"规则 {rule.rule_id} 没有配置条件")
        parsed: list[Condition] = []
        for _, condition_row in rows.sort_values("condition_order").iterrows():
            operator = _text(condition_row["comparison_operator"]).lower()
            threshold_type = _text(condition_row["threshold_type"]).lower()
            threshold_value = pd.to_numeric(condition_row["threshold_value"], errors="coerce")
            minimum_samples = pd.to_numeric(
                condition_row["minimum_sample_size"], errors="coerce"
            )
            if operator not in SUPPORTED_OPERATORS:
                raise ValueError(f"条件 {condition_row['condition_id']} 使用了不支持的运算符")
            if threshold_type not in {"static", "baseline_multiplier"}:
                raise ValueError(f"条件 {condition_row['condition_id']} 的阈值类型不受支持")
            if pd.isna(threshold_value):
                raise ValueError(f"条件 {condition_row['condition_id']} 缺少阈值")
            if pd.isna(minimum_samples) or int(minimum_samples) < 1:
                raise ValueError(f"条件 {condition_row['condition_id']} 的最小样本数无效")
            parsed.append(
                Condition(
                    condition_id=_text(condition_row["condition_id"]),
                    rule_id=rule.rule_id,
                    metric_name=_text(condition_row["metric_name"]),
                    comparison_operator=operator,
                    threshold_type=threshold_type,
                    threshold_value=float(threshold_value),
                    threshold_unit=_text(condition_row["threshold_unit"]),
                    baseline_window=_text(condition_row["baseline_window"]),
                    minimum_sample_size=int(minimum_samples),
                    condition_order=int(condition_row["condition_order"]),
                )
            )
        conditions_by_rule[rule.rule_id] = parsed

    return rules, conditions_by_rule


def _rolling_baseline(
    frame: pd.DataFrame,
    rule: CompositeRule,
    condition: Condition,
) -> pd.Series:
    match = re.fullmatch(r"previous_(\d+)_hours_(mean|median)", condition.baseline_window)
    if not match:
        raise ValueError(
            f"条件 {condition.condition_id} 的 baseline_window 格式不受支持："
            f"{condition.baseline_window}"
        )
    window, statistic = int(match.group(1)), match.group(2)
    if condition.minimum_sample_size > window:
        raise ValueError(f"条件 {condition.condition_id} 的最小样本数大于基线窗口")

    grouped = frame.groupby(rule.entity_field, sort=False)[condition.metric_name]
    shifted = grouped.shift(1)
    rolling = shifted.groupby(frame[rule.entity_field], sort=False).rolling(
        window=window,
        min_periods=condition.minimum_sample_size,
    )
    baseline = rolling.median() if statistic == "median" else rolling.mean()
    return baseline.reset_index(level=0, drop=True).sort_index()


def _evaluate_condition(
    frame: pd.DataFrame,
    rule: CompositeRule,
    condition: Condition,
) -> tuple[pd.Series, pd.Series]:
    if condition.metric_name not in frame.columns:
        raise ValueError(
            f"{rule.dataset}.csv 缺少规则 {rule.rule_id} 所需指标：{condition.metric_name}"
        )
    observed = pd.to_numeric(frame[condition.metric_name], errors="coerce")
    if condition.threshold_type == "static":
        threshold = pd.Series(condition.threshold_value, index=frame.index, dtype=float)
    else:
        threshold = _rolling_baseline(frame, rule, condition) * condition.threshold_value
    matched = SUPPORTED_OPERATORS[condition.comparison_operator](observed, threshold)
    return matched.fillna(False), threshold


def evaluate_rule(
    feature_frame: pd.DataFrame,
    rule: CompositeRule,
    conditions: list[Condition],
    severity_policy: list[SeverityBand],
) -> pd.DataFrame:
    """计算单条复合规则并返回已应用冷却窗口的告警。"""
    required = {rule.time_field, rule.entity_field}
    missing = required.difference(feature_frame.columns)
    if missing:
        raise ValueError(f"{rule.dataset}.csv 缺少字段：{sorted(missing)}")

    frame = feature_frame.copy()
    frame[rule.time_field] = pd.to_datetime(frame[rule.time_field], errors="coerce")
    frame = frame.dropna(subset=[rule.time_field]).sort_values(
        [rule.entity_field, rule.time_field]
    ).reset_index(drop=True)

    matches: list[pd.Series] = []
    thresholds: dict[str, pd.Series] = {}
    for condition in conditions:
        matched, threshold = _evaluate_condition(frame, rule, condition)
        matches.append(matched)
        thresholds[condition.condition_id] = threshold
    match_frame = pd.concat(matches, axis=1)
    rule_match = match_frame.all(axis=1) if rule.logical_operator == "all" else match_frame.any(axis=1)
    candidate_indices = frame.index[rule_match]

    kept_indices: list[int] = []
    last_alert: dict[str, pd.Timestamp] = {}
    cooldown = pd.Timedelta(minutes=rule.cooldown_minutes)
    for index in candidate_indices:
        entity = str(frame.at[index, rule.entity_field])
        detected_at = frame.at[index, rule.time_field]
        previous = last_alert.get(entity)
        if previous is None or detected_at - previous >= cooldown:
            kept_indices.append(index)
            last_alert[entity] = detected_at

    alerts: list[dict[str, object]] = []
    for ordinal, index in enumerate(kept_indices, start=1):
        observed_evidence: dict[str, float] = {}
        threshold_evidence: dict[str, float] = {}
        breach_ratios: list[float] = []
        matched_conditions = 0
        for position, condition in enumerate(conditions):
            observed = float(frame.at[index, condition.metric_name])
            threshold = float(thresholds[condition.condition_id].at[index])
            observed_evidence[condition.metric_name] = round(observed, 4)
            threshold_evidence[condition.metric_name] = round(threshold, 4)
            if bool(matches[position].at[index]):
                matched_conditions += 1
                breach_ratios.append(
                    breach_ratio(observed, threshold, condition.comparison_operator)
                )
        alert_level, severity_score, maximum_breach_ratio = grade_alert(
            base_severity=rule.severity,
            breach_ratios=breach_ratios,
            matched_conditions=matched_conditions,
            policy=severity_policy,
        )
        detected_at = pd.Timestamp(frame.at[index, rule.time_field])
        alerts.append(
            {
                "alert_id": f"COMP-{rule.rule_id}-{ordinal:04d}",
                "rule_id": rule.rule_id,
                "rule_name": rule.rule_name,
                "rule_name_cn": rule.rule_name_cn,
                "detected_at": detected_at,
                "entity_type": rule.entity_field,
                "entity_value": str(frame.at[index, rule.entity_field]),
                "base_severity": rule.severity,
                "alert_level": alert_level,
                "breach_ratio": maximum_breach_ratio,
                "severity_score": severity_score,
                "matched_conditions": matched_conditions,
                "config_version": rule.version,
                "observed_summary": json.dumps(observed_evidence, ensure_ascii=False),
                "threshold_summary": json.dumps(threshold_evidence, ensure_ascii=False),
                "description": rule.description,
                "recommended_action": rule.recommended_action,
                "alert_state": "open",
            }
        )
    return pd.DataFrame(alerts)


def run_engine(
    config_path: Path = DEFAULT_CONFIG,
    feature_dir: Path = DEFAULT_FEATURE_DIR,
) -> pd.DataFrame:
    rules, conditions_by_rule = load_composite_config(config_path)
    severity_policy = load_severity_policy(config_path)
    results: list[pd.DataFrame] = []
    loaded_features: dict[str, pd.DataFrame] = {}
    for rule in rules:
        if rule.dataset not in loaded_features:
            path = feature_dir / f"{rule.dataset}.csv"
            if not path.exists():
                raise FileNotFoundError(f"找不到特征表：{path}")
            loaded_features[rule.dataset] = pd.read_csv(path)
        alerts = evaluate_rule(
            loaded_features[rule.dataset],
            rule,
            conditions_by_rule[rule.rule_id],
            severity_policy,
        )
        if not alerts.empty:
            results.append(alerts)

    columns = [
        "alert_id", "rule_id", "rule_name", "rule_name_cn", "detected_at",
        "entity_type", "entity_value", "base_severity", "alert_level",
        "breach_ratio", "severity_score", "matched_conditions", "config_version",
        "observed_summary", "threshold_summary", "description",
        "recommended_action", "alert_state",
    ]
    if not results:
        return pd.DataFrame(columns=columns)
    return pd.concat(results, ignore_index=True)[columns].sort_values("detected_at")


def main() -> None:
    parser = argparse.ArgumentParser(description="执行指标字典中的复合异常规则")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    alerts = run_engine(args.config, args.feature_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    alerts.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"已生成复合告警：{args.output}")
    print(f"告警数量：{len(alerts):,}")
    if not alerts.empty:
        print(alerts.groupby(["rule_id", "rule_name_cn"]).size().to_string())


if __name__ == "__main__":
    main()
