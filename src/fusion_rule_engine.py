"""分层融合复合规则、MAD、STL 与 Isolation Forest，并生成分级告警。"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL

from severity_policy import SeverityBand, breach_ratio, grade_alert, load_severity_policy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "docs" / "ai_monitoring_metric_dictionary.xlsx"
DEFAULT_CUSTOMER_FEATURES = (
    PROJECT_ROOT / "outputs" / "features" / "customer_hourly_features.csv"
)
DEFAULT_COMPOSITE_ALERTS = PROJECT_ROOT / "outputs" / "composite_alerts.csv"
DEFAULT_TRUTH = PROJECT_ROOT / "data" / "ground_truth.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "fusion_alerts.csv"
DEFAULT_ALL_ALERTS = PROJECT_ROOT / "outputs" / "fusion_strategy_alerts.csv"
DEFAULT_SCORES = PROJECT_ROOT / "outputs" / "fusion_customer_scores.csv"
DEFAULT_BENCHMARK = (
    PROJECT_ROOT / "outputs" / "benchmark" / "fusion_strategy_results.csv"
)

DETECTOR_METRICS = [
    "server_error_rate",
    "tokens_per_request",
    "estimated_cost",
    "output_input_ratio",
]


def _text(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


@dataclass(frozen=True)
class FusionStrategy:
    strategy_id: str
    strategy_name: str
    strategy_name_cn: str
    detector_sources: str
    minimum_votes: int
    mad_threshold: float
    stl_threshold: float
    iforest_threshold: float
    minimum_requests: int
    server_error_threshold: float
    token_baseline_multiplier: float
    suppress_after_composite_minutes: int
    cooldown_minutes: int
    default_severity: str
    status: str
    is_default: bool
    version: str
    description: str


def load_fusion_strategies(
    path: Path = DEFAULT_CONFIG,
    *,
    as_of: date | None = None,
) -> list[FusionStrategy]:
    if not path.exists():
        raise FileNotFoundError(f"找不到指标字典：{path}")
    frame = pd.read_excel(path, sheet_name="Fusion Strategies")
    required = {
        "strategy_id", "strategy_name", "strategy_name_cn", "detector_sources",
        "minimum_votes", "mad_threshold", "stl_threshold", "iforest_threshold",
        "minimum_requests", "server_error_threshold", "token_baseline_multiplier",
        "suppress_after_composite_minutes", "cooldown_minutes", "default_severity",
        "status", "is_default", "version", "valid_from", "description",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("融合策略工作表缺少列：" + ", ".join(sorted(missing)))
    if frame["strategy_id"].duplicated().any():
        raise ValueError("融合策略存在重复 strategy_id")

    evaluation_date = pd.Timestamp(as_of or date.today()).normalize()
    frame["valid_from"] = pd.to_datetime(frame["valid_from"], errors="coerce")
    frame = frame[
        frame["status"].astype(str).str.lower().isin(["active", "experimental"])
        & (frame["valid_from"].isna() | (frame["valid_from"] <= evaluation_date))
    ].copy()
    if frame.empty:
        raise ValueError("没有可运行的融合策略")

    strategies: list[FusionStrategy] = []
    for _, row in frame.iterrows():
        minimum_votes = int(row["minimum_votes"])
        if minimum_votes not in {1, 2, 3}:
            raise ValueError(f"策略 {row['strategy_id']} 的 minimum_votes 必须为1到3")
        strategies.append(
            FusionStrategy(
                strategy_id=_text(row["strategy_id"]),
                strategy_name=_text(row["strategy_name"]),
                strategy_name_cn=_text(row["strategy_name_cn"]),
                detector_sources=_text(row["detector_sources"]),
                minimum_votes=minimum_votes,
                mad_threshold=float(row["mad_threshold"]),
                stl_threshold=float(row["stl_threshold"]),
                iforest_threshold=float(row["iforest_threshold"]),
                minimum_requests=int(row["minimum_requests"]),
                server_error_threshold=float(row["server_error_threshold"]),
                token_baseline_multiplier=float(row["token_baseline_multiplier"]),
                suppress_after_composite_minutes=int(
                    row["suppress_after_composite_minutes"]
                ),
                cooldown_minutes=int(row["cooldown_minutes"]),
                default_severity=_text(row["default_severity"]).lower(),
                status=_text(row["status"]).lower(),
                is_default=_text(row["is_default"]).lower() == "yes",
                version=_text(row["version"]),
                description=_text(row["description"]),
            )
        )
    defaults = [strategy for strategy in strategies if strategy.is_default]
    if len(defaults) != 1 or defaults[0].status != "active":
        raise ValueError("必须且只能有一条 active 的默认融合策略")
    return strategies


def _rolling_mad_score(group: pd.DataFrame, window: int = 24) -> pd.Series:
    scores: list[pd.Series] = []
    for metric in DETECTOR_METRICS:
        values = pd.to_numeric(group[metric], errors="coerce")
        baseline = values.shift(1).rolling(window, min_periods=12).median()
        deviation = (values.shift(1) - baseline).abs()
        mad = deviation.rolling(window, min_periods=12).median()
        fallback = values.shift(1).rolling(window, min_periods=12).std() / 1.4826
        scale = mad.where(mad > 1e-9, fallback).replace(0, np.nan)
        scores.append((0.6745 * (values - baseline).abs() / scale).rename(metric))
    return pd.concat(scores, axis=1).max(axis=1, skipna=True).fillna(0)


def _stl_score(group: pd.DataFrame) -> pd.Series:
    scores: list[pd.Series] = []
    for metric in DETECTOR_METRICS:
        values = pd.to_numeric(group[metric], errors="coerce").interpolate().bfill().ffill()
        residual = pd.Series(
            STL(values, period=24, robust=True).fit().resid,
            index=group.index,
        )
        center = residual.median()
        mad = (residual - center).abs().median()
        if mad <= 1e-9:
            mad = residual.std() / 1.4826
        scores.append(
            (0.6745 * (residual - center).abs() / max(float(mad), 1e-9)).rename(metric)
        )
    return pd.concat(scores, axis=1).max(axis=1, skipna=True).fillna(0)


def _iforest_score(group: pd.DataFrame, global_start: pd.Timestamp) -> pd.Series:
    training_mask = group["hour"] < global_start + pd.Timedelta(days=10)
    if training_mask.sum() < 24:
        return pd.Series(np.nan, index=group.index)
    pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        IsolationForest(
            n_estimators=300,
            contamination=0.02,
            random_state=42,
            n_jobs=-1,
        ),
    )
    pipeline.fit(group.loc[training_mask, DETECTOR_METRICS])
    return pd.Series(
        -pipeline.decision_function(group[DETECTOR_METRICS]),
        index=group.index,
    )


def build_customer_detector_scores(path: Path = DEFAULT_CUSTOMER_FEATURES) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到客户特征表：{path}")
    features = pd.read_csv(path, parse_dates=["hour"])
    required = {"hour", "customer_id", "request_count", *DETECTOR_METRICS}
    missing = required.difference(features.columns)
    if missing:
        raise ValueError("客户特征表缺少列：" + ", ".join(sorted(missing)))
    features = features.sort_values(["customer_id", "hour"]).reset_index(drop=True)
    global_start = features["hour"].min()
    groups: list[pd.DataFrame] = []
    for _, source in features.groupby("customer_id", sort=False):
        group = source.copy().reset_index(drop=True)
        group["score_mad"] = _rolling_mad_score(group)
        group["score_stl"] = _stl_score(group)
        group["score_isolation_forest"] = _iforest_score(group, global_start)
        group["token_baseline"] = (
            group["tokens_per_request"].shift(1).rolling(24, min_periods=12).median()
        )
        groups.append(group)
    return pd.concat(groups, ignore_index=True).sort_values(["hour", "customer_id"])


def _suppressed_by_composite(
    hour: pd.Timestamp,
    composite_hours: list[pd.Timestamp],
    suppress_minutes: int,
) -> bool:
    window = pd.Timedelta(minutes=suppress_minutes)
    return any(start <= hour <= start + window for start in composite_hours)


def _apply_cooldown(
    candidates: pd.DataFrame,
    strategy: FusionStrategy,
) -> pd.DataFrame:
    kept: list[int] = []
    last_alert: dict[tuple[str, str], pd.Timestamp] = {}
    cooldown = pd.Timedelta(minutes=strategy.cooldown_minutes)
    for index, row in candidates.sort_values("hour").iterrows():
        signal_type = "server_error_context" if row["error_context"] else "token_context"
        key = (str(row["customer_id"]), signal_type)
        previous = last_alert.get(key)
        if previous is None or row["hour"] - previous >= cooldown:
            kept.append(index)
            last_alert[key] = row["hour"]
    return candidates.loc[kept].copy()


def evaluate_strategy(
    scores: pd.DataFrame,
    composite_alerts: pd.DataFrame,
    strategy: FusionStrategy,
    severity_policy: list[SeverityBand],
) -> pd.DataFrame:
    data = scores.copy()
    data["pred_mad"] = data["score_mad"] >= strategy.mad_threshold
    data["pred_stl"] = data["score_stl"] >= strategy.stl_threshold
    data["pred_isolation_forest"] = (
        data["score_isolation_forest"] >= strategy.iforest_threshold
    )
    data["detector_votes"] = data[
        ["pred_mad", "pred_stl", "pred_isolation_forest"]
    ].sum(axis=1)
    sample_gate = data["request_count"] >= strategy.minimum_requests
    data["error_context"] = sample_gate & (
        data["server_error_rate"] >= strategy.server_error_threshold
    )
    data["token_context"] = sample_gate & (
        data["tokens_per_request"]
        >= data["token_baseline"] * strategy.token_baseline_multiplier
    )
    candidate_mask = (
        data["detector_votes"].ge(strategy.minimum_votes)
        & (data["error_context"] | data["token_context"])
    )
    candidates = data[candidate_mask].copy()

    composite_hours = sorted(
        set(pd.to_datetime(composite_alerts["detected_at"]).dt.floor("h"))
    )
    candidates = candidates[
        ~candidates["hour"].map(
            lambda hour: _suppressed_by_composite(
                hour, composite_hours, strategy.suppress_after_composite_minutes
            )
        )
    ]
    candidates = _apply_cooldown(candidates, strategy)

    rows: list[dict[str, object]] = []
    for _, alert in composite_alerts.iterrows():
        rows.append(
            {
                "fusion_alert_id": f"{strategy.strategy_id}-{alert['alert_id']}",
                "strategy_id": strategy.strategy_id,
                "strategy_name_cn": strategy.strategy_name_cn,
                "layer": "L1_COMPOSITE",
                "detected_at": alert["detected_at"],
                "entity_type": alert["entity_type"],
                "entity_value": alert["entity_value"],
                "signal_type": "composite_rule",
                "source_detectors": "CompositeRules",
                "detector_votes": 0,
                "base_severity": alert.get("base_severity", alert["alert_level"]),
                "alert_level": alert["alert_level"],
                "breach_ratio": float(alert.get("breach_ratio", 1.0)),
                "severity_score": float(alert.get("severity_score", 0.0)),
                "config_version": strategy.version,
                "evidence": alert["observed_summary"],
                "threshold_evidence": alert["threshold_summary"],
                "description": alert["description"],
                "recommended_action": alert["recommended_action"],
                "alert_state": "open",
            }
        )
    for ordinal, (_, alert) in enumerate(candidates.iterrows(), start=1):
        signal_type = "server_error_context" if alert["error_context"] else "token_context"
        if signal_type == "server_error_context":
            base_severity = "critical"
            threshold_value = strategy.server_error_threshold
            ratio = breach_ratio(alert["server_error_rate"], threshold_value, "gte")
        else:
            base_severity = strategy.default_severity
            threshold_value = alert["token_baseline"] * strategy.token_baseline_multiplier
            ratio = breach_ratio(alert["tokens_per_request"], threshold_value, "gte")
        severity, severity_score, ratio = grade_alert(
            base_severity=base_severity,
            breach_ratios=[ratio],
            matched_conditions=int(alert["detector_votes"]),
            policy=severity_policy,
        )
        source_detectors = [
            name
            for name, matched in [
                ("RollingMAD", alert["pred_mad"]),
                ("STLResidual", alert["pred_stl"]),
                ("IsolationForest", alert["pred_isolation_forest"]),
            ]
            if bool(matched)
        ]
        evidence = {
            "request_count": round(float(alert["request_count"]), 4),
            "server_error_rate": round(float(alert["server_error_rate"]), 4),
            "tokens_per_request": round(float(alert["tokens_per_request"]), 4),
            "token_baseline": round(float(alert["token_baseline"]), 4),
            "score_mad": round(float(alert["score_mad"]), 4),
            "score_stl": round(float(alert["score_stl"]), 4),
            "score_isolation_forest": round(
                float(alert["score_isolation_forest"]), 4
            ),
        }
        thresholds = {
            "minimum_votes": strategy.minimum_votes,
            "minimum_requests": strategy.minimum_requests,
            "server_error_threshold": strategy.server_error_threshold,
            "token_baseline_multiplier": strategy.token_baseline_multiplier,
        }
        rows.append(
            {
                "fusion_alert_id": f"{strategy.strategy_id}-ALG-{ordinal:04d}",
                "strategy_id": strategy.strategy_id,
                "strategy_name_cn": strategy.strategy_name_cn,
                "layer": "L2_CONTEXTUAL_FUSION",
                "detected_at": alert["hour"],
                "entity_type": "customer_id",
                "entity_value": alert["customer_id"],
                "signal_type": signal_type,
                "source_detectors": ";".join(source_detectors),
                "detector_votes": int(alert["detector_votes"]),
                "base_severity": base_severity,
                "alert_level": severity,
                "breach_ratio": ratio,
                "severity_score": severity_score,
                "config_version": strategy.version,
                "evidence": json.dumps(evidence, ensure_ascii=False),
                "threshold_evidence": json.dumps(thresholds, ensure_ascii=False),
                "description": (
                    "客户5xx错误率异常且多检测器一致"
                    if signal_type == "server_error_context"
                    else "客户单次Token显著偏离历史基线且多检测器一致"
                ),
                "recommended_action": (
                    "立即检查客户、供应商、路由和错误码，必要时降级或隔离"
                    if signal_type == "server_error_context"
                    else "检查提示词、批处理任务、上下文长度和预算保护"
                ),
                "alert_state": "open",
            }
        )
    return pd.DataFrame(rows).sort_values("detected_at").reset_index(drop=True)


def benchmark_strategies(
    alerts_by_strategy: dict[str, pd.DataFrame],
    truth: pd.DataFrame,
    total_days: float = 30.0,
) -> pd.DataFrame:
    truth_hours: set[pd.Timestamp] = set()
    for _, event in truth.iterrows():
        truth_hours.update(
            pd.date_range(
                event["start_time"].floor("h"),
                event["end_time"].floor("h"),
                freq="h",
            )
        )
    rows: list[dict[str, object]] = []
    for strategy_id, alerts in alerts_by_strategy.items():
        alert_hours = set(pd.to_datetime(alerts["detected_at"]).dt.floor("h"))
        true_positive = len(alert_hours & truth_hours)
        false_positive = len(alert_hours - truth_hours)
        false_negative = len(truth_hours - alert_hours)
        precision = true_positive / len(alert_hours) if alert_hours else 0.0
        recall = true_positive / len(truth_hours) if truth_hours else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        detected_events = 0
        for _, event in truth.iterrows():
            start_hour = event["start_time"].floor("h")
            end_hour = event["end_time"].floor("h")
            if any(start_hour <= hour <= end_hour for hour in alert_hours):
                detected_events += 1
        levels = alerts["alert_level"].value_counts().to_dict()
        rows.append(
            {
                "strategy_id": strategy_id,
                "strategy_name_cn": alerts["strategy_name_cn"].iloc[0],
                "alert_count": len(alerts),
                "unique_alert_hours": len(alert_hours),
                "true_positive_hours": true_positive,
                "false_positive_hours": false_positive,
                "false_negative_hours": false_negative,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "event_recall": round(detected_events / len(truth), 4),
                "false_alarms_per_day": round(false_positive / total_days, 4),
                "critical_alerts": int(levels.get("critical", 0)),
                "warning_alerts": int(levels.get("warning", 0)),
                "info_alerts": int(levels.get("info", 0)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["event_recall", "precision", "f1"], ascending=False
    )


def run_fusion(
    config_path: Path = DEFAULT_CONFIG,
    customer_feature_path: Path = DEFAULT_CUSTOMER_FEATURES,
    composite_alert_path: Path = DEFAULT_COMPOSITE_ALERTS,
    truth_path: Path = DEFAULT_TRUTH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    strategies = load_fusion_strategies(config_path)
    severity_policy = load_severity_policy(config_path)
    scores = build_customer_detector_scores(customer_feature_path)
    composite_alerts = pd.read_csv(composite_alert_path, parse_dates=["detected_at"])
    truth = pd.read_csv(truth_path, parse_dates=["start_time", "end_time"])

    alerts_by_strategy = {
        strategy.strategy_id: evaluate_strategy(
            scores, composite_alerts, strategy, severity_policy
        )
        for strategy in strategies
    }
    all_alerts = pd.concat(alerts_by_strategy.values(), ignore_index=True)
    benchmark = benchmark_strategies(alerts_by_strategy, truth)
    default_strategy = next(strategy for strategy in strategies if strategy.is_default)
    default_alerts = alerts_by_strategy[default_strategy.strategy_id]
    return default_alerts, all_alerts, scores, benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="运行分层融合告警策略")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--customer-features", type=Path, default=DEFAULT_CUSTOMER_FEATURES)
    parser.add_argument("--composite-alerts", type=Path, default=DEFAULT_COMPOSITE_ALERTS)
    parser.add_argument("--truth", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--all-alerts", type=Path, default=DEFAULT_ALL_ALERTS)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    args = parser.parse_args()

    default_alerts, all_alerts, scores, benchmark = run_fusion(
        args.config, args.customer_features, args.composite_alerts, args.truth
    )
    for path in [args.output, args.all_alerts, args.scores, args.benchmark]:
        path.parent.mkdir(parents=True, exist_ok=True)
    default_alerts.to_csv(args.output, index=False, encoding="utf-8-sig")
    all_alerts.to_csv(args.all_alerts, index=False, encoding="utf-8-sig")
    scores.to_csv(args.scores, index=False, encoding="utf-8-sig")
    benchmark.to_csv(args.benchmark, index=False, encoding="utf-8-sig")

    print(f"默认融合告警：{args.output}（{len(default_alerts)} 条）")
    print(f"策略对比结果：{args.benchmark}")
    print(benchmark.to_string(index=False))


if __name__ == "__main__":
    main()
