"""从指标字典读取规则，检测 AI 调用日志异常并生成告警表。"""

from __future__ import annotations

import argparse
import re
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from severity_policy import SeverityBand, breach_ratio, grade_alert, load_severity_policy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "sample_logs.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "alerts.csv"
DEFAULT_CONFIG = PROJECT_ROOT / "docs" / "ai_monitoring_metric_dictionary.xlsx"

SUPPORTED_RULES = {
    "TOKEN_SPIKE",
    "ERROR_RATE_HIGH",
    "P95_LATENCY_HIGH",
    "KEY_HIGH_FREQUENCY",
}


def _text(value: object) -> str:
    """把Excel空单元格稳定转换为空字符串。"""
    return "" if pd.isna(value) else str(value).strip()


@dataclass(frozen=True)
class RuleConfig:
    rule_id: str
    rule_name: str
    metric_id: str
    metric_name: str
    dimension_type: str
    comparison_operator: str
    threshold_type: str
    threshold_value: float
    threshold_unit: str
    baseline_window: str
    minimum_sample_size: int
    severity: str
    evaluation_frequency: str
    description: str
    recommended_action: str
    version: str


@dataclass
class Alert:
    alert_id: str
    rule_id: str
    detected_at: str
    dimension_type: str
    dimension_value: str
    rule_name: str
    metric_id: str
    metric_name: str
    metric_value: float
    threshold: float
    threshold_unit: str
    base_severity: str
    alert_level: str
    breach_ratio: float
    severity_score: float
    config_version: str
    description: str
    recommended_action: str


def load_logs(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"找不到 {path}，请先运行 python src/generate_sample_data.py"
        )

    logs = pd.read_csv(path, parse_dates=["timestamp"])
    required = {
        "request_id",
        "timestamp",
        "api_key",
        "total_tokens",
        "latency_ms",
        "status_code",
    }
    missing = required.difference(logs.columns)
    if missing:
        raise ValueError(f"日志缺少字段：{', '.join(sorted(missing))}")
    return logs


def load_rule_config(
    path: Path = DEFAULT_CONFIG,
    *,
    as_of: date | None = None,
) -> dict[str, RuleConfig]:
    """读取活动规则并转换为受控的运行时配置。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到指标字典配置：{path}")

    try:
        rules = pd.read_excel(path, sheet_name="Alert Rules")
        metrics = pd.read_excel(path, sheet_name="Metrics")
    except ValueError as exc:
        raise ValueError("Metric dictionary must contain Alert Rules and Metrics sheets") from exc

    required_rule_columns = {
        "rule_id",
        "rule_name",
        "metric_id",
        "dimension_type",
        "comparison_operator",
        "threshold_type",
        "threshold_value",
        "threshold_unit",
        "baseline_window",
        "minimum_sample_size",
        "severity",
        "evaluation_frequency",
        "description",
        "recommended_action",
        "status",
        "version",
        "valid_from",
    }
    missing_rule_columns = required_rule_columns.difference(rules.columns)
    if missing_rule_columns:
        raise ValueError(
            "异常规则工作表缺少列：" + ", ".join(sorted(missing_rule_columns))
        )

    required_metric_columns = {"metric_id", "metric_name_en", "status"}
    missing_metric_columns = required_metric_columns.difference(metrics.columns)
    if missing_metric_columns:
        raise ValueError(
            "指标字典工作表缺少列：" + ", ".join(sorted(missing_metric_columns))
        )

    evaluation_date = pd.Timestamp(as_of or date.today()).normalize()
    rules["valid_from"] = pd.to_datetime(rules["valid_from"], errors="coerce")
    active_mask = (
        rules["status"].astype(str).str.strip().str.lower().eq("active")
        & (rules["valid_from"].isna() | (rules["valid_from"] <= evaluation_date))
    )
    if "valid_to" in rules.columns:
        rules["valid_to"] = pd.to_datetime(rules["valid_to"], errors="coerce")
        active_mask &= rules["valid_to"].isna() | (
            rules["valid_to"] >= evaluation_date
        )
    active_rules = rules[active_mask].copy()

    if active_rules.empty:
        raise ValueError(f"{path} 中没有当前生效的活动规则")
    if active_rules["rule_id"].duplicated().any():
        duplicated = active_rules.loc[
            active_rules["rule_id"].duplicated(keep=False), "rule_id"
        ].tolist()
        raise ValueError(f"活动规则编号重复：{duplicated}")
    if active_rules["rule_name"].duplicated().any():
        duplicated = active_rules.loc[
            active_rules["rule_name"].duplicated(keep=False), "rule_name"
        ].tolist()
        raise ValueError(f"活动规则名称重复：{duplicated}")

    unsupported = set(active_rules["rule_name"]) - SUPPORTED_RULES
    if unsupported:
        raise ValueError(
            "以下活动规则尚无执行器，请先实现后再设为active："
            + ", ".join(sorted(unsupported))
        )

    active_metrics = metrics[
        metrics["status"].astype(str).str.strip().str.lower().eq("active")
    ].copy()
    if active_metrics["metric_id"].duplicated().any():
        raise ValueError("指标字典存在重复的活动 metric_id")
    metric_names = active_metrics.set_index("metric_id")["metric_name_en"].to_dict()

    configs: dict[str, RuleConfig] = {}
    for _, row in active_rules.iterrows():
        rule_name = _text(row["rule_name"])
        metric_id = _text(row["metric_id"])
        if metric_id not in metric_names:
            raise ValueError(f"规则 {rule_name} 关联的活动指标不存在：{metric_id}")

        operator = _text(row["comparison_operator"]).lower()
        if operator != "gt":
            raise ValueError(f"规则 {rule_name} 当前只支持 comparison_operator=gt")

        threshold = pd.to_numeric(row["threshold_value"], errors="coerce")
        minimum_sample_size = pd.to_numeric(
            row["minimum_sample_size"], errors="coerce"
        )
        if pd.isna(threshold) or float(threshold) <= 0:
            raise ValueError(f"规则 {rule_name} 的 threshold_value 必须大于0")
        if pd.isna(minimum_sample_size) or int(minimum_sample_size) < 1:
            raise ValueError(f"规则 {rule_name} 的 minimum_sample_size 必须至少为1")

        configs[rule_name] = RuleConfig(
            rule_id=_text(row["rule_id"]),
            rule_name=rule_name,
            metric_id=metric_id,
            metric_name=_text(metric_names[metric_id]),
            dimension_type=_text(row["dimension_type"]),
            comparison_operator=operator,
            threshold_type=_text(row["threshold_type"]),
            threshold_value=float(threshold),
            threshold_unit=_text(row["threshold_unit"]),
            baseline_window=_text(row["baseline_window"]),
            minimum_sample_size=int(minimum_sample_size),
            severity=_text(row["severity"]).lower(),
            evaluation_frequency=_text(row["evaluation_frequency"]),
            description=_text(row["description"]),
            recommended_action=_text(row["recommended_action"]),
            version=_text(row["version"]),
        )

    _validate_supported_rule_shapes(configs)
    return configs


def _parse_baseline_window(value: str, expected_statistic: str) -> int:
    match = re.fullmatch(r"previous_(\d+)_hours_(mean|median)", value)
    if not match:
        raise ValueError(
            f"不支持的 baseline_window={value}；格式应为 previous_24_hours_mean/median"
        )
    window, statistic = int(match.group(1)), match.group(2)
    if statistic != expected_statistic:
        raise ValueError(
            f"baseline_window={value} 应使用 {expected_statistic} 统计方式"
        )
    return window


def _validate_supported_rule_shapes(configs: dict[str, RuleConfig]) -> None:
    expected_threshold_types = {
        "TOKEN_SPIKE": "baseline_multiplier",
        "ERROR_RATE_HIGH": "static",
        "P95_LATENCY_HIGH": "baseline_multiplier",
        "KEY_HIGH_FREQUENCY": "static",
    }
    for rule_name, config in configs.items():
        expected = expected_threshold_types[rule_name]
        if config.threshold_type != expected:
            raise ValueError(
                f"规则 {rule_name} 的 threshold_type 应为 {expected}，"
                f"当前为 {config.threshold_type}"
            )

    if "TOKEN_SPIKE" in configs:
        window = _parse_baseline_window(
            configs["TOKEN_SPIKE"].baseline_window, "mean"
        )
        if configs["TOKEN_SPIKE"].minimum_sample_size > window:
            raise ValueError("TOKEN_SPIKE 的 minimum_sample_size 不能大于基线窗口")
    if "P95_LATENCY_HIGH" in configs:
        window = _parse_baseline_window(
            configs["P95_LATENCY_HIGH"].baseline_window, "median"
        )
        if configs["P95_LATENCY_HIGH"].minimum_sample_size > window:
            raise ValueError("P95_LATENCY_HIGH 的 minimum_sample_size 不能大于基线窗口")


def build_hourly_metrics(logs: pd.DataFrame) -> pd.DataFrame:
    data = logs.copy()
    data["hour"] = data["timestamp"].dt.floor("h")
    data["is_error"] = ~data["status_code"].between(200, 299)
    return (
        data.groupby("hour", as_index=False)
        .agg(
            request_count=("request_id", "count"),
            total_tokens=("total_tokens", "sum"),
            error_rate=("is_error", "mean"),
            p95_latency_ms=("latency_ms", lambda values: values.quantile(0.95)),
        )
        .sort_values("hour")
    )


def _create_alert(
    config: RuleConfig,
    severity_policy: list[SeverityBand],
    *,
    detected_at: object,
    dimension_value: str,
    metric_value: float,
    threshold: float,
) -> Alert:
    ratio = breach_ratio(metric_value, threshold, config.comparison_operator)
    level, score, ratio = grade_alert(
        base_severity=config.severity,
        breach_ratios=[ratio],
        matched_conditions=1,
        policy=severity_policy,
    )
    return Alert(
        alert_id="",
        rule_id=config.rule_id,
        detected_at=str(detected_at),
        dimension_type=config.dimension_type,
        dimension_value=dimension_value,
        rule_name=config.rule_name,
        metric_id=config.metric_id,
        metric_name=config.metric_name,
        metric_value=round(float(metric_value), 2),
        threshold=round(float(threshold), 2),
        threshold_unit=config.threshold_unit,
        base_severity=config.severity,
        alert_level=level,
        breach_ratio=ratio,
        severity_score=score,
        config_version=config.version,
        description=config.description,
        recommended_action=config.recommended_action,
    )


def detect_hourly_alerts(
    hourly: pd.DataFrame,
    configs: dict[str, RuleConfig],
    severity_policy: list[SeverityBand],
) -> list[Alert]:
    alerts: list[Alert] = []

    token_config = configs.get("TOKEN_SPIKE")
    if token_config:
        window = _parse_baseline_window(token_config.baseline_window, "mean")
        baseline = (
            hourly["total_tokens"]
            .rolling(window, min_periods=token_config.minimum_sample_size)
            .mean()
            .shift(1)
        )
        dynamic_threshold = baseline * token_config.threshold_value
        token_rows = hourly[baseline.notna() & (hourly["total_tokens"] > dynamic_threshold)]
        for index, row in token_rows.iterrows():
            alerts.append(
                _create_alert(
                    token_config,
                    severity_policy,
                    detected_at=row["hour"],
                    dimension_value="all",
                    metric_value=row["total_tokens"],
                    threshold=dynamic_threshold.loc[index],
                )
            )

    error_config = configs.get("ERROR_RATE_HIGH")
    if error_config:
        error_rate_percent = hourly["error_rate"] * 100
        error_rows = hourly[
            (hourly["request_count"] >= error_config.minimum_sample_size)
            & (error_rate_percent > error_config.threshold_value)
        ]
        for index, row in error_rows.iterrows():
            alerts.append(
                _create_alert(
                    error_config,
                    severity_policy,
                    detected_at=row["hour"],
                    dimension_value="all",
                    metric_value=error_rate_percent.loc[index],
                    threshold=error_config.threshold_value,
                )
            )

    latency_config = configs.get("P95_LATENCY_HIGH")
    if latency_config:
        window = _parse_baseline_window(latency_config.baseline_window, "median")
        baseline = (
            hourly["p95_latency_ms"]
            .rolling(window, min_periods=latency_config.minimum_sample_size)
            .median()
            .shift(1)
        )
        dynamic_threshold = baseline * latency_config.threshold_value
        latency_rows = hourly[
            baseline.notna() & (hourly["p95_latency_ms"] > dynamic_threshold)
        ]
        for index, row in latency_rows.iterrows():
            alerts.append(
                _create_alert(
                    latency_config,
                    severity_policy,
                    detected_at=row["hour"],
                    dimension_value="all",
                    metric_value=row["p95_latency_ms"],
                    threshold=dynamic_threshold.loc[index],
                )
            )

    return alerts


def detect_key_bursts(
    logs: pd.DataFrame,
    configs: dict[str, RuleConfig],
    severity_policy: list[SeverityBand],
) -> list[Alert]:
    config = configs.get("KEY_HIGH_FREQUENCY")
    if not config:
        return []

    data = logs.copy()
    data["minute"] = data["timestamp"].dt.floor("min")
    key_minute = (
        data.groupby(["minute", "api_key"], as_index=False)
        .agg(request_count=("request_id", "count"))
        .sort_values("request_count", ascending=False)
    )
    burst_rows = key_minute[
        (key_minute["request_count"] >= config.minimum_sample_size)
        & (key_minute["request_count"] > config.threshold_value)
    ]
    return [
        _create_alert(
            config,
            severity_policy,
            detected_at=row["minute"],
            dimension_value=str(row["api_key"]),
            metric_value=row["request_count"],
            threshold=config.threshold_value,
        )
        for _, row in burst_rows.iterrows()
    ]


def detect(
    logs: pd.DataFrame,
    configs: dict[str, RuleConfig] | None = None,
    *,
    config_path: Path = DEFAULT_CONFIG,
) -> pd.DataFrame:
    active_configs = configs if configs is not None else load_rule_config(config_path)
    severity_policy = load_severity_policy(config_path)
    hourly = build_hourly_metrics(logs)
    alerts = detect_hourly_alerts(hourly, active_configs, severity_policy) + detect_key_bursts(
        logs, active_configs, severity_policy
    )
    alerts.sort(key=lambda alert: (alert.detected_at, alert.rule_name))

    for number, alert in enumerate(alerts, start=1):
        alert.alert_id = f"ALT-{number:04d}"

    columns = list(Alert.__dataclass_fields__)
    return pd.DataFrame([asdict(alert) for alert in alerts], columns=columns)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检测 AI 调用日志异常")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="输入日志 CSV")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出告警 CSV")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="指标字典 Excel 配置",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logs = load_logs(args.input)
    configs = load_rule_config(args.config)
    alerts = detect(logs, configs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    alerts.to_csv(args.output, index=False)

    versions = sorted({config.version for config in configs.values()})
    print(f"指标字典：{args.config}")
    print(f"配置版本：{', '.join(versions)}")
    print(f"活动规则：{len(configs)} 条")
    print(f"检测日志：{len(logs):,} 条")
    print(f"生成告警：{len(alerts)} 条")
    if alerts.empty:
        print("未发现符合当前规则的异常")
    else:
        print("\n告警摘要：")
        print(
            alerts[
                [
                    "alert_id",
                    "rule_id",
                    "detected_at",
                    "rule_name",
                    "dimension_value",
                    "metric_value",
                    "threshold",
                    "alert_level",
                    "config_version",
                ]
            ].to_string(index=False)
        )
    print(f"\n告警文件：{args.output}")


if __name__ == "__main__":
    main()
