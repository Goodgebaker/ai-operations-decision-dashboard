"""根据连续失败、跨探针相关性与恢复状态生成主动拨测事件。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from severity_policy import SeverityBand, grade_alert, load_severity_policy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "docs" / "ai_monitoring_metric_dictionary.xlsx"
DEFAULT_INPUT = PROJECT_ROOT / "data" / "probe_runs.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "probe_alerts.csv"


@dataclass(frozen=True)
class IncidentWindow:
    provider: str
    start: pd.Timestamp
    end: pd.Timestamp
    failed_probes: str


def _provider_incident_windows(runs: pd.DataFrame) -> list[IncidentWindow]:
    data = runs.copy()
    data["bucket"] = data["started_at"].dt.floor("15min")
    failed = data[~data["success"].astype(bool)]
    correlated = (
        failed.groupby(["provider", "bucket"])["probe_id"]
        .agg(lambda values: ";".join(sorted(set(values))))
        .reset_index(name="failed_probes")
    )
    correlated["failed_count"] = correlated["failed_probes"].str.split(";").str.len()
    correlated = correlated[correlated["failed_count"] >= 2]

    windows: list[IncidentWindow] = []
    for provider, group in correlated.groupby("provider"):
        group = group.sort_values("bucket")
        cluster_start: pd.Timestamp | None = None
        cluster_end: pd.Timestamp | None = None
        cluster_probes: set[str] = set()
        for _, row in group.iterrows():
            bucket = pd.Timestamp(row["bucket"])
            probes = set(str(row["failed_probes"]).split(";"))
            if cluster_start is None or bucket - cluster_end > pd.Timedelta(minutes=60):
                if cluster_start is not None:
                    windows.append(
                        IncidentWindow(
                            provider,
                            cluster_start,
                            cluster_end + pd.Timedelta(minutes=30),
                            ";".join(sorted(cluster_probes)),
                        )
                    )
                cluster_start = cluster_end = bucket
                cluster_probes = probes
            else:
                cluster_end = bucket
                cluster_probes.update(probes)
        if cluster_start is not None and cluster_end is not None:
            windows.append(
                IncidentWindow(
                    provider,
                    cluster_start,
                    cluster_end + pd.Timedelta(minutes=30),
                    ";".join(sorted(cluster_probes)),
                )
            )
    return windows


def _inside_provider_incident(
    provider: str,
    timestamp: pd.Timestamp,
    windows: list[IncidentWindow],
) -> bool:
    return any(
        window.provider == provider and window.start <= timestamp <= window.end
        for window in windows
    )


def detect_probe_events(
    runs: pd.DataFrame,
    severity_policy: list[SeverityBand],
    probe_severities: dict[str, str],
) -> pd.DataFrame:
    data = runs.copy()
    data["started_at"] = pd.to_datetime(data["started_at"])
    data["success"] = data["success"].astype(str).str.lower().eq("true") if data["success"].dtype == object else data["success"].astype(bool)
    data = data.sort_values(["probe_id", "started_at"])
    windows = _provider_incident_windows(data)
    events: list[dict[str, object]] = []

    for window in windows:
        failed_probe_count = len(window.failed_probes.split(";"))
        incident_level, incident_score, incident_ratio = grade_alert(
            base_severity="critical",
            breach_ratios=[max(1.0, float(failed_probe_count))],
            matched_conditions=failed_probe_count,
            policy=severity_policy,
        )
        events.append(
            {
                "detected_at": window.start,
                "base_severity": "critical",
                "alert_level": incident_level,
                "breach_ratio": incident_ratio,
                "severity_score": incident_score,
                "event_type": "correlated_failure",
                "probe_id": "MULTI",
                "probe_name_cn": "多探针相关故障",
                "provider": window.provider,
                "model_id": "multiple",
                "region": "multiple",
                "title": f"{window.provider} 多个主动探针同时失败",
                "content": f"失败探针：{window.failed_probes}",
                "consecutive_failures": 3,
                "failed_probes": window.failed_probes,
                "recommended_action": "结合真实业务错误率和路由状态确认故障，必要时切换供应商",
                "alert_state": "open",
            }
        )
        events.append(
            {
                "detected_at": window.end,
                "base_severity": "recovery",
                "alert_level": "recovery",
                "breach_ratio": 0.0,
                "severity_score": 0.0,
                "event_type": "recovery",
                "probe_id": "MULTI",
                "probe_name_cn": "多探针恢复",
                "provider": window.provider,
                "model_id": "multiple",
                "region": "multiple",
                "title": f"{window.provider} 主动拨测恢复",
                "content": "多个探针已恢复正常",
                "consecutive_failures": 0,
                "failed_probes": window.failed_probes,
                "recommended_action": "继续观察两个拨测周期，并确认真实业务指标同步恢复",
                "alert_state": "recovered",
            }
        )

    for probe_id, group in data.groupby("probe_id"):
        state = "normal"
        recent: list[bool] = []
        consecutive_failures = 0
        consecutive_successes = 0
        for _, run in group.sort_values("started_at").iterrows():
            success = bool(run["success"])
            recent.append(success)
            recent = recent[-3:]
            if success:
                consecutive_successes += 1
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                consecutive_successes = 0

            if _inside_provider_incident(run["provider"], run["started_at"], windows):
                continue

            failures_in_three = sum(not value for value in recent)
            new_state = state
            if consecutive_failures >= 3:
                new_state = "critical"
            elif failures_in_three >= 2:
                new_state = "warning"
            elif not success:
                new_state = "observation"

            if new_state in {"warning", "critical"} and new_state != state:
                base_severity = probe_severities.get(str(probe_id), "warning")
                level, severity_score, failure_ratio = grade_alert(
                    base_severity=base_severity,
                    breach_ratios=[max(1.0, failures_in_three / 2)],
                    matched_conditions=consecutive_failures,
                    policy=severity_policy,
                )
                events.append(
                    {
                        "detected_at": run["started_at"],
                        "base_severity": base_severity,
                        "alert_level": level,
                        "breach_ratio": failure_ratio,
                        "severity_score": severity_score,
                        "event_type": "consecutive_failure",
                        "probe_id": probe_id,
                        "probe_name_cn": run["probe_name_cn"],
                        "provider": run["provider"],
                        "model_id": run["model_id"],
                        "region": run["region"],
                        "title": f"{run['probe_name_cn']} 连续失败",
                        "content": f"最近3次失败{failures_in_three}次，连续失败{consecutive_failures}次",
                        "consecutive_failures": consecutive_failures,
                        "failed_probes": probe_id,
                        "recommended_action": "检查接口状态、断言结果、网络和专用拨测Key",
                        "alert_state": "open",
                    }
                )
                state = new_state
            elif state in {"warning", "critical"} and consecutive_successes >= 2:
                events.append(
                    {
                        "detected_at": run["started_at"],
                        "base_severity": "recovery",
                        "alert_level": "recovery",
                        "breach_ratio": 0.0,
                        "severity_score": 0.0,
                        "event_type": "recovery",
                        "probe_id": probe_id,
                        "probe_name_cn": run["probe_name_cn"],
                        "provider": run["provider"],
                        "model_id": run["model_id"],
                        "region": run["region"],
                        "title": f"{run['probe_name_cn']} 已恢复",
                        "content": "连续2次拨测成功",
                        "consecutive_failures": 0,
                        "failed_probes": probe_id,
                        "recommended_action": "继续观察并关闭关联事件",
                        "alert_state": "recovered",
                    }
                )
                state = "normal"
            elif new_state == "observation" and state == "normal":
                state = "observation"
            elif state == "observation" and consecutive_successes >= 2:
                state = "normal"

    columns = [
        "probe_alert_id", "detected_at", "base_severity", "alert_level",
        "breach_ratio", "severity_score", "event_type", "probe_id",
        "probe_name_cn", "provider", "model_id", "region", "title", "content",
        "consecutive_failures", "failed_probes", "recommended_action", "alert_state",
    ]
    if not events:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(events).sort_values("detected_at").reset_index(drop=True)
    frame.insert(0, "probe_alert_id", [f"PALERT-{index:05d}" for index in range(1, len(frame) + 1)])
    return frame[columns]


def main() -> None:
    parser = argparse.ArgumentParser(description="生成主动拨测分级告警与恢复事件")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    runs = pd.read_csv(args.input, parse_dates=["started_at", "completed_at"])
    severity_policy = load_severity_policy(args.config)
    probe_frame = pd.read_excel(args.config, sheet_name="Active Probes")
    probe_severities = probe_frame.set_index("probe_id")["severity"].astype(str).str.lower().to_dict()
    events = detect_probe_events(runs, severity_policy, probe_severities)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"主动拨测事件：{args.output}（{len(events)} 条）")
    if not events.empty:
        print(events.groupby(["alert_level", "event_type"]).size().to_string())


if __name__ == "__main__":
    main()
