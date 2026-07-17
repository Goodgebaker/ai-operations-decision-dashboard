"""配置驱动的主动拨测执行器，支持历史模拟与OpenAI兼容接口实拨。"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "docs" / "ai_monitoring_metric_dictionary.xlsx"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "probe_runs.csv"
DEFAULT_HOURLY = PROJECT_ROOT / "outputs" / "probe_hourly_metrics.csv"

MODEL_LATENCY = {"gpt-4.1-mini": 1350, "qwen-plus": 1150, "deepseek-chat": 1050}
MODEL_PRICE = {"gpt-4.1-mini": 0.0000024, "qwen-plus": 0.0000016, "deepseek-chat": 0.0000012}
PROVIDER_ENDPOINT_ENV = {
    "OpenAI": "PROBE_ENDPOINT_OPENAI",
    "Alibaba Cloud": "PROBE_ENDPOINT_ALIBABA_CLOUD",
    "DeepSeek": "PROBE_ENDPOINT_DEEPSEEK",
}


def _text(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


@dataclass(frozen=True)
class ProbeConfig:
    probe_id: str
    probe_name: str
    probe_name_cn: str
    probe_type: str
    provider: str
    model_id: str
    region: str
    interval_minutes: int
    prompt_template: str
    expected_status: int
    timeout_ms: int
    max_ttft_ms: int
    max_latency_ms: int
    max_tokens: int
    stream: bool
    dedicated_api_key_ref: str
    severity: str
    cooldown_minutes: int
    version: str


def load_probe_config(path: Path = DEFAULT_CONFIG) -> tuple[list[ProbeConfig], pd.DataFrame]:
    probes = pd.read_excel(path, sheet_name="Active Probes")
    assertions = pd.read_excel(path, sheet_name="Probe Assertions")
    active = probes[probes["status"].astype(str).str.lower().eq("active")].copy()
    if active.empty:
        raise ValueError("没有生效的主动拨测任务")
    if active["probe_id"].duplicated().any():
        raise ValueError("主动拨测存在重复 probe_id")
    configs = [
        ProbeConfig(
            probe_id=_text(row["probe_id"]),
            probe_name=_text(row["probe_name"]),
            probe_name_cn=_text(row["probe_name_cn"]),
            probe_type=_text(row["probe_type"]),
            provider=_text(row["provider"]),
            model_id=_text(row["model_id"]),
            region=_text(row["region"]),
            interval_minutes=int(row["interval_minutes"]),
            prompt_template=_text(row["prompt_template"]),
            expected_status=int(row["expected_status"]),
            timeout_ms=int(row["timeout_ms"]),
            max_ttft_ms=int(row["max_ttft_ms"]),
            max_latency_ms=int(row["max_latency_ms"]),
            max_tokens=int(row["max_tokens"]),
            stream=bool(row["stream"]),
            dedicated_api_key_ref=_text(row["dedicated_api_key_ref"]),
            severity=_text(row["severity"]).lower(),
            cooldown_minutes=int(row["cooldown_minutes"]),
            version=_text(row["version"]),
        )
        for _, row in active.iterrows()
    ]
    assertions = assertions[
        assertions["status"].astype(str).str.lower().eq("active")
    ].copy()
    return configs, assertions


def _relevant_assertions(assertions: pd.DataFrame, probe_id: str) -> pd.DataFrame:
    return assertions[assertions["probe_id"].astype(str).isin([probe_id, "*"])]


def evaluate_assertions(
    config: ProbeConfig,
    assertions: pd.DataFrame,
    result: dict[str, object],
) -> tuple[bool, list[str]]:
    failed: list[str] = []
    for _, assertion in _relevant_assertions(assertions, config.probe_id).iterrows():
        assertion_type = _text(assertion["assertion_type"])
        field = _text(assertion["field_name"])
        operator = _text(assertion["comparison_operator"])
        expected = _text(assertion["expected_value"])

        if expected == "configured_max_ttft_ms":
            expected_value: object = config.max_ttft_ms
        elif expected == "configured_max_latency_ms":
            expected_value = config.max_latency_ms
        elif expected == "configured_expected_status":
            expected_value = config.expected_status
        else:
            expected_value = expected

        if assertion_type == "json":
            try:
                actual = json.loads(str(result.get("response_text", ""))).get(field)
            except (json.JSONDecodeError, AttributeError):
                actual = None
        else:
            actual = result.get(field)

        passed = False
        if operator == "contains":
            passed = str(expected_value).lower() in str(actual or "").lower()
        elif operator == "eq":
            passed = str(actual).lower() == str(expected_value).lower()
        elif operator in {"lte", "gte"}:
            try:
                left, right = float(actual), float(expected_value)
                passed = left <= right if operator == "lte" else left >= right
            except (TypeError, ValueError):
                passed = False
        if not passed:
            failed.append(_text(assertion["assertion_id"]))
    return not failed, failed


def _base_simulated_result(
    config: ProbeConfig,
    timestamp: pd.Timestamp,
    rng: np.random.Generator,
) -> dict[str, object]:
    type_factor = {"availability": 0.78, "json_schema": 1.05, "stream_ttft": 0.92, "tool_call": 1.25}[config.probe_type]
    region_factor = {"cn-east": 1.0, "cn-north": 1.08, "cn-south": 1.12}[config.region]
    latency = max(120, int((MODEL_LATENCY[config.model_id] * type_factor * region_factor) + rng.normal(0, 120)))
    ttft = max(60, int(latency * rng.uniform(0.25, 0.42)))
    status_code = 200
    error_type = ""
    response_text = "pong"
    function_name = ""
    if config.probe_type == "json_schema":
        response_text = '{"ok": true}'
    elif config.probe_type == "stream_ttft":
        response_text = "2"
    elif config.probe_type == "tool_call":
        response_text = ""
        function_name = "get_weather"

    congestion = (
        pd.Timestamp("2026-06-16 16:00") <= timestamp <= pd.Timestamp("2026-06-16 17:59:59")
        and config.model_id == "gpt-4.1-mini"
    )
    outage = (
        pd.Timestamp("2026-06-20 10:00") <= timestamp <= pd.Timestamp("2026-06-20 11:59:59")
        and config.provider == "Alibaba Cloud"
    )
    transient = (
        pd.Timestamp("2026-06-10 04:00") <= timestamp <= pd.Timestamp("2026-06-10 04:29:59")
        and config.probe_id == "PROBE-006"
    )
    if congestion:
        latency *= 4
        ttft *= 3
        error_type = "model_congestion"
    if outage:
        status_code = 503
        latency = int(rng.uniform(3200, 6500))
        ttft = latency
        response_text = ""
        error_type = "provider_unavailable"
    if transient:
        function_name = ""
        error_type = "tool_assertion_failed"
    if rng.random() < 0.0008 and not (congestion or outage or transient):
        status_code = 503
        response_text = ""
        error_type = "transient_network_error"

    input_tokens = max(4, len(config.prompt_template) // 2)
    output_tokens = int(rng.integers(2, max(3, min(config.max_tokens, 12))))
    completed_at = timestamp + pd.Timedelta(milliseconds=latency)
    return {
        "started_at": timestamp,
        "completed_at": completed_at,
        "status_code": status_code,
        "latency_ms": latency,
        "ttft_ms": ttft,
        "response_text": response_text,
        "function_name": function_name,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "error_type": error_type,
    }


def simulate_history(
    configs: list[ProbeConfig],
    assertions: pd.DataFrame,
    start: pd.Timestamp,
    days: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    run_number = 0
    end = start + pd.Timedelta(days=days)
    for config in configs:
        for timestamp in pd.date_range(
            start, end, freq=f"{config.interval_minutes}min", inclusive="left"
        ):
            run_number += 1
            result = _base_simulated_result(config, timestamp, rng)
            assertion_passed, failed_assertions = evaluate_assertions(
                config, assertions, result
            )
            success = (
                int(result["status_code"]) == config.expected_status
                and assertion_passed
            )
            total_tokens = int(result["input_tokens"]) + int(result["output_tokens"])
            rows.append(
                {
                    "probe_run_id": f"PRUN-{run_number:07d}",
                    "probe_id": config.probe_id,
                    "probe_name_cn": config.probe_name_cn,
                    "probe_type": config.probe_type,
                    "provider": config.provider,
                    "model_id": config.model_id,
                    "region": config.region,
                    **result,
                    "total_tokens": total_tokens,
                    "estimated_cost": round(total_tokens * MODEL_PRICE[config.model_id], 7),
                    "assertion_passed": assertion_passed,
                    "failed_assertions": ";".join(failed_assertions),
                    "success": success,
                    "traffic_type": "probe",
                    "config_version": config.version,
                }
            )
    return pd.DataFrame(rows).sort_values(["started_at", "probe_id"])


def _extract_openai_compatible(body: str) -> tuple[str, str]:
    try:
        payload = json.loads(body)
        message = payload.get("choices", [{}])[0].get("message", {})
        text = message.get("content") or ""
        tool_calls = message.get("tool_calls") or []
        function_name = tool_calls[0].get("function", {}).get("name", "") if tool_calls else ""
        return str(text), str(function_name)
    except (json.JSONDecodeError, AttributeError, IndexError, TypeError):
        return body[:500], ""


def run_live_probe(
    config: ProbeConfig,
    assertions: pd.DataFrame,
    timestamp: pd.Timestamp | None = None,
) -> dict[str, object]:
    endpoint_env = PROVIDER_ENDPOINT_ENV[config.provider]
    endpoint = os.getenv(endpoint_env, "")
    api_key = os.getenv(config.dedicated_api_key_ref, "")
    if not endpoint or not api_key:
        raise RuntimeError(
            f"{config.probe_id} 缺少环境变量 {endpoint_env} 或 {config.dedicated_api_key_ref}"
        )
    started_at = timestamp or pd.Timestamp.now().floor("s")
    payload = json.dumps(
        {
            "model": config.model_id,
            "messages": [{"role": "user", "content": config.prompt_template}],
            "max_tokens": config.max_tokens,
            "stream": False,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    clock_start = time.perf_counter()
    status_code = 0
    response_body = ""
    error_type = ""
    try:
        with urllib.request.urlopen(request, timeout=config.timeout_ms / 1000) as response:
            status_code = response.status
            first_byte = response.read(1)
            ttft_ms = int((time.perf_counter() - clock_start) * 1000)
            response_body = (first_byte + response.read()).decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        status_code = exc.code
        ttft_ms = int((time.perf_counter() - clock_start) * 1000)
        response_body = exc.read().decode("utf-8", errors="replace")
        error_type = "http_error"
    except (urllib.error.URLError, TimeoutError) as exc:
        ttft_ms = int((time.perf_counter() - clock_start) * 1000)
        error_type = type(exc).__name__
    latency_ms = int((time.perf_counter() - clock_start) * 1000)
    response_text, function_name = _extract_openai_compatible(response_body)
    result = {
        "started_at": started_at,
        "completed_at": started_at + pd.Timedelta(milliseconds=latency_ms),
        "status_code": status_code,
        "latency_ms": latency_ms,
        "ttft_ms": ttft_ms,
        "response_text": response_text,
        "function_name": function_name,
        "input_tokens": max(1, len(config.prompt_template) // 2),
        "output_tokens": max(0, len(response_text) // 2),
        "error_type": error_type,
    }
    assertion_passed, failed = evaluate_assertions(config, assertions, result)
    total_tokens = int(result["input_tokens"]) + int(result["output_tokens"])
    return {
        "probe_run_id": f"LIVE-{config.probe_id}-{started_at.strftime('%Y%m%d%H%M%S')}",
        "probe_id": config.probe_id,
        "probe_name_cn": config.probe_name_cn,
        "probe_type": config.probe_type,
        "provider": config.provider,
        "model_id": config.model_id,
        "region": config.region,
        **result,
        "total_tokens": total_tokens,
        "estimated_cost": round(total_tokens * MODEL_PRICE[config.model_id], 7),
        "assertion_passed": assertion_passed,
        "failed_assertions": ";".join(failed),
        "success": status_code == config.expected_status and assertion_passed,
        "traffic_type": "probe",
        "config_version": config.version,
    }


def build_hourly_metrics(runs: pd.DataFrame) -> pd.DataFrame:
    data = runs.copy()
    data["started_at"] = pd.to_datetime(data["started_at"])
    data["hour"] = data["started_at"].dt.floor("h")
    metrics = (
        data.groupby(
            ["hour", "probe_id", "probe_name_cn", "provider", "model_id", "region"],
            as_index=False,
        )
        .agg(
            run_count=("probe_run_id", "count"),
            success_rate=("success", "mean"),
            failure_count=("success", lambda values: int((~values.astype(bool)).sum())),
            p95_latency_ms=("latency_ms", lambda values: values.quantile(0.95)),
            p95_ttft_ms=("ttft_ms", lambda values: values.quantile(0.95)),
            estimated_cost=("estimated_cost", "sum"),
        )
        .sort_values(["hour", "probe_id"])
    )
    metrics["success_rate"] = (metrics["success_rate"] * 100).round(2)
    metrics[["p95_latency_ms", "p95_ttft_ms", "estimated_cost"]] = metrics[
        ["p95_latency_ms", "p95_ttft_ms", "estimated_cost"]
    ].round(4)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="运行主动拨测")
    parser.add_argument("--mode", choices=["simulate", "live"], default="simulate")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--hourly-output", type=Path, default=DEFAULT_HOURLY)
    parser.add_argument("--start", default="2026-06-01 00:00:00")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()

    configs, assertions = load_probe_config(args.config)
    if args.mode == "simulate":
        runs = simulate_history(
            configs, assertions, pd.Timestamp(args.start), args.days, args.seed
        )
    else:
        rows = [run_live_probe(config, assertions) for config in configs]
        runs = pd.DataFrame(rows)
        if args.output.exists():
            existing = pd.read_csv(args.output, parse_dates=["started_at", "completed_at"])
            runs = pd.concat([existing, runs], ignore_index=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.hourly_output.parent.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.output, index=False, encoding="utf-8-sig")
    hourly = build_hourly_metrics(runs)
    hourly.to_csv(args.hourly_output, index=False, encoding="utf-8-sig")
    print(f"拨测运行记录：{args.output}（{len(runs):,} 条）")
    print(f"拨测小时指标：{args.hourly_output}（{len(hourly):,} 行）")
    print(f"成功率：{runs['success'].mean() * 100:.2f}%")


if __name__ == "__main__":
    main()
