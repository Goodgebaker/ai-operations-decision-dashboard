"""生成30天AI中台模拟日志与独立异常真值。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .model_catalog import (
        MODEL_IDS,
        MODEL_LATENCY,
        MODEL_PRICE,
        MODEL_PROVIDER,
        load_observed_calibration,
    )
except ImportError:  # 支持 ``python src/generate_synthetic_v2.py`` 直接运行。
    from model_catalog import (
        MODEL_IDS,
        MODEL_LATENCY,
        MODEL_PRICE,
        MODEL_PROVIDER,
        load_observed_calibration,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_OUTPUT = PROJECT_ROOT / "data" / "synthetic_logs_v2.csv"
DEFAULT_TRUTH_OUTPUT = PROJECT_ROOT / "data" / "ground_truth.csv"


@dataclass(frozen=True)
class GroundTruth:
    anomaly_id: str
    anomaly_type: str
    start_time: str
    end_time: str
    entity_type: str
    entity_value: str
    severity: str
    affected_metrics: str
    evaluation_granularity: str
    description: str


MODEL_PRICE_PER_TOKEN = MODEL_PRICE


def _build_timestamps(
    rng: np.random.Generator,
    start: pd.Timestamp,
    days: int,
    base_rpm: float,
) -> pd.DatetimeIndex:
    minutes = pd.date_range(start, periods=days * 24 * 60, freq="min")
    hours = minutes.hour.to_numpy()
    weekdays = minutes.dayofweek.to_numpy()

    hour_factor = np.select(
        [hours < 6, hours < 9, hours < 18, hours < 22],
        [0.30, 0.85, 1.45, 0.95],
        default=0.50,
    )
    weekday_factor = np.where(weekdays < 5, 1.0, 0.72)
    gradual_growth = np.linspace(0.92, 1.08, len(minutes))
    rates = base_rpm * hour_factor * weekday_factor * gradual_growth
    counts = rng.poisson(rates)

    repeated = np.repeat(minutes.to_numpy(), counts)
    seconds = rng.integers(0, 60, size=len(repeated))
    return pd.DatetimeIndex(repeated + pd.to_timedelta(seconds, unit="s"))


def _customer_ip(customer_id: str, pool_index: int) -> str:
    customer_number = int(customer_id[1:])
    return f"10.{customer_number}.{pool_index + 1}.{20 + customer_number}"


def _base_logs(
    rng: np.random.Generator,
    start: pd.Timestamp,
    days: int,
    base_rpm: float,
) -> pd.DataFrame:
    timestamps = _build_timestamps(rng, start, days, base_rpm)
    rows = len(timestamps)

    customers = np.array([f"C{i:03d}" for i in range(1, 11)])
    customer_ids = rng.choice(
        customers,
        size=rows,
        p=[0.18, 0.16, 0.14, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04],
    )
    key_number = rng.integers(1, 4, size=rows)
    api_keys = np.array(
        [
            f"key_{customer.lower()}_{number}"
            for customer, number in zip(customer_ids, key_number)
        ]
    )
    ip_pool_index = rng.choice([0, 1, 2, 3, 4], rows, p=[0.48, 0.25, 0.14, 0.08, 0.05])
    source_ips = np.array(
        [
            _customer_ip(customer, int(pool_index))
            for customer, pool_index in zip(customer_ids, ip_pool_index)
        ]
    )

    model_ids, model_weights, ttft_samples, _ = load_observed_calibration()
    request_models = rng.choice(model_ids, size=rows, p=model_weights)
    response_models = request_models.copy()
    fallback_mask = rng.random(rows) < 0.012
    fallback_models = rng.choice(model_ids, size=fallback_mask.sum(), p=model_weights)
    response_models[fallback_mask] = fallback_models
    providers = pd.Series(response_models).map(MODEL_PROVIDER).to_numpy()

    customer_scale = pd.Series(customer_ids).map(
        {customer: 0.8 + index * 0.06 for index, customer in enumerate(customers)}
    ).to_numpy()
    input_tokens = np.maximum(
        (rng.lognormal(6.0, 0.60, rows) * customer_scale).astype(int), 20
    )
    output_tokens = np.maximum(rng.lognormal(5.15, 0.58, rows).astype(int), 10)

    cache_hit = rng.random(rows) < 0.18
    cache_read_tokens = np.where(
        cache_hit,
        np.minimum(input_tokens, (input_tokens * rng.uniform(0.25, 0.75, rows)).astype(int)),
        0,
    )
    cache_creation_tokens = np.where(
        (~cache_hit) & (rng.random(rows) < 0.05),
        (input_tokens * rng.uniform(0.15, 0.45, rows)).astype(int),
        0,
    )
    reasoning_tokens = np.where(
        np.isin(response_models, ["DeepSeek-V4", "Minimax-M2.5"]),
        (output_tokens * rng.uniform(0.05, 0.28, rows)).astype(int),
        0,
    )

    queue_latency_ms = np.maximum(rng.gamma(2.0, 35.0, rows).astype(int), 1)
    observed_ttft_ms = np.array(
        [float(rng.choice(ttft_samples[model_id])) for model_id in response_models]
    )
    observed_ttft_ms *= rng.normal(1.0, 0.05, rows).clip(0.75, 1.30)
    model_latency = pd.Series(response_models).map(MODEL_LATENCY).to_numpy()
    first_token_latency_ms = np.maximum(
        (observed_ttft_ms + queue_latency_ms + rng.normal(0, 35, rows)).astype(int),
        60,
    )
    generation_ms = output_tokens * rng.uniform(2.0, 4.5, rows)
    latency_ms = np.maximum(
        (
            np.maximum(model_latency, first_token_latency_ms)
            + queue_latency_ms
            + generation_ms
            + rng.normal(0, 180, rows)
        ).astype(int),
        first_token_latency_ms + 30,
    )

    status_codes = rng.choice(
        [200, 400, 401, 429, 500, 503],
        size=rows,
        p=[0.952, 0.012, 0.004, 0.012, 0.012, 0.008],
    )
    retry_count = np.where(
        status_codes == 200,
        rng.choice([0, 1], rows, p=[0.97, 0.03]),
        rng.choice([0, 1, 2], rows, p=[0.70, 0.22, 0.08]),
    )

    quota_map = {
        customer: int(4_000_000 + index * 1_000_000)
        for index, customer in enumerate(customers)
    }
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "customer_id": customer_ids,
            "api_key": api_keys,
            "source_ip": source_ips,
            "operation_name": rng.choice(
                ["chat", "text_completion", "embeddings"],
                rows,
                p=[0.82, 0.12, 0.06],
            ),
            "request_model": request_models,
            "response_model": response_models,
            "model_id": response_models,
            "provider": providers,
            "channel_id": rng.choice(
                ["web", "app", "api"], rows, p=[0.18, 0.27, 0.55]
            ),
            "region": rng.choice(
                ["cn-east", "cn-north", "cn-south"], rows, p=[0.52, 0.28, 0.20]
            ),
            "stream": rng.random(rows) < 0.72,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_input_tokens": cache_read_tokens,
            "cache_creation_input_tokens": cache_creation_tokens,
            "reasoning_output_tokens": reasoning_tokens,
            "queue_latency_ms": queue_latency_ms,
            "first_token_latency_ms": first_token_latency_ms,
            "latency_ms": latency_ms,
            "status_code": status_codes,
            "retry_count": retry_count,
            "quota_limit": pd.Series(customer_ids).map(quota_map).to_numpy(),
            "safety_hit": rng.random(rows) < 0.012,
        }
    )


def _append_key_leak_burst(
    logs: pd.DataFrame,
    rng: np.random.Generator,
    start: pd.Timestamp,
) -> pd.DataFrame:
    burst_size = 120
    sampled = logs.sample(n=burst_size, replace=True, random_state=17).copy()
    sampled["timestamp"] = start + pd.to_timedelta(
        rng.integers(0, 10 * 60, burst_size), unit="s"
    )
    sampled["customer_id"] = "C007"
    sampled["api_key"] = "key_c007_2"
    suspicious_ips = [f"203.0.113.{value}" for value in range(80, 96)]
    sampled["source_ip"] = rng.choice(suspicious_ips, burst_size)
    sampled["channel_id"] = "api"
    sampled["region"] = rng.choice(["cn-north", "cn-south"], burst_size)
    sampled["status_code"] = rng.choice([200, 429], burst_size, p=[0.86, 0.14])
    sampled["retry_count"] = rng.choice([0, 1], burst_size, p=[0.75, 0.25])
    return pd.concat([logs, sampled], ignore_index=True)


def _inject_anomalies(
    logs: pd.DataFrame,
    rng: np.random.Generator,
    start: pd.Timestamp,
) -> tuple[pd.DataFrame, list[GroundTruth]]:
    truth: list[GroundTruth] = []

    key_start = start + pd.Timedelta(days=11, hours=3, minutes=15)
    logs = _append_key_leak_burst(logs, rng, key_start)
    truth.append(
        GroundTruth(
            "GT-001", "key_leak_suspected", str(key_start),
            str(key_start + pd.Timedelta(minutes=9, seconds=59)), "api_key",
            "key_c007_2", "critical",
            "request_count;distinct_ip_count;new_ip_ratio;off_hours",
            "minute", "非工作时间内单Key从大量新IP高频调用",
        )
    )

    congestion_start = start + pd.Timedelta(days=15, hours=16)
    congestion_end = congestion_start + pd.Timedelta(hours=1, minutes=59, seconds=59)
    mask = logs["timestamp"].between(congestion_start, congestion_end) & logs[
        "model_id"
    ].eq("DeepSeek-V4")
    logs.loc[mask, "queue_latency_ms"] *= 6
    logs.loc[mask, ["first_token_latency_ms", "latency_ms"]] *= 4
    slow_indices = logs.index[mask]
    if len(slow_indices):
        failed = rng.choice(slow_indices, size=max(1, len(slow_indices) // 5), replace=False)
        logs.loc[failed, "status_code"] = 503
    truth.append(
        GroundTruth(
            "GT-002", "model_congestion", str(congestion_start), str(congestion_end),
            "model", "DeepSeek-V4", "critical",
            "p95_latency_ms;queue_p95_ms;error_rate", "hour",
            "模型队列和响应时延同时升高并伴随部分503",
        )
    )

    outage_start = start + pd.Timedelta(days=19, hours=10)
    outage_end = outage_start + pd.Timedelta(hours=1, minutes=59, seconds=59)
    mask = logs["timestamp"].between(outage_start, outage_end) & logs[
        "provider"
    ].eq("Qwen")
    outage_indices = logs.index[mask]
    if len(outage_indices):
        failed = rng.choice(
            outage_indices, size=max(1, int(len(outage_indices) * 0.72)), replace=False
        )
        logs.loc[failed, "status_code"] = 503
        logs.loc[failed, "latency_ms"] *= 2
    truth.append(
        GroundTruth(
            "GT-003", "provider_outage", str(outage_start), str(outage_end),
            "provider", "Qwen", "critical",
            "error_rate;server_error_rate;p95_latency_ms", "hour",
            "单一供应商大面积503，其他供应商相对正常",
        )
    )

    token_start = start + pd.Timedelta(days=23, hours=14)
    token_end = token_start + pd.Timedelta(hours=1, minutes=59, seconds=59)
    mask = logs["timestamp"].between(token_start, token_end) & logs[
        "customer_id"
    ].eq("C003")
    logs.loc[mask, ["input_tokens", "output_tokens"]] *= 12
    logs.loc[mask, "reasoning_output_tokens"] *= 8
    truth.append(
        GroundTruth(
            "GT-004", "token_spike", str(token_start), str(token_end),
            "customer", "C003", "warning",
            "total_tokens;tokens_per_request;estimated_cost", "hour",
            "客户单次请求Token显著上升并带来成本突增",
        )
    )

    error_start = start + pd.Timedelta(days=26, hours=9)
    error_end = error_start + pd.Timedelta(minutes=59, seconds=59)
    mask = logs["timestamp"].between(error_start, error_end) & logs[
        "customer_id"
    ].eq("C002")
    error_indices = logs.index[mask]
    if len(error_indices):
        failed = rng.choice(
            error_indices, size=max(1, int(len(error_indices) * 0.65)), replace=False
        )
        logs.loc[failed, "status_code"] = 500
    truth.append(
        GroundTruth(
            "GT-005", "customer_error_burst", str(error_start), str(error_end),
            "customer", "C002", "critical",
            "error_rate;success_rate", "hour",
            "单个客户的500错误率突然升高",
        )
    )

    cost_start = start + pd.Timedelta(days=28, hours=1)
    cost_end = cost_start + pd.Timedelta(hours=2, minutes=59, seconds=59)
    mask = logs["timestamp"].between(cost_start, cost_end) & logs[
        "customer_id"
    ].eq("C004")
    logs.loc[mask, "output_tokens"] *= 9
    logs.loc[mask, "reasoning_output_tokens"] *= 9
    truth.append(
        GroundTruth(
            "GT-006", "cost_runaway", str(cost_start), str(cost_end),
            "customer", "C004", "warning",
            "tokens_per_request;output_input_ratio;estimated_cost", "hour",
            "调用量未明显变化，但输出Token和推理Token持续放大",
        )
    )

    return logs, truth


def build_dataset(
    days: int = 30,
    base_rpm: float = 1.35,
    seed: int = 20260714,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if days < 30:
        raise ValueError("为保证基线和全部异常场景存在，days必须至少为30")
    if base_rpm <= 0:
        raise ValueError("base_rpm必须大于0")

    rng = np.random.default_rng(seed)
    _, _, _, latest_observed = load_observed_calibration()
    start = (
        latest_observed.normalize() - pd.Timedelta(days=days - 1)
        if latest_observed is not None
        else pd.Timestamp("2026-06-01 00:00:00")
    )
    logs = _base_logs(rng, start, days, base_rpm)
    logs, truth = _inject_anomalies(logs, rng, start)

    logs["total_tokens"] = logs["input_tokens"] + logs["output_tokens"]
    logs["estimated_cost"] = [
        round(tokens * MODEL_PRICE_PER_TOKEN[model], 6)
        for tokens, model in zip(logs["total_tokens"], logs["response_model"])
    ]
    logs["data_origin"] = "synthetic_calibrated"
    logs["cost_origin"] = "synthetic_assumption"
    error_map = {
        400: "BAD_REQUEST",
        401: "UNAUTHORIZED",
        429: "RATE_LIMIT",
        500: "INTERNAL_ERROR",
        503: "SERVICE_UNAVAILABLE",
    }
    logs["error_code"] = logs["status_code"].map(error_map).fillna("")
    logs = logs.sort_values("timestamp").reset_index(drop=True)
    logs.insert(0, "request_id", [f"v2_req_{index:07d}" for index in range(1, len(logs) + 1)])

    ground_truth = pd.DataFrame([asdict(item) for item in truth])
    return logs, ground_truth


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成30天模拟日志及异常真值")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--base-rpm", type=float, default=1.35)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--logs-output", type=Path, default=DEFAULT_LOG_OUTPUT)
    parser.add_argument("--truth-output", type=Path, default=DEFAULT_TRUTH_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logs, ground_truth = build_dataset(args.days, args.base_rpm, args.seed)
    args.logs_output.parent.mkdir(parents=True, exist_ok=True)
    args.truth_output.parent.mkdir(parents=True, exist_ok=True)
    logs.to_csv(args.logs_output, index=False)
    ground_truth.to_csv(args.truth_output, index=False)

    print(f"生成日志：{len(logs):,} 条")
    print(f"时间范围：{logs['timestamp'].min()} 至 {logs['timestamp'].max()}")
    print(f"异常场景：{len(ground_truth)} 类")
    print(f"日志文件：{args.logs_output}")
    print(f"真值文件：{args.truth_output}")


if __name__ == "__main__":
    main()
