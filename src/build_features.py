"""从v2模拟日志构建平台、客户、模型、供应商和Key特征表。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "synthetic_logs_v2.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "features"


def load_logs(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"找不到 {path}，请先运行 python src/generate_synthetic_v2.py"
        )
    logs = pd.read_csv(path, parse_dates=["timestamp"])
    required = {
        "request_id", "timestamp", "customer_id", "api_key", "source_ip",
        "model_id", "provider", "input_tokens", "output_tokens", "total_tokens",
        "estimated_cost", "latency_ms", "first_token_latency_ms",
        "queue_latency_ms", "status_code", "retry_count", "safety_hit",
        "cache_read_input_tokens", "reasoning_output_tokens", "request_model",
        "response_model",
    }
    missing = required.difference(logs.columns)
    if missing:
        raise ValueError(f"v2日志缺少字段：{', '.join(sorted(missing))}")

    logs["is_success"] = logs["status_code"].between(200, 299)
    logs["is_error"] = ~logs["is_success"]
    logs["is_rate_limited"] = logs["status_code"].eq(429)
    logs["is_server_error"] = logs["status_code"].between(500, 599)
    logs["has_retry"] = logs["retry_count"].gt(0)
    logs["has_cache_read"] = logs["cache_read_input_tokens"].gt(0)
    logs["is_fallback"] = logs["request_model"].ne(logs["response_model"])
    logs["hour"] = logs["timestamp"].dt.floor("h")
    logs["minute"] = logs["timestamp"].dt.floor("min")
    return logs


def build_hourly_features(
    logs: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    grouping = ["hour", *group_columns]
    features = (
        logs.groupby(grouping, as_index=False)
        .agg(
            request_count=("request_id", "count"),
            total_tokens=("total_tokens", "sum"),
            input_tokens=("input_tokens", "sum"),
            output_tokens=("output_tokens", "sum"),
            estimated_cost=("estimated_cost", "sum"),
            success_rate=("is_success", "mean"),
            error_rate=("is_error", "mean"),
            rate_limit_rate=("is_rate_limited", "mean"),
            server_error_rate=("is_server_error", "mean"),
            p50_latency_ms=("latency_ms", lambda values: values.quantile(0.50)),
            p95_latency_ms=("latency_ms", lambda values: values.quantile(0.95)),
            p99_latency_ms=("latency_ms", lambda values: values.quantile(0.99)),
            p95_first_token_latency_ms=(
                "first_token_latency_ms", lambda values: values.quantile(0.95)
            ),
            queue_p95_ms=("queue_latency_ms", lambda values: values.quantile(0.95)),
            active_customers=("customer_id", "nunique"),
            active_keys=("api_key", "nunique"),
            distinct_ips=("source_ip", "nunique"),
            safety_hit_rate=("safety_hit", "mean"),
            retry_rate=("has_retry", "mean"),
            cache_hit_rate=("has_cache_read", "mean"),
            fallback_rate=("is_fallback", "mean"),
            reasoning_tokens=("reasoning_output_tokens", "sum"),
        )
        .sort_values(grouping)
    )

    features["tokens_per_request"] = (
        features["total_tokens"] / features["request_count"]
    )
    features["output_input_ratio"] = (
        features["output_tokens"] / features["input_tokens"].replace(0, np.nan)
    )
    features["cost_per_request"] = (
        features["estimated_cost"] / features["request_count"]
    )
    successful_requests = features["request_count"] * features["success_rate"]
    features["cost_per_success"] = (
        features["estimated_cost"] / successful_requests.replace(0, np.nan)
    )
    features["reasoning_token_share"] = (
        features["reasoning_tokens"] / features["output_tokens"].replace(0, np.nan)
    )

    percentage_columns = [
        "success_rate", "error_rate", "rate_limit_rate", "server_error_rate",
        "safety_hit_rate", "retry_rate", "cache_hit_rate", "fallback_rate",
        "reasoning_token_share",
    ]
    features[percentage_columns] = features[percentage_columns] * 100
    numeric_columns = features.select_dtypes(include="number").columns
    features[numeric_columns] = features[numeric_columns].round(4)
    return features


def build_key_minute_features(logs: pd.DataFrame) -> pd.DataFrame:
    data = logs.copy()
    first_seen = data.groupby(["api_key", "source_ip"])["timestamp"].transform("min")
    data["is_new_ip"] = first_seen.dt.floor("min").eq(data["minute"])

    key_features = (
        data.groupby(["minute", "api_key"], as_index=False)
        .agg(
            customer_id=("customer_id", "first"),
            request_count=("request_id", "count"),
            distinct_ip_count=("source_ip", "nunique"),
            new_ip_count=("is_new_ip", "sum"),
            total_tokens=("total_tokens", "sum"),
            error_rate=("is_error", "mean"),
            rate_limit_rate=("is_rate_limited", "mean"),
            estimated_cost=("estimated_cost", "sum"),
            avg_latency_ms=("latency_ms", "mean"),
        )
        .sort_values(["api_key", "minute"])
    )
    key_features["new_ip_ratio"] = (
        key_features["new_ip_count"] / key_features["distinct_ip_count"].replace(0, np.nan)
    )
    key_features["tokens_per_request"] = (
        key_features["total_tokens"] / key_features["request_count"]
    )
    key_features["off_hours"] = (
        key_features["minute"].dt.hour.lt(6)
        | key_features["minute"].dt.hour.ge(23)
    ).astype(int)

    ip_counts = (
        data.groupby(["minute", "api_key", "source_ip"])
        .size()
        .rename("ip_requests")
        .reset_index()
    )

    def entropy(values: pd.Series) -> float:
        probabilities = values / values.sum()
        return float(-(probabilities * np.log2(probabilities)).sum())

    entropies = (
        ip_counts.groupby(["minute", "api_key"])["ip_requests"]
        .apply(entropy)
        .rename("ip_entropy")
        .reset_index()
    )
    key_features = key_features.merge(entropies, on=["minute", "api_key"], how="left")
    key_features[["error_rate", "rate_limit_rate", "new_ip_ratio"]] *= 100
    numeric_columns = key_features.select_dtypes(include="number").columns
    key_features[numeric_columns] = key_features[numeric_columns].round(4)
    return key_features


def build_all_features(logs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "hourly_features.csv": build_hourly_features(logs, []),
        "customer_hourly_features.csv": build_hourly_features(logs, ["customer_id"]),
        "model_hourly_features.csv": build_hourly_features(logs, ["model_id"]),
        "provider_hourly_features.csv": build_hourly_features(logs, ["provider"]),
        "key_minute_features.csv": build_key_minute_features(logs),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建v2监控特征表")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logs = load_logs(args.input)
    feature_sets = build_all_features(logs)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"读取日志：{len(logs):,} 条")
    for filename, frame in feature_sets.items():
        output_path = args.output_dir / filename
        frame.to_csv(output_path, index=False)
        print(f"{filename}: {len(frame):,} 行 × {len(frame.columns)} 列")


if __name__ == "__main__":
    main()
