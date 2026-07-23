"""生成用于 AI 中台运营监控练习的模拟调用日志。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .model_catalog import MODEL_IDS, MODEL_PRICE, MODEL_PROVIDER
except ImportError:  # 支持 ``python src/generate_sample_data.py`` 直接运行。
    from model_catalog import MODEL_IDS, MODEL_PRICE, MODEL_PROVIDER


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "sample_logs.csv"


def build_sample_logs(rows: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """生成包含正常业务波动及人为异常的数据集。"""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-07-01 00:00:00")
    duration_seconds = 7 * 24 * 60 * 60

    timestamps = start + pd.to_timedelta(
        rng.integers(0, duration_seconds, size=rows), unit="s"
    )

    customers = np.array([f"C{i:03d}" for i in range(1, 11)])
    customer_ids = rng.choice(
        customers,
        size=rows,
        p=[0.18, 0.16, 0.14, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04],
    )
    key_number = rng.integers(1, 4, size=rows)
    api_keys = np.array(
        [f"key_{customer.lower()}_{number}" for customer, number in zip(customer_ids, key_number)]
    )

    models = rng.choice(
        MODEL_IDS,
        size=rows,
        p=[0.93, 0.04, 0.03],
    )
    providers = pd.Series(models).map(MODEL_PROVIDER).to_numpy()

    input_tokens = np.maximum(rng.lognormal(6.0, 0.65, rows).astype(int), 20)
    output_tokens = np.maximum(rng.lognormal(5.2, 0.60, rows).astype(int), 10)
    total_tokens = input_tokens + output_tokens

    first_token_latency_ms = np.maximum(rng.normal(480, 130, rows).astype(int), 80)
    latency_ms = np.maximum(
        first_token_latency_ms + rng.normal(1_000, 300, rows).astype(int),
        first_token_latency_ms + 50,
    )

    status_codes = rng.choice(
        [200, 400, 401, 429, 500, 503],
        size=rows,
        p=[0.94, 0.015, 0.005, 0.015, 0.015, 0.01],
    )

    data = pd.DataFrame(
        {
            "request_id": [f"req_{i:06d}" for i in range(1, rows + 1)],
            "timestamp": timestamps,
            "customer_id": customer_ids,
            "api_key": api_keys,
            "source_ip": [f"10.{a}.{b}.{c}" for a, b, c in rng.integers(1, 255, (rows, 3))],
            "model_id": models,
            "provider": providers,
            "channel_id": rng.choice(["web", "app", "api"], rows, p=[0.2, 0.25, 0.55]),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "first_token_latency_ms": first_token_latency_ms,
            "latency_ms": latency_ms,
            "status_code": status_codes,
        }
    )

    # 人为注入三类异常，便于后续验证规则是否能被解释。
    token_spike = (
        data["timestamp"].between("2026-07-07 14:00:00", "2026-07-07 14:59:59")
        & data["customer_id"].eq("C003")
    )
    # 只影响一个客户，因此放大到 20 倍，确保在全平台小时汇总中也清晰可见。
    data.loc[token_spike, ["input_tokens", "output_tokens", "total_tokens"]] *= 20

    error_spike = data["timestamp"].between(
        "2026-07-06 10:00:00", "2026-07-06 10:59:59"
    )
    error_indices = data.index[error_spike]
    if len(error_indices):
        selected = rng.choice(error_indices, size=max(1, len(error_indices) // 2), replace=False)
        data.loc[selected, "status_code"] = 500

    latency_spike = (
        data["timestamp"].between("2026-07-05 16:00:00", "2026-07-05 16:59:59")
        & data["model_id"].eq("DeepSeek-V4")
    )
    data.loc[latency_spike, ["first_token_latency_ms", "latency_ms"]] *= 4

    # 模拟某个 Key 在一分钟内集中发起大量请求。
    burst_indices = rng.choice(data.index, size=min(40, rows), replace=False)
    burst_seconds = rng.integers(0, 60, size=len(burst_indices))
    data.loc[burst_indices, "timestamp"] = pd.Timestamp("2026-07-04 03:15:00") + pd.to_timedelta(
        burst_seconds, unit="s"
    )
    data.loc[burst_indices, "customer_id"] = "C007"
    data.loc[burst_indices, "api_key"] = "key_c007_2"
    data.loc[burst_indices, "source_ip"] = "203.0.113.77"

    error_map = {400: "BAD_REQUEST", 401: "UNAUTHORIZED", 429: "RATE_LIMIT", 500: "INTERNAL_ERROR", 503: "SERVICE_UNAVAILABLE"}
    data["error_code"] = data["status_code"].map(error_map).fillna("")
    data["estimated_cost"] = [
        round(tokens * MODEL_PRICE[model_id], 6)
        for tokens, model_id in zip(data["total_tokens"], data["model_id"], strict=False)
    ]
    data["data_origin"] = "synthetic_calibrated"
    data["cost_origin"] = "synthetic_assumption"
    data["safety_hit"] = rng.random(rows) < 0.012

    return data.sort_values("timestamp").reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 AI 调用模拟日志")
    parser.add_argument("--rows", type=int, default=10_000, help="生成行数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出 CSV 路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rows <= 0:
        raise ValueError("--rows 必须大于 0")

    logs = build_sample_logs(rows=args.rows, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    logs.to_csv(args.output, index=False)

    print(f"已生成 {len(logs):,} 条模拟日志")
    print(f"时间范围：{logs['timestamp'].min()} 至 {logs['timestamp'].max()}")
    print(f"文件位置：{args.output}")
    print("已注入：Token 突增、错误率升高、模型时延升高、单 Key 高频调用")


if __name__ == "__main__":
    main()
