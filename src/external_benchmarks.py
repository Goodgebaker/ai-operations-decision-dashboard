"""标准化外部模型压测汇总，并生成可供路由参考的容量画像。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


SOURCE_NAME = "商业模型测试汇总"
SOURCE_VERSION = "0630版本"
DATA_QUALITY_STATUS = "external_summary_without_raw_runs"
REQUIRED_SOURCE_FIELDS = {
    "vendor",
    "model",
    "io",
    "concurrency",
    "ttft",
    "speed",
    "rate_limited",
}
REQUIRED_STANDARD_FIELDS = {
    "provider",
    "model_id",
    "io_profile",
    "input_tokens_target",
    "output_tokens_target",
    "concurrency",
    "ttft_ms",
    "total_output_tokens_per_second",
    "per_request_output_tokens_per_second",
    "rate_limited",
}


class BenchmarkDataError(ValueError):
    """外部压测数据缺失必要字段或存在无法解释的值。"""


def _parse_io_profile(value: object) -> tuple[int, int]:
    match = re.fullmatch(r"\s*(\d+)k\s*/\s*(\d+)k\s*", str(value), flags=re.IGNORECASE)
    if not match:
        raise BenchmarkDataError(f"无法解析输入输出档位：{value!r}")
    return int(match.group(1)) * 1_000, int(match.group(2)) * 1_000


def normalize_benchmark_records(
    records: list[dict[str, object]],
    *,
    source_file: str = "newdata/model_test_data.json",
) -> pd.DataFrame:
    """把来源 JSON 的汇总记录转换成含义明确、单位统一的标准表。"""
    if not records:
        return pd.DataFrame(columns=sorted(REQUIRED_STANDARD_FIELDS))

    missing = REQUIRED_SOURCE_FIELDS.difference(records[0])
    if missing:
        raise BenchmarkDataError(f"来源记录缺少字段：{', '.join(sorted(missing))}")

    normalized: list[dict[str, object]] = []
    for index, record in enumerate(records, start=1):
        row_missing = REQUIRED_SOURCE_FIELDS.difference(record)
        if row_missing:
            raise BenchmarkDataError(
                f"第 {index} 条来源记录缺少字段：{', '.join(sorted(row_missing))}"
            )
        input_tokens, output_tokens = _parse_io_profile(record["io"])
        concurrency = int(record["concurrency"])
        total_speed = pd.to_numeric(record["speed"], errors="coerce")
        ttft_seconds = pd.to_numeric(record["ttft"], errors="coerce")
        rate_limited = bool(record["rate_limited"])
        per_request_speed = (
            float(total_speed) / concurrency
            if pd.notna(total_speed) and concurrency > 0
            else float("nan")
        )
        normalized.append(
            {
                "benchmark_id": f"EXT-{index:04d}",
                "source_name": SOURCE_NAME,
                "source_version": SOURCE_VERSION,
                "provider": str(record["vendor"]).strip(),
                "model_id": str(record["model"]).strip(),
                "io_profile": str(record["io"]).strip(),
                "input_tokens_target": input_tokens,
                "output_tokens_target": output_tokens,
                "concurrency": concurrency,
                "ttft_ms": float(ttft_seconds) * 1_000 if pd.notna(ttft_seconds) else float("nan"),
                "total_output_tokens_per_second": (
                    float(total_speed) if pd.notna(total_speed) else float("nan")
                ),
                "per_request_output_tokens_per_second": per_request_speed,
                "rate_limited": rate_limited,
                "measured_at": pd.NA,
                "repeat_count": pd.NA,
                "aggregation_method": "unknown",
                "test_environment": "unknown",
                "source_file": source_file,
                "data_quality_status": DATA_QUALITY_STATUS,
            }
        )

    frame = pd.DataFrame(normalized)
    duplicate_key = ["provider", "model_id", "io_profile", "concurrency"]
    if frame.duplicated(duplicate_key).any():
        raise BenchmarkDataError("来源数据存在重复的供应商、模型、输入输出档位和并发组合。")
    return frame


def load_benchmark_json(path: str | Path) -> pd.DataFrame:
    """读取并标准化外部压测 JSON。"""
    source_path = Path(path)
    with source_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload.get("records")
    if not isinstance(records, list):
        raise BenchmarkDataError("来源 JSON 缺少 records 列表。")
    return normalize_benchmark_records(records, source_file=source_path.as_posix())


def build_capacity_profiles(benchmarks: pd.DataFrame) -> pd.DataFrame:
    """按供应商、模型和输入输出档位汇总稳定并发及观测到的 429 边界。"""
    if benchmarks.empty:
        return pd.DataFrame()
    missing = REQUIRED_STANDARD_FIELDS.difference(benchmarks.columns)
    if missing:
        raise BenchmarkDataError(f"标准压测表缺少字段：{', '.join(sorted(missing))}")

    profiles: list[dict[str, object]] = []
    group_columns = ["provider", "model_id", "io_profile", "input_tokens_target", "output_tokens_target"]
    for keys, group in benchmarks.groupby(group_columns, dropna=False, sort=True):
        provider, model_id, io_profile, input_tokens, output_tokens = keys
        group = group.sort_values("concurrency")
        valid = group[
            ~group["rate_limited"].astype(bool)
            & group["ttft_ms"].notna()
            & group["total_output_tokens_per_second"].notna()
        ]
        limited = group[group["rate_limited"].astype(bool)]
        tested_max = int(group["concurrency"].max())

        if valid.empty:
            max_stable = 0
            stable_row = None
        else:
            max_stable = int(valid["concurrency"].max())
            stable_row = (
                valid[valid["concurrency"].eq(max_stable)]
                .sort_values("ttft_ms")
                .iloc[0]
            )

        rate_limit_boundary = (
            int(limited["concurrency"].min()) if not limited.empty else pd.NA
        )
        if not limited.empty:
            capacity_state = "观测到 429"
        elif max_stable == tested_max and max_stable > 0:
            capacity_state = "测试范围内稳定"
        else:
            capacity_state = "有效结果不足"

        profiles.append(
            {
                "provider": provider,
                "model_id": model_id,
                "io_profile": io_profile,
                "input_tokens_target": int(input_tokens),
                "output_tokens_target": int(output_tokens),
                "tested_max_concurrency": tested_max,
                "max_stable_concurrency": max_stable,
                "stability_coverage_pct": round(max_stable / tested_max * 100, 1),
                "observed_rate_limit_concurrency": rate_limit_boundary,
                "rate_limit_observed": not limited.empty,
                "ttft_ms_at_max_stable": (
                    float(stable_row["ttft_ms"]) if stable_row is not None else float("nan")
                ),
                "total_output_tokens_per_second_at_max_stable": (
                    float(stable_row["total_output_tokens_per_second"])
                    if stable_row is not None
                    else float("nan")
                ),
                "per_request_output_tokens_per_second_at_max_stable": (
                    float(stable_row["per_request_output_tokens_per_second"])
                    if stable_row is not None
                    else float("nan")
                ),
                "capacity_state": capacity_state,
                "capacity_confidence": "低",
                "sample_count": int(len(group)),
            }
        )
    return pd.DataFrame(profiles)


def write_standardized_benchmarks(input_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    """标准化 JSON 并写入可移植的 UTF-8 CSV。"""
    frame = load_benchmark_json(input_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, encoding="utf-8")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="来源 model_test_data.json")
    parser.add_argument("--output", type=Path, required=True, help="标准化 CSV 输出路径")
    args = parser.parse_args()
    frame = write_standardized_benchmarks(args.input, args.output)
    print(f"已写入 {len(frame)} 条外部压测记录：{args.output}")


if __name__ == "__main__":
    main()
