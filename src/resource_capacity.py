"""每日真实资源工作簿的校验、脱敏、历史追加与容量诊断。"""

from __future__ import annotations

import argparse
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
import re
import secrets
import shutil

import numpy as np
import pandas as pd

from .model_catalog import MODEL_BY_SOURCE_GROUP, canonical_model_id


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INBOX = PROJECT_ROOT / "newdata" / "01_每日三份Excel放这里"
DEFAULT_ARCHIVE = PROJECT_ROOT / "newdata" / "archive"
DEFAULT_SALT = PROJECT_ROOT / "newdata" / ".instance_salt"
DEFAULT_MODEL_OUTPUT = PROJECT_ROOT / "data" / "resource_model_timeseries.csv"
DEFAULT_INSTANCE_OUTPUT = PROJECT_ROOT / "data" / "resource_instance_hourly.csv"
DEFAULT_CAPACITY_OUTPUT = PROJECT_ROOT / "outputs" / "resource_capacity_daily.csv"
DEFAULT_AUDIT_OUTPUT = PROJECT_ROOT / "outputs" / "resource_import_audit.csv"

FILE_PATTERNS = {
    "detail": re.compile(r"^模型性能中间明细_(\d{8})\.xlsx$"),
    "busy": re.compile(r"^模型性能忙时对比_(\d{8})\.xlsx$"),
    "npu": re.compile(r"^NPU中间统计表_(\d{8})\.xlsx$"),
}


@dataclass(frozen=True)
class WorkbookBatch:
    source_date: pd.Timestamp
    detail: Path
    busy: Path
    npu: Path


def discover_batches(source_dir: Path) -> list[WorkbookBatch]:
    """按日期发现完整三表批次；发现残缺批次时直接拒绝。"""

    grouped: dict[str, dict[str, Path]] = {}
    for path in source_dir.glob("*.xlsx"):
        for kind, pattern in FILE_PATTERNS.items():
            match = pattern.match(path.name)
            if match:
                grouped.setdefault(match.group(1), {})[kind] = path
                break
    if not grouped:
        raise FileNotFoundError(f"{source_dir} 中没有符合命名规则的每日 Excel")

    incomplete = {
        date: sorted(set(FILE_PATTERNS).difference(files))
        for date, files in grouped.items()
        if set(files) != set(FILE_PATTERNS)
    }
    if incomplete:
        details = "；".join(f"{date} 缺少 {','.join(kinds)}" for date, kinds in incomplete.items())
        raise ValueError(f"每日文件批次不完整：{details}")

    return [
        WorkbookBatch(
            source_date=pd.to_datetime(date, format="%Y%m%d"),
            detail=files["detail"],
            busy=files["busy"],
            npu=files["npu"],
        )
        for date, files in sorted(grouped.items())
    ]


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{label} 缺少字段：{', '.join(sorted(missing))}")


def _salt(path: Path) -> bytes:
    if path.exists():
        value = path.read_text(encoding="ascii").strip()
        if len(value) < 32:
            raise ValueError("实例脱敏盐文件无效")
        return bytes.fromhex(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = secrets.token_bytes(32)
    path.write_text(value.hex(), encoding="ascii")
    return value


def _anonymous_instance(model_id: str, raw_ip: object, salt: bytes) -> str:
    digest = hashlib.sha256(salt + str(raw_ip).strip().encode("utf-8")).hexdigest()[:10]
    prefix = {
        "DeepSeek-V4": "DSV4",
        "Minimax-M2.5": "MM25",
        "Qwen3.6-35B-A3B": "QW36",
    }[model_id]
    return f"{prefix}-{digest}"


def _validate_source_date(timestamps: pd.Series, source_date: pd.Timestamp) -> None:
    observed = pd.to_datetime(timestamps, errors="coerce")
    if observed.isna().any():
        raise ValueError("中间明细包含无法解析的时间点")
    dates = observed.dt.normalize().drop_duplicates()
    if len(dates) != 1 or dates.iloc[0] != source_date:
        raise ValueError("文件名日期与中间明细时间点不一致")


def _model_timeseries(detail: pd.DataFrame, source_date: pd.Timestamp) -> pd.DataFrame:
    required = {
        "模型组", "instance/IP", "时间点", "running", "waiting", "ttft",
        "tokens_s", "npu_usage",
    }
    _require_columns(detail, required, "模型性能中间明细")
    _validate_source_date(detail["时间点"], source_date)
    paid = detail[detail["模型组"].isin(MODEL_BY_SOURCE_GROUP)].copy()
    if paid.empty:
        raise ValueError("中间明细中没有三个受监控商用模型")
    duplicate_key = ["模型组", "instance/IP", "时间点"]
    if paid.duplicated(duplicate_key).any():
        raise ValueError("中间明细存在重复的模型、实例、时间点")

    metric_columns = ["running", "waiting", "ttft", "tokens_s"]
    distinct = paid.groupby(["模型组", "时间点"])[metric_columns].nunique(dropna=False)
    if distinct.gt(1).any().any():
        raise ValueError("同一模型时间点在不同实例上的模型级指标不一致")

    model = paid.drop_duplicates(["模型组", "时间点"])[
        ["模型组", "时间点", *metric_columns]
    ].copy()
    model["model_id"] = model["模型组"].map(canonical_model_id)
    model["timestamp"] = pd.to_datetime(model["时间点"])
    model["date"] = model["timestamp"].dt.normalize()
    model["ttft_ms"] = pd.to_numeric(model["ttft"], errors="coerce") * 1000
    model["data_origin"] = "observed"
    model["source_date"] = source_date
    return model[
        [
            "timestamp", "date", "source_date", "model_id", "running", "waiting",
            "ttft_ms", "tokens_s", "data_origin",
        ]
    ].rename(columns={"tokens_s": "tokens_per_second"})


def _instance_hourly(
    detail: pd.DataFrame,
    source_date: pd.Timestamp,
    salt: bytes,
) -> pd.DataFrame:
    paid = detail[detail["模型组"].isin(MODEL_BY_SOURCE_GROUP)].copy()
    paid["model_id"] = paid["模型组"].map(canonical_model_id)
    paid["instance_id"] = [
        _anonymous_instance(model_id, raw_ip, salt)
        for model_id, raw_ip in zip(paid["model_id"], paid["instance/IP"], strict=False)
    ]
    paid["hour"] = pd.to_datetime(paid["时间点"]).dt.floor("h")
    paid["npu_usage"] = pd.to_numeric(paid["npu_usage"], errors="coerce")
    hourly = (
        paid.groupby(["hour", "model_id", "instance_id"], as_index=False)
        .agg(
            npu_mean=("npu_usage", "mean"),
            npu_p95=("npu_usage", lambda values: values.quantile(0.95)),
            npu_max=("npu_usage", "max"),
            high_npu_samples=("npu_usage", lambda values: int(values.ge(70).sum())),
            observed_samples=("npu_usage", "count"),
        )
    )
    hourly["date"] = source_date
    hourly["data_origin"] = "observed_anonymized"
    return _round_numeric(hourly)


def _capacity_daily(
    busy_path: Path,
    npu_path: Path,
    model_series: pd.DataFrame,
    instance_hourly: pd.DataFrame,
    source_date: pd.Timestamp,
) -> pd.DataFrame:
    busy = pd.read_excel(busy_path, sheet_name="忙时对比", header=1)
    _require_columns(
        busy,
        {
            "模型组", "实例数量", "State-running", "State-waiting", "TTFT",
            "Token/s每秒吞吐量", "Cache%", "NPU", "HBM%",
        },
        "模型性能忙时对比",
    )
    busy = busy[busy["模型组"].isin(MODEL_BY_SOURCE_GROUP)].copy()
    if set(busy["模型组"]) != set(MODEL_BY_SOURCE_GROUP):
        raise ValueError("忙时对比未覆盖三个受监控商用模型")
    busy["model_id"] = busy["模型组"].map(canonical_model_id)

    npu = pd.read_excel(npu_path, sheet_name="NPU中间统计")
    _require_columns(npu, {"model", "instance/ip", "时段1峰值平均值"}, "NPU中间统计表")
    npu = npu[npu["model"].isin(MODEL_BY_SOURCE_GROUP)].copy()
    npu["model_id"] = npu["model"].map(canonical_model_id)
    npu_summary = (
        npu.groupby("model_id", as_index=False)
        .agg(
            npu_peak_average_mean=("时段1峰值平均值", "mean"),
            npu_summary_instance_count=("instance/ip", "nunique"),
        )
    )

    detail_summary = (
        instance_hourly.groupby("model_id", as_index=False)
        .agg(
            instance_count=("instance_id", "nunique"),
            npu_mean=("npu_mean", "mean"),
            npu_p95=("npu_p95", "max"),
            npu_max=("npu_max", "max"),
            high_npu_samples=("high_npu_samples", "sum"),
        )
    )
    model_summary = (
        model_series.groupby("model_id", as_index=False)
        .agg(
            running_mean=("running", "mean"),
            running_max_detail=("running", "max"),
            waiting_max_detail=("waiting", "max"),
            ttft_mean_ms=("ttft_ms", "mean"),
            ttft_p95_ms=("ttft_ms", lambda values: values.quantile(0.95)),
            tokens_per_second_mean=("tokens_per_second", "mean"),
        )
    )
    result = (
        busy.rename(
            columns={
                "实例数量": "instance_count_busy",
                "State-running": "running_max_busy",
                "State-waiting": "waiting_max_busy",
                "TTFT": "ttft_busy_s",
                "Token/s每秒吞吐量": "tokens_per_second_busy",
                "Cache%": "cache_pct",
                "NPU": "npu_busy_pct",
                "HBM%": "hbm_pct",
            }
        )
        .merge(detail_summary, on="model_id", validate="one_to_one")
        .merge(model_summary, on="model_id", validate="one_to_one")
        .merge(npu_summary, on="model_id", validate="one_to_one")
    )
    if not result["instance_count"].eq(result["instance_count_busy"]).all():
        raise ValueError("中间明细实例数与忙时对比不一致")
    if not result["instance_count"].eq(result["npu_summary_instance_count"]).all():
        raise ValueError("中间明细实例数与 NPU 统计表不一致")

    result["date"] = source_date
    result["concurrency_ratio"] = (
        result["running_max_busy"] / result["instance_count"].replace(0, np.nan)
    )
    result["hbm_headroom_pct"] = 100 - result["hbm_pct"]
    result["baseline_ready"] = False
    result["capacity_state"] = result.apply(_capacity_state, axis=1)
    result["diagnosis"] = result.apply(_capacity_diagnosis, axis=1)
    result["data_origin"] = "observed"
    columns = [
        "date", "model_id", "instance_count", "running_mean", "running_max_busy",
        "waiting_max_busy", "concurrency_ratio", "ttft_mean_ms", "ttft_p95_ms",
        "tokens_per_second_mean", "cache_pct", "npu_busy_pct", "npu_mean",
        "npu_p95", "npu_max", "npu_peak_average_mean", "high_npu_samples",
        "hbm_pct", "hbm_headroom_pct", "baseline_ready", "capacity_state",
        "diagnosis", "data_origin",
    ]
    return _round_numeric(result[columns].sort_values("model_id"))


def _capacity_state(row: pd.Series) -> str:
    if float(row["waiting_max_busy"]) > 0 or float(row["hbm_pct"]) >= 95:
        return "容量风险"
    if (
        float(row["concurrency_ratio"]) >= 0.8
        or float(row["npu_p95"]) >= 70
        or float(row["hbm_pct"]) >= 90
    ):
        return "需要关注"
    return "容量充足"


def _capacity_diagnosis(row: pd.Series) -> str:
    signals: list[str] = []
    if float(row["waiting_max_busy"]) > 0:
        signals.append("出现等待队列，需评估扩容或分流")
    if float(row["concurrency_ratio"]) >= 0.8:
        signals.append("忙时并发接近实例数量")
    if float(row["npu_p95"]) >= 70:
        signals.append("存在 NPU 高负载尖峰")
    if float(row["hbm_pct"]) >= 90 and float(row["npu_mean"]) < 30:
        signals.append("平均 NPU 较低但 HBM 很高，更像模型驻留内存压力")
    elif float(row["hbm_pct"]) >= 90:
        signals.append("HBM 余量不足")
    if not signals:
        signals.append("当前未发现明确资源瓶颈")
    return "；".join(signals)


def process_batch(
    batch: WorkbookBatch,
    salt_path: Path = DEFAULT_SALT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    detail = pd.read_excel(batch.detail, sheet_name="中间明细")
    salt = _salt(salt_path)
    model = _model_timeseries(detail, batch.source_date)
    instance = _instance_hourly(detail, batch.source_date, salt)
    capacity = _capacity_daily(
        batch.busy, batch.npu, model, instance, batch.source_date
    )
    audit = {
        "source_date": batch.source_date,
        "detail_file": batch.detail.name,
        "busy_file": batch.busy.name,
        "npu_file": batch.npu.name,
        "source_checksum": _batch_checksum(batch),
        "model_timeseries_rows": len(model),
        "instance_hourly_rows": len(instance),
        "capacity_rows": len(capacity),
        "excluded_platform_rows": int(detail["模型组"].eq("中台模型").sum()),
        "status": "success",
    }
    return model, instance, capacity, audit


def _batch_checksum(batch: WorkbookBatch) -> str:
    digest = hashlib.sha256()
    for path in (batch.detail, batch.busy, batch.npu):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _replace_dates(existing: pd.DataFrame, incoming: pd.DataFrame, date_column: str) -> pd.DataFrame:
    if existing.empty:
        return incoming.copy()
    current = existing.copy()
    current[date_column] = pd.to_datetime(current[date_column])
    dates = pd.to_datetime(incoming[date_column]).dt.normalize().unique()
    current = current[~current[date_column].dt.normalize().isin(dates)]
    return pd.concat([current, incoming], ignore_index=True)


def _read_existing(path: Path, date_columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=date_columns)


def _atomic_csv(frame: pd.DataFrame, path: Path, sort_columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.sort_values(sort_columns).reset_index(drop=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    output.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def import_batches(
    source_dir: Path = DEFAULT_INBOX,
    archive_dir: Path = DEFAULT_ARCHIVE,
    archive: bool = True,
) -> list[dict[str, object]]:
    batches = discover_batches(source_dir)
    processed: list[tuple[WorkbookBatch, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]] = []
    for batch in batches:
        model, instance, capacity, audit = process_batch(batch)
        processed.append((batch, model, instance, capacity, audit))

    model_all = pd.concat([item[1] for item in processed], ignore_index=True)
    instance_all = pd.concat([item[2] for item in processed], ignore_index=True)
    capacity_all = pd.concat([item[3] for item in processed], ignore_index=True)
    audit_all = pd.DataFrame([item[4] for item in processed])

    model_history = _replace_dates(
        _read_existing(DEFAULT_MODEL_OUTPUT, ["timestamp", "date", "source_date"]),
        model_all,
        "source_date",
    )
    instance_history = _replace_dates(
        _read_existing(DEFAULT_INSTANCE_OUTPUT, ["hour", "date"]),
        instance_all,
        "date",
    )
    capacity_history = _replace_dates(
        _read_existing(DEFAULT_CAPACITY_OUTPUT, ["date"]), capacity_all, "date"
    )
    capacity_history["observed_days"] = capacity_history.groupby("model_id")[
        "date"
    ].transform("nunique")
    capacity_history["baseline_ready"] = capacity_history["observed_days"].ge(7)
    audit_history = _replace_dates(
        _read_existing(DEFAULT_AUDIT_OUTPUT, ["source_date"]), audit_all, "source_date"
    )

    _atomic_csv(model_history, DEFAULT_MODEL_OUTPUT, ["timestamp", "model_id"])
    _atomic_csv(instance_history, DEFAULT_INSTANCE_OUTPUT, ["hour", "model_id", "instance_id"])
    _atomic_csv(capacity_history, DEFAULT_CAPACITY_OUTPUT, ["date", "model_id"])
    _atomic_csv(audit_history, DEFAULT_AUDIT_OUTPUT, ["source_date"])

    if archive:
        for batch, *_ in processed:
            target = archive_dir / batch.source_date.strftime("%Y%m%d")
            target.mkdir(parents=True, exist_ok=True)
            for source in (batch.detail, batch.busy, batch.npu):
                destination = target / source.name
                if destination.exists():
                    checksum = hashlib.sha256(source.read_bytes()).hexdigest()[:8]
                    destination = target / f"{source.stem}_{checksum}{source.suffix}"
                shutil.move(str(source), str(destination))
    return [item[4] for item in processed]


def _round_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    numeric = result.select_dtypes(include="number").columns
    result[numeric] = result[numeric].round(4)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="导入每日真实资源工作簿并生成脱敏看板数据")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_INBOX)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--no-archive", action="store_true")
    args = parser.parse_args()
    audits = import_batches(args.source_dir, args.archive_dir, not args.no_archive)
    for audit in audits:
        print(
            f"已导入 {pd.Timestamp(audit['source_date']):%Y-%m-%d}："
            f"模型时序 {audit['model_timeseries_rows']:,} 行，"
            f"实例小时 {audit['instance_hourly_rows']:,} 行"
        )


if __name__ == "__main__":
    main()
