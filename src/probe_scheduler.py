"""轻量主动拨测调度器；生产环境可替换为Celery、Airflow或Kubernetes CronJob。"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from probe_runner import (
    DEFAULT_CONFIG,
    DEFAULT_HOURLY,
    DEFAULT_OUTPUT,
    build_hourly_metrics,
    load_probe_config,
    run_live_probe,
)


def append_rows(path: Path, rows: list[dict[str, object]]) -> pd.DataFrame:
    current = pd.read_csv(path) if path.exists() else pd.DataFrame()
    combined = pd.concat([current, pd.DataFrame(rows)], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False, encoding="utf-8-sig")
    return combined


def run_once(config_path: Path, output_path: Path, hourly_path: Path) -> None:
    configs, assertions = load_probe_config(config_path)
    rows = [run_live_probe(config, assertions) for config in configs]
    combined = append_rows(output_path, rows)
    metrics = build_hourly_metrics(combined)
    hourly_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(hourly_path, index=False, encoding="utf-8-sig")
    print(f"本轮完成 {len(rows)} 个拨测任务")


def run_loop(config_path: Path, output_path: Path, hourly_path: Path) -> None:
    configs, assertions = load_probe_config(config_path)
    next_due = {config.probe_id: pd.Timestamp.min for config in configs}
    print("主动拨测调度器已启动，按 Ctrl+C 停止")
    while True:
        now = pd.Timestamp.now().floor("s")
        due = [config for config in configs if now >= next_due[config.probe_id]]
        if due:
            rows = []
            for config in due:
                rows.append(run_live_probe(config, assertions, now))
                next_due[config.probe_id] = now + pd.Timedelta(
                    minutes=config.interval_minutes
                )
            combined = append_rows(output_path, rows)
            build_hourly_metrics(combined).to_csv(
                hourly_path, index=False, encoding="utf-8-sig"
            )
            print(f"{now}: 完成 {len(rows)} 个拨测任务")
        time.sleep(15)


def main() -> None:
    parser = argparse.ArgumentParser(description="定时执行真实主动拨测")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--hourly-output", type=Path, default=DEFAULT_HOURLY)
    parser.add_argument("--once", action="store_true", help="只执行一轮")
    args = parser.parse_args()
    if args.once:
        run_once(args.config, args.output, args.hourly_output)
    else:
        run_loop(args.config, args.output, args.hourly_output)


if __name__ == "__main__":
    main()
