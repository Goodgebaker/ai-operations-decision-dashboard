"""一键导入每日三份真实资源工作簿并重建完整看板数据。"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAPACITY_OUTPUT = PROJECT_ROOT / "outputs" / "resource_capacity_daily.csv"


def _run(arguments: list[str]) -> None:
    print(f"\n> {' '.join(arguments)}", flush=True)
    subprocess.run(arguments, cwd=PROJECT_ROOT, check=True)


def _simulation_start() -> str:
    capacity = pd.read_csv(CAPACITY_OUTPUT, parse_dates=["date"])
    if capacity.empty:
        raise RuntimeError("资源容量汇总为空，无法确定模拟数据日期")
    latest = pd.Timestamp(capacity["date"].max()).normalize()
    return (latest - pd.Timedelta(days=29)).strftime("%Y-%m-%d 00:00:00")


def main() -> None:
    parser = argparse.ArgumentParser(description="导入每日资源数据并重建看板")
    parser.add_argument("--skip-import", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    python = sys.executable
    if not args.skip_import:
        _run([python, "-m", "src.resource_capacity"])
    start = _simulation_start()
    commands = [
        [python, "src/generate_sample_data.py"],
        [python, "src/calculate_metrics.py"],
        [python, "src/generate_synthetic_v2.py"],
        [python, "src/build_features.py"],
        [python, "src/composite_rule_engine.py"],
        [python, "src/model_benchmark.py"],
        [python, "src/fusion_rule_engine.py"],
        [python, "src/probe_runner.py", "--start", start, "--days", "30"],
        [python, "src/detect_probe_alerts.py"],
        [python, "src/capability_calibration.py", "--start", start, "--days", "30"],
        [python, "src/model_operations.py"],
        [python, "src/model_profile.py"],
        [python, "src/model_health_risk.py"],
        [python, "scripts/check_deployment.py"],
    ]
    if not args.skip_tests:
        commands.append([python, "-m", "unittest", "discover", "-s", "tests", "-v"])
    for command in commands:
        _run(command)
    print("\n每日数据更新、脱敏、看板重建和测试全部完成。")


if __name__ == "__main__":
    main()
