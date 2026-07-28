"""Cross-platform entry point for rebuilding calibrated synthetic demo data."""

from __future__ import annotations

import argparse
import os
import secrets
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RebuildStep:
    label: str
    command: tuple[str, ...]


class DemoRebuildError(RuntimeError):
    """Raised when one step of the synthetic-data rebuild fails."""


def create_seed() -> int:
    """Return a fresh seed that is accepted by NumPy's default RNG."""

    return secrets.randbelow(2_000_000_000) + 1


def build_rebuild_steps(seed: int, python_executable: str = sys.executable) -> tuple[RebuildStep, ...]:
    """Build the existing demo pipeline, varying every simulated source."""

    if seed <= 0:
        raise ValueError("seed 必须大于 0")

    def python_step(label: str, script: str, *args: str) -> RebuildStep:
        return RebuildStep(label, (python_executable, script, *args))

    return (
        python_step("生成基础模拟调用日志", "src/generate_sample_data.py", "--seed", str(seed)),
        python_step("计算基础监控指标", "src/calculate_metrics.py"),
        python_step("生成校准模拟日志与异常真值", "src/generate_synthetic_v2.py", "--days", "90", "--seed", str(seed + 1)),
        python_step("构建分钟级特征", "src/build_features.py"),
        python_step("运行复合异常规则", "src/composite_rule_engine.py"),
        python_step("评估异常检测模型", "src/model_benchmark.py"),
        python_step("融合异常判断", "src/fusion_rule_engine.py"),
        python_step("生成主动检查记录", "src/probe_runner.py", "--days", "90", "--seed", str(seed + 2)),
        python_step("检测主动检查异常", "src/detect_probe_alerts.py"),
        python_step("生成能力校准记录", "src/capability_calibration.py", "--days", "90", "--seed", str(seed + 3)),
        python_step("计算模型运营评分", "src/model_operations.py"),
        python_step("生成模型画像", "src/model_profile.py"),
        python_step("生成模型健康风险", "src/model_health_risk.py"),
        python_step("校验部署文件", "scripts/check_deployment.py"),
    )


def rebuild_demo_data(
    *,
    seed: int | None = None,
    project_root: Path = PROJECT_ROOT,
    python_executable: str = sys.executable,
    on_step: Callable[[int, int, str], None] | None = None,
) -> int:
    """Regenerate synthetic inputs and all of their derived dashboard artifacts.

    Real-source resource files and ``newdata`` are deliberately not part of this
    pipeline. The returned seed makes a generated dataset reproducible.
    """

    selected_seed = seed if seed is not None else create_seed()
    steps = build_rebuild_steps(selected_seed, python_executable)
    environment = os.environ.copy()
    environment.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

    for index, step in enumerate(steps, start=1):
        if on_step is not None:
            on_step(index, len(steps), step.label)
        result = subprocess.run(
            step.command,
            cwd=project_root,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "没有错误输出").strip()
            raise DemoRebuildError(f"{step.label}失败：{detail[-2000:]}")

    return selected_seed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="重新生成全部校准模拟数据与看板衍生产物")
    parser.add_argument("--seed", type=int, help="可选；用于复现某次生成结果")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    seed = rebuild_demo_data(
        seed=args.seed,
        on_step=lambda index, total, label: print(f"[{index}/{total}] {label}"),
    )
    print(f"模拟数据与衍生产物已全部重新生成；本次种子：{seed}")


if __name__ == "__main__":
    main()
