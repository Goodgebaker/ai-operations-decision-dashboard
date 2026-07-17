"""读取模拟日志，计算基础监控指标并生成第一份实验图表。"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "sample_logs.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"

REQUIRED_COLUMNS = {
    "request_id",
    "timestamp",
    "model_id",
    "total_tokens",
    "latency_ms",
    "status_code",
}


def load_logs(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"找不到 {path}，请先运行 python src/generate_sample_data.py"
        )

    logs = pd.read_csv(path, parse_dates=["timestamp"])
    missing = REQUIRED_COLUMNS.difference(logs.columns)
    if missing:
        raise ValueError(f"日志缺少字段：{', '.join(sorted(missing))}")

    logs["is_success"] = logs["status_code"].between(200, 299)
    logs["hour"] = logs["timestamp"].dt.floor("h")
    return logs


def calculate_hourly_metrics(logs: pd.DataFrame) -> pd.DataFrame:
    hourly = (
        logs.groupby("hour", as_index=False)
        .agg(
            request_count=("request_id", "count"),
            total_tokens=("total_tokens", "sum"),
            success_rate=("is_success", "mean"),
            p95_latency_ms=("latency_ms", lambda values: values.quantile(0.95)),
        )
        .sort_values("hour")
    )
    hourly["success_rate"] = (hourly["success_rate"] * 100).round(2)
    hourly["p95_latency_ms"] = hourly["p95_latency_ms"].round(0).astype(int)
    return hourly


def calculate_model_metrics(logs: pd.DataFrame) -> pd.DataFrame:
    model_metrics = (
        logs.groupby("model_id", as_index=False)
        .agg(
            request_count=("request_id", "count"),
            total_tokens=("total_tokens", "sum"),
            success_rate=("is_success", "mean"),
            p95_latency_ms=("latency_ms", lambda values: values.quantile(0.95)),
        )
        .sort_values("request_count", ascending=False)
    )
    model_metrics["success_rate"] = (model_metrics["success_rate"] * 100).round(2)
    model_metrics["p95_latency_ms"] = model_metrics["p95_latency_ms"].round(0).astype(int)
    return model_metrics


def build_report(hourly: pd.DataFrame, model_metrics: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("每小时调用量", "每小时 Token 消耗", "各模型成功率", "各模型 P95 时延"),
        vertical_spacing=0.17,
    )
    fig.add_trace(
        go.Scatter(x=hourly["hour"], y=hourly["request_count"], mode="lines", name="调用量"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=hourly["hour"], y=hourly["total_tokens"], mode="lines", name="Token"),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(x=model_metrics["model_id"], y=model_metrics["success_rate"], name="成功率"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=model_metrics["model_id"], y=model_metrics["p95_latency_ms"], name="P95 时延"),
        row=2,
        col=2,
    )
    fig.update_yaxes(title_text="请求数", row=1, col=1)
    fig.update_yaxes(title_text="Token", row=1, col=2)
    fig.update_yaxes(title_text="成功率（%）", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="毫秒", row=2, col=2)
    fig.update_layout(
        title="AI 中台调用日志：第一次指标实验",
        template="plotly_white",
        height=760,
        showlegend=False,
    )
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="计算 AI 调用日志基础指标")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="输入日志 CSV")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="结果目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logs = load_logs(args.input)
    hourly = calculate_hourly_metrics(logs)
    model_metrics = calculate_model_metrics(logs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    hourly_path = args.output_dir / "hourly_metrics.csv"
    model_path = args.output_dir / "model_metrics.csv"
    report_path = args.output_dir / "first_experiment_report.html"

    hourly.to_csv(hourly_path, index=False)
    model_metrics.to_csv(model_path, index=False)
    build_report(hourly, model_metrics).write_html(report_path, include_plotlyjs=True)

    print(f"读取日志：{len(logs):,} 条")
    print(f"总体成功率：{logs['is_success'].mean():.2%}")
    print(f"总 Token：{logs['total_tokens'].sum():,}")
    print("\n各模型指标：")
    print(model_metrics.to_string(index=False))
    print(f"\n小时指标：{hourly_path}")
    print(f"模型指标：{model_path}")
    print(f"交互图表：{report_path}")


if __name__ == "__main__":
    main()
