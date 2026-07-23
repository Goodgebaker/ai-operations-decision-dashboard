"""标准化模型能力拨测与对称评价数据生成。

能力拨测与现有可用性探针独立，所有启用模型使用相同任务、输入、执行频率、
环境标识和规则评测器，输出可直接用于后续模型能力画像与融合诊断。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

try:
    from .model_catalog import (
        MODEL_LATENCY,
        MODEL_PRICE,
        PROVIDER_ENDPOINT_ENV,
        SIMULATED_CAPABILITY,
        load_observed_calibration,
    )
except ImportError:  # 支持 ``python src/capability_calibration.py`` 直接运行。
    from model_catalog import (
        MODEL_LATENCY,
        MODEL_PRICE,
        PROVIDER_ENDPOINT_ENV,
        SIMULATED_CAPABILITY,
        load_observed_calibration,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "docs" / "ai_monitoring_metric_dictionary.xlsx"
DEFAULT_RUN_OUTPUT = PROJECT_ROOT / "data" / "capability_probe_runs.csv"
DEFAULT_SCORE_OUTPUT = PROJECT_ROOT / "outputs" / "model_capability_scores.csv"
ENVIRONMENT_ID = "calibration-standard-v1"

VALID_EVALUATORS = {"exact_match", "json_exact", "numeric_exact", "tool_name"}
DIMENSION_LATENCY_FACTOR = {
    "instruction_following": 0.90,
    "structured_output": 1.05,
    "reasoning": 1.20,
    "tool_call": 1.25,
}
def _text(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


@dataclass(frozen=True)
class CapabilityTask:
    task_id: str
    task_name_cn: str
    capability_dimension: str
    difficulty: str
    prompt_template: str
    evaluator_type: str
    expected_value: str
    pass_threshold: float
    task_weight: float
    interval_hours: int
    repeat_count: int
    max_tokens: int
    timeout_ms: int
    expected_output_version: str
    version: str


@dataclass(frozen=True)
class ModelTarget:
    model_id: str
    provider: str
    region: str
    dedicated_api_key_ref: str


def load_calibration_config(
    path: Path = DEFAULT_CONFIG,
) -> tuple[list[CapabilityTask], list[ModelTarget]]:
    task_frame = pd.read_excel(path, sheet_name="Capability Tasks")
    probe_frame = pd.read_excel(path, sheet_name="Active Probes")
    active_tasks = task_frame[
        task_frame["status"].astype(str).str.lower().eq("active")
    ].copy()
    if active_tasks.empty:
        raise ValueError("没有生效的标准能力任务")
    if active_tasks["task_id"].duplicated().any():
        raise ValueError("Capability Tasks 存在重复 task_id")

    tasks = [
        CapabilityTask(
            task_id=_text(row["task_id"]),
            task_name_cn=_text(row["task_name_cn"]),
            capability_dimension=_text(row["capability_dimension"]),
            difficulty=_text(row["difficulty"]),
            prompt_template=_text(row["prompt_template"]),
            evaluator_type=_text(row["evaluator_type"]),
            expected_value=_text(row["expected_value"]),
            pass_threshold=float(row["pass_threshold"]),
            task_weight=float(row["task_weight"]),
            interval_hours=int(row["interval_hours"]),
            repeat_count=int(row["repeat_count"]),
            max_tokens=int(row["max_tokens"]),
            timeout_ms=int(row["timeout_ms"]),
            expected_output_version=_text(row["expected_output_version"]),
            version=_text(row["version"]),
        )
        for _, row in active_tasks.iterrows()
    ]
    for task in tasks:
        validate_task(task)

    active_probes = probe_frame[
        probe_frame["status"].astype(str).str.lower().eq("active")
    ].copy()
    active_probes["availability_priority"] = active_probes["probe_type"].ne(
        "availability"
    )
    canonical = (
        active_probes.sort_values(
            ["model_id", "availability_priority", "probe_id"]
        )
        .groupby("model_id", as_index=False)
        .first()
    )
    targets = [
        ModelTarget(
            model_id=_text(row["model_id"]),
            provider=_text(row["provider"]),
            region=_text(row["region"]),
            dedicated_api_key_ref=_text(row["dedicated_api_key_ref"]),
        )
        for _, row in canonical.iterrows()
    ]
    if not targets:
        raise ValueError("Active Probes 中没有可用于能力校准的模型")
    validate_symmetric_definition(tasks, targets)
    return tasks, targets


def validate_task(task: CapabilityTask) -> None:
    if task.evaluator_type not in VALID_EVALUATORS:
        raise ValueError(
            f"{task.task_id} evaluator_type 必须为 {sorted(VALID_EVALUATORS)} 之一"
        )
    if not 0 <= task.pass_threshold <= 100:
        raise ValueError(f"{task.task_id} pass_threshold 必须处于0到100")
    if task.task_weight <= 0:
        raise ValueError(f"{task.task_id} task_weight 必须大于0")
    if task.interval_hours <= 0 or task.repeat_count <= 0:
        raise ValueError(f"{task.task_id} 执行频率和重复次数必须大于0")
    if task.max_tokens <= 0 or task.timeout_ms <= 0:
        raise ValueError(f"{task.task_id} max_tokens和timeout_ms必须大于0")


def validate_symmetric_definition(
    tasks: list[CapabilityTask], targets: list[ModelTarget]
) -> None:
    if not tasks or not targets:
        raise ValueError("标准任务和模型目标不能为空")
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("标准任务 task_id 必须唯一")
    if len({target.model_id for target in targets}) != len(targets):
        raise ValueError("能力校准模型 model_id 必须唯一")


def evaluate_task_result(
    task: CapabilityTask,
    response_text: str,
    function_name: str = "",
    status_code: int = 200,
) -> tuple[float, bool, str]:
    """使用确定性规则评测器返回分数、是否通过和评测证据。"""

    if not 200 <= int(status_code) <= 299:
        return 0.0, False, f"HTTP {status_code}"

    if task.evaluator_type == "exact_match":
        matched = response_text.strip() == task.expected_value.strip()
        score = 100.0 if matched else 0.0
        evidence = "严格文本匹配" if matched else "响应与期望文本不一致"
    elif task.evaluator_type == "json_exact":
        score, evidence = _score_json(response_text, task.expected_value)
    elif task.evaluator_type == "numeric_exact":
        score, evidence = _score_numeric(response_text, task.expected_value)
    else:
        matched = function_name.strip() == task.expected_value.strip()
        score = 100.0 if matched else 0.0
        evidence = "工具名称匹配" if matched else "工具名称不匹配"

    return score, score >= task.pass_threshold, evidence


def _score_json(response_text: str, expected_value: str) -> tuple[float, str]:
    try:
        actual = json.loads(response_text)
        expected = json.loads(expected_value)
    except (json.JSONDecodeError, TypeError):
        return 0.0, "响应不是合法JSON"
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        return (100.0, "JSON值完全匹配") if actual == expected else (0.0, "JSON值不匹配")
    if not expected:
        return 100.0, "期望JSON为空对象"
    matched = sum(actual.get(key) == value for key, value in expected.items())
    score = matched / len(expected) * 100.0
    return score, f"JSON字段匹配 {matched}/{len(expected)}"


def _score_numeric(response_text: str, expected_value: str) -> tuple[float, str]:
    match = re.search(r"-?\d+(?:\.\d+)?", response_text.replace(",", ""))
    if not match:
        return 0.0, "响应中没有可解析数值"
    try:
        actual = float(match.group(0))
        expected = float(expected_value)
    except ValueError:
        return 0.0, "期望值或响应数值无效"
    matched = abs(actual - expected) <= 1e-9
    return (100.0, "数值答案匹配") if matched else (0.0, "数值答案不匹配")


def simulate_history(
    tasks: list[CapabilityTask],
    targets: list[ModelTarget],
    start: pd.Timestamp,
    days: int,
    seed: int,
) -> pd.DataFrame:
    if days <= 0:
        raise ValueError("days 必须大于0")
    validate_symmetric_definition(tasks, targets)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    run_number = 0
    end = start + pd.Timedelta(days=days)

    for target in sorted(targets, key=lambda item: item.model_id):
        if target.model_id not in SIMULATED_CAPABILITY:
            raise ValueError(f"缺少 {target.model_id} 的模拟能力参数")
        for task in sorted(tasks, key=lambda item: item.task_id):
            timestamps = pd.date_range(
                start,
                end,
                freq=f"{task.interval_hours}h",
                inclusive="left",
            )
            for timestamp in timestamps:
                for repeat_index in range(1, task.repeat_count + 1):
                    run_number += 1
                    rows.append(
                        _simulate_one(
                            run_number,
                            task,
                            target,
                            timestamp + pd.Timedelta(seconds=repeat_index - 1),
                            repeat_index,
                            rng,
                            start,
                        )
                    )
    frame = pd.DataFrame(rows).sort_values(
        ["started_at", "task_id", "model_id", "repeat_index"]
    )
    validate_symmetric_results(frame)
    return frame.reset_index(drop=True)


def _simulate_one(
    run_number: int,
    task: CapabilityTask,
    target: ModelTarget,
    timestamp: pd.Timestamp,
    repeat_index: int,
    rng: np.random.Generator,
    simulation_start: pd.Timestamp,
) -> dict[str, object]:
    ability = SIMULATED_CAPABILITY[target.model_id][task.capability_dimension]
    passed_latent = rng.random() < ability
    status_code = 200
    error_type = ""

    provider_outage = (
        simulation_start + pd.Timedelta(days=19, hours=10)
        <= timestamp
        <= simulation_start + pd.Timedelta(days=19, hours=11, minutes=59, seconds=59)
        and target.provider == "Qwen"
    )
    congestion = (
        simulation_start + pd.Timedelta(days=15, hours=16)
        <= timestamp
        <= simulation_start + pd.Timedelta(days=15, hours=17, minutes=59, seconds=59)
        and target.model_id == "DeepSeek-V4"
    )
    transient = rng.random() < 0.001
    if provider_outage or transient:
        status_code = 503
        passed_latent = False
        error_type = "provider_unavailable" if provider_outage else "transient_network_error"

    response_text, function_name = _simulated_response(task, passed_latent)
    base_latency = MODEL_LATENCY[target.model_id] * DIMENSION_LATENCY_FACTOR[
        task.capability_dimension
    ]
    latency_ms = max(150, int(base_latency + rng.normal(0, 130)))
    if congestion:
        latency_ms *= 4
        error_type = "model_congestion"
    if status_code != 200:
        latency_ms = max(latency_ms, int(rng.uniform(3000, 6000)))
    ttft_ms = max(60, int(latency_ms * rng.uniform(0.26, 0.40)))
    task_score, passed, evidence = evaluate_task_result(
        task, response_text, function_name, status_code
    )
    input_tokens = max(4, len(task.prompt_template) // 2)
    output_tokens = max(1, (len(response_text) + len(function_name)) // 2)
    total_tokens = input_tokens + output_tokens

    return {
        "capability_run_id": f"CAPRUN-{run_number:07d}",
        "task_id": task.task_id,
        "task_name_cn": task.task_name_cn,
        "capability_dimension": task.capability_dimension,
        "difficulty": task.difficulty,
        "task_weight": task.task_weight,
        "model_id": target.model_id,
        "provider": target.provider,
        "region": target.region,
        "started_at": timestamp,
        "completed_at": timestamp + pd.Timedelta(milliseconds=latency_ms),
        "repeat_index": repeat_index,
        "status_code": status_code,
        "latency_ms": latency_ms,
        "ttft_ms": ttft_ms,
        "response_text": response_text,
        "function_name": function_name,
        "task_score": task_score,
        "passed": passed,
        "evaluation_evidence": evidence,
        "evaluator_type": task.evaluator_type,
        "evaluator_confidence": 100.0,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "estimated_cost": round(total_tokens * MODEL_PRICE[target.model_id], 7),
        "error_type": error_type,
        "environment_id": ENVIRONMENT_ID,
        "input_hash": hashlib.sha256(
            task.prompt_template.encode("utf-8")
        ).hexdigest()[:16],
        "expected_output_version": task.expected_output_version,
        "traffic_type": "capability_probe",
        "data_origin": "synthetic_assumption",
        "cost_origin": "synthetic_assumption",
        "config_version": task.version,
    }


def _simulated_response(task: CapabilityTask, passed: bool) -> tuple[str, str]:
    if task.evaluator_type == "tool_name":
        return "", task.expected_value if passed else "unknown_tool"
    if passed:
        return task.expected_value, ""
    if task.evaluator_type == "json_exact":
        return '{"status":"wrong"}', ""
    if task.evaluator_type == "numeric_exact":
        return "390", ""
    return "CALIBRATION_FAIL", ""


def validate_symmetric_results(runs: pd.DataFrame) -> None:
    required = {"model_id", "task_id", "started_at", "repeat_index"}
    missing = required.difference(runs.columns)
    if missing:
        raise ValueError(f"能力拨测结果缺少字段：{', '.join(sorted(missing))}")
    task_sets = runs.groupby("model_id")["task_id"].agg(lambda values: frozenset(values))
    if task_sets.empty or task_sets.nunique() != 1:
        raise ValueError("各模型覆盖的标准任务集合不一致")
    counts = runs.groupby(["model_id", "task_id"]).size().unstack(fill_value=0)
    if counts.nunique(axis=0).max() != 1:
        raise ValueError("各模型的标准任务样本数不一致")


def build_dimension_scores(runs: pd.DataFrame) -> pd.DataFrame:
    validate_symmetric_results(runs)
    data = runs.copy()
    data["weighted_score"] = data["task_score"] * data["task_weight"]

    def summarize(group: pd.DataFrame) -> pd.Series:
        total_weight = group["task_weight"].sum()
        quality_score = group["weighted_score"].sum() / total_weight
        score_std = group["task_score"].std(ddof=0)
        return pd.Series(
            {
                "run_count": len(group),
                "task_count": group["task_id"].nunique(),
                "pass_rate": group["passed"].astype(bool).mean() * 100,
                "quality_score": quality_score,
                "score_std": score_std,
                "consistency_score": max(0.0, 100.0 - score_std),
                "p50_latency_ms": group["latency_ms"].quantile(0.50),
                "p95_latency_ms": group["latency_ms"].quantile(0.95),
                "p95_ttft_ms": group["ttft_ms"].quantile(0.95),
                "estimated_cost": group["estimated_cost"].sum(),
                "latest_run_at": group["started_at"].max(),
            }
        )

    scores = (
        data.groupby(
            ["model_id", "provider", "capability_dimension"],
            as_index=False,
            group_keys=False,
        )
        .apply(summarize, include_groups=False)
        .reset_index()
    )
    numeric = scores.select_dtypes(include="number").columns
    scores[numeric] = scores[numeric].round(4)
    return scores.sort_values(["model_id", "capability_dimension"])


def _extract_openai_compatible(body: str) -> tuple[str, str]:
    try:
        payload = json.loads(body)
        message = payload.get("choices", [{}])[0].get("message", {})
        text = message.get("content") or ""
        tool_calls = message.get("tool_calls") or []
        function_name = (
            tool_calls[0].get("function", {}).get("name", "") if tool_calls else ""
        )
        return str(text), str(function_name)
    except (json.JSONDecodeError, AttributeError, IndexError, TypeError):
        return body[:500], ""


def run_live_task(
    task: CapabilityTask,
    target: ModelTarget,
    timestamp: pd.Timestamp | None = None,
    repeat_index: int = 1,
) -> dict[str, object]:
    endpoint = os.getenv(PROVIDER_ENDPOINT_ENV[target.provider], "")
    api_key = os.getenv(target.dedicated_api_key_ref, "")
    if not endpoint or not api_key:
        raise RuntimeError(
            f"{target.model_id} 缺少端点或专用拨测Key环境变量"
        )
    started_at = timestamp or pd.Timestamp.now().floor("s")
    payload: dict[str, object] = {
        "model": target.model_id,
        "messages": [{"role": "user", "content": task.prompt_template}],
        "max_tokens": task.max_tokens,
        "stream": False,
        "temperature": 0,
    }
    if task.evaluator_type == "tool_name":
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": task.expected_value,
                    "description": "标准能力校准工具",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]
        payload["tool_choice"] = "auto"

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    clock_start = time.perf_counter()
    status_code = 0
    response_body = ""
    error_type = ""
    try:
        with urllib.request.urlopen(
            request, timeout=task.timeout_ms / 1000
        ) as response:
            status_code = response.status
            first_byte = response.read(1)
            ttft_ms = int((time.perf_counter() - clock_start) * 1000)
            response_body = (first_byte + response.read()).decode(
                "utf-8", errors="replace"
            )
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
    task_score, passed, evidence = evaluate_task_result(
        task, response_text, function_name, status_code
    )
    input_tokens = max(1, len(task.prompt_template) // 2)
    output_tokens = max(0, (len(response_text) + len(function_name)) // 2)
    total_tokens = input_tokens + output_tokens
    return {
        "capability_run_id": (
            f"LIVE-{task.task_id}-{target.model_id}-R{repeat_index}-"
            f"{started_at:%Y%m%d%H%M%S}"
        ),
        "task_id": task.task_id,
        "task_name_cn": task.task_name_cn,
        "capability_dimension": task.capability_dimension,
        "difficulty": task.difficulty,
        "task_weight": task.task_weight,
        "model_id": target.model_id,
        "provider": target.provider,
        "region": target.region,
        "started_at": started_at,
        "completed_at": started_at + pd.Timedelta(milliseconds=latency_ms),
        "repeat_index": repeat_index,
        "status_code": status_code,
        "latency_ms": latency_ms,
        "ttft_ms": ttft_ms,
        "response_text": response_text,
        "function_name": function_name,
        "task_score": task_score,
        "passed": passed,
        "evaluation_evidence": evidence,
        "evaluator_type": task.evaluator_type,
        "evaluator_confidence": 100.0,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "estimated_cost": round(total_tokens * MODEL_PRICE[target.model_id], 7),
        "error_type": error_type,
        "environment_id": ENVIRONMENT_ID,
        "input_hash": hashlib.sha256(
            task.prompt_template.encode("utf-8")
        ).hexdigest()[:16],
        "expected_output_version": task.expected_output_version,
        "traffic_type": "capability_probe",
        "data_origin": "observed_probe",
        "cost_origin": "synthetic_assumption",
        "config_version": task.version,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="运行标准化模型能力校准任务")
    parser.add_argument("--mode", choices=["simulate", "live"], default="simulate")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-output", type=Path, default=DEFAULT_RUN_OUTPUT)
    parser.add_argument("--score-output", type=Path, default=DEFAULT_SCORE_OUTPUT)
    parser.add_argument("--start")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()

    tasks, targets = load_calibration_config(args.config)
    if args.mode == "simulate":
        _, _, _, latest_observed = load_observed_calibration()
        simulation_start = (
            pd.Timestamp(args.start)
            if args.start
            else latest_observed.normalize() - pd.Timedelta(days=args.days - 1)
            if latest_observed is not None
            else pd.Timestamp("2026-06-01 00:00:00")
        )
        runs = simulate_history(
            tasks, targets, simulation_start, args.days, args.seed
        )
    else:
        rows = [
            run_live_task(task, target, repeat_index=repeat_index)
            for target in targets
            for task in tasks
            for repeat_index in range(1, task.repeat_count + 1)
        ]
        runs = pd.DataFrame(rows)
        if args.run_output.exists():
            existing = pd.read_csv(
                args.run_output, parse_dates=["started_at", "completed_at"]
            )
            runs = pd.concat([existing, runs], ignore_index=True)
        validate_symmetric_results(runs)

    scores = build_dimension_scores(runs)
    args.run_output.parent.mkdir(parents=True, exist_ok=True)
    args.score_output.parent.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.run_output, index=False, encoding="utf-8-sig")
    scores.to_csv(args.score_output, index=False, encoding="utf-8-sig")
    print(f"能力拨测记录：{args.run_output}（{len(runs):,} 条）")
    print(f"能力维度评分：{args.score_output}（{len(scores):,} 行）")
    print(
        runs.groupby("model_id")["task_id"]
        .agg(["nunique", "count"])
        .rename(columns={"nunique": "任务数", "count": "样本数"})
        .to_string()
    )


if __name__ == "__main__":
    main()
