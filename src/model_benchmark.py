"""比较复合规则、滚动 MAD、STL 残差和 Isolation Forest 的检测效果。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = PROJECT_ROOT / "outputs" / "features" / "hourly_features.csv"
DEFAULT_TRUTH = PROJECT_ROOT / "data" / "ground_truth.csv"
DEFAULT_COMPOSITE_ALERTS = PROJECT_ROOT / "outputs" / "composite_alerts.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "benchmark"

MAD_METRICS = [
    "request_count",
    "total_tokens",
    "error_rate",
    "server_error_rate",
    "p95_latency_ms",
    "tokens_per_request",
    "output_input_ratio",
]
STL_METRICS = [
    "request_count",
    "total_tokens",
    "error_rate",
    "p95_latency_ms",
    "tokens_per_request",
]
ISOLATION_FEATURES = [
    "request_count",
    "total_tokens",
    "estimated_cost",
    "error_rate",
    "rate_limit_rate",
    "server_error_rate",
    "p95_latency_ms",
    "p99_latency_ms",
    "p95_first_token_latency_ms",
    "queue_p95_ms",
    "retry_rate",
    "fallback_rate",
    "tokens_per_request",
    "output_input_ratio",
    "cost_per_request",
    "reasoning_token_share",
]


def load_inputs(
    feature_path: Path,
    truth_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = pd.read_csv(feature_path, parse_dates=["hour"]).sort_values("hour")
    truth = pd.read_csv(truth_path, parse_dates=["start_time", "end_time"])
    if features["hour"].duplicated().any():
        raise ValueError("平台小时特征存在重复 hour")
    required_features = set(MAD_METRICS + STL_METRICS + ISOLATION_FEATURES)
    missing = required_features.difference(features.columns)
    if missing:
        raise ValueError("小时特征缺少字段：" + ", ".join(sorted(missing)))
    return features.reset_index(drop=True), truth


def attach_ground_truth(features: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    result = features.copy()
    truth_ids: list[str] = []
    truth_types: list[str] = []
    for hour in result["hour"]:
        hour_end = hour + pd.Timedelta(hours=1) - pd.Timedelta(microseconds=1)
        overlapping = truth[
            (truth["start_time"] <= hour_end) & (truth["end_time"] >= hour)
        ]
        truth_ids.append(";".join(overlapping["anomaly_id"].astype(str)))
        truth_types.append(";".join(overlapping["anomaly_type"].astype(str)))
    result["truth_ids"] = truth_ids
    result["truth_types"] = truth_types
    result["is_ground_truth"] = result["truth_ids"].ne("")
    return result


def rolling_mad_scores(frame: pd.DataFrame, window: int = 24) -> pd.DataFrame:
    """仅使用过去窗口计算鲁棒 Z 分数，适合流式实现。"""
    metric_scores: list[pd.Series] = []
    for metric in MAD_METRICS:
        values = pd.to_numeric(frame[metric], errors="coerce")
        baseline = values.shift(1).rolling(window, min_periods=12).median()
        absolute_deviation = (values.shift(1) - baseline).abs()
        mad = absolute_deviation.rolling(window, min_periods=12).median()
        fallback = values.shift(1).rolling(window, min_periods=12).std() / 1.4826
        scale = mad.where(mad > 1e-9, fallback).replace(0, np.nan)
        score = 0.6745 * (values - baseline).abs() / scale
        metric_scores.append(score.rename(metric))
    scores = pd.concat(metric_scores, axis=1)
    top_metric = scores.fillna(-np.inf).idxmax(axis=1)
    top_metric = top_metric.mask(scores.isna().all(axis=1), "insufficient_history")
    return pd.DataFrame(
        {
            "score_mad": scores.max(axis=1, skipna=True).fillna(0),
            "top_metric_mad": top_metric,
        }
    )


def stl_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """离线实验基线；STL 使用完整序列，不应直接当作线上无泄漏实现。"""
    metric_scores: list[pd.Series] = []
    for metric in STL_METRICS:
        values = pd.to_numeric(frame[metric], errors="coerce").interpolate().bfill().ffill()
        residual = pd.Series(
            STL(values, period=24, robust=True).fit().resid,
            index=frame.index,
        )
        center = residual.median()
        mad = (residual - center).abs().median()
        if mad <= 1e-9:
            mad = residual.std() / 1.4826
        score = 0.6745 * (residual - center).abs() / max(float(mad), 1e-9)
        metric_scores.append(score.rename(metric))
    scores = pd.concat(metric_scores, axis=1)
    return pd.DataFrame(
        {
            "score_stl": scores.max(axis=1, skipna=True).fillna(0),
            "top_metric_stl": scores.idxmax(axis=1).fillna("unknown"),
        }
    )


def isolation_forest_scores(frame: pd.DataFrame, train_days: int = 10) -> pd.DataFrame:
    training_cutoff = frame["hour"].min() + pd.Timedelta(days=train_days)
    training_mask = frame["hour"] < training_cutoff
    pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        IsolationForest(
            n_estimators=300,
            contamination=0.02,
            max_samples="auto",
            random_state=42,
            n_jobs=-1,
        ),
    )
    pipeline.fit(frame.loc[training_mask, ISOLATION_FEATURES])
    decision = pipeline.decision_function(frame[ISOLATION_FEATURES])
    prediction = pipeline.predict(frame[ISOLATION_FEATURES]) == -1
    return pd.DataFrame(
        {
            "score_isolation_forest": -decision,
            "pred_isolation_forest": prediction,
        }
    )


def composite_predictions(frame: pd.DataFrame, alert_path: Path) -> pd.Series:
    prediction = pd.Series(False, index=frame.index)
    if not alert_path.exists():
        return prediction
    alerts = pd.read_csv(alert_path, parse_dates=["detected_at"])
    if alerts.empty:
        return prediction
    alert_hours = set(alerts["detected_at"].dt.floor("h"))
    return frame["hour"].isin(alert_hours)


def event_metrics(
    scores: pd.DataFrame,
    truth: pd.DataFrame,
    prediction_column: str,
) -> tuple[float, float | None]:
    detected_events = 0
    delays: list[float] = []
    predicted_hours = scores.loc[scores[prediction_column], "hour"]
    for _, event in truth.iterrows():
        start_hour = event["start_time"].floor("h")
        end_hour = event["end_time"].floor("h")
        matches = predicted_hours[(predicted_hours >= start_hour) & (predicted_hours <= end_hour)]
        if not matches.empty:
            detected_events += 1
            delays.append(max(0.0, (matches.min() - event["start_time"]).total_seconds() / 60))
    recall = detected_events / len(truth) if len(truth) else 0.0
    return recall, (float(np.mean(delays)) if delays else None)


def evaluate_algorithms(scores: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    algorithms = {
        "CompositeRules": "pred_composite_rules",
        "RollingMAD": "pred_mad",
        "STLResidual": "pred_stl",
        "IsolationForest": "pred_isolation_forest",
    }
    truth_mask = scores["is_ground_truth"].astype(bool)
    day_count = max(1.0, (scores["hour"].max() - scores["hour"].min()).total_seconds() / 86400 + 1 / 24)
    rows: list[dict[str, object]] = []
    for algorithm, column in algorithms.items():
        prediction = scores[column].astype(bool)
        true_positive = int((prediction & truth_mask).sum())
        false_positive = int((prediction & ~truth_mask).sum())
        false_negative = int((~prediction & truth_mask).sum())
        precision = true_positive / (true_positive + false_positive) if prediction.sum() else 0.0
        recall = true_positive / (true_positive + false_negative) if truth_mask.sum() else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        event_recall, delay = event_metrics(scores, truth, column)
        rows.append(
            {
                "algorithm": algorithm,
                "predicted_hours": int(prediction.sum()),
                "true_positive_hours": true_positive,
                "false_positive_hours": false_positive,
                "false_negative_hours": false_negative,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "false_alarms_per_day": round(false_positive / day_count, 4),
                "event_recall": round(event_recall, 4),
                "mean_detection_delay_minutes": None if delay is None else round(delay, 2),
            }
        )
    return pd.DataFrame(rows).sort_values(["event_recall", "f1"], ascending=False)


def run_benchmark(
    feature_path: Path = DEFAULT_FEATURES,
    truth_path: Path = DEFAULT_TRUTH,
    composite_alert_path: Path = DEFAULT_COMPOSITE_ALERTS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features, truth = load_inputs(feature_path, truth_path)
    scores = attach_ground_truth(features, truth)

    mad = rolling_mad_scores(scores)
    stl = stl_scores(scores)
    isolation = isolation_forest_scores(scores)
    scores = pd.concat([scores, mad, stl, isolation], axis=1)
    scores["pred_composite_rules"] = composite_predictions(scores, composite_alert_path)
    scores["score_composite_rules"] = scores["pred_composite_rules"].astype(float)
    scores["pred_mad"] = scores["score_mad"] >= 6.0
    scores["pred_stl"] = scores["score_stl"] >= 6.0

    results = evaluate_algorithms(scores, truth)
    return scores, results


def main() -> None:
    parser = argparse.ArgumentParser(description="运行异常检测算法对比实验")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--truth", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--composite-alerts", type=Path, default=DEFAULT_COMPOSITE_ALERTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    scores, results = run_benchmark(args.features, args.truth, args.composite_alerts)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    score_path = args.output_dir / "anomaly_scores.csv"
    result_path = args.output_dir / "model_benchmark_results.csv"
    scores.to_csv(score_path, index=False, encoding="utf-8-sig")
    results.to_csv(result_path, index=False, encoding="utf-8-sig")
    print(f"已生成算法分数：{score_path}")
    print(f"已生成评测结果：{result_path}")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
