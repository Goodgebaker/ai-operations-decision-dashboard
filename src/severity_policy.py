"""Shared three-level alert grading based on rule risk and threshold breach magnitude."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "docs" / "ai_monitoring_metric_dictionary.xlsx"


@dataclass(frozen=True)
class SeverityBand:
    level: str
    rank: int
    base_risk_weight: float
    min_severity_score: float
    max_severity_score: float
    condition_bonus_per_extra_match: float
    max_condition_bonus: float


def load_severity_policy(path: Path = DEFAULT_CONFIG) -> list[SeverityBand]:
    frame = pd.read_excel(path, sheet_name="Severity Policy")
    required = {
        "level",
        "rank",
        "base_risk_weight",
        "min_severity_score",
        "max_severity_score",
        "condition_bonus_per_extra_match",
        "max_condition_bonus",
        "status",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "Severity Policy sheet is missing columns: " + ", ".join(sorted(missing))
        )
    active = frame[
        frame["status"].astype(str).str.strip().str.lower().eq("active")
    ].copy()
    active["level"] = active["level"].astype(str).str.strip().str.lower()
    if set(active["level"]) != {"info", "warning", "critical"}:
        raise ValueError("Severity Policy must contain active info, warning and critical rows")
    if active["level"].duplicated().any():
        raise ValueError("Severity Policy contains duplicate active levels")

    bands: list[SeverityBand] = []
    for _, row in active.sort_values("rank").iterrows():
        maximum = pd.to_numeric(row["max_severity_score"], errors="coerce")
        bands.append(
            SeverityBand(
                level=str(row["level"]),
                rank=int(row["rank"]),
                base_risk_weight=float(row["base_risk_weight"]),
                min_severity_score=float(row["min_severity_score"]),
                max_severity_score=float(maximum) if pd.notna(maximum) else np.inf,
                condition_bonus_per_extra_match=float(
                    row["condition_bonus_per_extra_match"]
                ),
                max_condition_bonus=float(row["max_condition_bonus"]),
            )
        )
    return bands


def breach_ratio(observed: float, threshold: float, operator: str) -> float:
    """Return a normalized ratio where 1.0 is exactly at the configured threshold."""
    observed_value = float(observed)
    threshold_value = float(threshold)
    operator = operator.lower()
    if operator in {"gt", "gte"}:
        if threshold_value == 0:
            ratio = 5.0 if observed_value > 0 else 1.0
        else:
            ratio = observed_value / threshold_value
    elif operator in {"lt", "lte"}:
        if observed_value == 0:
            ratio = 5.0 if threshold_value > 0 else 1.0
        else:
            ratio = threshold_value / observed_value
    else:
        ratio = 1.0
    if not np.isfinite(ratio):
        return 5.0
    return round(float(np.clip(ratio, 1.0, 5.0)), 4)


def grade_alert(
    *,
    base_severity: str,
    breach_ratios: Iterable[float],
    matched_conditions: int,
    policy: list[SeverityBand],
) -> tuple[str, float, float]:
    """Grade an alert using risk prior + breach margin + multi-condition evidence."""
    if not policy:
        raise ValueError("Severity policy is empty")
    by_level = {band.level: band for band in policy}
    base_level = base_severity.strip().lower()
    if base_level not in by_level:
        raise ValueError(f"Unsupported base severity: {base_severity}")

    ratios = [max(1.0, float(value)) for value in breach_ratios]
    maximum_ratio = max(ratios, default=1.0)
    base_band = by_level[base_level]
    extra_matches = max(0, int(matched_conditions) - 1)
    condition_bonus = min(
        base_band.max_condition_bonus,
        base_band.condition_bonus_per_extra_match * extra_matches,
    )
    score = round(
        max(0.0, maximum_ratio - 1.0)
        + base_band.base_risk_weight
        + condition_bonus,
        4,
    )
    for band in sorted(policy, key=lambda item: item.rank):
        if band.min_severity_score <= score < band.max_severity_score:
            return band.level, score, round(maximum_ratio, 4)
    highest = max(policy, key=lambda item: item.rank)
    return highest.level, score, round(maximum_ratio, 4)
