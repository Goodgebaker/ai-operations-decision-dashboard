"""模型运营评分的配置模型与纯计算函数。

评分规则由指标字典 ``Scoring Policy`` 工作表提供。模块本身不依赖
Streamlit，便于在特征流水线、离线评测和单元测试中复用。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, Mapping


VALID_DIRECTIONS = {
    "higher_better",
    "lower_better",
    "volatility",
    "passthrough",
}


@dataclass(frozen=True)
class ComponentRule:
    """一个复合评分中的单项指标规则。"""

    policy_id: str
    score_family: str
    component: str
    direction: str
    weight: float
    target_value: float | None = None
    tolerance_value: float | None = None


@dataclass(frozen=True)
class ScoreBand:
    """一个评分区间，区间采用左闭右开，100 分属于最高区间。"""

    policy_id: str
    score_family: str
    min_score: float
    max_score: float
    label_cn: str


@dataclass(frozen=True)
class ScoringPolicy:
    """经过校验的评分组件和分级区间集合。"""

    component_rules: tuple[ComponentRule, ...]
    score_bands: tuple[ScoreBand, ...]

    @classmethod
    def from_rows(cls, rows: Iterable[Mapping[str, object]]) -> "ScoringPolicy":
        component_rules: list[ComponentRule] = []
        score_bands: list[ScoreBand] = []

        for row in rows:
            if _text(row.get("status", "active")).lower() != "active":
                continue
            policy_type = _required_text(row, "policy_type").lower()
            policy_id = _required_text(row, "policy_id")
            score_family = _required_text(row, "score_family")

            if policy_type == "component":
                direction = _required_text(row, "direction").lower()
                if direction not in VALID_DIRECTIONS:
                    raise ValueError(
                        f"{policy_id} direction 必须为 {sorted(VALID_DIRECTIONS)} 之一"
                    )
                rule = ComponentRule(
                    policy_id=policy_id,
                    score_family=score_family,
                    component=_required_text(row, "component"),
                    direction=direction,
                    weight=_required_number(row, "weight"),
                    target_value=_optional_number(row.get("target_value")),
                    tolerance_value=_optional_number(row.get("tolerance_value")),
                )
                _validate_component_rule(rule)
                component_rules.append(rule)
            elif policy_type == "band":
                band = ScoreBand(
                    policy_id=policy_id,
                    score_family=score_family,
                    min_score=_required_number(row, "min_score"),
                    max_score=_required_number(row, "max_score"),
                    label_cn=_required_text(row, "label_cn"),
                )
                score_bands.append(band)
            else:
                raise ValueError(f"{policy_id} policy_type 仅支持 component 或 band")

        policy = cls(tuple(component_rules), tuple(score_bands))
        policy.validate()
        return policy

    def validate(self) -> None:
        if not self.component_rules:
            raise ValueError("评分配置没有生效的 component 记录")

        seen_components: set[tuple[str, str]] = set()
        families: dict[str, list[ComponentRule]] = {}
        for rule in self.component_rules:
            key = (rule.score_family, rule.component)
            if key in seen_components:
                raise ValueError(f"评分组件重复：{rule.score_family}.{rule.component}")
            seen_components.add(key)
            families.setdefault(rule.score_family, []).append(rule)

        for family, rules in families.items():
            total_weight = sum(rule.weight for rule in rules)
            if not math.isclose(total_weight, 1.0, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(
                    f"{family} 的生效权重合计必须为1，当前为 {total_weight:g}"
                )

        bands_by_family: dict[str, list[ScoreBand]] = {}
        for band in self.score_bands:
            _validate_band(band)
            bands_by_family.setdefault(band.score_family, []).append(band)
        for family, bands in bands_by_family.items():
            _validate_contiguous_bands(family, bands)

    def rules_for(self, score_family: str) -> tuple[ComponentRule, ...]:
        rules = tuple(
            rule for rule in self.component_rules if rule.score_family == score_family
        )
        if not rules:
            raise KeyError(f"没有评分族配置：{score_family}")
        return rules

    def bands_for(self, score_family: str) -> tuple[ScoreBand, ...]:
        bands = tuple(
            sorted(
                (band for band in self.score_bands if band.score_family == score_family),
                key=lambda item: item.min_score,
            )
        )
        if not bands:
            raise KeyError(f"没有评分分级配置：{score_family}")
        return bands


def clamp_score(value: float) -> float:
    """将有限数值限制到 0 至 100。"""

    number = _finite_number(value, "score")
    return min(100.0, max(0.0, number))


def score_higher_is_better(actual: float, target: float) -> float:
    """正向指标达标得100分，未达标按完成比例计分。"""

    actual_value = _finite_number(actual, "actual")
    target_value = _positive_number(target, "target")
    if actual_value < 0:
        raise ValueError("actual 不能小于0")
    return clamp_score(actual_value / target_value * 100.0)


def score_lower_is_better(actual: float, target: float) -> float:
    """逆向指标不高于目标得100分，超目标后按目标/实际值衰减。"""

    actual_value = _finite_number(actual, "actual")
    target_value = _positive_number(target, "target")
    if actual_value < 0:
        raise ValueError("actual 不能小于0")
    if actual_value <= target_value:
        return 100.0
    return clamp_score(target_value / actual_value * 100.0)


def score_volatility(actual: float, tolerance: float) -> float:
    """波动为0得100分，达到容忍上限时降为0分。"""

    actual_value = _finite_number(actual, "actual")
    tolerance_value = _positive_number(tolerance, "tolerance")
    if actual_value < 0:
        raise ValueError("actual 不能小于0")
    return clamp_score((1.0 - actual_value / tolerance_value) * 100.0)


def calculate_family_score(
    score_family: str,
    component_values: Mapping[str, float],
    policy: ScoringPolicy,
) -> float:
    """按配置转换每个组件并计算加权得分。"""

    total = 0.0
    for rule in policy.rules_for(score_family):
        if rule.component not in component_values:
            raise KeyError(f"{score_family} 缺少组件值：{rule.component}")
        value = component_values[rule.component]
        component_score = _apply_rule(value, rule)
        total += component_score * rule.weight
    return clamp_score(total)


def classify_score(
    score_family: str,
    score: float,
    policy: ScoringPolicy,
) -> str:
    """根据配置返回中文等级标签。"""

    value = clamp_score(score)
    bands = policy.bands_for(score_family)
    for index, band in enumerate(bands):
        is_last = index == len(bands) - 1
        if band.min_score <= value < band.max_score or (
            is_last and value == band.max_score
        ):
            return band.label_cn
    raise ValueError(f"{score_family} 的分级区间没有覆盖 {value:g} 分")


def load_scoring_policy(
    workbook_path: Path,
    sheet_name: str = "Scoring Policy",
) -> ScoringPolicy:
    """从指标字典读取评分策略；I/O 与纯计算入口保持分离。"""

    import pandas as pd

    frame = pd.read_excel(workbook_path, sheet_name=sheet_name)
    rows = frame.where(frame.notna(), None).to_dict(orient="records")
    return ScoringPolicy.from_rows(rows)


def _apply_rule(value: float, rule: ComponentRule) -> float:
    if rule.direction == "higher_better":
        return score_higher_is_better(value, _required_rule_target(rule))
    if rule.direction == "lower_better":
        return score_lower_is_better(value, _required_rule_target(rule))
    if rule.direction == "volatility":
        if rule.tolerance_value is None:
            raise ValueError(f"{rule.policy_id} 缺少 tolerance_value")
        return score_volatility(value, rule.tolerance_value)
    return clamp_score(value)


def _validate_component_rule(rule: ComponentRule) -> None:
    if not 0 < rule.weight <= 1:
        raise ValueError(f"{rule.policy_id} weight 必须大于0且不超过1")
    if rule.direction in {"higher_better", "lower_better"}:
        _required_rule_target(rule)
    if rule.direction == "volatility":
        if rule.tolerance_value is None:
            raise ValueError(f"{rule.policy_id} 缺少 tolerance_value")
        _positive_number(rule.tolerance_value, "tolerance_value")


def _validate_band(band: ScoreBand) -> None:
    if not 0 <= band.min_score < band.max_score <= 100:
        raise ValueError(f"{band.policy_id} 的分数区间必须满足 0<=min<max<=100")


def _validate_contiguous_bands(family: str, bands: list[ScoreBand]) -> None:
    ordered = sorted(bands, key=lambda item: item.min_score)
    if ordered[0].min_score != 0 or ordered[-1].max_score != 100:
        raise ValueError(f"{family} 的分级区间必须完整覆盖0至100")
    for previous, current in zip(ordered, ordered[1:]):
        if not math.isclose(
            previous.max_score, current.min_score, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(f"{family} 的分级区间存在空档或重叠")


def _required_rule_target(rule: ComponentRule) -> float:
    if rule.target_value is None:
        raise ValueError(f"{rule.policy_id} 缺少 target_value")
    return _positive_number(rule.target_value, "target_value")


def _required_text(row: Mapping[str, object], key: str) -> str:
    value = _text(row.get(key))
    if not value:
        raise ValueError(f"评分配置缺少 {key}")
    return value


def _text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _required_number(row: Mapping[str, object], key: str) -> float:
    value = _optional_number(row.get(key))
    if value is None:
        raise ValueError(f"评分配置缺少 {key}")
    return value


def _optional_number(value: object) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return _finite_number(value, "value")


def _positive_number(value: object, name: str) -> float:
    number = _finite_number(value, name)
    if number <= 0:
        raise ValueError(f"{name} 必须大于0")
    return number


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是有限数值")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是有限数值") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} 必须是有限数值")
    return number
