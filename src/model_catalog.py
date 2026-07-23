"""项目统一模型目录与真实资源数据中的名称映射。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESOURCE_SERIES = PROJECT_ROOT / "data" / "resource_model_timeseries.csv"


@dataclass(frozen=True)
class ModelDefinition:
    model_id: str
    provider: str
    source_group: str
    synthetic_price_per_token: float
    fallback_ttft_ms: int
    simulated_capability: dict[str, float]
    endpoint_env: str
    probe_key_env: str


MODEL_DEFINITIONS = (
    ModelDefinition(
        model_id="DeepSeek-V4",
        provider="DeepSeek",
        source_group="商用付费deepseek-v4",
        synthetic_price_per_token=0.0000012,
        fallback_ttft_ms=4_278,
        simulated_capability={
            "instruction_following": 0.960,
            "structured_output": 0.930,
            "reasoning": 0.970,
            "tool_call": 0.920,
        },
        endpoint_env="PROBE_ENDPOINT_DEEPSEEK",
        probe_key_env="PROBE_DEEPSEEK_KEY",
    ),
    ModelDefinition(
        model_id="Minimax-M2.5",
        provider="MiniMax",
        source_group="商用付费minimax25",
        synthetic_price_per_token=0.0000024,
        fallback_ttft_ms=413,
        simulated_capability={
            "instruction_following": 0.985,
            "structured_output": 0.970,
            "reasoning": 0.940,
            "tool_call": 0.960,
        },
        endpoint_env="PROBE_ENDPOINT_MINIMAX",
        probe_key_env="PROBE_MINIMAX_KEY",
    ),
    ModelDefinition(
        model_id="Qwen3.6-35B-A3B",
        provider="Qwen",
        source_group="商用付费qwen36-35b",
        synthetic_price_per_token=0.0000016,
        fallback_ttft_ms=437,
        simulated_capability={
            "instruction_following": 0.970,
            "structured_output": 0.950,
            "reasoning": 0.920,
            "tool_call": 0.900,
        },
        endpoint_env="PROBE_ENDPOINT_QWEN",
        probe_key_env="PROBE_QWEN_KEY",
    ),
)

MODEL_IDS = tuple(item.model_id for item in MODEL_DEFINITIONS)
MODEL_BY_ID = {item.model_id: item for item in MODEL_DEFINITIONS}
MODEL_BY_SOURCE_GROUP = {item.source_group: item for item in MODEL_DEFINITIONS}
MODEL_PROVIDER = {item.model_id: item.provider for item in MODEL_DEFINITIONS}
MODEL_PRICE = {
    item.model_id: item.synthetic_price_per_token for item in MODEL_DEFINITIONS
}
MODEL_LATENCY = {
    item.model_id: max(450, int(item.fallback_ttft_ms * 1.35))
    for item in MODEL_DEFINITIONS
}
PROVIDER_ENDPOINT_ENV = {
    item.provider: item.endpoint_env for item in MODEL_DEFINITIONS
}
SIMULATED_CAPABILITY = {
    item.model_id: item.simulated_capability for item in MODEL_DEFINITIONS
}


def canonical_model_id(source_group: str) -> str:
    """将来源工作簿的模型组名称转换为项目统一模型 ID。"""

    definition = MODEL_BY_SOURCE_GROUP.get(str(source_group).strip())
    if definition is None:
        raise ValueError(f"不支持的模型组：{source_group}")
    return definition.model_id


def load_observed_calibration(
    path: Path = DEFAULT_RESOURCE_SERIES,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], pd.Timestamp | None]:
    """读取真实 running 占比和 TTFT 样本，缺失时回退到保守默认值。"""

    fallback_weights = np.array([0.93, 0.04, 0.03], dtype=float)
    fallback_samples = {
        item.model_id: np.array([item.fallback_ttft_ms], dtype=float)
        for item in MODEL_DEFINITIONS
    }
    if not path.exists():
        return np.array(MODEL_IDS), fallback_weights, fallback_samples, None

    data = pd.read_csv(path, parse_dates=["timestamp"])
    required = {"timestamp", "model_id", "running", "ttft_ms"}
    if data.empty or not required.issubset(data.columns):
        return np.array(MODEL_IDS), fallback_weights, fallback_samples, None

    data = data[data["model_id"].isin(MODEL_IDS)].copy()
    means = data.groupby("model_id")["running"].mean().reindex(MODEL_IDS).fillna(0)
    if float(means.sum()) > 0:
        weights = (means / means.sum()).to_numpy(dtype=float)
    else:
        weights = fallback_weights

    samples: dict[str, np.ndarray] = {}
    for model_id in MODEL_IDS:
        values = pd.to_numeric(
            data.loc[data["model_id"].eq(model_id), "ttft_ms"], errors="coerce"
        ).dropna()
        values = values[values.gt(0)]
        samples[model_id] = (
            values.to_numpy(dtype=float)
            if not values.empty
            else fallback_samples[model_id]
        )
    latest = pd.Timestamp(data["timestamp"].max()) if not data.empty else None
    return np.array(MODEL_IDS), weights, samples, latest
