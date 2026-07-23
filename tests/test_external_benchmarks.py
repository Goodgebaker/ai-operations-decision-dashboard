"""外部模型压测标准化与容量画像测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.external_benchmarks import (
    build_capacity_profiles,
    write_standardized_benchmarks,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STANDARD_CSV = PROJECT_ROOT / "data" / "external_model_benchmarks.csv"


class ExternalBenchmarkTests(unittest.TestCase):
    def test_real_source_is_normalized_without_losing_test_combinations(self) -> None:
        benchmarks = pd.read_csv(STANDARD_CSV)

        self.assertEqual(138, len(benchmarks))
        self.assertEqual(3, benchmarks["provider"].nunique())
        self.assertEqual(10, int(benchmarks["rate_limited"].sum()))
        self.assertEqual(0, int(benchmarks["benchmark_id"].duplicated().sum()))
        valid = benchmarks[benchmarks["total_output_tokens_per_second"].notna()]
        derived = valid["total_output_tokens_per_second"] / valid["concurrency"]
        pd.testing.assert_series_equal(
            valid["per_request_output_tokens_per_second"].reset_index(drop=True),
            derived.reset_index(drop=True),
            check_names=False,
        )

    def test_capacity_profiles_preserve_observed_rate_limit_boundary(self) -> None:
        capacity = build_capacity_profiles(pd.read_csv(STANDARD_CSV))

        self.assertEqual(45, len(capacity))
        self.assertEqual(5, int(capacity["rate_limit_observed"].sum()))
        selected = capacity[
            capacity["provider"].eq("阿里百炼")
            & capacity["model_id"].eq("deepseek-v4-pro")
            & capacity["io_profile"].eq("16k/1k")
        ].iloc[0]
        self.assertEqual(30, selected["max_stable_concurrency"])
        self.assertEqual(50, selected["observed_rate_limit_concurrency"])
        self.assertEqual("低", selected["capacity_confidence"])

    def test_standardized_csv_can_be_reloaded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            source = temporary_path / "source.json"
            destination = temporary_path / "benchmarks.csv"
            source.write_text(
                json.dumps(
                    {
                        "records": [
                            {
                                "vendor": "测试供应商",
                                "model": "test-model",
                                "io": "6k/1k",
                                "concurrency": 10,
                                "ttft": 1.25,
                                "speed": 200.0,
                                "per_speed": 20.0,
                                "rate_limited": False,
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            written = write_standardized_benchmarks(source, destination)
            reloaded = pd.read_csv(destination)

        self.assertEqual(1, len(written))
        self.assertEqual(1, len(reloaded))
        self.assertEqual(list(written.columns), list(reloaded.columns))


if __name__ == "__main__":
    unittest.main()
