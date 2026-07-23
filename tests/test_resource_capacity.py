from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.resource_capacity import discover_batches, process_batch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ResourceCapacityImportTests(unittest.TestCase):
    def test_real_batch_is_deduplicated_anonymized_and_excludes_platform_pool(self) -> None:
        batch = discover_batches(PROJECT_ROOT / "newdata")[0]
        with tempfile.TemporaryDirectory() as temporary:
            model, instances, capacity, audit = process_batch(
                batch, Path(temporary) / ".instance_salt"
            )

        self.assertEqual(len(model), 2160)
        self.assertEqual(len(instances), 240)
        self.assertEqual(len(capacity), 3)
        self.assertEqual(audit["excluded_platform_rows"], 5760)
        expected_models = {"DeepSeek-V4", "Minimax-M2.5", "Qwen3.6-35B-A3B"}
        self.assertEqual(set(model["model_id"]), expected_models)
        self.assertEqual(set(capacity["model_id"]), expected_models)
        self.assertNotIn("instance/IP", instances.columns)
        self.assertTrue(instances["instance_id"].str.match(r"^[A-Z0-9]+-[0-9a-f]{10}$").all())
        self.assertEqual(model.groupby(["model_id", "timestamp"]).size().max(), 1)
        self.assertEqual(int(capacity["high_npu_samples"].sum()), 25)

    def test_incomplete_batch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "模型性能中间明细_20260721.xlsx").touch()
            with self.assertRaisesRegex(ValueError, "批次不完整"):
                discover_batches(root)


if __name__ == "__main__":
    unittest.main()
