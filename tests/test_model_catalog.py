from __future__ import annotations

import unittest

from src.model_catalog import MODEL_IDS, MODEL_PRICE, load_observed_calibration


class ModelCatalogTests(unittest.TestCase):
    def test_catalog_contains_only_current_monitored_models(self) -> None:
        self.assertEqual(
            set(MODEL_IDS),
            {"DeepSeek-V4", "Minimax-M2.5", "Qwen3.6-35B-A3B"},
        )
        self.assertEqual(set(MODEL_PRICE), set(MODEL_IDS))

    def test_observed_resource_data_drives_weights_and_ttft_samples(self) -> None:
        models, weights, samples, latest = load_observed_calibration()
        self.assertEqual(models.tolist(), list(MODEL_IDS))
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertGreater(weights[0], 0.9)
        self.assertTrue(all(len(samples[model]) > 0 for model in MODEL_IDS))
        self.assertEqual(str(latest.date()), "2026-07-21")


if __name__ == "__main__":
    unittest.main()
