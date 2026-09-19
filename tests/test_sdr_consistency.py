import unittest

import torch

from ctta.modules.consistency import (
    reliability_guided_source_consistency,
    source_prediction_consistency,
)


class ReliabilityGuidedSourceConsistencyTest(unittest.TestCase):
    def setUp(self):
        self.adapted = torch.tensor(
            [
                [[0.8, 0.6], [0.4, 0.2]],
                [[0.2, 0.4], [0.6, 0.8]],
            ],
            dtype=torch.float32,
            requires_grad=True,
        )
        self.source = torch.tensor(
            [
                [[0.7, 0.5], [0.5, 0.3]],
                [[0.3, 0.5], [0.5, 0.7]],
            ],
            dtype=torch.float32,
        )

    def test_min_weight_one_recovers_uniform_gsc(self):
        reliability = torch.tensor(
            [[0.0, 0.3], [0.8, 1.0]], dtype=torch.float32
        )
        uniform = source_prediction_consistency(self.adapted, self.source)
        guided = reliability_guided_source_consistency(
            self.adapted,
            self.source,
            reliability,
            min_weight=1.0,
        )
        self.assertTrue(torch.allclose(uniform, guided, atol=1e-6, rtol=1e-6))

    def test_sdr_keeps_gradients_on_adapted_prediction(self):
        reliability = torch.tensor(
            [[0.1, 0.2], [0.9, 1.0]], dtype=torch.float32
        )
        loss = reliability_guided_source_consistency(
            self.adapted,
            self.source,
            reliability,
            min_weight=0.5,
        )
        loss.backward()
        self.assertIsNotNone(self.adapted.grad)
        self.assertTrue(torch.isfinite(self.adapted.grad).all())

    def test_invalid_floor_is_rejected(self):
        reliability = torch.zeros((2, 2), dtype=torch.float32)
        with self.assertRaises(ValueError):
            reliability_guided_source_consistency(
                self.adapted,
                self.source,
                reliability,
                min_weight=-0.1,
            )


if __name__ == '__main__':
    unittest.main()
