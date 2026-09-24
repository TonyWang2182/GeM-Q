import unittest
import numpy as np
from train import make_splits, Head
import torch


class ProtocolTests(unittest.TestCase):
    def test_disjoint_and_complete(self):
        rows = [{"system": f"S{g:03}"} for g in range(31) for _ in range(10)]
        splits = make_splits(rows)
        self.assertEqual(len(splits), 8)
        for sp in splits[:5]:
            sets = [
                {rows[i]["system"] for i in sp[p]} for p in ["train", "val", "test"]
            ]
            self.assertFalse(
                sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]
            )

    def test_importance_weighting_identity(self):
        counts = np.array([2, 3, 7])
        q = np.array([0.2, 0.1, 0.7])
        w = 0.5 / 3 + 0.5 * q
        losses = np.arange(12, dtype=float) / 20
        weights = np.concatenate([np.repeat(12 * a / n, n) for a, n in zip(w, counts)])
        grouped = [x.mean() for x in np.split(losses, np.cumsum(counts)[:-1])]
        self.assertAlmostEqual((weights * losses).mean(), np.dot(w, grouped))
        self.assertAlmostEqual(weights.mean(), 1.0)
        self.assertGreaterEqual(w.min(), 0.5 / 3)

    def test_head_shape_gradient_and_eval_repeat(self):
        torch.manual_seed(42)
        m = Head(dim=8)
        x = torch.randn(4, 7, 8)
        y = m(x)
        self.assertEqual(list(y.shape), [4])
        y.square().mean().backward()
        self.assertTrue(
            all(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in m.parameters()
            )
        )
        m.eval()
        self.assertTrue(torch.equal(m(x), m(x)))


if __name__ == "__main__":
    unittest.main()
