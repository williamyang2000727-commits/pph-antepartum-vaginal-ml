"""不使用病人資料的指標與失敗路徑測試。"""
import unittest
import numpy as np
from submission_pipeline import evaluate, verify

class VerificationTests(unittest.TestCase):
    def test_reject_wrong_or_nonfinite_results(self):
        for actual in ({"auc": .5}, {"auc": float("nan")}, {"auc": float("inf")}, {}):
            with self.assertRaises(ValueError):
                verify(actual, {"auc": .74}, 1e-8)
        verify({"auc": .74}, {"auc": .74}, 1e-8)

    def test_confusion_matrix_and_probability_metrics(self):
        y = np.array([1, 1, 0, 0])
        p = np.array([.9, .4, .6, .1])
        m = evaluate(y, p, .5)
        for k in ("tp", "fp", "fn", "tn"):
            self.assertEqual(m[k], 1)
        self.assertAlmostEqual(m["auc"], .75)
        self.assertAlmostEqual(m["brier"], .185)
        self.assertAlmostEqual(m["sensitivity"], .5)
        self.assertAlmostEqual(m["mcc"], 0.)

if __name__ == "__main__":
    unittest.main()
