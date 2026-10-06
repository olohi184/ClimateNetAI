import math
import unittest

from rbe import (
    RBE_COVERAGE_FLOOR,
    RBE_ERROR_THRESHOLD_DB,
    RBE_WINDOW,
    compute_rbe_state,
)


def record(error, covered):
    return {"Absolute_Error_dB": error, "Interval_Covered": covered}


class ReliabilityBoundaryEngineTests(unittest.TestCase):
    def test_warm_up_before_five_completed_observations(self):
        result = compute_rbe_state([record(2.0, 1)] * (RBE_WINDOW - 1))
        self.assertEqual(result["state"], "WARM-UP")
        self.assertTrue(math.isnan(result["mae"]))
        self.assertEqual(result["n_prior"], RBE_WINDOW - 1)

    def test_trust_when_both_criteria_are_supported(self):
        result = compute_rbe_state([record(4.0, 1)] * RBE_WINDOW)
        self.assertEqual(result["state"], "TRUST")
        self.assertLessEqual(result["mae"], RBE_ERROR_THRESHOLD_DB)
        self.assertGreaterEqual(result["coverage"], RBE_COVERAGE_FLOOR)

    def test_caution_when_error_criterion_alone_fails(self):
        result = compute_rbe_state([record(10.0, 1)] * RBE_WINDOW)
        self.assertEqual(result["state"], "CAUTION")

    def test_caution_when_coverage_criterion_alone_fails(self):
        rows = [record(4.0, 1), record(4.0, 1), record(4.0, 1),
                record(4.0, 0), record(4.0, 0)]
        result = compute_rbe_state(rows)
        self.assertEqual(result["state"], "CAUTION")

    def test_abstain_when_both_criteria_fail(self):
        rows = [record(10.0, 1), record(10.0, 1), record(10.0, 1),
                record(10.0, 0), record(10.0, 0)]
        result = compute_rbe_state(rows)
        self.assertEqual(result["state"], "ABSTAIN")

    def test_only_latest_five_completed_observations_are_used(self):
        old = [record(50.0, 0)] * 4
        latest = [record(2.0, 1)] * RBE_WINDOW
        result = compute_rbe_state(old + latest)
        self.assertEqual(result["state"], "TRUST")
        self.assertAlmostEqual(result["mae"], 2.0)
        self.assertAlmostEqual(result["coverage"], 1.0)

    def test_future_current_outcome_cannot_change_already_assigned_state(self):
        prior = [record(2.0, 1)] * RBE_WINDOW
        assigned = compute_rbe_state(prior)
        current_outcome = record(100.0, 0)
        self.assertEqual(assigned["state"], "TRUST")
        after_completion = compute_rbe_state(prior + [current_outcome])
        self.assertNotEqual(
            assigned["mae"],
            after_completion["mae"],
            "The completed outcome may affect the NEXT prediction, not its own state.",
        )


if __name__ == "__main__":
    unittest.main()
