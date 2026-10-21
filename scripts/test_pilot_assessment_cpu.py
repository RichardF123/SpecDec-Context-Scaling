"""CPU tests for offline pilot state separation."""
import unittest
from scripts.analyze_teaching_pilot import assess_pilot_unit


class AssessmentTests(unittest.TestCase):
    def test_new_boundary_evidence_is_not_ignored(self):
        r = assess_pilot_unit([.1], .1, True, "OK")
        self.assertEqual(r["boundary_environment_status"], "OK")
        self.assertEqual(r["environment_status"], "BOUNDARY_ONLY")
        self.assertNotIn("runtime_boundary_evidence_missing", r["speedup_claim_blockers"])
        self.assertFalse(r["speedup_claim_allowed"])

    def test_low_boundary_does_not_prove_environment_failure(self):
        r = assess_pilot_unit([.1], .1, True, "CHECK")
        self.assertEqual(r["environment_status"], "NOT_ESTABLISHED")
        self.assertIn("boundary_clock_check", r["speedup_claim_blockers"])

    def test_stable_is_not_formal_admission(self):
        r = assess_pilot_unit([.1, .2], .2, True)
        self.assertEqual(r["within_run_status"], "OK")
        self.assertEqual(r["between_block_status"], "OK")
        self.assertEqual(r["environment_status"], "NOT_ESTABLISHED")
        self.assertFalse(r["speedup_claim_allowed"])

    def test_within_and_between_reasons_remain_separate(self):
        for within, between in ((.3, .1), (.1, .3), (.3, .3)):
            with self.subTest(within=within, between=between):
                r = assess_pilot_unit([within], between, True)
                self.assertEqual(r["within_run_status"], "CHECK" if within > .2 else "OK")
                self.assertEqual(r["between_block_status"], "CHECK" if between > .2 else "OK")
                self.assertEqual("within_run_spread_gt_20pct" in r["speedup_claim_blockers"], within > .2)
                self.assertEqual("between_block_difference_gt_20pct" in r["speedup_claim_blockers"], between > .2)

    def test_failed_preflight_not_missing_evidence(self):
        r = assess_pilot_unit([.1], .1, False)
        self.assertEqual(r["environment_status"], "ENV_INVALID")
        self.assertFalse(r["speedup_claim_allowed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
