"""Boundary telemetry tests with no real GPU sampling."""
import unittest
from unittest.mock import patch
from .runtime_audit import capture_boundary, assess_boundaries, validate_boundary_audit


def point(sm=2500, mem=8000, available=True):
    return {"snapshot": {"available": available,
                         "clock_sm_MHz": sm, "clock_mem_MHz": mem}}


class RuntimeAuditTests(unittest.TestCase):
    def test_saved_schema_and_legacy_policy(self):
        from .test_configuration_cpu import sample
        from .records import validate_record
        row = sample()
        validate_boundary_audit(row["runtime_environment_audit"])
        validate_record(row, 151645)
        del row["runtime_environment_audit"]
        with self.assertRaises(ValueError):
            validate_record(row, 151645)
        validate_record(row, 151645, require_runtime_audit=False)

    def test_bad_saved_audits(self):
        from copy import deepcopy
        from .test_configuration_cpu import sample
        original = sample()["runtime_environment_audit"]
        for case in ("missing", "reverse", "overlap", "nonfinite", "claim", "forged_verdict"):
            with self.subTest(case=case):
                a = deepcopy(original)
                if case == "missing": del a["monitor_start"]
                if case == "reverse": a["monitor_start"]["query_end_monotonic_s"] = 0
                if case == "overlap": a["monitor_end"]["query_start_monotonic_s"] = 0
                if case == "nonfinite": a["monitor_end"]["query_end_monotonic_s"] = float("nan")
                if case == "claim": a["continuous_active_load_verified"] = True
                if case == "forged_verdict": a["boundary_clock_qualified"] = False
                with self.assertRaises(ValueError): validate_boundary_audit(a)

    def test_good_boundaries_not_continuous_proof(self):
        r = assess_boundaries(point(), point())
        self.assertIs(r["boundary_clock_qualified"], True)
        self.assertIs(r["continuous_active_load_verified"], False)

    def test_low_boundary_is_recorded(self):
        r = assess_boundaries(point(210, 405), point(210, 405))
        self.assertIs(r["boundary_clock_qualified"], False)
        self.assertEqual(r["evidence_status"], "COMPLETE")

    def test_missing_and_nonfinite_are_unknown(self):
        for p in (point(available=False), point(sm=None), point(mem=float("nan"))):
            with self.subTest(point=p):
                r = assess_boundaries(p, point())
                self.assertIsNone(r["boundary_clock_qualified"])
                self.assertEqual(r["evidence_status"], "UNAVAILABLE")

    def test_capture_is_timestamped(self):
        with patch("experiments.formal.runtime_audit.g5_driver.system_snapshot", return_value=point()["snapshot"]) as query:
            r = capture_boundary()
            query.assert_called_once_with()
            self.assertGreaterEqual(r["query_end_monotonic_s"], r["query_start_monotonic_s"])
            self.assertFalse(r["active_load_verified"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
