"""Metadata-only input-set checks; no tokenization or model inference."""
import unittest
from unittest.mock import patch

from . import input_sets as inputs
from .plan import build_jobs, validate_plan


def manifest(ids):
    return {"families": [{"family_id": family} for family in ids]}


class InputSetTests(unittest.TestCase):
    def test_real_manifests_and_plan_counts(self):
        for name, count, jobs_count, measured in (
            ("pilot", 4, 48, 120), ("formal", 16, 192, 480)
        ):
            with self.subTest(name=name):
                path, families = inputs.select_input_set(name)
                self.assertTrue(path.is_file())
                self.assertEqual(len(families), count)
                jobs = build_jobs(families)
                validate_plan(jobs, families)
                self.assertEqual(len(jobs), jobs_count)
                self.assertEqual(sum(j.measured_reps for j in jobs), measured)

    def test_unknown_name_rejected_before_read(self):
        with patch.object(inputs, "load_manifest") as read:
            with self.assertRaises(ValueError):
                inputs.select_input_set("unknown")
            read.assert_not_called()

    def test_duplicates_rejected(self):
        ids = inputs.PILOT_FAMILIES + (inputs.PILOT_FAMILIES[0],)
        with patch.object(inputs, "load_manifest", return_value=manifest(ids)):
            with self.assertRaisesRegex(ValueError, "重复"):
                inputs.select_input_set("pilot")

    def test_missing_families_rejected(self):
        for name, ids in (("pilot", inputs.PILOT_FAMILIES),
                          ("formal", inputs.FORMAL_FAMILIES)):
            with self.subTest(name=name):
                with patch.object(inputs, "load_manifest", return_value=manifest(ids[:-1])):
                    with self.assertRaisesRegex(ValueError, "缺少"):
                        inputs.select_input_set(name)

    def test_extra_formal_family_rejected(self):
        ids = inputs.FORMAL_FAMILIES + ("FF17",)
        with patch.object(inputs, "load_manifest", return_value=manifest(ids)):
            with self.assertRaisesRegex(ValueError, "恰好"):
                inputs.select_input_set("formal")

    def test_extra_pilot_manifest_family_is_allowed(self):
        ids = inputs.PILOT_FAMILIES + ("F01",)
        with patch.object(inputs, "load_manifest", return_value=manifest(ids)):
            _, families = inputs.select_input_set("pilot")
            self.assertEqual(families, inputs.PILOT_FAMILIES)


if __name__ == "__main__":
    unittest.main(verbosity=2)
