"""Commit-gate boundary checks with simulated system memory."""

import unittest
from unittest.mock import patch
from .environment import check_environment, EnvironmentRefused


class CommitGateTests(unittest.TestCase):
    def test_boundaries(self):
        for threshold in (16, 18):
            for free in (threshold - 0.01, threshold, threshold + 0.01):
                with self.subTest(threshold=threshold, free=free):
                    memory = {"available": True, "commit_free_mib": free * 1024}
                    with patch("experiments.formal.environment.system_memory", return_value=memory):
                        if free < threshold:
                            with self.assertRaises(EnvironmentRefused):
                                check_environment(threshold)
                        else:
                            result = check_environment(threshold)
                            self.assertEqual(result["commit_min_gib"], threshold)
                            self.assertTrue(result["commit_gate_passed"])

    def test_worker_default_remains_16(self):
        memory = {"available": True, "commit_free_mib": 17 * 1024}
        with patch("experiments.formal.environment.system_memory", return_value=memory):
            self.assertEqual(check_environment()["commit_min_gib"], 16)

    def test_unavailable_memory_refused(self):
        with patch("experiments.formal.environment.system_memory", return_value={"available": False}):
            with self.assertRaises(EnvironmentRefused):
                check_environment()


if __name__ == "__main__":
    unittest.main(verbosity=2)
