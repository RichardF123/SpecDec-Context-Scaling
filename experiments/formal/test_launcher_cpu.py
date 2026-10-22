"""Launcher regression tests using simulated workers, never GPU processes."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from . import run as launcher


class LauncherTests(unittest.TestCase):
    def exercise(self, case):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake_file = root / "experiments" / "formal" / "run.py"
            code = 7 if case == "worker_failed" else 0
            with (
                patch.object(launcher, "__file__", str(fake_file)),
                patch.object(launcher.sys, "argv", ["run", "--execute", "--limit", "2"]),
                patch.object(launcher.subprocess, "run", return_value=
                             subprocess.CompletedProcess(["mock"], code)) as child,
                patch.object(launcher, "verify_job_results") as verify,
                patch.object(launcher.time, "sleep"),
                patch.object(launcher, "collect_runtime_info", return_value={
                    "git_commit": "cpu-test", "dirty_worktree": False,
                }),
                patch.object(launcher, "check_environment", return_value={
                    "commit_gate_passed": True, "commit_min_gib": 18,
                }) as environment,
                redirect_stdout(io.StringIO()),
            ):
                verify.return_value = {"verified": True, "n_measured": 3}
                if case == "verification_failed":
                    verify.side_effect = ValueError("synthetic invalid result")
                if case == "success":
                    launcher.main()
                elif case == "worker_failed":
                    with self.assertRaises(SystemExit):
                        launcher.main()
                else:
                    with self.assertRaisesRegex(ValueError, "synthetic"):
                        launcher.main()

                summaries = list(root.glob("output/formal_batch_development/*/summary.json"))
                self.assertEqual(len(summaries), 1)
                summary = json.loads(summaries[0].read_text(encoding="utf-8"))
                environment.assert_called_once_with(min_commit_gib=18)
                self.assertEqual(summary["entry_environment"]["commit_min_gib"], 18)
                if case == "success":
                    self.assertEqual(summary["status"], "development_batch_verified")
                    self.assertEqual(len(summary["completed"]), 2)
                    self.assertIsNone(summary["active_index"])
                    self.assertNotIn("failed", summary)
                    self.assertEqual(child.call_count, 2)
                    self.assertEqual(verify.call_count, 2)
                else:
                    self.assertEqual(summary["status"], "aborted")
                    self.assertEqual(summary["completed"], [])
                    self.assertEqual(summary["active_index"], 0)
                    failure = summary["failed"]
                    self.assertEqual(failure["index"], 0)
                    self.assertEqual(failure["job"], summary["jobs"][0])
                    launch_dir = Path(failure["launch_dir"])
                    self.assertEqual(launch_dir.name, failure["launch_id"])
                    exit_record = json.loads((launch_dir / "exit.json").read_text(encoding="utf-8"))
                    expected = ("worker_failed" if case == "worker_failed"
                                else "result_verification_failed")
                    self.assertEqual(exit_record["status"], expected)
                    self.assertEqual(exit_record["launch_id"], failure["launch_id"])
                    self.assertTrue((launch_dir / "worker.log").is_file())
                    self.assertEqual(child.call_count, 1)
                    self.assertEqual(verify.call_count, 0 if case == "worker_failed" else 1)

    def test_success(self):
        self.exercise("success")

    def test_refusal_does_not_launch_or_create_batch(self):
        from .environment import EnvironmentRefused
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch.object(launcher, "__file__", str(root / "experiments/formal/run.py")),
                patch.object(launcher.sys, "argv", ["run", "--execute"]),
                patch.object(launcher, "check_environment", side_effect=EnvironmentRefused("low commit")),
                patch.object(launcher.subprocess, "run") as child,
                redirect_stdout(io.StringIO()),
            ):
                with self.assertRaises(EnvironmentRefused):
                    launcher.main()
                child.assert_not_called()
                self.assertFalse((root / "output").exists())

    def test_dry_run_skips_environment_and_worker(self):
        with (
            patch.object(launcher.sys, "argv", ["run", "--limit", "2"]),
            patch.object(launcher, "check_environment") as environment,
            patch.object(launcher.subprocess, "run") as child,
            redirect_stdout(io.StringIO()),
        ):
            launcher.main()
            environment.assert_not_called()
            child.assert_not_called()

    def test_pilot_dry_run_has_complete_plan(self):
        output = io.StringIO()
        with (
            patch.object(launcher.sys, "argv", ["run", "--pilot"]),
            patch.object(launcher, "check_environment") as environment,
            patch.object(launcher.subprocess, "run") as child,
            redirect_stdout(output),
        ):
            launcher.main()
            environment.assert_not_called()
            child.assert_not_called()
        text = output.getvalue()
        self.assertIn("本次任务数： 48", text)
        self.assertIn("Measured 次数： 120", text)
        self.assertEqual(text.count("命令："), 48)

    def test_pilot_and_limit_are_mutually_exclusive(self):
        from contextlib import redirect_stderr
        with (
            patch.object(launcher.sys, "argv", ["run", "--pilot", "--limit", "2"]),
            patch.object(launcher.subprocess, "run") as child,
            redirect_stderr(io.StringIO()),
        ):
            with self.assertRaises(SystemExit) as result:
                launcher.main()
            self.assertEqual(result.exception.code, 2)
            child.assert_not_called()

    def test_worker_failure_stops_batch_and_links_evidence(self):
        self.exercise("worker_failed")

    def test_verification_failure_stops_batch_and_links_evidence(self):
        self.exercise("verification_failed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
