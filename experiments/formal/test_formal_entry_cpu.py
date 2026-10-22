"""Full formal orchestration using fake workers and temporary synthetic records."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from . import run as launcher
from .plan import Condition, Job
from .records import save_record, save_diagnostic, validate_record
from .runtime_info import require_formal_version
from .test_configuration_cpu import sample

COMMIT = "a" * 40


class FormalEntryTests(unittest.TestCase):
    def test_clean_version_gate(self):
        self.assertEqual(require_formal_version({"git_commit": COMMIT, "dirty_worktree": False}), COMMIT)
        for runtime, expected in (({"git_commit": COMMIT, "dirty_worktree": True}, COMMIT),
                                  ({"git_commit": COMMIT, "dirty_worktree": False}, "other")):
            with self.assertRaises(RuntimeError):
                require_formal_version(runtime, expected)

    def formal_record(self):
        row = sample()
        row.update(experiment_stage="formal", input={"input_set": "formal", "hash_verified": True},
                   runtime={"git_commit": COMMIT, "dirty_worktree": False},
                   collection={"expected_commit": COMMIT, "speedup_claim_allowed": False})
        row["job"]["family_id"] = "FF01"
        return row

    def test_formal_record_contract(self):
        validate_record(self.formal_record(), 151645)
        mutations = (("job", "family_id", "F09"), ("input", "input_set", "pilot"),
                     ("input", "hash_verified", False), ("runtime", "dirty_worktree", True),
                     ("collection", "expected_commit", "other"),
                     ("collection", "speedup_claim_allowed", True))
        for parent, field, value in mutations:
            with self.subTest(field=field):
                row = deepcopy(self.formal_record())
                row[parent][field] = value
                with self.assertRaises(ValueError): validate_record(row, 151645)

    def test_full_192_job_batch_with_fake_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generated_records = []
            def fake_worker(command, **kwargs):
                def arg(name): return command[command.index(name) + 1]
                family, ctx, mode, block = arg("--family"), int(arg("--ctx")), arg("--mode"), int(arg("--block"))
                self.assertEqual(arg("--input-set"), "formal")
                self.assertEqual(arg("--expected-commit"), COMMIT)
                job = Job(family, Condition(f"L{ctx}_{mode}_base_bf16", ctx, mode, "base_bf16"), block, 3 if block == 0 else 2)
                out = Path(arg("--output-dir"))
                run_id = f"{family}-{ctx}-{mode}-{block}"
                diagnostic = {"diagnostic_only": True, "job": asdict(job),
                              "experiment_stage": "formal", "run_id": run_id,
                              "generated_ids": [10]*64,
                              "finiteness_coverage": {"n_checked": 64, "first_bad": None},
                              "decision_coverage": {"ok": True}}
                save_diagnostic(diagnostic, out/"diagnostics/result.json")
                for rep in range(job.measured_reps):
                    row = sample(mode)
                    row.update(self.formal_record())
                    row["configuration"] = sample(mode)["configuration"]
                    row.update(job=asdict(job), run_id=run_id, rep_index=rep,
                               diagnostic_file="diagnostics/result.json", same_mode_diagnostic_tokens_equal=True,
                               run_kind="measured", eos_id=151645)
                    save_record(row, out/f"rep{rep}.json", 151645)
                    generated_records.append(row)
                return subprocess.CompletedProcess(command, 0)

            with (
                patch.object(launcher, "__file__", str(root/"experiments/formal/run.py")),
                patch("sys.argv", ["run", "--formal", "--execute"]),
                patch.object(launcher, "check_environment", return_value={"commit_gate_passed": True}),
                patch.object(launcher, "collect_runtime_info", return_value={"git_commit": COMMIT, "dirty_worktree": False}),
                patch.object(launcher.time, "sleep"),
                patch.object(launcher.subprocess, "run", side_effect=fake_worker) as workers,
                redirect_stdout(io.StringIO()),
            ):
                launcher.main()
                self.assertEqual(workers.call_count, 192)
            summary = json.loads(next(root.glob("output/formal_batches/*/summary.json")).read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "formal_batch_verified")
            self.assertEqual(len(summary["completed"]), 192)
            self.assertEqual(len(generated_records), 480)
            self.assertFalse(summary["speedup_claim_allowed"])
            self.assertEqual({r["job"]["family_id"] for r in generated_records}, {f"FF{i:02d}" for i in range(1,17)})


if __name__ == "__main__":
    unittest.main(verbosity=2)
