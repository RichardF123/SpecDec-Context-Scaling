"""Worker input routing with mocked tokenization, no corpus/model inference."""
from contextlib import redirect_stderr
import io
import unittest
from unittest.mock import patch

from . import worker


class WorkerInputTests(unittest.TestCase):
    def argv(self, *extra, family="F09"):
        return ["worker", "--family", family, "--ctx", "2048",
                "--mode", "AR", "--block", "0", *extra]

    def test_pilot_default_is_unchanged(self):
        with patch("sys.argv", self.argv()):
            self.assertEqual(worker.parse_args().input_set, "pilot")

    def test_formal_dry_run_allowed(self):
        with patch("sys.argv", self.argv("--input-set", "formal", "--dry-run", family="FF01")):
            self.assertEqual(worker.parse_args().family, "FF01")

    def test_formal_execution_requires_expected_commit(self):
        with patch("sys.argv", self.argv("--input-set", "formal", "--expected-commit", "abc", family="FF01")):
            self.assertEqual(worker.parse_args().expected_commit, "abc")

    def test_cross_set_and_unapproved_execution_rejected(self):
        for args in (self.argv(family="FF01"),
                     self.argv("--input-set", "formal", "--dry-run"),
                     self.argv("--input-set", "formal", family="FF01")):
            with self.subTest(args=args), patch("sys.argv", args), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as exc:
                    worker.parse_args()
                self.assertEqual(exc.exception.code, 2)

    def test_family_rejected_before_tokenizer(self):
        with patch.object(worker.AutoTokenizer, "from_pretrained") as tokenizer:
            with self.assertRaises(ValueError):
                worker.load_input("FF01", 2048)
            tokenizer.assert_not_called()

    def test_selected_manifest_and_hash_check_forwarded(self):
        for name, family in (("pilot", "F09"), ("formal", "FF01")):
            with self.subTest(name=name):
                path, _ = worker.select_input_set(name)
                with (
                    patch.object(worker.AutoTokenizer, "from_pretrained") as tokenizer,
                    patch.object(worker, "load_family_ids", return_value=([10]*2048, {"manifest_path": "old"})) as load,
                    patch.object(worker, "eos_config", return_value={"matches_expected": True}),
                ):
                    ids, meta = worker.load_input(family, 2048, input_set=name)
                    self.assertEqual(len(ids), 2048)
                    self.assertTrue(load.call_args.kwargs["verify_hash"])
                    self.assertEqual(load.call_args.kwargs["ctx"], 2048)
                    self.assertEqual(load.call_args.kwargs["manifest"]["manifest_version"],
                                     worker.load_manifest(path)["manifest_version"])
                    self.assertEqual(meta["input_set"], name)
                    self.assertEqual(meta["manifest_path"], str(path))
                    self.assertEqual(len(meta["manifest_sha256"]), 64)
                    tokenizer.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
