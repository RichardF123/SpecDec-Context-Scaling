"""CPU contract tests; no model loading or CUDA workload."""

import ast
from copy import deepcopy
from pathlib import Path
import unittest


def load_check_round():
    # Extract the production checker without importing the model dependencies.
    path = Path(__file__).with_name("diagnostics.py")
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    diagnose = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "diagnose_spec"
    )
    checker = next(
        node for node in diagnose.body
        if isinstance(node, ast.FunctionDef) and node.name == "check_round"
    )
    namespace = {"trace": []}
    module = ast.Module(body=[checker], type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["check_round"]


class RoundContractTests(unittest.TestCase):
    def setUp(self):
        self.row = {
            "Kp": 4, "cands": [10, 20, 30, 40],
            "g": [10, 20, 99, 40], "m": 2,
            "emitted": [10, 20, 99], "pre": 2048,
            "tgt_len": 2051, "drf_len": 2051,
        }

    def reject(self, key, value, message):
        row = deepcopy(self.row)
        row[key] = value
        with self.assertRaisesRegex(RuntimeError, message):
            load_check_round()(row)

    def test_replacement(self):
        load_check_round()(self.row)

    def test_all_accepted(self):
        row = deepcopy(self.row)
        row.update(g=list(row["cands"]), m=4,
                   emitted=list(row["cands"]), tgt_len=2052, drf_len=2052)
        load_check_round()(row)

    def test_bad_accept_count(self):
        self.reject("m", 3, "接受长度")

    def test_bad_replacement(self):
        self.reject("emitted", [10, 20, 30], "替换")

    def test_bad_target_length(self):
        self.reject("tgt_len", 2050, "缓存长度")

    def test_bad_draft_length(self):
        self.reject("drf_len", 2050, "缓存长度")

    def test_bad_candidate_count(self):
        self.reject("Kp", 0, "候选数量")

    def test_missing_decision_row(self):
        self.reject("g", [10], "判断行数量")

    def test_continuous_rounds(self):
        checker = load_check_round()
        checker(deepcopy(self.row))
        row = deepcopy(self.row)
        row.update(pre=2051, tgt_len=2054, drf_len=2054)
        checker(row)

    def test_discontinuous_rounds(self):
        checker = load_check_round()
        checker(deepcopy(self.row))
        with self.assertRaisesRegex(RuntimeError, "跨轮"):
            checker(deepcopy(self.row))


class HookRestorationTests(unittest.TestCase):
    def check_restoration(self, fail):
        from specdec.engine import model_self, spec_dec
        from specdec.engine.smoke_checks import CheckRegistry
        from specdec.engine.smoke_hooks import DecisionHooks, _HOOK_REGISTRY

        names = ("_prefill", "_propose", "_verify", "_step")
        owner = model_self.QwenLM
        originals = tuple(getattr(spec_dec, name) for name in names)
        original_run = owner._run
        previous_registry = dict(_HOOK_REGISTRY)
        hooks = DecisionHooks(
            tgt=object(), drf=object(), registry=CheckRegistry(),
            prefill_impl=lambda *args: None,
        )
        try:
            try:
                hooks.install()
                self.assertIsNot(spec_dec._prefill, originals[0])
                self.assertIsNot(owner._run, original_run)
                if fail:
                    raise RuntimeError("synthetic failure")
            except RuntimeError as exc:
                if not fail or str(exc) != "synthetic failure":
                    raise
            finally:
                hooks.remove()
            for name, original in zip(names, originals):
                self.assertIs(getattr(spec_dec, name), original)
            self.assertIs(owner._run, original_run)
            self.assertFalse(hooks.installed)
        finally:
            # Keep these tests isolated from other tests in the same process.
            for name, original in zip(names, originals):
                setattr(spec_dec, name, original)
            owner._run = original_run
            _HOOK_REGISTRY.clear()
            _HOOK_REGISTRY.update(previous_registry)

    def test_normal_exit(self):
        self.check_restoration(False)

    def test_exception_exit(self):
        self.check_restoration(True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
