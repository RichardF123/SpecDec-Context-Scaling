"""CPU-only tests for configuration validation and its record entry point."""

from copy import deepcopy
import unittest

from .records import validate_configuration, validate_record


def sample(mode="AR"):
    return {
        "job": {"condition": {"mode": mode}},
        "configuration": {
            "attention_strategy": "cudnn_prefill",
            "diag_precision": {"qk_fp32": False, "prob_fp32": False},
            "target_verify_mode": "block",
            "target_kv_engine": "fp16",
            "draft_kv_engine": "fp16" if mode == "SpecDec" else None,
            "compute_dtype": "bfloat16",
            "draft_resident": mode == "SpecDec",
        },
        "generated_ids": [10] * 64,
        "generated_length": 64,
        "max_new": 64,
        "stop_reason": "max_new_reached",
        "prefill_s": 1.0, "decode_s": 2.0, "total_s": 3.0,
        "decode_tokens_per_s": 32.0,
        "peak_allocated_bytes": 1024,
        "peak_reserved_bytes": 2048,
        "diagnostic_checks_executed": False,
    }


class ConfigurationTests(unittest.TestCase):
    def test_valid_modes(self):
        for mode in ("AR", "SpecDec"):
            with self.subTest(mode=mode):
                validate_record(sample(mode), 151645)

    def test_missing_configuration(self):
        row = sample()
        del row["configuration"]
        with self.assertRaisesRegex(ValueError, "configuration"):
            validate_record(row, 151645)

    def test_missing_each_field(self):
        for key in sample()["configuration"]:
            with self.subTest(key=key):
                row = sample()
                del row["configuration"][key]
                with self.assertRaisesRegex(ValueError, "缺少字段"):
                    validate_configuration(row)

    def test_wrong_values(self):
        for mode in ("AR", "SpecDec"):
            for key in sample(mode)["configuration"]:
                with self.subTest(mode=mode, key=key):
                    row = sample(mode)
                    row["configuration"][key] = "invalid"
                    with self.assertRaisesRegex(ValueError, "配置不符"):
                        validate_record(row, 151645)

    def test_unknown_mode(self):
        row = sample()
        row["job"]["condition"]["mode"] = "unknown"
        with self.assertRaisesRegex(ValueError, "未知"):
            validate_configuration(row)

    def test_integer_is_not_residency_boolean(self):
        for mode, value in (("AR", 0), ("SpecDec", 1)):
            with self.subTest(mode=mode):
                row = deepcopy(sample(mode))
                row["configuration"]["draft_resident"] = value
                with self.assertRaisesRegex(ValueError, "布尔值"):
                    validate_configuration(row)

    def test_bad_throughput(self):
        for value in (True, "32", float("nan"), float("inf"), 0, -1, 31):
            with self.subTest(value=value):
                row = sample()
                row["decode_tokens_per_s"] = value
                with self.assertRaises(ValueError):
                    validate_record(row, 151645)

    def test_bad_memory(self):
        for key in ("peak_allocated_bytes", "peak_reserved_bytes"):
            for value in (True, 1.5, "1024", 0, -1):
                with self.subTest(key=key, value=value):
                    row = sample()
                    row[key] = value
                    with self.assertRaises(ValueError):
                        validate_record(row, 151645)
        row = sample()
        row["peak_allocated_bytes"] = 4096
        with self.assertRaises(ValueError):
            validate_record(row, 151645)

    def test_measured_diagnostic_flag(self):
        for value in (True, None, 0, "False"):
            with self.subTest(value=value):
                row = sample()
                row["diagnostic_checks_executed"] = value
                with self.assertRaises(ValueError):
                    validate_record(row, 151645)
        row = sample()
        del row["diagnostic_checks_executed"]
        with self.assertRaises(ValueError):
            validate_record(row, 151645)


if __name__ == "__main__":
    unittest.main(verbosity=2)
