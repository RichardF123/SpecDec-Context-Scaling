"""RunConfig + reproducibility hashing (Phase 0 scaffold).

Reproducibility chain (runbook v3):
  run_id = sha256(config_hash | env_profile)
  config_hash = sha256(canonical config JSON WITHOUT seed)   # seed is a knob, not identity
  env_profile = python/system/machine snapshot (extend later with torch/CUDA/git)
The fixed seed is stored INSIDE each results row, never part of run_id.
"""
from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone


@dataclass
class RunConfig:
    # ladder stage: L1..L5, L4b
    stage: str = "L1"
    # target model (Apache-2.0 preferred)
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    seed: int = 42
    # weight precision: fp16 | int8 | int4
    dtype: str = "fp16"
    # KV-cache precision: fp16 | fp8 | int4 | int2
    kv_quant: str = "fp16"
    max_seq_len: int = 2048
    # draft: none | small_model | ngram | prompt_lookup | eagle_head
    draft: str = "none"
    draft_model_id: str = ""
    # fixed-chain draft length (MUST L5 path)
    spec_k: int = 4
    benchmark: dict = field(
        default_factory=lambda: {
            "prompts": 32,
            "max_new_tokens": 128,
            "warmup": 2,
            "reps": 3,
        }
    )
    created_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # --- helpers -------------------------------------------------------

    def canonical_json(self, *, include_seed: bool = True) -> str:
        d = asdict(self)
        d.pop("created_utc", None)  # timestamps never participate in hashing
        if not include_seed:
            d.pop("seed", None)
        return json.dumps(d, sort_keys=True, ensure_ascii=False, default=str)

    def config_hash(self) -> str:
        """Identity of the experiment config; the seed is deliberately excluded."""
        return hashlib.sha256(
            self.canonical_json(include_seed=False).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def env_profile() -> str:
        return json.dumps(
            {
                "python": platform.python_version(),
                "system": platform.system(),
                "machine": platform.machine(),
            },
            sort_keys=True,
        )

    def run_id(self) -> str:
        core = self.config_hash() + "|" + self.env_profile()
        return hashlib.sha256(core.encode("utf-8")).hexdigest()[:32]

    def to_record(self) -> dict:
        rec = asdict(self)
        rec["config_hash"] = self.config_hash()
        rec["run_id"] = self.run_id()
        return rec
