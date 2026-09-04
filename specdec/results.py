"""Append-only results ledger (results.jsonl). Never overwrite history."""
from __future__ import annotations

import json
import os
import threading
from typing import Iterator


class ResultsLedger:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()

    def append(self, record: dict) -> None:
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, default=str, sort_keys=True)
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def read(self) -> Iterator[dict]:
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
