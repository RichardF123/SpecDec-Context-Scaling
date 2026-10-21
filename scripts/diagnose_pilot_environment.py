"""One monitored development task; never part of primary timing evidence."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]


def main():
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    out = ROOT / "output/pilot_environment_diagnostic" / tag
    out.mkdir(parents=True, exist_ok=False)
    fields = ["timestamp", "pstate", "clocks.sm", "clocks.mem", "power.draw",
              "temperature.gpu", "utilization.gpu", "memory.used", "memory.total"]
    command = [str(ROOT / ".venv/Scripts/python.exe"), "-u", "-X", "utf8", "-m",
               "experiments.formal.worker", "--family", "F09", "--ctx", "4096",
               "--mode", "SpecDec", "--block", "0", "--output-dir", str(out / "worker_results")]
    metadata = {
        "diagnostic_only": True, "eligible_for_primary_timing": False,
        "selection": "Largest within-task spread in completed pilot: F09/4096/SpecDec/block0",
        "source_batch": "1f6b884a4b5946b09d4ee9a381507b21",
        "command": command, "sampling_interval_ms": 500,
        "limitations": "External sampling perturbs execution. Output receipt times are approximate phase anchors, not CUDA events. No automatic retries.",
    }
    (out / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Diagnostic directory:", out, flush=True)
    start = time.perf_counter()
    samples, errors = [], []
    monitor = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits", "-lms", "500"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
    )

    def collect():
        for line in monitor.stdout:
            values = next(csv.reader([line.strip()]))
            if len(values) == len(fields):
                samples.append(dict(elapsed_s=time.perf_counter()-start,
                                    **dict(zip(fields, [v.strip() for v in values]))))
            else:
                errors.append(line.strip())

    thread = threading.Thread(target=collect, daemon=True)
    thread.start()
    worker = None
    events = []
    try:
        worker = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True,
                                  encoding="utf-8", errors="replace")
        with (out / "worker.log").open("x", encoding="utf-8") as log:
            for line in worker.stdout:
                elapsed = time.perf_counter()-start
                log.write(f"[{elapsed:.6f}] {line}")
                log.flush()
                if "Measured " in line or "warm_up" in line:
                    events.append({"elapsed_s": elapsed, "text": line.strip()})
                print(line.rstrip(), flush=True)
        metadata["returncode"] = worker.wait()
    finally:
        if worker is not None and worker.poll() is None:
            worker.terminate()
            worker.wait()
        monitor.terminate()
        monitor.wait()
        thread.join(timeout=5)
        metadata.update(sample_count=len(samples), telemetry_errors=errors,
                        phase_output_events=events, elapsed_s=time.perf_counter()-start)
        (out / "summary.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        if samples:
            with (out / "telemetry.csv").open("x", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(samples[0]))
                writer.writeheader()
                writer.writerows(samples)
    print("Samples:", len(samples), "returncode:", metadata.get("returncode"), flush=True)
    return metadata.get("returncode", 1)


if __name__ == "__main__":
    raise SystemExit(main())
