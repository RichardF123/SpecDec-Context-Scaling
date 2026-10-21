import hashlib
import os
import subprocess
import sys
from pathlib import Path

import torch

def collect_runtime_info():
    root = Path(__file__).resolve().parents[2]

    def git(*arguments):
        return subprocess.check_output(
            ["git", *arguments],
            cwd = root,
            text = True,
            encoding = "utf-8",
        ).strip()

    commit = git("rev-parse", "HEAD")
    changes = git("status", "--porcelain", "--untracked-files=all")

    sources = {}
    for path in sorted((root / "experiments" / "formal").glob("*.py")):
        relative = path.relative_to(root).as_posix()
        sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()

    return {
        "git_commit": commit,
        "dirty_worktree": bool(changes),
        "git_changes": changes.splitlines(),
        "runner_source_sha256": sources,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "torch_version": str(torch.__version__),
        "torch_cuda_version": torch.version.cuda,
        "torch_num_threads": torch.get_num_threads(),
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
    }

if __name__ == "__main__":
    import json
    print(json.dumps(collect_runtime_info(), ensure_ascii=False, indent=2))