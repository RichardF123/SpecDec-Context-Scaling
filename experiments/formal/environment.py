from specdec.engine.g5_8k_spec_variance_probe import system_memory
import torch
from specdec.engine import gpu_pstate_preflight

class EnvironmentRefused(RuntimeError):
    pass

def check_environment(min_commit_gib=16):
    memory = system_memory()

    if not memory.get("available"):
        raise EnvironmentRefused(
            f"无法读取系统内存： {memory.get('error','未知原因')}"
        )

    free_gib= memory["commit_free_mib"] / 1024

    if free_gib < min_commit_gib:
        raise EnvironmentRefused(
            f"可用 commit {free_gib:.2f} GiB, 低于 {min_commit_gib} GiB 安全线"
        )

    return {
        "commit_gate_passed": True,
        "commit_min_gib": min_commit_gib,
        "system_memory": memory
    }
def check_gpu_environment():
    if not torch.cuda.is_available():
        raise EnvironmentRefused("CUDA不可用")

    verdict = gpu_pstate_preflight.qualify()

    if not verdict.get("available") or not verdict.get("qualified"):
        error = EnvironmentRefused(
            f"GPU 负载期预检未通过： {verdict.get('reason')}"
        )
        error.evidence = verdict
        raise error

    return {
        "gpu_name": torch.cuda.get_device_name(0),
        "pstate_preflight": verdict,
        "thresholds": {
            "min_sm_mhz": gpu_pstate_preflight.MIN_SM,
            "min_mem_mhz": gpu_pstate_preflight.MIN_MEM,
        },
    }

if __name__ == "__main__":
    print(check_environment())
