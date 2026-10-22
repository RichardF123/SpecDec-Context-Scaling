import json
import math
from pathlib import Path
from dataclasses import asdict
from .runtime_audit import validate_boundary_audit

def validate_configuration(record):
    config = record.get("configuration")
    if not isinstance(config, dict):
        raise ValueError("缺少 configuration 配置记录")

    mode = record["job"]["condition"]["mode"]
    if mode not in ("AR", "SpecDec"):
        raise ValueError("未知解码模式")

    expected = {
        "attention_strategy": "cudnn_prefill",
        "diag_precision": {
            "qk_fp32": False,
            "prob_fp32": False,
        },
        "target_verify_mode": "block",
        "target_kv_engine": "fp16",
        "draft_kv_engine": "fp16" if mode == "SpecDec" else None,
        "compute_dtype": "bfloat16",
        "draft_resident": mode == "SpecDec",
    }

    for key, value in expected.items():
        if key not in config:
            raise ValueError(f"配置缺少字段： {key}")
        actual = config[key]
        if actual != value:
            raise ValueError(
                f"配置不符： {key},"
                f"预期 {value!r}, 实际 {actual!r}"
            )
    if type(config["draft_resident"]) is not bool:
        raise ValueError("draft_resident 必须是布尔值")
def validate_record(record, eos_id, *, require_runtime_audit=True):
    validate_configuration(record)
    if record.get("experiment_stage") == "formal":
        from .input_sets import FORMAL_FAMILIES
        if record["job"]["family_id"] not in FORMAL_FAMILIES:
            raise ValueError("正式记录混入非正式 family")
        if record["input"]["input_set"] != "formal" or record["input"]["hash_verified"] is not True:
            raise ValueError("正式输入集合或 hash 校验不合规")
        if record["runtime"]["dirty_worktree"] is not False:
            raise ValueError("正式记录来自 dirty 工作区")
        if not record["collection"]["expected_commit"] or record["collection"]["expected_commit"] != record["runtime"]["git_commit"]:
            raise ValueError("正式记录代码版本与冻结版本不一致")
        if record["collection"]["speedup_claim_allowed"] is not False:
            raise ValueError("原始正式采集不能直接声明加速")
    if require_runtime_audit or "runtime_environment_audit" in record:
        validate_boundary_audit(record.get("runtime_environment_audit"))
    ids = record["generated_ids"]
    length = record["generated_length"]
    limit = record["max_new"]
    if type(limit) is not int or limit != 64:
        raise ValueError("当前实验要求 max_new=64")

    if length != len(ids) or not 0 < length <= limit:
        raise ValueError("生成长度不合法")

    reason = record["stop_reason"]

    if reason == "eos":
        if ids[-1] != eos_id or eos_id in ids[:-1]:
            raise ValueError("EOS 停止记录不合法")
    elif reason == "max_new_reached":
        if length != limit or eos_id in ids:
            raise ValueError("长度上限停止记录不合法")
    else:
        raise ValueError(f"未知停止原因： {reason}")

    times = [
        record["prefill_s"],
        record["decode_s"],
        record["total_s"],
    ]
    if not all(math.isfinite(t) and t > 0 for t in times):
        raise ValueError("时间必须是有限正数")

    if not math.isclose(
        times[2], times[0] + times[1],
        rel_tol = 1e-9, abs_tol = 1e-9,
    ):
        raise ValueError("总时间与阶段时间不一致")
    throughput = record["decode_tokens_per_s"]
    if(
        isinstance(throughput, bool)
        or not isinstance(throughput,(int,float))
        or not math.isfinite(throughput)
        or throughput <= 0
    ):
        raise ValueError("Decode 吞吐量必须是有限正数")

    expected_throughput = length / record["decode_s"]
    if not math.isclose(
        throughput,
        expected_throughput,
        rel_tol = 1e-9,
        abs_tol = 1e-9,
    ):
        raise ValueError("吞吐量与生成长度、decode时间不一致")
    allocated = record["peak_allocated_bytes"]
    reserved = record["peak_reserved_bytes"]

    if type(allocated) is not int or allocated <= 0:
        raise ValueError("活跃分配峰值必须是正整数字节数")

    if type(reserved) is not int or reserved <= 0:
        raise ValueError("保留内存峰值必须是正整数字节数")

    if allocated > reserved:
        raise ValueError("活跃分配峰值不能大于保留内存峰值")

    if record.get("diagnostic_checks_executed") is not False:
        raise ValueError("Measured 记录必须明确关闭诊断检查")
def save_record(record, output_path, eos_id):
    validate_record(record, eos_id)
    payload = json.dumps(
        record, ensure_ascii= False, indent = 2, allow_nan = False
    )

    path = Path(output_path)
    path.parent.mkdir(parents = True, exist_ok = True)

    with path.open("x", encoding = "utf-8") as file:
        file.write(payload)

def save_failure(failure, output_path):
    required = (
        "run_id",
        "job",
        "stage",
        "failure_class",
        "error_type",
        "error_message",
    )

    missing = [key for key in required if key not in failure]
    if missing:
        raise ValueError(f"失败记录缺少字段：{missing}")

    payload = json.dumps(
        failure,
        ensure_ascii = False,
        indent = 2,
        allow_nan = False,
    )

    path = Path(output_path)
    path.parent.mkdir(parents = True, exist_ok = True)

    with path.open("x", encoding = "utf-8") as file:
        file.write(payload)

def verify_job_results(output_dir, job, eos_id, *, require_runtime_audit=True,
                       experiment_stage="development"):
    if experiment_stage not in ("development", "formal"):
        raise ValueError("未知实验阶段")
    directory = Path(output_dir)

    if (directory / "failure.json").exists():
        raise ValueError("任务存在失败记录")

    files = sorted(directory.glob("*.json"))
    if len(files) != job.measured_reps:
        raise ValueError("Measured 文件数量不符合要求")
    diagnostic_path = directory / "diagnostics" / "result.json"
    diagnostic = json.loads(
        diagnostic_path.read_text(encoding="utf-8")
    )

    if diagnostic.get("diagnostic_only") is not True:
        raise ValueError("诊断文件标识错误")
    if diagnostic["job"] != asdict(job):
        raise ValueError("诊断任务身份不一致")
    if diagnostic["experiment_stage"] != experiment_stage:
        raise ValueError("诊断阶段标识不一致")
    coverage = diagnostic["finiteness_coverage"]
    if coverage["n_checked"] <= 0 or coverage["first_bad"] is not None:
        raise ValueError("诊断有限性证据为空或存在失败")
    if job.condition.mode == "SpecDec":
        if diagnostic["decision_coverage"]["ok"] is not True:
            raise ValueError("SpecDec 决策覆盖未通过")
    seen = set()
    run_ids = set()

    for path in files:
        record = json.loads(path.read_text(encoding = "utf-8"))
        validate_record(record, eos_id, require_runtime_audit=require_runtime_audit)
        if record["diagnostic_file"] != "diagnostics/result.json":
            raise ValueError("诊断文件关联错误")
        if record["run_id"] != diagnostic["run_id"]:
            raise ValueError("诊断与 measured 不属于同一次任务")
        if record["same_mode_diagnostic_tokens_equal"] is not True:
            raise ValueError("记录声明同模式输出不一致")
        if record["generated_ids"] != diagnostic["generated_ids"]:
            raise ValueError("实际 token 序列与诊断不一致")
        if record["job"] != asdict(job):
            raise ValueError("结果与启动任务不一致")
        if record["run_kind"] != "measured":
            raise ValueError("混入非 measured 记录")
        if record["experiment_stage"] != experiment_stage:
            raise ValueError("测量记录阶段与批次不一致")
        if record["eos_id"] != eos_id:
            raise ValueError("EOS 配置不一致")

        index = record["rep_index"]
        if type(index) is not int or index in seen:
            raise ValueError("重复编号类型错误或恢复")
        seen.add(index)
        run_ids.add(record["run_id"])

    if seen != set(range(job.measured_reps)):
        raise ValueError("重复编号遗漏或超范围")
    if len(run_ids) != 1 or not next(iter(run_ids)):
        raise ValueError ("运行标识缺失或混入不同运行")
    
    return {"verified": True, "n_measured": len(files)}

def save_diagnostic(record, output_path):
    if record.get("diagnostic_only") is not True:
        raise ValueError("不是独立诊断记录")

    payload = json.dumps(
        record, ensure_ascii = False, indent = 2, allow_nan = False
    )
    path = Path(output_path)
    path.parent.mkdir(parents = True, exist_ok = True)

    with path.open("x", encoding = "utf-8") as file:
        file.write(payload)
