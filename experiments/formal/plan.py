from dataclasses import dataclass, asdict
import json
import random
from pathlib import Path

@dataclass(frozen=True)
class Condition:
    condition_id: str
    context_tokens: int
    mode: str
    target_kv: str

@dataclass(frozen=True)
class Job:
    family_id: str
    condition: Condition
    measurement_block: int
    measured_reps: int

def build_condition():
    conditions = []

    for context_tokens in (2048, 4096, 8192):
        for mode in("AR", "SpecDec"):
            condition = Condition(
                condition_id = f"L{context_tokens}_{mode}_base_bf16",
                context_tokens = context_tokens,
                mode = mode,
                target_kv = "base_bf16"
            )
            conditions.append(condition)
    return conditions

def build_jobs(family_ids, seed = 4207):
    jobs = []
    conditions = build_condition()

    for block_id, reps in ((0,3), (1,2)):
        block_jobs = []

        for family_id in family_ids:
            for condition in conditions:
                job = Job(
                    family_id = family_id,
                    condition = condition,
                    measurement_block = block_id,
                    measured_reps = reps
                )
                block_jobs.append(job)
        rng = random.Random(seed + block_id)
        rng.shuffle(block_jobs)
        jobs.extend(block_jobs)
    return jobs

def validate_plan(jobs, family_ids):
    families = tuple(family_ids)

    if not families or len(set(families)) != len(families):
        raise ValueError("family 名单为空或存在重复")

    conditions = build_condition()
    expected = {
        (family, condition, block): reps
        for family in families
        for condition in conditions
        for block, reps in ((0,3),(1,2))
    }

    seen = set()
    for job in jobs:
        key = (
            job.family_id,
            job.condition,
            job.measurement_block,
        )
        if key in seen:
            raise ValueError(f"重复任务: {key}")
        if key not in expected:
            raise ValueError(f"未知任务: {key}")
        if job.measured_reps != expected[key]:
            raise ValueError(f"测量次数错误：{key}")
        seen.add(key)
    missing = set(expected) - seen
    if missing:
        raise ValueError(f"缺少任务: {missing}")

def test_validate_plan():
    families = ("F09", "F03", "F11", "F04")
    jobs = build_jobs(families)

    validate_plan(jobs, families)

    def expect_error(name, bad_jobs, bad_families):
        try:
            validate_plan(bad_jobs, bad_families)
        except ValueError as exc:
            print(f"{name}：成功拦截，{exc}")
        else:
            raise AssertionError(f"{name}：错误计划竟然通过了")

    expect_error("重复任务", jobs + [jobs[0]], families)
    expect_error("遗漏任务", jobs[1:], families)

    wrong_job = Job(
        family_id=jobs[0].family_id,
        condition=jobs[0].condition,
        measurement_block=jobs[0].measurement_block,
        measured_reps=99,
    )
    expect_error("次数错误", [wrong_job] + jobs[1:], families)

    unknown_job = Job(
        family_id="F99",
        condition=jobs[0].condition,
        measurement_block=jobs[0].measurement_block,
        measured_reps=jobs[0].measured_reps,
    )
    expect_error("计划外任务", [unknown_job] + jobs[1:], families)
    expect_error("Family 重复", jobs, families + ("F09",))
    expect_error("Family 为空", jobs, ())

    print("计划校验自测：全部通过")
    

def save_plan(jobs, family_ids, seed, output_path):
    validate_plan(jobs, family_ids)

    data = {
        "schema_version": "teaching-plan-v1",
        "phase": "pilot",
        "seed": seed,
        "family_ids": list(family_ids),
        "n_jobs": len(jobs),
        "n_measured": sum(job.measured_reps for job in jobs),
        "jobs": [asdict(job) for job in jobs],
    }

    path = Path(output_path)
    path.parent.mkdir(parents = True, exist_ok = True)
    with path.open("x",encoding = "utf-8") as file:
        json.dump(data, file, ensure_ascii = False, indent = 2)
def test_saved_plan(jobs, output_path):
    path = Path(output_path)

    with path.open("r", encoding = "utf-8") as file:
        data = json.load(file)
    assert data["n_jobs"] == len(jobs)
    assert data["n_measured"] == sum(
        job.measured_reps for job in jobs
    )
    assert data["jobs"] == [asdict(job) for job in jobs]
    print("JSON 读回检查： 内容和任务顺序一致")

    original = path.read_bytes()

    try:
        save_plan(
            jobs,
            data["family_ids"],
            data["seed"],
            path,
        )
    except FileExistsError:
        print("JSON 读回检查： 文件已存在，保存失败")
    else:
        raise AssertionError("已有文件竟然被允许覆盖")
    print("原文件检查：内容未改变")

if __name__ == "__main__":
    families = ("F09", "F03", "F11", "F04")
    jobs = build_jobs(families)
    assert len(jobs) == 48
    assert sum(job.measured_reps for job in jobs) == 120
    assert jobs == build_jobs(families, seed=4207)

    assert all(job.measurement_block == 0 for job in jobs[:24])
    assert all(job.measurement_block == 1 for job in jobs[24:])
    jobs = build_jobs(families)
    validate_plan(jobs, families)
    print("计划校验通过")
    test_validate_plan()
    test_saved_plan(
    jobs,
    "output/formal_teaching/pilot_plan_001.json",
)