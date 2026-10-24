"""experiments/confirmatory/plan.py — confirmatory measurement plan builder。

统计单位仍是 family。每个 `(family, context)` 形成一个 AR/Spec 配对单元：
* 两种模式仍跑在独立进程；
* 每个 context 内平衡 AB（AR→SpecDec）与 BA（SpecDec→AR），由固定 seed 决定；
* 每个 measurement block 内，family/context 配对顺序独立随机化；
* AR/Spec 以完整相邻 pair 为调度单位；两个 block 在 pair 单位上交错（不是"先跑完 block0
  再统一跑 block1"），避免 block 身份成为早期/晚期运行时的代理变量。
* 每个 family-condition 仍为 5 次 measured：block0 = 3 reps、block1 = 2 reps；每进程 1 次 warm-up。

本模块只做计划与校验；模型计算/生成逻辑全部复用 `experiments/formal/worker.py`。
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from experiments.formal.plan import Condition, Job

ROOT = Path(__file__).resolve().parents[2]
CONTEXTS = (2048, 4096, 8192)
MODES = ("AR", "SpecDec")
BLOCK_REPS = ((0, 3), (1, 2))
SCHEMA_VERSION = "confirmatory-plan-v1"


@dataclass(frozen=True)
class Task:
    family_id: str
    context_tokens: int
    mode: str
    measurement_block: int
    measured_reps: int
    pair_id: str
    pair_order: str          # "AB"（AR 先）或 "BA"（SpecDec 先）
    pair_position: int       # 0=配对内先执行，1=后执行
    planned_order: int       # 全序列中的执行位置（0 起）

    @property
    def condition_id(self) -> str:
        return f"L{self.context_tokens}_{self.mode}_base_bf16"

    def to_job(self) -> Job:
        """复用正式 `Job`/`Condition`，worker 与 records 校验原样使用。"""
        return Job(
            family_id=self.family_id,
            condition=Condition(self.condition_id, self.context_tokens, self.mode,
                                "base_bf16"),
            measurement_block=self.measurement_block,
            measured_reps=self.measured_reps,
        )

    def key(self):
        return (self.family_id, self.context_tokens, self.mode, self.measurement_block)


def build_conditions() -> list:
    return [Condition(f"L{ctx}_{mode}_base_bf16", ctx, mode, "base_bf16")
            for ctx in CONTEXTS for mode in MODES]


def _pair_order_map(families, contexts, seed: int) -> dict:
    """按 context 分层平衡 AB/BA，并让同一 (family, context) 两个 block 共用顺序。"""
    mapping = {}
    for context_index, ctx in enumerate(contexts):
        order = list(families)
        random.Random(seed * 1000 + int(ctx)).shuffle(order)
        n = len(order)
        # family 数为奇数时，在 context 间交替多出的 AB/BA，保持全局尽量平衡。
        n_ab = n // 2 + (1 if n % 2 and context_index % 2 == 0 else 0)
        for idx, family in enumerate(order):
            mapping[(family, ctx)] = "AB" if idx < n_ab else "BA"
    return mapping


def build_plan(families, *, contexts=CONTEXTS, seed: int = 4207) -> list:
    families = tuple(families)
    contexts = tuple(contexts)
    if not families or len(set(families)) != len(families):
        raise ValueError("family 名单为空或存在重复")
    if len(set(contexts)) != len(contexts):
        raise ValueError("context 名单存在重复")
    pair_order = _pair_order_map(families, contexts, seed)

    block_pairs = {0: [], 1: []}
    for block, reps in BLOCK_REPS:
        pairs = [(f, c) for f in families for c in contexts]
        rng = random.Random(seed + block)      # 每个 block 独立随机化配对顺序
        rng.shuffle(pairs)
        for family, ctx in pairs:
            order = pair_order[(family, ctx)]
            modes = ("AR", "SpecDec") if order == "AB" else ("SpecDec", "AR")
            pair_tasks = []
            for position, mode in enumerate(modes):
                pair_tasks.append(Task(
                    family_id=family, context_tokens=ctx, mode=mode,
                    measurement_block=block, measured_reps=reps,
                    pair_id=f"{family}_{ctx}", pair_order=order,
                    pair_position=position, planned_order=-1))
            block_pairs[block].append(pair_tasks)

    # 交错单位是完整 AR/Spec pair：pair 内连续，两个 block 在 pair 之间交替。
    interleaved = []
    for i in range(max(len(block_pairs[0]), len(block_pairs[1]))):
        if i < len(block_pairs[0]):
            interleaved.extend(block_pairs[0][i])
        if i < len(block_pairs[1]):
            interleaved.extend(block_pairs[1][i])
    return [Task(**{**asdict(task), "planned_order": idx})
            for idx, task in enumerate(interleaved)]


def validate_plan(tasks, families, *, contexts=CONTEXTS) -> dict:
    """校验任务集合、配对、AB/BA 平衡、block 交错、重复次数。"""
    families = tuple(families)
    contexts = tuple(contexts)
    expected_keys = {
        (family, ctx, mode, block)
        for family in families for ctx in contexts for block, _ in BLOCK_REPS
        for mode in MODES
    }
    seen = set()
    for task in tasks:
        key = task.key()
        if key in seen:
            raise ValueError(f"重复任务: {key}")
        if key not in expected_keys:
            raise ValueError(f"计划外任务: {key}")
        if task.measured_reps != dict(BLOCK_REPS)[task.measurement_block]:
            raise ValueError(f"measurement reps 错误: {key}")
        if task.pair_order not in ("AB", "BA"):
            raise ValueError(f"非法 pair_order: {task}")
        if task.mode == "AR":
            if (task.pair_order == "AB") != (task.pair_position == 0):
                raise ValueError(f"AB/BA 与 pair_position 不一致: {key}")
        else:
            if (task.pair_order == "BA") != (task.pair_position == 0):
                raise ValueError(f"AB/BA 与 pair_position 不一致: {key}")
        seen.add(key)
    missing = expected_keys - seen
    if missing:
        raise ValueError(f"缺少任务: {sorted(missing)}")

    # 每个 (family, context, block) 恰含 AR 与 SpecDec 各一
    pair_blocks = {}
    for task in tasks:
        pair_blocks.setdefault((task.family_id, task.context_tokens, task.measurement_block),
                               set()).add(task.mode)
    bad = [k for k, modes in pair_blocks.items() if modes != set(MODES)]
    if bad:
        raise ValueError(f"配对不完整: {bad}")

    # 配对内 pair_id 一致；AB/BA 全局与每个 context 均平衡。
    pair_orders = {}
    for task in tasks:
        pair_orders.setdefault(task.pair_id, set()).add(task.pair_order)
    if any(len(v) != 1 for v in pair_orders.values()):
        raise ValueError("同一配对出现多种 pair_order")
    n_ab = sum(1 for v in pair_orders.values() if "AB" in v)
    n_ba = len(pair_orders) - n_ab
    if abs(n_ab - n_ba) > 1:
        raise ValueError(f"AB/BA 不平衡: AB={n_ab} BA={n_ba}")
    order_by_context = {}
    for ctx in contexts:
        ctx_orders = {
            pair_id: next(iter(values))
            for pair_id, values in pair_orders.items()
            if pair_id.endswith(f"_{ctx}")
        }
        ctx_ab = sum(value == "AB" for value in ctx_orders.values())
        ctx_ba = len(ctx_orders) - ctx_ab
        if abs(ctx_ab - ctx_ba) > 1:
            raise ValueError(f"context={ctx} 的 AB/BA 不平衡: AB={ctx_ab} BA={ctx_ba}")
        order_by_context[str(ctx)] = {"AB": ctx_ab, "BA": ctx_ba}

    # planned_order 连续且唯一
    orders = sorted(task.planned_order for task in tasks)
    if orders != list(range(len(tasks))):
        raise ValueError("planned_order 不连续/不唯一")

    # 每个 (family, context, block) 的 AR/Spec 必须在全局序列中相邻。
    ordered_tasks = sorted(tasks, key=lambda t: t.planned_order)
    pair_positions = {}
    for task in ordered_tasks:
        key = (task.family_id, task.context_tokens, task.measurement_block)
        pair_positions.setdefault(key, []).append(task.planned_order)
    non_adjacent = [key for key, pos in pair_positions.items()
                    if sorted(pos)[1] - sorted(pos)[0] != 1]
    if non_adjacent:
        raise ValueError(f"AR/Spec pair 未相邻: {non_adjacent[:4]}")

    # block 交错：不出现连续 >2 的同 block；前 8 与后 8 个任务都包含两个 block
    blocks = [task.measurement_block for task in sorted(tasks, key=lambda t: t.planned_order)]
    run = 1
    max_run = 1
    for a, b in zip(blocks, blocks[1:]):
        run = run + 1 if a == b else 1
        max_run = max(max_run, run)
    if max_run > 2:
        raise ValueError(f"block 出现 {max_run} 连排，交错不足")
    if len(blocks) >= 8:
        if set(blocks[:8]) != {0, 1} or set(blocks[-8:]) != {0, 1}:
            raise ValueError("block 在序列首尾没有交错出现")
    transitions = sum(1 for a, b in zip(blocks, blocks[1:]) if a != b)
    if len(blocks) >= 4 and transitions < len(blocks) // 4:
        raise ValueError("block 切换次数过少（可能退化为前后两段）")

    return dict(n_tasks=len(tasks),
                n_measured=sum(t.measured_reps for t in tasks),
                n_pairs=len(pair_orders), n_ab=n_ab, n_ba=n_ba,
                pair_order_by_context=order_by_context,
                adjacent_pair_units=len(pair_positions),
                block_interleave={"max_same_block_run": max_run,
                                  "transitions": transitions,
                                  "blocks": blocks})


def save_plan(tasks, families, output_path, *, contexts=CONTEXTS, seed=4207,
              batch_id=None, provenance=None, input_manifest=None) -> Path:
    summary = validate_plan(tasks, families, contexts=contexts)
    manifest_hash = None
    manifest_path = None
    if input_manifest is not None:
        manifest_path = Path(input_manifest)
        manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    data = {
        "schema_version": SCHEMA_VERSION,
        "phase": "confirmatory",
        "experiment": "specdec_confirmatory_pilot",
        "batch_id": batch_id,
        "seed": seed,
        "family_ids": list(families),
        "contexts": list(contexts),
        "modes": list(MODES),
        "block_reps": [list(x) for x in BLOCK_REPS],
        "input_manifest": str(manifest_path) if manifest_path else None,
        "input_manifest_sha256": manifest_hash,
        "n_tasks": summary["n_tasks"],
        "n_measured": summary["n_measured"],
        "n_pairs": summary["n_pairs"],
        "pair_order_counts": {"AB": summary["n_ab"], "BA": summary["n_ba"]},
        "block_interleave": summary["block_interleave"],
        "tasks": [asdict(task) for task in tasks],
        "claims": {
            "speedup_claim_allowed": False,
            "family_is_statistical_unit": True,
            "8k_speedup_ratio": ("confirmatory batch 满足完整性/环境门/有限值/测量稳定性后"
                                 "才可计算 family-level T_AR/T_Spec；否则 8K 保留绝对指标并"
                                 "标为 restricted；不回填第一轮正式批次 8K ratio"),
        },
        "provenance": provenance or {},
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    return path


def load_plan(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
