"""stage2b_preflight_cpu.py — **Stage 2b 开跑前只读预检**（纯 CPU；不加载模型、不使用 GPU）。

回答四个问题：
1. 新 family 的**来源与 hash**：manifest 条目齐全、`prefixes[8192].token_sha1` 可**复现**（`verify_hash=True`）；
2. **独立性**：窗口两两不重叠；按仓库口径（`SHINGLE=32`）与其它新 family、与已用 family、
   与调试输入**零交集**；且**从未被任何产物消费**；
3. **守卫登记**：这些 family 当前是否在授权范围内（不在 ⇒ 必须由 D-3 批准登记，否则跑不了）；
4. **工具就绪**：Stage 2b 的判决/核对/runner 是否就位，且 `--tag/--families/--reps` 是否**必填**
   （防止静默读错 tag 或忽略 family）。

```powershell
python -X utf8 -m specdec.engine.stage2b_preflight_cpu
python -X utf8 -m specdec.engine.stage2b_preflight_cpu --families F09,F03,F11,F04
```

退出码：`0` = 可开跑（且 family 已登记）；`3` = **工具就绪但 family 未登记（待 D-3 批准）**；`1` = 存在硬问题。
"""
from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_NEW = ("F09", "F03", "F11", "F04")
USED = ("F13", "F01", "F02", "F08", "F10")
CTX = 8192


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default=",".join(DEFAULT_NEW))
    ap.add_argument("--ctx", type=int, default=CTX)
    a = ap.parse_args()
    new = tuple(x.strip() for x in a.families.split(",") if x.strip())

    from . import verify_family_independence_cpu as G
    from . import stage2b_verdict as SV
    from . import family_input as FI

    hard: list = []
    print(f"== Stage 2b 预检（只读）== 待新增 family={list(new)} ctx={a.ctx}")

    print("-- 1) manifest 来源与 token hash")
    ids_by_fam: dict = {}
    man = FI.load_manifest()
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct",
                                            trust_remote_code=True, local_files_only=True)
    except Exception as e:                                                # noqa: BLE001
        tok = None
        print(f"   （无法加载 tokenizer，跳过 token_sha1 复现：{type(e).__name__}: {str(e)[:80]}）")
    for fam in new + USED:
        rec = FI.family_record(fam, man)
        line = (f"   {fam}: stratum={rec.get('stratum')} window={rec.get('window_token_start')} "
                f"text_sha1={rec.get('text_sha1')}")
        if tok is not None and fam in new:
            try:
                ids_list, meta = FI.load_family_ids(tok, fam, manifest=man, ctx=a.ctx, verify_hash=True)
                ids_by_fam[fam] = list(ids_list)
                line += f" token_sha1={meta.get('token_sha1')} hash_verified={meta.get('hash_verified')}"
                if meta.get("hash_verified") is not True:
                    hard.append(f"{fam}: token_sha1 未通过校验")
            except Exception as e:                                        # noqa: BLE001
                hard.append(f"{fam}: token_sha1 复现失败 {type(e).__name__}: {str(e)[:80]}")
        print(line)

    print("-- 2) 独立性（窗口 + 32-gram，仓库口径 SHINGLE=%d）" % G.SHINGLE)
    starts = {f: (int(FI.family_record(f, man)["window_token_start"]),
                  int(FI.family_record(f, man)["window_token_start"]) + a.ctx) for f in new + USED}
    for i, x in enumerate(new):
        for y in new[i + 1:]:
            ov = max(0, min(starts[x][1], starts[y][1]) - max(starts[x][0], starts[y][0]))
            print(f"   窗口重叠 {x} vs {y} = {ov}")
            if ov:
                hard.append(f"{x}/{y}: 窗口重叠 {ov} token")
    if ids_by_fam:
        dbg_sh = None
        try:
            dbg, _ = G._debug_ids(tok)
            dbg_sh = G._shingles(dbg)
        except Exception as e:                                            # noqa: BLE001
            print(f"   （无法构造调试输入：{type(e).__name__}）")
        for fam, ids in ids_by_fam.items():
            sh = G._shingles(ids)
            others = {f: len(sh & G._shingles(ids_by_fam[f])) for f in new if f != fam and f in ids_by_fam}
            used_ov = {f: len(sh & G._shingles(ids_by_fam[f])) for f in USED if f in ids_by_fam}
            dbg_ov = (len(sh & dbg_sh) if dbg_sh is not None else None)
            print(f"   {fam}: 32-gram 交集 新={others} 已用={used_ov or '（未重建）'} 调试输入={dbg_ov}")
            if any(others.values()) or any(used_ov.values()) or dbg_ov:
                hard.append(f"{fam}: 存在非零 32-gram 交集")
    # 2026-10-04（批 1 已批准）：Stage 2b 的 24 次 run **合法地**消费这四个 family
    # （运行器记录 + 其 `smoke_artifact` 链接的 harness smoke 产物）。
    # 2026-10-05：**授权 tag 集合**扩为 {div8209_p2（批 1）, div8209_b6（B6 轮，P-A/P-B 授权）}；
    # `tool` 与 `families` 口径不变（见守卫的 `AUTHORIZED_ROUNDS`）。
    # ⇒ 这里只统计**授权范围之外**的消费；授权口径与守卫**完全一致**（逐份链接，不按 family 泛指）：
    #    * 文件属于任一授权 tag（`<tag>_*`）；
    #    * 或文件名出现在被授权 run 记录 `smoke_artifact` 指向的集合里。
    _auth_prefixes = tuple(f"{t}_" for t in G.AUTHORIZED_RUN_TAGS)
    _auth_linked = G._stage2b_authorized_smoke_files()
    # 本仓库**自己的 CPU 工具**产出的汇报文件（例如守卫的 `family_independence_cpu_check.json`）
    # 会列出候选 family 名字，但那**不是**"消费 family 数据"（也不是 run）⇒ 单列为"仅提及"。
    # ⚠️ 白名单**只含本仓库 CPU 工具的汇报产物**；任何**数据/运行产物**在授权范围之外出现即硬失败。
    _report_like = ("family_independence_cpu_check.json",)
    consumed, authorized_hits, mention_only = [], [], []
    for p in sorted((ROOT / "output" / "g5").rglob("*.json")):
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:                                                 # noqa: BLE001
            continue
        for fam in new:
            if f'"{fam}"' in txt:
                if p.name.startswith(_auth_prefixes) or p.name in _auth_linked:
                    authorized_hits.append((p.name, fam))
                elif p.name in _report_like:
                    mention_only.append((p.name, fam))
                else:
                    consumed.append((p.name, fam))
    print(f"   授权范围内的消费（Stage 2b 授权 tag + 其链接的 smoke）= {len(authorized_hits)} 份"
          f"（示例 {authorized_hits[:2]}）")
    print(f"   本仓库 CPU 工具汇报的**仅提及**（不计为消费）= {sorted({x[0] for x in mention_only})}")
    print(f"   **授权范围之外**的消费 = {len(consumed)} {consumed[:3]}")
    if consumed:
        hard.append(f"新 family 被授权范围之外的产物消费：{consumed[:3]}")

    print("-- 3) 守卫登记（D-3）")
    # ⚠️ 单一授权来源：全部取自守卫模块的常量（`REGISTERED_DIAG_FAMILIES` /
    # `PROTOCOL_ROUND_FAMILIES` / `SERIAL_ORACLE_ROUND` / `STAGE2B_ROUND`），
    # **不再硬编码** family 列表，避免预检与守卫的口径漂移。
    reg = {f: dict(diag=(f in G.REGISTERED_DIAG_FAMILIES),
                   proto=(f in G.PROTOCOL_ROUND_FAMILIES),
                   round=(f in G.SERIAL_ORACLE_ROUND["families"]),
                   stage2b=(f in G.STAGE2B_ROUND["families"]),
                   b6=(f in G.B6_ROUND["families"])) for f in new}
    for fam, v in reg.items():
        print(f"   {fam}: 定点诊断={v['diag']} 协议轮={v['proto']} 串行oracle轮={v['round']} "
              f"Stage2b授权={v['stage2b']} B6轮授权={v['b6']} "
              f"⇒ 会被判越界={not (v['diag'] or v['proto'] or v['round'] or v['stage2b'] or v['b6'])}")
    unregistered = [f for f, v in reg.items()
                    if not (v["diag"] or v["proto"] or v["round"] or v["stage2b"] or v["b6"])]
    print(f"   授权范围（批 1 批准，2026-10-04）：tag={G.STAGE2B_ROUND['tag']} · "
          f"tool={G.STAGE2B_ROUND['tool']} · families={list(G.STAGE2B_ROUND['families'])}")
    print(f"   授权范围（B6 轮批准，2026-10-05）：tag={G.B6_ROUND['tag']} · "
          f"tool={G.B6_ROUND['tool']} · families={list(G.B6_ROUND['families'])}")

    print("-- 4) 工具就绪（tag/family/reps 必填）")
    src = (ROOT / "specdec" / "engine" / "stage2b_verdict.py").read_text(encoding="utf-8")
    src_ck = (ROOT / "specdec" / "engine" / "stage_check.py").read_text(encoding="utf-8")
    for probe, ok in (("--tag 必填", 'add_argument("--tag", required=True' in src),
                      ("--families 必填", 'add_argument("--families", required=True' in src),
                      ("--reps 必填", 'add_argument("--reps", required=True' in src),
                      ("stage_check tag 必填", 'add_argument("--tag", required=True' in src_ck),
                      ("stage_check 家族参数化", 'add_argument("--family", required=True' in src_ck)):
        print(f"   {'OK  ' if ok else 'FAIL'} {probe}")
        if not ok:
            hard.append(f"工具未就绪：{probe}")
    for mod in ("specdec/engine/stage2b_verdict.py", "specdec/engine/stage_check.py",
                "specdec/engine/serial_oracle_gpu.py", "scripts/so_batch.ps1"):
        ex = (ROOT / mod).exists()
        print(f"   {'OK  ' if ex else '缺失'} {mod}")
        if not ex:
            hard.append(f"缺文件 {mod}")

    print("== 结论")
    for h in hard:
        print(f"   [硬问题] {h}")
    if hard:
        print("   ★ PREFLIGHT_FAIL")
        return 1
    if unregistered:
        print(f"   ★ PREFLIGHT_OK_TOOLS_BUT_FAMILIES_UNREGISTERED：工具就绪；"
              f"{unregistered} 尚未登记（**待 D-3 批准**）⇒ 现在开跑会被守卫正确判越界")
        return 3
    print("   ★ PREFLIGHT_OK（工具与 family 均就绪）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
