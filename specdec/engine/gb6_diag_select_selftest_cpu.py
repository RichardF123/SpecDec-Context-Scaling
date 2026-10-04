"""gb6_diag_select_selftest_cpu.py — GB6 诊断**产物配对/来源核对**的 CPU 自测（纯 CPU；不加载模型）。

## 目的
反例 CF3：诊断工具原按旧模式名 `aligned`/`spec` 取产物，Stage 2b 用 `prod`/`serial`；
找不到 divergent 侧时它把位置**退化为末 token**（真实 F13：pos=63 而真值 47）。
本自测在**临时目录**合成产物（**绝不触碰真实产物**），证明配对逻辑：

| 用例 | 构造 | 期望 |
| --- | --- | --- |
| T1 | 同 tag/同 family/同配置，prod 在 index 7 分歧 | `DIVERGENT`，`first_divergence_index=7`，`position_origin` 含 `first_divergence`，`round_index` 由 prod 轮记录推出 |
| **T2** | **只有 AR，没有 prod** | **抛 `MISSING_ARTIFACT`**（**不得**退化为末 token） |
| T3 | 记录内 `tag` 是旧 tag（文件名叫新 tag） | 抛 `PROVENANCE_MISMATCH` |
| T4 | 只有**旧 tag** 的 prod 产物（请求新 tag） | 抛 `MISSING_ARTIFACT`（**不得**借用旧 tag） |
| T5 | prod 记录的 family/case 是别的 family | 抛 `PROVENANCE_MISMATCH`（**不得**借别的 family） |
| T6 | 两侧 `ids_hash` 不同 | 抛 `CONFIG_MISMATCH` |
| T7 | 两侧 `ctx` 不同 | 抛 `CONFIG_MISMATCH` |
| T8 | 两侧 `kv_format` 不同（旧 backend） | 抛 `CONFIG_MISMATCH` |
| T9 | 两侧 `prompt_len` 不同 | 抛 `CONFIG_MISMATCH` |
| T10 | 两侧 `smoke_config` 不同（计算配置） | 抛 `CONFIG_MISMATCH` |
| T11 | 两侧 `dtypes` 不同 | 抛 `CONFIG_MISMATCH` |
| T12 | `family.hash_verified=False` | 抛 `CONFIG_MISMATCH` |
| T13 | AR 与 prod 逐 token 相同（无分歧） | `TOKEN_EQUAL`、`position=None`（**不**伪造位置） |
| T14 | 无分歧 + 显式 legacy 开关 | 末位位置，但 `position_origin` 标注 `legacy`（**不**含 `first_divergence`） |
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from . import gb6_diag_select as G

TAG = "div8209_p2"
OLD_TAG = "div8209_proto"
FAM = "F09"
OTHER_FAM = "F03"
RUNID = "20261004T120000Z"
CTX = 8192
IDS_HASH = "9a278731d744a859"
SHA = "e5dbe83616ea348a"
KV = {"target": "fp16", "draft": "not_loaded"}
DTYPES = {"model": "torch.bfloat16", "target_kv_payload": "base_bf16", "draft_kv_payload": "base_bf16"}
CFG = {"ctx": 8192, "chunk": 2048, "k": 4, "max_new": 64, "kv": "base_bf16", "arm": "eager",
       "mode": "prod", "family": "F09", "target_rope": "cpu_fp32", "bf16_red_red": "1",
       "struct_check": True}
RULES = "冻结配置：ctx=8192/chunk=2048/k=4/max_new=64/base_bf16/eager/target_rope=cpu_fp32"


def _base_gen(n: int = 64, *, base: int = 500) -> list:
    return [base + i for i in range(n)]


def _prod_rounds(gen: list, *, kp: int = 4) -> list:
    """把 prod 输出切成 kp 大小的轮（用于 round_index 推导）。"""
    return [dict(round=i + 1, emitted=list(gen[i * kp:(i + 1) * kp]))
            for i in range(len(gen) // kp)]


def _artifact(tag: str, family: str, mode: str, *, gen: list, ctx: int = CTX,
              ids_hash: str = IDS_HASH, token_sha1: str = SHA, prompt_len: int = 8192,
              kv=None, cfg=None, rules: str = RULES, dtypes=None, verified: bool = True,
              case: str | None = None, tag_in_record: str | None = None,
              rounds: list | None = None) -> dict:
    c = dict(cfg or CFG)
    c["mode"] = mode
    return dict(tool="serial_oracle_gpu", tag=(tag_in_record or tag), case=(case or family),
                mode=mode, rep=1, run_id=RUNID, stage="complete", rules=rules,
                smoke_config=c,
                smoke=dict(gen=list(gen), kv_format=dict(kv or KV),
                           rounds=list(rounds or []),
                           input=dict(ctx=ctx, ids_hash=ids_hash,
                                      dtypes=dict(dtypes or DTYPES),
                                      family=dict(family_id=family, token_sha1=token_sha1,
                                                  hash_verified=verified)),
                           end_state=dict(prompt_len=prompt_len, gen_len=len(gen))))


def _write(outdir: Path, tag: str, family: str, mode: str, rec: dict) -> None:
    (outdir / f"{tag}_{family}_{mode}_rep1_{RUNID}.json").write_text(
        json.dumps(rec, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    checks: list = []

    def ck(name: str, ok: bool, extra=None) -> None:
        checks.append((name, bool(ok), extra))

    tmp = Path(tempfile.mkdtemp(prefix="gb6sel_"))
    try:
        # ---------- T1 正常配对：prod 在 index 7 分歧 ----------
        d = tmp / "t1"; d.mkdir()
        ar_gen = _base_gen()
        pr_gen = _base_gen(); pr_gen[7] = 4242
        _write(d, TAG, FAM, "ar", _artifact(TAG, FAM, "ar", gen=ar_gen))
        _write(d, TAG, FAM, "prod", _artifact(TAG, FAM, "prod", gen=pr_gen,
                                             rounds=_prod_rounds(pr_gen)))
        sel = G.select_divergent_pair(tag=TAG, family=FAM, outdir=d)
        ck("T1_状态为DIVERGENT", sel["status"] == "DIVERGENT", sel["status"])
        ck("T1_首分歧位置=7", sel["first_divergence_index"] == 7, sel["first_divergence_index"])
        ck("T1_position=首分歧下标", sel["position"] == 7, sel["position"])
        ck("T1_位置来源标注为首分歧",
           "first_divergence" in sel["position_origin"] and "prod" in sel["position_origin"],
           sel["position_origin"])
        ck("T1_轮号由prod轮记录推出=2", sel["round_index"] == 2, sel["round_index"])
        ck("T1_两侧token与产物一致",
           sel["ar_token"] == ar_gen[7] and sel["prod_token"] == 4242,
           (sel["ar_token"], sel["prod_token"]))
        ck("T1_配置核对全部通过",
           set(sel["config_checks"]) and all(v is True for v in sel["config_checks"].values()),
           {k: v for k, v in sel["config_checks"].items() if v is not True})
        ck("T1_记录选中的是prod侧文件", sel["div"]["file"].startswith(f"{TAG}_{FAM}_prod"),
           sel["div"]["file"])

        # ---------- T2 **只有 AR、没有 prod** ⇒ 必须拒绝（反例 CF3 的核心） ----------
        d2 = tmp / "t2"; d2.mkdir()
        _write(d2, TAG, FAM, "ar", _artifact(TAG, FAM, "ar", gen=ar_gen))
        code2, detail2 = None, ""
        try:
            G.select_divergent_pair(tag=TAG, family=FAM, outdir=d2)
        except G.PairSelectionError as e:
            code2, detail2 = e.code, e.detail
        ck("T2_缺prod时抛MISSING_ARTIFACT", code2 == "MISSING_ARTIFACT", (code2, detail2[:120]))
        ck("T2_拒绝信息禁止退化末token",
           ("不允许" in detail2) and ("末 token" in detail2), detail2[:160])

        # ---------- T3 记录内 tag 是旧 tag ----------
        d3 = tmp / "t3"; d3.mkdir()
        _write(d3, TAG, FAM, "ar", _artifact(TAG, FAM, "ar", gen=ar_gen))
        _write(d3, TAG, FAM, "prod", _artifact(TAG, FAM, "prod", gen=pr_gen, tag_in_record=OLD_TAG))
        code3 = _code(lambda: G.select_divergent_pair(tag=TAG, family=FAM, outdir=d3))
        ck("T3_记录内旧tag被拒", code3 == "PROVENANCE_MISMATCH", code3)

        # ---------- T4 只有旧 tag 的产物：不得借用 ----------
        d4 = tmp / "t4"; d4.mkdir()
        _write(d4, OLD_TAG, FAM, "ar", _artifact(OLD_TAG, FAM, "ar", gen=ar_gen))
        _write(d4, OLD_TAG, FAM, "prod", _artifact(OLD_TAG, FAM, "prod", gen=pr_gen))
        code4 = _code(lambda: G.select_divergent_pair(tag=TAG, family=FAM, outdir=d4))
        ck("T4_只有旧tag时拒绝（不借用旧tag）", code4 == "MISSING_ARTIFACT", code4)

        # ---------- T5 别的 family ----------
        d5 = tmp / "t5"; d5.mkdir()
        _write(d5, TAG, FAM, "ar", _artifact(TAG, FAM, "ar", gen=ar_gen))
        _write(d5, TAG, FAM, "prod", _artifact(TAG, FAM, "prod", gen=pr_gen, case=OTHER_FAM))
        code5 = _code(lambda: G.select_divergent_pair(tag=TAG, family=FAM, outdir=d5))
        ck("T5_别的family被拒", code5 == "PROVENANCE_MISMATCH", code5)

        # ---------- T6..T12 配置不一致 ----------
        for name, kw, expect in (("T6_ids_hash不同", dict(ids_hash="deadbeefdeadbeef"), "CONFIG_MISMATCH"),
                                 ("T7_ctx不同", dict(ctx=4096), "CONFIG_MISMATCH"),
                                 ("T8_kv_format不同", dict(kv={"target": "fp8", "draft": "not_loaded"}),
                                  "CONFIG_MISMATCH"),
                                 ("T9_prompt_len不同", dict(prompt_len=4096), "CONFIG_MISMATCH"),
                                 ("T10_计算配置不同", dict(cfg=dict(CFG, max_new=32)), "CONFIG_MISMATCH"),
                                 ("T11_dtypes不同", dict(dtypes={"model": "torch.float32"}), "CONFIG_MISMATCH"),
                                 ("T12_hash未验证", dict(verified=False), "CONFIG_MISMATCH")):
            dd = tmp / name.split("_")[0]; dd.mkdir()
            _write(dd, TAG, FAM, "ar", _artifact(TAG, FAM, "ar", gen=ar_gen))
            _write(dd, TAG, FAM, "prod", _artifact(TAG, FAM, "prod", gen=pr_gen, **kw))
            got = _code(lambda dd=dd: G.select_divergent_pair(tag=TAG, family=FAM, outdir=dd))
            ck(name + "被拒", got == expect, got)

        # ---------- T13 无分歧 ⇒ 不记录位置 ----------
        d13 = tmp / "t13"; d13.mkdir()
        _write(d13, TAG, FAM, "ar", _artifact(TAG, FAM, "ar", gen=ar_gen))
        _write(d13, TAG, FAM, "prod", _artifact(TAG, FAM, "prod", gen=list(ar_gen)))
        sel13 = G.select_divergent_pair(tag=TAG, family=FAM, outdir=d13)
        ck("T13_无分歧状态为TOKEN_EQUAL", sel13["status"] == "TOKEN_EQUAL", sel13["status"])
        ck("T13_无分歧时不记录位置", sel13["position"] is None
           and sel13["first_divergence_index"] is None, sel13["position"])
        ck("T13_来源标注为无分歧",
           "token_equal" in sel13["position_origin"]
           and "first_divergence" not in sel13["position_origin"], sel13["position_origin"])

        # ---------- T14 legacy 开关：末位但来源标 legacy ----------
        d14 = tmp / "t14"; d14.mkdir()
        _write(d14, TAG, FAM, "ar", _artifact(TAG, FAM, "ar", gen=ar_gen))
        _write(d14, TAG, FAM, "prod", _artifact(TAG, FAM, "prod", gen=list(ar_gen)))
        sel14 = G.select_divergent_pair(tag=TAG, family=FAM, outdir=d14,
                                        allow_last_index_when_equal=True)
        ck("T14_legacy末位位置=len-1", sel14["position"] == len(ar_gen) - 1, sel14["position"])
        ck("T14_legacy来源不含first_divergence",
           "legacy" in sel14["position_origin"]
           and "first_divergence" not in sel14["position_origin"], sel14["position_origin"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    npass = sum(1 for _, ok, _ in checks if ok)
    nfail = len(checks) - npass
    print("== GB6 诊断配对自测（纯 CPU；合成产物在临时目录，未触碰真实产物）==")
    for nm, ok, extra in checks:
        print(f"  [{'ok' if ok else 'FAIL'}] {nm}" + (f"  {str(extra)[:170]}" if not ok else ""))
    print(f"\nGB6 诊断配对自测：PASS {npass} / FAIL {nfail}")
    return 0 if nfail == 0 else 1


def _code(fn) -> str | None:
    """跑一次并返回错误码（无异常则返回 None）。"""
    try:
        fn()
        return None
    except G.PairSelectionError as e:
        return e.code


if __name__ == "__main__":
    raise SystemExit(main())
