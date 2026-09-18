# G4 cache API 框架指导（供用户实现 `specdec/engine/spec_cache.py`）

> 定位：核心代码由**你实现**；本文件给接口契约/数据流/不变式/边界用例。测试（B4）由我按此契约编写（`g4_b4_cache_tests.py`），实现与测试签名不一致会立刻暴露。
> 关联：`design_freeze_g4_specdec.md` §4/§5；引擎缓存结构见 `model_self.py`。

## 1. cache 是什么（数据流角度）
一次 `_run(x, cache_list)` 后，`cache_list` = 每层一个 dict：
- base（fmt 'fp16'）：`{"k": (1,kv,T,d) bf16, "v": (1,kv,T,d) bf16}`（无 fmt 键；kv_format 语义决定）；
- v_fp8（fmt 'v_fp8'）：`{"k": bf16 全长, "v": fp8 payload, "v_scale": (1,kv,T,1) fp32, "fmt": "v_fp8"}`；
- full-FP8 ablation（fmt 'fp8'）另有 `k_scale`（本轮正式 SpecDec 不用，但通用实现应能处理任意含/不含 scale 键的层，避免写死）。
`T` = 该模型已处理的 token 数；**不变式：所有 token 相关张量的 dim=2 长度处处相等**（同一层内、跨层都一致）。

## 2. 接口契约（实现须满足，测试据此写）
```python
# spec_cache.py
def cache_length(cache) -> int
def trim_cache(cache, new_len: int) -> list[dict]      # 返回新 cache（不原地改入参）
def check_consistent(cache) -> None                    # 违反不变式时抛 RuntimeError
```
语义细则：
1. `cache_length(None)` → 0；`cache_length(cache)` → `cache[0]["k"].shape[2]`（两种 KV 通用）。
2. `trim_cache`：对每层、每张量（键含 `"k"/"v"/"k_scale"/"v_scale"`，跳过 `"fmt"`），`t = t[:, :, :new_len]`；
   返回 **浅拷贝**（新 list + 每层新 dict），内部张量可用视图（torch 切片即视图，禁止原地改原 cache——状态机里原 cache 可能还要被别处引用）；
   保留 `"fmt"` 键。
3. 越界：`new_len <= 0` 或 `new_len > cache_length(cache)` → 抛 `ValueError`（不做静默钳制）。
   **`new_len == 0` 非法**：状态机里"真实 token 数"恒 >0（prefix 至少 1 token），允许 0 只会掩盖
   `pre + m` 被算成非正数的 bug。`None` cache 且 `new_len==0` → 返回 `None`（唯一例外，见 §2 第 2 条）。
4. `check_consistent`：`None` ok；否则断言每层 `k/v`（含可选 scale）dim=2 相等，且各层一致；不等抛 `RuntimeError("cache token dim mismatch at layer i")`。

## 3. 在 SpecDec 主循环里的用法（设计意图，供实现时想清楚）
每轮（target 与 draft 各持一个 cache）：
- 进入验证前：`pre_len = cache_length(tgt_cache)`；
- block forward 后 target cache 长度 = pre_len + k（包含未确认尾部）→ 接受 m 后：
  - 若 m < k：`tgt_cache = trim_cache(tgt_cache, pre_len + m + 1)`；`drf_cache = trim_cache(drf_cache, pre_len + m)`；
  - 若 m == k（无 bonus）：`tgt_cache` 已是 pre_len+k 有效 → 直接推进；`drf_cache = trim_cache(drf_cache, pre_len + k)`。
- 原因：cache 长度 ≡ 已确认"真实"token 数；被拒的假 draft 若留在 cache，后续 softmax 会向它们分配概率 → 破坏 exact equivalence（不只性能问题）。
- 两个模型的 cache 必须各自独立 trim（长度不一定相等：draft 落后一个替换 token）。

## 4. 边界用例（B4 测试覆盖）
- None/空 cache；trim 到当前长度（幂等）；trim 1 token / 多 token；
- trim 到 0（当前>0 时抛错）；trim 超过长度抛错；trim 负数抛错；- mismatch 发生在第 1/2/3/4 个 draft token（即 m=0/1/2/3）后 trim 结果长度正确；
- base（bf16 k/v）与 v_fp8（k + fp8 v + v_scale）都能正确 trim；
- fp8 payload 与 v_scale 同步（trim 后 `v_scale.shape[2]==v.shape[2]`）；
- check_consistent 能发现"人为把某层 v 切片长度弄错"的坏 cache。
- 真实路径验证：小模型（0.5B）真 prefill→真 decode→trim，长度与 finite 保持。

## 5. 实现提示（不是代码）
- 不要改 `model_self.py` 的缓存内部（append 由 `_run` 负责）；本模块只做读长度/裁剪/一致性检查。
- 张量沿 `dim=2` 裁剪对 v_scale 同样成立（其 shape (1,kv,T,1)，T 在 dim=2）。
- 若某层缺某键（如 base 无 v_scale）→ 只处理存在的张量，不要 KeyError。
- 测试契约：`cache_length(None)==0`；`trim_cache(None, 0) is None`。
- **`v_fp8` 是 target 专属格式**：`GROUP_V=128` 要求 `head_dim % 128 == 0`（1.5B: head_dim=128 ✓；
  0.5B draft: head_dim=64 ✗）。B4/B5 测试里 **draft 恒用 base bf16**，`v_fp8` 只在 1.5B target 上测。
