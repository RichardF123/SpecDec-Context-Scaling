# Prompt families（≥16 个 nested-prefix 族）— **草案 v1，待定稿**

> 状态：**DRAFT**。构建脚本与 manifest 已入库，规则可复现；**需你确认后才作为正式实验输入**。
> 产物：`docs/prompt_families_manifest.json`（版本化 + 哈希）、构建器 `scripts/build_prompt_families.py`。
> 生成命令：`python scripts/build_prompt_families.py`

## 1. 为什么需要 family，而不是一堆 prompt

RQ1 的自变量是**上下文长度**。如果 2K/4K/8K 用不同文本，测到的差异里混着"内容效应"，无法归因。
所以每个 family 定义为一个 **8192 token 的真实文本窗口**，2K/4K/8K 三个条件取它的**嵌套前缀**
（2048 ⊂ 4096 ⊂ 8192），续写任务在三个长度上完全相同 —— 长度是唯一变化量。

## 2. 取材与不重叠

| 项 | 取值 | 理由 |
|---|---|---|
| family 语料 | `wikitext-2-raw-v1/wiki.train.raw` | 真实文本，不插合成填充 |
| 质量窗语料（G3） | `wikitext-2-raw-v1/wiki.test.raw` | **分文件 ⇒ 天然不重叠**，不用维护偏移保留区 |
| 窗口规则 | 每 300k 字符一个 block；块内按 8192 token **对齐切分** | 互不重叠、可复现、窗口不跨 block |
| 覆盖 | 4 个 block，共 32 个候选窗口，选 16 个 | 实测 per-block 分布 `{'0':6,'1':3,'2':1,'3':6}` |

## 3. 预登记分层（先验假设，pilot 负责证伪）

按窗口文本特征分 4 层、每层 4 个：

| 层 | 定义（特征规则） | 先验预期 |
|---|---|---|
| `S1_low_entropy` | distinct-token 比例最低（模板化/重复文本） | 接受率**高** |
| `S3_prose` | 字母比例高、标记少（纯散文） | 接受率中高 |
| `S4_mixed` | 其余 | 中 |
| `S2_high_markup` | wikitext 标记符号比例最高（表格/信息框） | 接受率**低** |

**先验假设（写进 manifest，pilot 后回填实测值）**：

> 接受率排序 **S1 > S3 > S4 > S2**

若被证伪，同样是有价值的结论（说明"文本可预测性"不是接受率的主导因素）。
已知弱点（已写进 manifest 的 `caveat`）：`markup_ratio` 的层间分离较弱
（S2 ≈ 0.004–0.005 vs S3 ≈ 0.002），所以 **S2 更应视为排序而非独立文本体制**；
pilot 出数后按实测接受率重新分层，走 G3 那套 pilot-driven amendment 流程登记。

## 4. 实测特征（16 个 family）

| 项 | 范围 |
|---|---|
| distinct-token 比例 | 0.185 – 0.258 |
| markup 比例 | 0.001 – 0.005 |
| digit 比例 | 0.002 – 0.029 |
| 每 family token 数 | 8192（三段前缀 2048 / 4096 / 8192，均已校验长度一致） |

每个 family 在 manifest 中带：`text_sha1`、三段前缀各自的 `token_sha1`、特征值、分层、
先验接受率序、来源 block/windows 下标、以及前 100 字符的可读预览（便于人工抽查）。

## 5. 怎么用（接进正式矩阵）

- 正式矩阵 = `16 families × 12 conditions（3 ctx × 2 KV × 2 decode）× ≥5 reps`
  = **≥960 次运行**；family 决定"测什么文本"，ctx 决定"取多长的前缀"。
- G5 的 `results.jsonl` 每行必须带 `family_id` + `ctx`；这两者与 `config_hash` 一起构成主键。
- `expected_acceptance_rank` 用于**分层报告**（按层汇总接受率与加速比），
  而不是把 16 个族当 16 个独立样本求均值 —— 这也是抵消"接受率强依赖 prompt"的办法
  （Part 4 首跑实测：同档内接受率 0.465 – 0.925）。

## 6. 需要你定稿的点

1. **N=16 是否够了**（协议写"≥16"；按当前单条件成本外推，加族数线性涨价，见 Part 4 标定）；
2. **分层规则是否接受**（我按"文本可预测性"这一物理假设分的层，也接受你改成别的维度）；
3. **是否允许 family 用 train.raw**（我选它是为了与 G3 质量窗天然不重叠；若你想全部用 test.raw，
   我就改成在同一文件里划保留区并维护偏移表）；
4. **是否把 2K/4K/8K 之外的中间长度**（如 6K）也做成 family 条件，用于 break-even 区间的加密
   （协议里"若加速比在测试长度间穿过 1×，可加密 1–2 个中间长度"）。
