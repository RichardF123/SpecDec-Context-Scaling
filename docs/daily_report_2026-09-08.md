# SpecDec 项目日报 —— 2026-09-08

## 一、今日一句话
**分块 prefill 完工（8K/18 格硬前提清除）+ decode 基线经库参考验证可信 + 方案 v6.1 吸收评审 + 全套文档更新。进度 ≈ 42–45%。**

## 二、今日完成 ✅

### 1. 方案/决策（v6.1 修订，吸收评审 #3–#9）
- ppl 定位 = 量化质量指标（投机解码质量由接受率+净加速衡量）；
- 增量定位：8GB 从零 eager 引擎上的组合实证与扩展（非新方法，与 MagicDec 关系诚实化）；
- draft 定案 **Qwen2.5-0.5B**（同词表、免训练）；K1 = 第一优先砍项；
- M1 归因重心改为"草稿收益 vs 验证开销"，KV 带宽仅理论估计列（局限写明）；
- 承诺收敛：18(15/18) 格 + M1 + 诚实分析 + 报告（短文/演示可选）；
- 决策：**不做 demo**（课程 demo 不必须）；shape/数据流分析助手仅为可选演示项。

### 2. 课程对齐
- 确认落在课程 J（Efficient LLM Inference）+ Improve；按 rubric 权重（实现20/评测25/分析20）评估；
- 课程提案 `course_proposal_1page.md`（1–2 页、五问格式）就绪待填队名分工；
- 评分自查：满足度设计上 100%，剩余 45% 权重待实验+分析兑现。

### 3. 工程/测量
- **decode 基线 sanity**：transformers eager fp16 = 8.29–9.40 tok/s；我们 = 10.58–13.82 tok/s（快 27–47%）→ 基线可信，L5 分母有库参考背书（data_ledger §4.3）；
- **8K OOM 定位**：单次 eager prefill 因 scores(1.75GiB)+logits(2.5GiB) OOM → 动机素材 + 分块必要性（§4.4）；
- **分块 prefill 完工**：
  - SelfAttention 块级因果 mask（统一 diag=p+1 三分支）；
  - `_run` 加 `need_logits`；新增 `prefill_chunked(ids, chunk=1024)`；
  - 验证：L1 回归 0.000 ✅；缓存等价 PASS ✅；4K 分块等价 PASS（12 步 decode 全同）；**fp32 隔离 7.8e-4 → 0.55 缓存差归因为 fp16 取整累积（非 bug）**。

### 4. 文档/仓库
- 更新：data_ledger（§4.3/4.4）、status_checkpoint、README（中文+项目简介）；
- 新建：course_proposal_1page、project_proposal、engine_guide_for_team（给协作者的全解析）；
- commits（均已 push）：2fb6ff4（v6.1）→ e0d902a（sanity）→ a682270（快照）→ c8b259f（探针）→ 3908248（分块 prefill）→ 30344cf（引擎解析）。

## 三、进度
| 阶段 | 状态 |
|---|---|
| 规划/方案（v6.1）| ≈95%（10/7 提案待填队名）|
| 环境 | 100% |
| L1 引擎 | 100%（0.000）|
| L2 KV-cache + 生成 | 100%（生成对齐 PASS）|
| 分块 prefill（测量前置）| 100%（行为等价验证）|
| L4 / L4b / L5 / 18 格 / M1 / 报告 | 0%（下一站 L4）|
| **整体** | **≈42–45%** |

## 四、待办
- [ ] 修 `model_self.py` 末尾 `__main__` 残留
- [ ] 10/7 前：定 2–3 人团队与分工（贡献声明）
- [ ] L4 int8/int4 权重量化（第一课已讲，待写 QuantLinear）
- [ ] 8K 分块 prefill 实测（探针 C 用 prefill_chunked 重跑，确认显存达标）

## 五、下一步
1. L4 量化实现（QuantLinear int8 → 验证 → int4 对比）；
2. 之后 L4b fp8 KV、L5 固定链投机解码、2K/4K/8K 18 格、M1 归因；
3. 10/7 课程提案提交。
