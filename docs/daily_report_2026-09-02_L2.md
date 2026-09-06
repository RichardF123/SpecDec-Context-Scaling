# SpecDec 项目日报 —— L2 通关日（KV-cache + 生成 + 对齐 PASS）

## 一、今日一句话
**L2 主体完工：KV-cache 缓存解码 + 贪心生成 + 生成对齐门 PASS（整条序列与官方模型数学逐 token 一致），引擎第一次"自己开口说话"。**

## 二、今日完成 ✅

### 实现（全部你亲手写，AI 只讲解/审阅/建测试）
1. **T1：RoPE 支持 start_pos** —— decode 新词拿到真实绝对位置（=缓存长度）
2. **T2：SelfAttention 支持 KV-cache** —— 读档+追加、mask 只在 prefill、GQA 展开挪到 concat 后、返回 2 头新档案
3. **T3：DecoderLayer / QwenLM 透传** —— `_run(x, cache_list)` 统一 prefill/decode；`forward` 变薄壳保持 L1 接口
4. **generate()** —— prefill + 循环（取最后位置 argmax → EOS 停词 → 拼接 → 回喂 1 词）
5. **bench_basic.py** —— 第一个性能测量脚本（prefill ms + decode tokens/s，CPU/GPU 双跑）

### 测试与验收
| 测试 | 结果 |
|---|---|
| L1 回归 align_ours | 0.000 ✅（重构没破坏 prefill）|
| 缓存等价 test_cache_equiv | ~2e-5 + 预测一致 ✅（fp 噪声级，非逻辑错）|
| 真权重多深度 cache==full | 各深度 ~2e-5，argmax 全 same ✅ |
| **生成对齐 align_generate** | **PASS（4/4 序列一致）** ✅ |

### 关键发现（写报告素材）
1. **hf.generate ≠ 模型全量前向**：同一前缀下官方 logits 第一是 ' is'（21.75 vs 20.15），`hf.generate` 却输出 ' was' → 参考改用"HF 全量逐步贪心"（纯模型数学）→ PASS。教训：**裁判要选"声称复刻的对象"（前向数学），别选带隐藏管线的黑盒**。
2. **缓存路径与全量路径差 ~2e-5 是正常 fp kernel 顺序噪声**（matmul 形状不同），非 bug；判据：逐层差 ~1e-7 且不放大 = 正常。
3. 自建引擎的价值再次被验证：只有掌控解码循环，才知道 token 从哪来、谁能信。

## 三、进度总览

| 阶段 | 状态 |
|---|---|
| 规划/方案 | ≈95% |
| 环境 | ≈100% |
| **L1 引擎 + 对齐** | **100%** |
| **L2 KV-cache + generate + 对齐** | **≈90%**（差计时数字实跑）|
| L4 量化 / L4b / L5 / 基准 / CUDA / 报告 | 0% |

**整体进度 ≈ 36%**。L2 提前于 v6 周表完成；下一个里程碑 = L4 权重量化（int8/int4）。

## 四、待办
- [ ] 跑 bench_basic 拿到第一个 decode tokens/s（本里程碑收尾）
- [ ] commit（L2 PASS 值得存档）
- [ ] 修 model_self.py 末尾 __main__ 残留（model 未定义）
- [ ] 下一站：L4 权重量化
