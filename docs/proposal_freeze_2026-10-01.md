# Proposal 定稿记录（2026-10-01）

状态：用户已确认，当前提交稿 FROZEN。此状态不代表导师已批准修改后的范围，也不代表实验正确性或运行验收通过。

## 当前版本

- PDF：[五页提交稿](../output/pdf/specdec_course_proposal_main6_ablation2_2026-10-01.pdf)
- 源文件：[LaTeX](../output/pdf/specdec_course_proposal_main6_ablation2_2026-10-01.tex)
- PDF SHA256：`A6568D7C64BDE9C84AEF4273B62041639569C88C9E832D06B7CE08F802113F4C`
- TEX SHA256：`0DAAE14AC39ECE07A6E59F7937CC84AAF6AF08475E1B4D13E416CAD353BBE0D4`

后续提交与回复导师以此稿为准。不再自行迭代措辞或扩大研究范围；需修改时另行取得用户确认并登记新版本。旧稿保留，不覆盖。

## 冻结范围

主问题：固定模型对和硬件下，上下文长度如何影响贪心投机解码相对 AR 的性能。

| 层次 | 条件 | 定位 |
| --- | --- | --- |
| 主实验 | BF16 KV，2K/4K/8K × AR/Spec，共 6 条件 | 回答主问题 |
| 补充消融 | 预先指定 8K，V-only FP8 × AR/Spec，共 2 条件 | 最长上下文的存储方案敏感性检查；不推断完整压缩交互曲线 |
| 已完成工作 | 引擎、chunked prefill、FP8 质量与显存结果、量化负结果、测量基础设施 | 保留为已有成果，标明配置和证据边界 |

模型对固定为 Qwen2.5-1.5B-Instruct / 0.5B-Instruct；BF16 compute；k=4；chunk=2048；最多生成 64 token，遵循冻结 EOS 规则。Draft KV 始终为未量化 BF16。

主指标为 decode speedup，端到端延迟、接受统计、组件耗时和显存为支持指标。速度比较使用同 KV 的 AR 分母；AR 为 target-only。FP8 补充使用相同正式 families 的 8K 前缀及重复协议。

## 尚未冻结或完成的事项

- 最终 attention backend/阶段策略仍须通过正确性和运行验证，不能将 proposal 定稿当作后端验收。
- 严格 AR/Spec 输出一致性按各 KV 表示分别验证，不放宽门槛；FP8 与 BF16 之间不要求逐 token 相同。
- 若后端改变，旧 FP8 质量结果不能直接外推；按提交稿重复既有质量检查。
- 补充条件受正确性和可行性门约束；失败如实记录，不阻塞已验证的主实验。
- 本次未改生产代码、runner、manifest 或历史结果，也未启动实验。
- 旧 v7.1/v7.2 的 12 条件设计属于历史范围。当前 proposal 范围以本记录为准；执行协议和汇总工具的 6+2 配套更新尚待实施，不得直接沿用旧矩阵作为新方案执行清单。
