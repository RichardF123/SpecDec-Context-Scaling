SpecDec final 4K/8K PagedAttention validation archive
=====================================================

Hardware:
- NVIDIA GeForce RTX 4060 8GB

Real dual-model validation:
- target Qwen/Qwen2.5-1.5B-Instruct
- draft Qwen/Qwen2.5-0.5B-Instruct
- batch=1, SpecDec k=4, 16 generated tokens
- paged_v0 + final direct PagedAttention q_len=1..4
- target BF16 and asymmetric V-FP8
- contexts 4096 and 8192
- chunked prefill=512
- 3 fresh direct runs per context/KV condition
- one materialized reference per condition

Final stability:
- 4K BF16 direct 3/3 PASS
- 4K V-FP8 direct 3/3 PASS
- 8K BF16 direct 3/3 PASS
- 8K V-FP8 direct 3/3 PASS
- materialized references PASS
- direct/materialized output and SpecDec-stat parity PASS

Boundary:
This supports the tested configuration only. The current direct PagedAttention
kernel reduces full-history materialization temporary allocation but is not a
latency speedup on the tested RTX 4060.
