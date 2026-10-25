"""execution_optimization 包：Phase O 的优化执行器 / 参照 fixture / 开发集基准 / 分析。

约束（与冻结配置一致）：
* 默认 `SPECDEC_EXECUTION_MODE=reference`，即本包不接管任何默认行为；
* 只有显式 `SPECDEC_EXECUTION_MODE=optimized`（或调用方显式传 flags）时，才走优化执行器；
* 不修改 `specdec/engine/*` 的任何默认行为，不复制第二套模型实现（全部复用 engine 的计算件）。
"""
from . import config  # noqa: F401
