"""Generate a source-anchored Chinese learning guide, without importing the runner."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess
import textwrap
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, PageBreak,
    Table, TableStyle, Preformatted,
)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/pdf"
NAME = "specdec_runner_learning_guide_2026-10-21"
MAIN = ["plan.py", "worker.py", "run.py", "records.py", "diagnostics.py",
        "environment.py", "runtime_audit.py", "runtime_info.py"]
PATHS = ["experiments/formal/" + p for p in MAIN] + [
    "scripts/analyze_teaching_pilot.py", "scripts/diagnose_pilot_environment.py"]
DEPENDENCIES = ["family_input.py", "lowmem_load.py", "target_rope_scope.py",
                "model_self.py", "spec_dec.py", "spec_cache.py",
                "smoke_hooks.py", "smoke_checks.py", "g5_driver.py",
                "gpu_pstate_preflight.py"]

FUNCTIONS = {
    "plan.py": {
        "build_condition": ("无输入参数。", "返回 6 个 Condition 对象的列表。", "外层遍历三个上下文长度，内层遍历 AR/SpecDec。固定 target_kv=base_bf16，用统一命名建立六条件全集，避免不同文件各自手写后不一致。"),
        "build_jobs": ("family_ids：family 编号序列；seed=4207：仅控制任务排列。", "返回按两个 block 拼接的 Job 列表。", "block 0 每任务测 3 次，block 1 测 2 次。每个 block 用独立 random.Random(seed+block_id) 打乱，不改变全局随机状态。先完整生成再调度，不能按模型结果选择下一条任务。"),
        "validate_plan": ("jobs：待检查的任务表；family_ids：本批应使用的 family。", "通过返回 None；错误抛 ValueError。", "先构造应有的 (family, condition, block) 到 reps 映射，再逐条核对重复、未知任务、次数和遗漏。这检查覆盖完整性，不检查 GPU 或生成质量。"),
        "save_plan": ("jobs、family_ids、seed 与 output_path：计划及目标 JSON 路径。", "独占创建计划 JSON；无显式返回值。", "先 validate_plan，再用 asdict 把 dataclass 转为普通字典。open('x') 防止覆盖。schema_version=teaching-plan-v1，phase 写死 pilot：这是教学期保存工具，不是已经实现的正式入口。"),
    },
    "worker.py": {
        "load_input": ("family_id：如 F09；context_tokens：2048/4096/8192。", "(ids, metadata)：Python token 列表及输入来源/hash 元数据。", "固定本地 tokenizer，读取 MANIFEST 后调用既有 load_family_ids(verify_hash=True)。核对长度和 EOS。当前 MANIFEST 是 pilot 使用的旧清单；没有实现 formal manifest 自动切换。"),
        "load_target": ("无参数，模型 ID 在函数中固定。", "(model, tokenizer, info)：CUDA BF16 target、分词器及加载证据。", "复用低内存加载器并启用逐张量校验。检查 n_params_checked>0、weights_bitmatch、设备与 dtype。权重相同只说明复制正确，不说明每次前向与 HF 完全相同。"),
        "load_draft": ("无参数，模型 ID 固定为 0.5B。", "(model, tokenizer, info)：CUDA BF16 draft 及加载证据。", "与 target 一样核对权重，设 kv_format='fp16'。这里只表示 BF16 基线缓存的历史接口名；不对 draft 安装 target 专用 RoPE 来源。"),
        "configure_target": ("model：本次 target 模型实例。", "rope_info：可序列化的 RoPE 安装元数据；同时修改实例/模块配置。", "设未量化 KV、关闭额外 QK/概率 FP32 干预、设 cudnn_prefill，再按真实实例安装 CPU FP32 inv_freq 来源。它有副作用，不是只读查询；主程序 finally 要 restore。"),
        "check_runtime_config": ("target、draft（AR 为 None）、job。", "configuration 字典；不符合冻结值则抛异常。", "读取真实 attention/verify/precision 状态，遍历两模型所有参数检查 CUDA 和 BF16，并核对常驻模型与任务模式。返回值是检查后的快照，不是又执行一遍推理。"),
        "test_chunked_prefill": ("model；input_ids=[1,L]；chunk=2048；diagnostic=True 默认开启额外检查。", "(next_logits, cache)：最后位置分数 [1,V] 与全层 KV。", "这是实际运行 helper，不是独立测试文件。cache 从 None 开始，每块把旧缓存传给 _run。只有最后一块 need_logits=True；中间块只建 cache。最后一行 clone 防止保留整块 logits 的底层存储。diagnostic=False 关闭额外检查/打印，但模型计算仍执行。"),
        "test_ar_decode": ("model；next_logits=[1,V]；cache；eos_id；max_new=16 为函数默认，测量调用显式传 64；diagnostic=True。", "(generated, stop_reason)：仅新生成 token 列表与停止原因。", "先从 prefill 的最后一行选首 token；不是先把 prompt 重算一次。每轮把刚选 token 喂回模型得到下一行；遇 EOS 或输出预算耗尽先 break，因此生成 64 个 token 通常只需 63 次 decode 前向。函数不把最后 cache 返回给调用者。"),
        "spec_prefill_scope": ("target、draft 两个真实模型实例。", "上下文管理器 yield calls；列表记录 target/draft 的 prefill 秒数。", "临时把 spec_dec._prefill 换成分块适配器，按对象身份区分角色。把 [1,V] unsqueeze 为算法要求的 [1,1,V]，并恢复原函数。这样复用算法而不复制一份投机循环。"),
        "spec_generation_scope": ("无显式参数。", "上下文管理器，无业务返回值。", "包装 _propose/_verify/_step，把阶段标为 propose/verify/repair。阶段决定 attention 策略和统计归属，不是另一个算法。finally 恢复三个入口。这里的轻量阶段包装仍在生产测量路径，不等同于已关闭的诊断有限性 hooks。"),
        "run_ar_once": ("model、tokenizer、token_ids、job。", "一次 AR 结果字典：token、时间、峰值、attention 统计等。", "输入 GPU 张量在计时前准备；同步并重置峰值后开始。prefill 后再次同步划分阶段，再生成最多 64 token。decode_s=total_s-prefill_s。没有把模型加载、输入重建、独立诊断和外部快照算进内部计时。"),
        "run_spec_once": ("target、draft、tokenizer、token_ids、job。", "一次 SpecDec 结果字典，额外含 prefill_components 和 spec_stats。", "拒绝 serial oracle 和额外 verify wrapper。固定 k=4，调用现有 spec_decode_greedy；两个模型 prefill 时间相加。decode 是 total 减两个 prefill 包装区间，包含循环、同步及控制开销，不是 GPU kernel 时间之和。"),
        "run_once": ("model、draft、tokenizer、token_ids、job。", "返回对应模式的一次结果字典。", "AR 要求 draft 为 None；SpecDec 转到 run_spec_once；未知 mode 抛错。这是分派函数，使诊断后预热和测量调用同一条模式入口。"),
        "run_diagnostic": ("model、draft、tokenizer、token_ids、job。", "独立诊断字典。", "先准备 [1,L] 输入，定义分块 prefill_impl，返回 [1,1,V]。按 mode 调 diagnose_ar 或 diagnose_spec。它会执行实际模型推理，只是不进入 measured 数据。"),
        "execute_job": ("模型/分词器/输入/Job；provenance 来源证据；run_id 唯一编号；output_dir；progress 可变阶段字典。", "返回本任务 measured 结果列表，同时逐条落盘。", "独立诊断并保存后运行一次 warm-up 丢弃。每个 rep：开始边界快照、run_once、结束快照、合并证据、保存、核对与同模式诊断 token 是否一致。先保存再对分歧抛错，确保异常样本留证。"),
        "parse_args": ("从命令行读取参数，无 Python 形参。", "argparse.Namespace。", "必选 family/ctx/mode/block；--dry-run 只展示 Job；--output-dir 支持父进程指定位置。当前 family choices 只有 F09/F03/F11/F04，正式 FF 编号尚不接受。"),
    },
    "run.py": {
        "launch_job": ("job；command 命令参数列表；root 项目根目录。", "成功返回 metadata；失败记录 exit.json 后抛异常。", "创建唯一 launch 目录，追加 --output-dir，把 stdout/stderr 收到 worker.log。subprocess.run 串行等待；非零退出不能当成功，零退出后还要 verify_job_results。异常携带 launch_id/launch_dir，批次可追到失败证据。"),
        "main": ("--execute、--seed、--limit 或 --pilot。", "dry-run 打印任务；execute 写计划、各 launch 与批次摘要。", "--pilot 是完整开发 pilot，不是正式数据模式。执行前检查 18 GiB commit；每任务新子进程，任务间冷却 10 秒。失败即停，finally 保存 aborted 摘要；当前没有自动续跑、超时管理或按 FF 集合正式采集入口。"),
    },
    "records.py": {
        "validate_configuration": ("record：含 configuration 与 job 的字典。", "None 或 ValueError。", "核对 backend、dtype、KV、verify 模式、诊断开关和 draft 常驻状态。draft_resident 必须真布尔值，不能用 0/1 冒充。它检查记录一致性，不会重新查询活模型。"),
        "validate_record": ("record；eos_id；require_runtime_audit=True。", "None 或异常。", "校验配置及边界审计、max_new=64、长度/EOS、正有限时间、阶段求和、吞吐量、整数字节峰值 allocated<=reserved、diagnostic_checks_executed=False。历史读取可显式关闭审计必需项，但存在的审计仍须自洽；并非通用的所有字段完整性证明。"),
        "save_record": ("record、output_path、eos_id。", "写出一个 measured JSON。", "先校验再 json.dumps(allow_nan=False)，最后独占创建文件。减少半写入的非法记录，但不是事务式原子写入：写盘中断仍可能留下部分文件。"),
        "save_failure": ("failure：运行/任务/阶段/分类/异常；output_path。", "独占创建 failure.json。", "失败也属于实验结果，必须留来源、阶段和 traceback，而不是只显示终端报错。检查的是必需字段存在，不代表根因已经查明。"),
        "verify_job_results": ("output_dir、job、eos_id；require_runtime_audit=True。", "{'verified': True, 'n_measured': n} 或异常。", "父进程核对文件数量、诊断来源和覆盖、每条 measured 内容、任务身份、同模式完整 token、rep 编号和 run_id。一旦发现 failure.json 就拒绝。verified 是执行与文件契约通过，不是 speedup 声明资格。"),
        "save_diagnostic": ("record、output_path。", "写出 diagnostics/result.json。", "要求 diagnostic_only=True，序列化时拒绝 NaN 并防覆盖。诊断独立子目录使顶层 measured 文件计数不会把诊断混进去。"),
    },
    "diagnostics.py": {
        "diagnose_spec": ("target、draft、input_ids=[1,L]、eos_id、prefill_impl 回调。", "gen/stats/trace/决策与有限性覆盖/attention 统计的诊断字典。", "只允许生产 block verifier，装 DecisionHooks 检查真正进入 argmax 的行。check_round 重新计算连续接受前缀、替换 token、缓存长度和跨轮连续性；还检查 prefill 缓存与 verify 前缓存。finally 移除 hooks 并按身份核对恢复。它不证明 block 结果必须逐 token 等于 AR。"),
        "diagnose_ar": ("target、input_ids、eos_id、prefill_impl。", "生成 token、停止原因、decode_calls、有限性覆盖和 attention 统计。", "逐个决策前检查 logits 有限性，核对每次 decode 后缓存增长。正常输出 64 token 可有 63 次 decode。异常给 exc.evidence 留当时证据，不把这次诊断的时间当 measured。"),
    },
    "environment.py": {
        "check_environment": ("min_commit_gib=16；批次入口显式传 18。", "commit_gate_passed、阈值及 system_memory；失败抛 EnvironmentRefused。", "读系统提交额度而非 GPU VRAM。commit_free_mib/1024 得 GiB。18/16 是已登记的入口/任务安全线，不保证绝不 OOM。读不到数据也拒绝，不能把未知当合格。"),
        "check_gpu_environment": ("无参数。", "GPU 名称、前置负载预检 verdict 与阈值。", "先检查 CUDA，再复用 qualify 运行短负载采样。available 与 qualified 必须均成立。低频预检拒绝会携带 evidence。此项发生在加载模型前，不是测量全程频率认证。"),
    },
    "runtime_audit.py": {
        "capture_boundary": ("无参数。", "含 snapshot、查询起止单调时钟、scope 和 active_load_verified=False 的字典。", "调用已有 g5_driver.system_snapshot 查询 nvidia-smi；在每次 run_once 调用前后执行，不在内部计时区间轮询。记录查询耗时使观察者开销可见。"),
        "assess_boundaries": ("before、after：两个边界点字典。", "boundary_audit_v1 字典，含 COMPLETE/UNAVAILABLE、时钟 verdict 与解释。", "要求两点都有有限的 SM/memory 频率才可判定，复用既有时钟阈值。低频为边界判定不合格，不自动推断整个 decode 环境失败。连续负载验证始终为 False。"),
        "validate_boundary_audit": ("audit：一条保存的边界审计。", "None 或 ValueError。", "检查版本、快照结构、非负有限时间、query 内顺序和两点先后、scope，再从快照重算 verdict 核对结果。重算使用当前进程阈值，因此跨阈值配置读取历史数据需要显式版本治理，不能随意更改环境变量。"),
    },
    "runtime_info.py": {
        "collect_runtime_info": ("无参数。", "git/Python/PyTorch/线程环境/runner 源码 SHA256 的字典。", "通过 git 子进程记录 HEAD 与修改，遍历 formal/*.py 摘要源码。版本+hash 辅助复现；不等于记录整个操作系统或保证显存状态相同。函数自身不执行模型推理。"),
    },
    "analyze_teaching_pilot.py": {
        "assess_pilot_unit": ("within_spreads；between_spread；preflight_ok；boundary_state='UNAVAILABLE'。", "分层状态及 speedup_claim_blockers 字典。", "20% 只用于预登记诊断标记。分开 within、between、边界证据；缺证据用 NOT_ESTABLISHED，合格边界为 BOUNDARY_ONLY，不冒充全程认证。pilot 的 speedup_claim_allowed 永远 False。"),
        "main": ("batch 批次目录；--out 新报告目录。", "tasks.csv、blocks.csv、assessment.json、report.md。", "逐任务只读重验收旧结果，历史模式显式 require_runtime_audit=False。按 family/ctx/mode 跨 block 合并。复用既有 MAD/IQR，不因 IQR 样本不足就填零。不覆盖源结果，也不重跑 GPU。当前面向已完成的双 block pilot，不是任意缺失批次的通用报告器。"),
    },
    "diagnose_pilot_environment.py": {
        "main": ("无 CLI 参数；固定案例 F09/4096/SpecDec/block0。", "唯一诊断目录中的 manifest、worker.log、telemetry.csv、summary。", "父进程启动一个 -u worker，同时独立 nvidia-smi 每 500 ms 采样；线程读遥测，主线程给日志打接收时间。结束清理自己启动的进程。此工具实际使用 GPU，但采样扰动下的计时不进入主结果；不是正式 runner，也不是可重复直到满意的循环。"),
    },
}

OVERVIEWS = {
    "plan.py": ("第一优先级：先看任务是什么", "把研究设计变成可验证的数据结构。Condition 是一个实验条件，Job 是某 family 在某条件、某 block 下的一次独立进程任务。", "输入 family 列表 -> 六条件 -> 每 block 任务 -> seed 打乱 -> 完整性校验 -> Job 列表。", "dataclasses 的 frozen=True 防止属性被意外修改；Condition 可作为集合/字典键。asdict 递归展开。random 只用于任务顺序；json/Path 用于教学计划落盘。"),
    "worker.py": ("第一优先级：全文重点", "真正执行一个 Job；负责输入、加载、配置、独立诊断、warm-up、测量、边界审计及写盘。模型算法主要由已有 engine 模块提供。", "CLI -> Job -> 环境 -> 输入重建 -> 模型 -> 配置 -> 诊断 -> warm-up -> 3 或 2 次 measured -> JSON。", "torch 管张量、CUDA 与 inference_mode；transformers 只提供 tokenizer/加载器依赖；time.perf_counter 计时；contextmanager 管临时包装；uuid4 命名；traceback 保存异常。多个本地模块承担输入、模型、验证和记录，避免全写进一个函数。"),
    "run.py": ("第一优先级：理解父子进程", "父进程编排，不亲自执行模型生成。它会导入环境模块及其依赖，所以不应把它描述为完全不导入 torch；关键是它不加载 LLM。", "完整 Job 表 -> 选择开发范围或 pilot -> 每 Job 一个子进程 -> 退出码 -> 记录验收 -> 批次摘要。", "argparse 解析 CLI；subprocess 启动独立 worker；sys.executable 保证父子使用同一解释器；time.sleep 冷却；Path/json/uuid4 留日志。copy command 防止追加 output-dir 污染调用者列表。"),
    "records.py": ("第一优先级：读懂成功的含义", "这是文件契约校验器，不是性能统计器，也不是模型精度裁判。worker 保存前和父进程读取后均复用相同规则。", "结果字典 -> 参数/审计/长度/数值校验 -> 严格 JSON 序列化 -> 独占写盘；目录 -> 重验收 -> verified。", "json 负责结构化输出；math 做有限性和容差；Path 管路径；asdict 与启动 Job 对照；本地 runtime_audit 复算边界结论。"),
    "diagnostics.py": ("第二优先级：诊断与计时分离", "使用真实模型、真实算法走一遍带检查的路径。它不是 test_*.py，但自身属于诊断用途，因此本报告仍讲清它对测量的支持关系。", "模型+输入+prefill 回调 -> 临时 hooks -> 真正生成 -> 检查覆盖/逐轮不变量 -> 证据字典 -> 恢复 hooks。", "torch 的 inference_mode 避免梯度；spec_cache 查结构；CheckRegistry 记有限性；DecisionHooks 检查决策前张量；expected_map 使 repair 在出现 mismatch 时才必需。全接受时没 repair 不是失败。"),
    "environment.py": ("第二优先级：资源门", "把环境不具备与算法失败分开。系统 commit 和 GPU 显存不同；不要见 OOM 就默认是 attention 张量太大。", "OS 提交额度 -> 16/18 GiB 门；CUDA 可用 -> 负载期预检 -> 环境字典或拒绝。", "system_memory 复用历史工具的只读系统查询；torch 查 CUDA；gpu_pstate_preflight 提供短负载预检；EnvironmentRefused 是可以分类保存的业务异常。"),
    "runtime_audit.py": ("第二优先级：观察的边界", "在 run_once 外侧保留两点遥测，不更改内部数学路径或计时公式。不把边界读数升级成全程高频认证。", "开始/结束快照 -> 完整性与阈值判定 -> runtime_environment_audit -> 保存和重验收。", "time 提供单调查询时标；math 检查有限值；g5_driver 复用 nvidia-smi 字段和阈值判定。不启动自己的常驻采样线程。"),
    "runtime_info.py": ("第三优先级：复现来源", "记录这条结果由哪个代码/解释器/库环境产生。工作树干净与结果稳定是两个独立概念。", "Git+Python 环境+源文件 -> provenance.runtime。", "hashlib 给源码计算摘要；os 读线程环境变量；subprocess 调 git；sys 读解释器；torch 读版本与 CPU 线程数。"),
    "analyze_teaching_pilot.py": ("第二优先级：离线结果解读", "对完整 pilot 做只读分层审计。报表中的 NOT_ESTABLISHED/BOUNDARY_ONLY 是这份分析 schema 的状态，不是静默改写 canonical 协议状态。", "summary.json -> 各 launch 的 measured JSON -> 重验收 -> 每任务/跨 block 统计 -> 新报表目录。", "statistics 做中位数；csv/json 输出可复核数据；ast 不参与本脚本；results_schema.mad/iqr 复用已有口径。没有启动模型的调用。"),
    "diagnose_pilot_environment.py": ("第三优先级：知道它不能当主计时", "为已发现的波动案例收集独立环境证据，不筛掉慢样本，不替代 pilot。", "固定案例 -> worker 子进程+遥测进程 -> 日志时间锚点+CSV -> 诊断总结。", "subprocess 管两个外部程序；threading 读遥测避免阻塞 worker 输出；csv 解析；datetime(timezone.utc)/uuid4 标识产物；time.perf_counter 做同进程相对时间轴。"),
}

NESTED = {
    "adapted_prefill": "作用域内的 prefill 替身：识别 target/draft，分块并累计秒数，返回算法契约要求的形状。",
    "wrap": "闭包工厂：把具体原函数和阶段名称绑定给 wrapped。",
    "wrapped": "用 *args/**kwargs 原样转发调用，只增加 attention_phase 上下文。",
    "prefill_impl": "给诊断器传入分块实现；把 worker 的 [1,V] 返回值适配为 [1,1,V]。",
    "check_round": "重新从 cands/g 推导 m/emitted，核对本轮和跨轮缓存长度；不是再次计算 target logits。",
    "check_prefill_cache": "target/draft 各自 prefill 后查所有层缓存结构与 prompt 长度。",
    "check_verify_cache": "target 批量验证前检查缓存结构。",
    "git": "绑定项目根目录的命令包装，返回 git 标准输出字符串。",
    "collect": "诊断工具后台线程读取遥测进程输出，收集样本和解析错误。",
}

class GuideDoc(BaseDocTemplate):
    def __init__(self, filename):
        super().__init__(filename, pagesize=A4, leftMargin=20*mm, rightMargin=20*mm,
                         topMargin=22*mm, bottomMargin=19*mm,
                         title="SpecDec 实验代码学习手册", author="SpecDec project")
        self.addPageTemplates(PageTemplate(id="normal", frames=[Frame(
            self.leftMargin, self.bottomMargin, self.width, self.height,
            leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)], onPage=self.decorate))
        self.heading_id = 0

    def beforeDocument(self):
        self.heading_id = 0

    def decorate(self, canvas, doc):
        canvas.setFont("CN", 8)
        canvas.setFillColor(colors.HexColor("#65717a"))
        canvas.drawString(self.leftMargin, A4[1]-13*mm, "SpecDec | 实验代码学习手册 | 2026-10-21")
        canvas.drawRightString(A4[0]-self.rightMargin, 10*mm, str(doc.page))

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name == "H1":
            if flowable.getPlainText() == "目录":
                return
            self.heading_id += 1
            key = "section-" + str(self.heading_id)
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(flowable.getPlainText(), key, level=0)
            self.notify("TOCEntry", (0, flowable.getPlainText(), self.page, key))


def main():
    pdfmetrics.registerFont(TTFont("CN", "C:/Windows/Fonts/msyh.ttc"))
    pdfmetrics.registerFont(TTFont("CNBold", "C:/Windows/Fonts/msyhbd.ttc"))
    pdfmetrics.registerFont(TTFont("Mono", "C:/Windows/Fonts/consola.ttf"))
    styles = {
        "Body": ParagraphStyle("Body", fontName="CN", fontSize=9.2, leading=15, spaceAfter=6, wordWrap="CJK"),
        "H1": ParagraphStyle("H1", fontName="CNBold", fontSize=17, leading=24, spaceAfter=12, wordWrap="CJK"),
        "H2": ParagraphStyle("H2", fontName="CNBold", fontSize=11, leading=17, spaceBefore=10, spaceAfter=6, keepWithNext=True, wordWrap="CJK"),
        "Cell": ParagraphStyle("Cell", fontName="CN", fontSize=8.5, leading=13, wordWrap="CJK"),
        "Title": ParagraphStyle("Title", fontName="CNBold", fontSize=27, leading=39, spaceAfter=18),
        "Code": ParagraphStyle("Code", fontName="Mono", fontSize=7.5, leading=10.5, spaceAfter=8),
    }
    story, markdown = [], []
    def p(text, style="Body"):
        story.append(Paragraph(escape(text).replace("\n", "<br/>"), styles[style]))
        markdown.append(text + "\n")
    def h(text): p(text, "H2")
    def chapter(text):
        if story: story.append(PageBreak())
        p(text, "H1")
    def code(text):
        wrapped = "\n".join(part for line in text.splitlines()
                            for part in (textwrap.wrap(line, width=92, subsequent_indent="    ", replace_whitespace=False) or [""]))
        story.append(Preformatted(wrapped, styles["Code"]))
        markdown.append("```python\n" + text + "\n```\n")
    def table(headers, rows, widths=None):
        data = [[Paragraph(escape(str(c)), styles["Cell"]) for c in row]
                for row in [headers] + rows]
        t = Table(data, colWidths=widths or [145, A4[0]-40*mm-145], repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#e8eff0")),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("LINEBELOW", (0,0), (-1,0), .6, colors.HexColor("#8ea6ad")),
            ("LINEBELOW", (0,1), (-1,-1), .25, colors.HexColor("#d7dfe2")),
            ("LEFTPADDING", (0,0), (-1,-1), 7), ("RIGHTPADDING", (0,0), (-1,-1), 7),
            ("TOPPADDING", (0,0), (-1,-1), 6), ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ]))
        story.append(t)
        story.append(Spacer(1, 8))
        markdown.append(" | ".join(headers))
        markdown.extend(" | ".join(str(c) for c in row) for row in rows)
        markdown.append("")

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    source_meta = {}
    for rel in PATHS:
        raw = (ROOT/rel).read_bytes()
        source_meta[rel] = {"sha256": hashlib.sha256(raw).hexdigest(), "lines": len(raw.splitlines())}
    p("实验代码学习手册", "Title")
    p("从一张 Job 到一份可验收的测量记录", "H2")
    p("SpecDec | 当前非测试性实验模块、运行依赖与参数词典")
    p("代码快照日期：2026-10-21（香港时区）。基础提交：" + commit[:12] + "。")
    p("本手册以本地实际代码为准；不执行模型、不修改 runner。函数签名、导入及行号从 AST 读取，讲解按真实执行顺序整理。")
    p("阅读重点：先 plan.py，再 worker.py，接着 run.py 和 records.py。之后再读 diagnostics.py、environment.py、runtime_audit.py；runtime_info.py 与离线工具用于追溯和分析。")
    h("范围说明")
    p("“所有非测试性文件”在此明确限定为：本轮逐步编写的 experiments/formal 下 8 个非空业务模块，以及两个新增的分析/环境诊断脚本。它们全部逐函数讲解。两个空 __init__.py 仅用于包结构。test_*.py 和 plan.py 内自测函数不展开。worker 中 test_chunked_prefill/test_ar_decode 虽叫 test，实际在测量路径执行，必须阅读。")
    p("另外补充 10 个既有 engine 依赖的重点导读，解释模型数据流和算法接口；不把仓库所有历史诊断脚本逐一当成当前生产模块。input_sets.py 尚未存在，正式 FF01-FF16 接入仍是计划，不写成已完成。")
    h("当前完成到哪里")
    p("完整 pilot 已完成 48 个任务 / 120 条 measured，输入为已开发的 4 个 family。新边界审计也已在一个 F09/4K/SpecDec 任务上联调。当前 worker 仍只接受 pilot family；experiment_stage 仍是 development；正式采集尚未放行。")
    chapter("目录")
    toc = TableOfContents()
    toc.levelStyles = [ParagraphStyle("TOC", fontName="CN", fontSize=9, leading=14, leftIndent=0, firstLineIndent=0, spaceBefore=1, wordWrap="CJK")]
    story.append(toc)

    chapter("01 / 先理解整个系统")
    h("三种“参数”不是一回事")
    p("模型参数是训练得到的权重，例如 embedding 和 q_proj.weight；函数参数是调用时传入的 target/job/cache；实验参数是 ctx/k/chunk/reps 等控制量。我们的 runner 不训练权重，主要控制后两类。")
    table(["层次", "职责 / 不负责什么"], [
        ("研究设计", "主线六条件：2K/4K/8K × AR/SpecDec；模型对、BF16 和其他配置固定。"),
        ("计划与编排", "plan 生成任务；run 启动独立子进程。不是用一个循环长期保留所有条件的 cache。"),
        ("单任务执行", "worker 负责环境、输入、模型、诊断、warm-up、measured、写盘。"),
        ("数学与算法", "model_self 做前向；spec_dec 组织投机；spec_cache 裁剪/检查 KV。"),
        ("证据与统计", "records 验合同，runtime_* 留来源和边界证据，离线脚本汇总。通过文件校验不等于有速度声明资格。"),
    ])
    code("run.main()\n  -> build_jobs() + validate_plan()\n  -> launch_job() -> new Python worker process\n     -> environment -> input -> models -> configuration\n     -> run_diagnostic() -> save_diagnostic()\n     -> run_once()                       # warm-up, discarded\n     -> capture_boundary()\n     -> run_once()                       # measured timing inside\n     -> capture_boundary() -> assess_boundaries()\n     -> save_record() -> validate_record()\n  -> verify_job_results() -> summary.json")
    h("Job、rep、round 的区别")
    p("Job 是一个独立进程任务；rep 是该进程内一整次 prefill+生成；round 是一次 SpecDec 候选提出与验证。一个 Job 可以有 3 个 rep，一个 rep 又可有 20 个 round。不要把这三层计数相加或互换。")
    p("同一 family 的 2K/4K/8K 是同一 8K 输入的嵌套前缀。它们不是三个任意文本，更不能把 ctx 当作生成长度。")

    chapter("02 / 参数总表：控制、形状与命令")
    table(["参数", "当前值 / 类型 / 作用"], [
        ("family_id", "pilot：F09/F03/F11/F04；正式 manifest 预留 FF01-FF16，但 worker 当前未接入。"),
        ("context_tokens / ctx", "2048/4096/8192，输入 token 数，不含新生成。"),
        ("mode / target_kv", "AR 或 SpecDec；主线固定 base_bf16。"),
        ("measurement_block", "0/1 是随机测量轮次，不是 transformer layer 或 prefill chunk。"),
        ("measured_reps", "block0=3，block1=2；每 family×condition 合计 5 次。"),
        ("seed", "4207，控制任务顺序；block 使用 seed+block_id。生成仍是 greedy。"),
        ("k / Kp / k_eff", "k=4 是上限；k_eff 受剩余预算限制；Kp 是考虑 draft EOS 后实际候选数。"),
        ("chunk", "2048，prefill 单块大小；8K 分四块，历史 cache 累积。"),
        ("max_new / max_new_tokens", "测量固定 64。helper test_ar_decode 默认 16，但运行入口显式传 64。"),
        ("eos_id", "151645；命中时停止并保留这个新输出 token。未达预算且无 EOS 不能伪装正常短输出。"),
        ("target / draft", "1.5B 最终判断模型 / 0.5B 候选模型；AR 进程 draft=None。"),
        ("dtype / device", "权重 BF16 / cuda。input_ids 是 torch.long 整数索引，不是 BF16。"),
        ("kv_format='fp16'", "历史引擎字符串，当前表示不量化的基线 KV；BF16 模型路径下 KV 是 BF16。"),
        ("need_logits", "中间 prefill 块 False，最后块 True；不是不运行中间 decoder 层。"),
        ("diagnostic", "helper 的额外有限性/缓存检查开关，测量设 False；独立诊断另跑。"),
        ("theta / head_dim", "target RoPE 基数 1000000；每头维度 128；不要把 head_dim 当 heads=12。"),
        ("min_commit_gib", "批次入口 18，worker 默认 16，单位 GiB。指系统可用提交额度，不是显存。"),
    ])
    h("命令行参数")
    table(["入口 / 参数", "作用"], [
        ("run --execute", "真的创建批次并启动子进程；不加则 dry-run。"),
        ("run --pilot", "选全部 48 个 pilot Job；仍是 development 记录，不是 formal 模式。"),
        ("run --limit 1/2", "只选开发任务前 1 或 2 个；与 --pilot 互斥；默认 1。"),
        ("run --seed", "重建可复现顺序；需要在结果计划中保存。"),
        ("worker --family/--ctx/--mode/--block", "唯一定位一张 Job；目前必须属于现有 pilot choices。"),
        ("worker --dry-run", "显示任务，不进入环境检查与模型加载。"),
        ("worker --output-dir", "父进程指定 worker 结果目录；不提供时用 UUID 建 formal_development 目录。"),
        ("analysis batch --out", "读取已有批次；输出新派生目录，存在则拒绝覆盖。"),
    ])

    chapter("03 / 张量在推理中如何变化")
    p("以下示例是 target：batch B=1，hidden H=1536，Q heads=12，KV heads=2，head_dim D=128，layers=28，词表 V=151936。L 是全部输入长度；s 是本次送进 _run 的长度；T 是可见历史加当前块总长度。")
    table(["变量 / 操作", "形状与含义"], [
        ("ids -> input_ids", "Python list[int] -> [1,L]，dtype long；每个数是词表索引。"),
        ("piece", "[1,s]；例如 chunk=2048。最后一个块可以短于 chunk。"),
        ("embed_tokens(piece)", "[1,s,1536]；按 token ID 查 embedding 行，不是 one-hot 必须显式构造。"),
        ("RMSNorm / residual", "形状仍 [1,s,1536]；归一化改变数值尺度，残差保留信息通路。"),
        ("Q projection + reshape", "[1,s,1536] -> [1,s,12,128] -> [1,12,s,128]。"),
        ("K/V projection", "[1,s,256] -> [1,2,s,128]；GQA 让 12 个 Q 头共享 2 个 KV 头。"),
        ("RoPE", "Q/K 旋转后形状不变；V 不做同样的位置旋转。start_pos 来自已有缓存长度。"),
        ("cache per layer", "K/V 典型为 [1,2,T,128]；列表共 28 层。第 2 维是 token 轴（从 0 起编号）。"),
        ("eager scores", "[1,12,s,T]；先 QK^T/sqrt(D)，再因果 mask 和 softmax。"),
        ("attention output", "各头加权 V 后 [1,12,s,128]，合并为 [1,s,1536]，再 o_proj。"),
        ("DecoderLayer + MLP", "attention、残差、第二次 norm、MLP、残差；层输出仍 [1,s,1536]。"),
        ("LM head", "最终 norm 后乘 embedding.weight.T，得到 [1,s,151936]。不是再乘 attention 的 V。"),
        ("next_logits", "取最后位置 clone -> [1,151936]；argmax -> 一个新 token ID。"),
    ])
    h("两种 attention 与因果约束")
    p("eager 显式分开做 QK、mask、softmax、乘 V。融合 prefill 使用既有 cuDNN 路径，避免部分大临时张量；这不是另一种投机解码。chunk 后面的 query 可看之前全部块和当前块左侧，不能看当前块未来 token。mask 偏移必须计入历史长度 T-s。")
    h("为什么只最后一块需要词表 logits")
    p("每块仍完整更新 28 层 KV。下一 token 只由全部 prompt 的最后位置预测；不需要把之前每个 prompt 位置映射到 151936 个词分数。但当前实现最后块仍生成该块全部位置 logits，再取最后一行，并不是只计算一个位置的 LM head。")
    h("为什么 clone 最后一行")
    p("logits[:, -1, :] 是视图，若长期保存可能连同整块 [1,s,V] 的底层存储一起保留。clone 得到独立 [1,V] 小张量，使原块 logits 可释放。它不意味着 CUDA reserved 会立即下降。")

    for index, rel in enumerate(PATHS, start=4):
        name = Path(rel).name
        source = (ROOT/rel).read_text(encoding="utf-8-sig")
        tree = ast.parse(source)
        title, summary, flow, libraries = OVERVIEWS[name]
        chapter(f"{index:02d} / {name}")
        p("路径：" + rel)
        p("阅读等级：" + title)
        p(summary)
        h("输入输出与执行设计")
        p(flow)
        p(libraries)
        h("实际导入清单")
        imports = [ast.unparse(node) for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        code("\n".join(imports))
        p("上方是实际导入而不是建议列表；from .xxx 是项目内相对导入，不是需要 pip 安装的新库。__future__.annotations 只影响注解求值方式；重复或未使用的导入不等于多执行一次模型。")
        if name == "plan.py":
            h("两个 dataclass 的字段")
            table(["对象", "字段 / 设计理由"], [
                ("Condition", "condition_id 可读编号；context_tokens 输入长度；mode 解码方式；target_kv 缓存表示。frozen=True 使对象字段不可改。"),
                ("Job", "family_id + condition + measurement_block 唯一定位进程任务；measured_reps 控制本进程重复次数。"),
            ])
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name.startswith("test_") and name != "worker.py":
                continue
            if node.name not in FUNCTIONS[name]:
                raise RuntimeError(f"Missing explanation: {name}:{node.name}")
            inputs, outputs, why = FUNCTIONS[name][node.name]
            h(f"{node.name}()  |  源码第 {node.lineno} 行")
            code("def " + node.name + "(" + ast.unparse(node.args) + "):")
            p("输入参数：" + inputs)
            p("输出 / 副作用：" + outputs)
            p("执行与设计原因：" + why)
            nested = [n for n in ast.walk(node) if isinstance(n, ast.FunctionDef) and n is not node and not n.name.startswith("test_")]
            for helper in nested:
                if helper.name in NESTED:
                    p("内部函数 " + helper.name + "(" + ast.unparse(helper.args) + ")：" + NESTED[helper.name])
        if name == "worker.py":
            h("主程序不是一个命名函数，但同样重要")
            p("if __name__ == '__main__' 下解析参数，用 pilot family 构造任务并查找唯一 Job。dry-run 在模型加载前退出。之后依次记 progress.stage，检查 commit / GPU，加载输入和模型，安装配置；provenance 合并 input/load/draft_load/rope/environment/runtime/configuration；最后 execute_job。异常按 ENV_REFUSED/OOM/ERROR 落 failure.json；finally 恢复 target RoPE。")
            p("provenance 是每条结果附带的来源证据，不是模型输入。result.update(provenance) 把这些字段复制到顶层字典。progress 是异常现场标签，例如 load_target、monitor_start、measured、save_record，不属于模型状态。")
        if name == "plan.py":
            p("注意：本文件仍含 test_validate_plan/test_saved_plan 与 __main__ 教学自测。它们不是父进程正常导入时的业务路径。本手册排除其逐行教学；不要为查看计划随意执行此模块，因为其主程序依赖教学期 JSON 已存在。")
        if name == "runtime_audit.py":
            p("重要：capture_boundary 不运行额外 CUDA 负载。GPU 可能已在结束快照前转为空闲，因此低读数不能自动当作整个计时区间低频；高读数也不能排除区间中途降频。这个限制是证据能力边界，不是尚需无限追加测试的借口。")

    chapter("14 / 一次 AR 和一次 SpecDec 如何计时")
    code("# AR, simplified\nprepare_input_tensor()            # outside timer\nsynchronize(); reset_peak()\nstart = perf_counter()\nprefill()\nsynchronize(); prefill_end = perf_counter()\ndecode()\nsynchronize(); end = perf_counter()\nprefill_s = prefill_end - start\ntotal_s = end - start\ndecode_s = total_s - prefill_s")
    p("CUDA 通常异步执行：如果不同步，CPU 可能只测到提交 kernel 的时间。这里的 synchronize 把已提交工作等完，形成可理解的墙钟边界。CPU 控制循环、argmax 取标量及相关开销仍属于实际生成路径，不会因为使用 CUDA 就自动消失。")
    h("SpecDec 的拆分口径")
    p("run_spec_once 在完整 spec_decode_greedy 外计 total。两个 adapted_prefill 各自同步并记录秒数，prefill_s 是两段之和；decode_s 是 total 减这两段。它是当前协议的残差式口径，不能直接解读成所有 verify kernel 的纯耗时，也不能把 draft prefill 隐藏掉。")
    table(["在 measured 计时内", "在计时外"], [
        ("prefill + 实际生成、生产阶段包装及控制开销", "模型/tokenizer 加载、输入重建与配置安装"),
        ("本模式需要的 target/draft 算法工作", "独立诊断、warm-up、边界 nvidia-smi 查询"),
        ("含调用所需的同步", "写 JSON、父进程验收、10 秒 cooldown"),
    ])
    p("当前记录没有独立命名的 TTFT 测量边界。prefill_s 不应直接改名 TTFT：首 token 选择/主机可见的边界和双模型调度含义不同。若未来报告 TTFT，必须先定义并实现相应边界。")

    chapter("15 / 投机算法与 KV 回滚的接口导读")
    p("重点文件：specdec/engine/spec_dec.py 与 specdec/engine/spec_cache.py。以下是已有代码的解释，不是 runner 新写了一套算法。")
    table(["函数 / 参数", "输入输出与为什么这样做"], [
        ("_prefill(model, ids)", "算法默认单次前向返回 logits/cache；worker 作用域把它换成 chunked 实现，仍保持接口。"),
        ("_propose(draft, cache, lg, k_eff, eos_id)", "从 draft 下一 token 分数连续猜候选并更新 draft cache；返回 cands/cache/lg。EOS 可使 Kp 少于 k_eff。"),
        ("_verify(target, cache, cands)", "一次送入候选块，返回 rows=[Kp,V] 与包含整个候选的 target cache。当前 block 模式；serial 仅诊断。"),
        ("_step(model, token, cache)", "喂一个替换 token，返回下一预测行 [1,1,V] 与新 cache，用于 repair。"),
        ("spec_decode_greedy(...)", "target/draft/prompt_ids/max_new_tokens；关键字 k/eos_id/两侧 KV/log。返回 ids（prompt+gen）、gen（仅新输出）、stats。"),
        ("cache_length(cache)", "None -> 0；否则读第 0 层 K 的 shape[2]。不单独证明所有层一致。"),
        ("trim_cache(cache, new_len)", "每层 token 轴切到 new_len 并返回新容器；张量切片可能共享底层存储，不保证立即释放物理显存。"),
        ("check_consistent(cache)", "查每层 k/v/scale 的 token 轴一致及跨层一致；通过返回 None。不是检查每个缓存数值完全正确。"),
    ])
    h("关键的预测行对齐")
    code("g = [argmax(saved_next)] + [argmax(rows[j]) for j in range(Kp-1)]")
    p("saved_next 预测第一个候选；喂入候选 0 后得到的 rows[0] 预测候选 1，而非候选 0。最后 rows[Kp-1] 用于全接受后的下一轮。把候选输入行和被预测位置混淆，会造成 off-by-one。")
    h("拒绝分支示例")
    code("pre = 2048\ncands = [10, 20, 30, 40]\ng     = [10, 20, 99, 40]\nm = 2\nemitted = [10, 20, 99]\ntrim both caches to pre + m       # 2050\nfeed replacement 99 to both       # final length 2051")
    p("第 3 个不匹配后，后面的候选建立在错误历史上，即使某个位置 token 碰巧相同也不能跳着接受。全接受分支输出全部 Kp 候选；本实现没有无条件额外发一个 bonus token。正常完成一轮时 accepted + mismatch_rounds 累计对应输出长度。")
    p("AR 最后一个输出通常没有再喂回模型，SpecDec 最后接受块可能已经写进缓存。因此“生成长度相同”不要求两条实现最终 cache_length 完全相同；缓存检查必须按各自算法语义写。")

    chapter("16 / 既有运行依赖：导入它们意味着什么")
    dependency_rows = [
        ("family_input.py", "标准库 hashlib/json/os/zipfile/Path。load_manifest 读元数据；load_family_ids 按源 block/window 重建 ids 并校验 hash；eos_config 核对 EOS；不运行模型。"),
        ("lowmem_load.py", "torch/gc + align_ours 构建 + transformers 加载。load_ours_bf16(model_name, verify=True) 返回 CUDA 模型、tokenizer、加载证据。先让自建模型转 BF16 再复制权重，减少与 HF state dict 同时驻留的体积；构建阶段仍可能短暂有 FP32 分配，不可把说明当实测峰值。"),
        ("target_rope_scope.py", "torch/hashlib 等。install(target, theta, head_dim) 按实例身份给 target 提供 CPU FP32 inv_freq；draft 保留原来源。restore 还原包装。不是全模型 FP32，也不是用参考 logits 替换。"),
        ("model_self.py", "torch/nn/F 等提供 Embedding、RMSNorm、RoPE、GQA、MLP、DecoderLayer、QwenLM；策略接口控制 prefill 融合。QwenLM._run 输入 embedding 张量而不是 token IDs。"),
        ("spec_dec.py / spec_cache.py", "torch 与上下文/缓存工具。组织候选、验证、拒绝、裁剪、修复与统计。没有重训模型或假定 draft 必须与 target 相同。"),
        ("smoke_hooks.py", "torch/time + engine 模块。DecisionHooks.install/remove 临时替换入口，在 argmax 前查有限性和阶段。属于真实诊断，不在 measured 中保留。"),
        ("smoke_checks.py", "CheckRegistry 记录 checked/finite/角色/轮次，expected_map 和 verify_decision_coverage 检查必须项。条件不需要且未执行，不等于失败。"),
        ("g5_driver.py", "提供 system_snapshot、pstate_verdict 等。nvidia-smi 读不到应为 unavailable/None，不能用 0 伪造观测。"),
        ("gpu_pstate_preflight.py", "torch + threading + 时间/统计等，短固定 CUDA 负载配采样；默认本机 SM>=2000 MHz、memory>=7500 MHz、5 秒、0.5 秒采样、2048 方阵。阈值是本机经验值而非通用标准。"),
    ]
    table(["依赖文件", "导入 / 作用 / 边界"], dependency_rows)
    h("为什么会导入名字含 tests 的历史文件")
    p("lowmem_load 当前从 g4_b4_cache_tests 导入 DTYPE 与加载兼容 helper。这是既有耦合，不代表导入时就执行全部历史测试。是否执行取决于顶层代码和 __main__ 守卫；本手册没有为整理文档而重构它。")
    h("量化功能和当前主线的关系")
    p("引擎已有 V-only FP8 能力：按 group 对 V 做 max-abs scale，q=cast_E4M3(V/scale)，使用时反量化；K 留 BF16。它是压缩缓存表示，不把整模型变为 FP8。当前 runner 只放行 base_bf16，FP8 消融的正式接入和后端质量回归不能当作已完成。")

    chapter("17 / 所有结果参数如何读")
    table(["字段 / 类型", "含义、单位和常见误读"], [
        ("job / configuration", "前者是任务要求，后者是活模型检查后的配置快照；二者都不是权重。"),
        ("input / load / draft_load / rope", "输入来源和 token hash、权重复制证据、RoPE 来源；weights_bitmatch 不等于 AR==Spec。"),
        ("generated_ids / generated_length", "仅新生成 token / 实际个数，不包含 prompt。整数 ID 不是词文本；tokenizer.decode 才转文本。"),
        ("stop_reason / max_new / eos_id", "eos 或 max_new_reached；64 是上限不是强制无视 EOS 生成 64。"),
        ("prefill_s / decode_s / total_s", "秒。total=prefill+decode；模型加载时间不在这个 total 内。"),
        ("decode_tokens_per_s", "实际生成个数 / decode 秒数；不是 inverse(total)，也不直接等于一次 forward 的吞吐。"),
        ("peak_allocated_bytes", "PyTorch 活跃分配峰值，单位字节；除 2^30 得 GiB。只覆盖重置后测量区间，不是进程一生最大峰值。"),
        ("peak_reserved_bytes", "分配器保留池峰值，含空闲块；不能仅凭超过标称 VRAM 证明物理超限或迁移。"),
        ("runtime_environment_audit", "两个计时外查询及重算 verdict；continuous_active_load_verified=False 是诚实边界，不代表程序失败。"),
        ("diagnostic_checks_executed", "measured 内额外诊断是否执行；要求严格 False，不接受 0 或字符串 'False'。"),
        ("same_mode_diagnostic_tokens_equal", "本次 measured 对同模式独立诊断的 token 一致性；不是比较 AR 和 Spec。"),
        ("spec_stats", "rounds/proposed/accepted/target_block_calls/draft_calls/mismatch_rounds/all_accept_rounds/rollbacks。接受率应按 accepted/proposed，替换 token 不是 accepted 候选。"),
        ("attention_stats", "按 phase 统计 fused/eager 调用。target 每块 28 层；双模型每块 28+24=52 次。计数按层调用，不是生成 token 数。"),
        ("run_id / rep_index", "一个 worker 执行的 UUID / 该任务内从 0 起的测量编号。rep_index 在另一个 block 重新起算并不重复任务身份。"),
        ("launch_id / batch_id", "父进程每次启动编号 / 本批编号。失败关联 launch_dir，可找 worker.log 与 exit.json。"),
        ("runtime.git_commit / dirty_worktree", "实际代码提交及是否有修改。干净工作区有助追溯，但不保证时间稳定。"),
        ("runner_source_sha256 / token_sha1", "前者识别源码内容，后者核对输入 token 内容。摘要不可还原原文，不是加密文本。"),
    ])

    chapter("18 / 状态、统计和证据的边界")
    table(["术语", "应该如何解释"], [
        ("development_job_verified", "父进程文件契约验收成功。不能解读成已经正式准入或已经加速。"),
        ("execution SUCCESS", "任务完成；与 measurement 有效、环境可认证是不同轴。"),
        ("within_run_status", "同一进程重复之间的差异；本项目既有报表用吞吐量 range/median。"),
        ("between_block_status", "两个 block 吞吐中位数相对差，用 (hi-lo)/hi。不能用另一个分母暗换阈值。"),
        ("MAD", "median(abs(x-median(x)))，不是标准差；抵抗个别离群值但不自动修复漂移。"),
        ("IQR", "既有 results_schema 定义上下半样本中位差；少于 4 个样本返回 None。2/3 reps 的每 block IQR 不可假装为 0，合并 5 次可以报告。"),
        ("CHECK", "触发诊断线，需完整展示并限制声明；不是删样本或重跑直到好看的许可。"),
        ("NOT_ESTABLISHED", "离线分析中的证据不足；不是已经观测到环境失败。"),
        ("BOUNDARY_ONLY", "两点边界时钟合格；不等于全程活跃频率已经证明。"),
        ("speedup_claim_allowed", "pilot 分析固定 False；正式声明还要满足完整配对、环境、计时和状态等条件。"),
    ])
    h("为什么两个“波动数”可能不同")
    p("早期 v1 报表用时间极差/中位数与非对称 block 比值区间，得到 13/48 和 13/24；按既有吞吐量口径的 v2 得到 12/48 和 12/24。原始结果没变，分母和变量变了。当前应明确使用 protocol_* 字段，不把两份数字当两次实验。")
    h("速度比与输出不一致")
    p("speedup=T_AR/T_Spec，>1 才是 Spec 更快。必须匹配同 family、同 ctx、KV、预算与计时边界；不能拿 F11/4K/Spec 除 F04/2K/AR。AR 与 Spec 分歧后，工作量和接受率可能变化，速度比是不同生成轨迹的比较，不可声称 lossless same-output speedup。")
    p("本轮 pilot 在旧配置 a8a8ba4 上完成，后续边界审计版本为 4c2147c。旧样本缺审计不可补填；新版本对一个任务的验证也不能替换全部旧样本。")

    chapter("19 / 按这个顺序学习，不必一次读全库")
    table(["优先级", "文件与掌握目标"], [
        ("必须精读 1", "plan.py：能手算 4×6×2=48 个 Job，4×6×5=120 条 measured；能解释 Condition/Job。"),
        ("必须精读 2", "worker.py：先 load_input -> execute_job -> run_once，再看 AR/Spec 与 scope。能画出计时内外边界。"),
        ("必须精读 3", "spec_dec.py + spec_cache.py：能解释 saved_next/rows 对齐、m、回滚长度和替换 token。"),
        ("必须精读 4", "run.py：理解全新进程、日志、失败停止、目录关联。records.py：理解为什么退出码 0 仍需验收。"),
        ("第二轮阅读", "diagnostics.py：有限性与控制不变量；model_self.py：按一个 DecoderLayer 到 _run 追张量，不必先看全部诊断开关。"),
        ("第三轮阅读", "environment.py / runtime_audit.py / runtime_info.py：分别处理准入、边界观察和版本来源。"),
        ("出报告时阅读", "analyze_teaching_pilot.py：统计口径与状态分层；diagnose_pilot_environment.py：为什么插桩数据不能混入主计时。"),
    ])
    h("建议用这 8 个问题检查自己")
    for q in [
        "1. 一个 Job 为什么不是只生成一个 token？为什么 48 个 Job 不是 48 条 measured？",
        "2. 同一个 8K prompt 用四个 chunk 时，旧 cache 是丢弃还是继续累积？",
        "3. 第一个生成 token 为什么不需要额外 decode 前向？",
        "4. saved_next 与 rows[0] 分别预测候选的哪个位置？",
        "5. m=2 时，为什么不把整个四候选 cache 留下来？",
        "6. 为什么 weights_bitmatch=True 不证明每条生成轨迹相同？",
        "7. 为什么边界查询 60 ms 不在 decode_s 里，却仍可能改变后续环境状态？",
        "8. verified=True、CHECK、BOUNDARY_ONLY 分别属于哪一层证据？",
    ]: p(q)

    chapter("20 / 现有实现、限制与下一步")
    p("已实现：pilot 任务规划、独立 worker、低内存权重校验加载、BF16/策略/RoPE 配置、分块 prefill、AR/Spec 执行、独立诊断与 hooks 恢复、warm-up、3+2 测量、JSON 防覆盖、父进程验收、18/16 GiB 门、边界审计、只读分层分析。")
    p("尚未实现：input_sets.py 未落盘；FF01-FF16 正式输入选择与明确 formal 入口未接入；当前 records 只认可 development 阶段。没有可用的自动续跑/完整超时管理，也没有声称全程频率已获认证。FP8 正式消融与可选生成质量研究不能视为已完成。")
    p("已知实现限制：部分 measured 数值和 token 字段未进行最严格的所有类型/范围校验；计划类 frozen 不是磁盘数据签名；独占写文件不是原子事务；边界 verdict 重算依赖当前阈值；低内存加载的 CPU 峰值说明不是本次实测。理解边界比把每个 PASS 当最终证明更重要。")
    p("后续教学的第一步是将 pilot/formal manifest 选择集中为一个只读接口，然后让调度器和 worker 共用，而不是仅把 family choices 扩大。这一步先读元数据、核对集合，不运行正式模型样本。")

    chapter("附录 / 源码快照与范围清单")
    p("下表给出生成文档时实际读取的业务文件及 SHA256 前 16 位。完整摘要保存在同名 sources.json，便于确认手册对应哪一版源文件。新生成的本 PDF 脚本属于文档工具，不是实验 runner。")
    table(["源码 / 行数", "SHA256 前 16 位"], [(rel + " / " + str(m["lines"]), m["sha256"][:16]) for rel,m in source_meta.items()], widths=[350, A4[0]-40*mm-350])
    p("空包文件：experiments/__init__.py、experiments/formal/__init__.py，只为 python -m 包导入提供结构，无业务参数和函数。")
    p("不展开测试文件：test_configuration_cpu.py、test_diagnostics_cpu.py、test_environment_cpu.py、test_launcher_cpu.py、test_runtime_audit_cpu.py、scripts/test_pilot_assessment_cpu.py。测试能证明契约反例被拦截，不替代 GPU 数值证据。")
    h("依赖文件定位")
    for dep in DEPENDENCIES:
        path = ROOT / "specdec/engine" / dep
        p("specdec/engine/" + dep + "  |  " + hashlib.sha256(path.read_bytes()).hexdigest()[:16])

    OUT.mkdir(parents=True, exist_ok=True)
    pdf = OUT / (NAME + ".pdf")
    GuideDoc(str(pdf)).multiBuild(story)
    (OUT/(NAME+".md")).write_text("\n".join(markdown), encoding="utf-8")
    (OUT/(NAME+".sources.json")).write_text(json.dumps({"git_commit":commit, "snapshot_date":"2026-10-21", "files":source_meta}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pdf)


if __name__ == "__main__":
    main()
