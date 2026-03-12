# Translatorle — 语音识别 + 机器翻译桌面应用

## 环境

- **OS**: Windows 11 + WSL2
- **Python**: 3.12, 用 `uv` 管理依赖
- **uv 路径**: `C:\Users\taowen\.local\bin\uv.exe`
- **推理硬件**: Intel Core Ultra 7 258V (Lunar Lake) — NPU + GPU

## Python 虚拟环境

项目有两个独立的 uv venv：

| venv | 路径 | transformers | openvino | 其他 | 用途 |
|------|------|-------------|----------|------|------|
| **根目录** | `.venv/` | 4.57.6 (<5) | 2026.0.0 | openvino_genai 2026.0, optimum-intel 1.15.0 (broken) | ASR、MT 推理、App GUI |
| **qwen35** | `qwen35/.venv/` | 5.3.0 (>=5) | 2026.0.0 | 无 genai/optimum | Qwen3.5 模型导出 |

关键注意：
- **推理脚本**用根 venv 运行：`uv run python -m ...`
- **qwen35 导出**用子 venv 运行：`uv run --project qwen35 python -m ...`
- **optimum-intel 1.15.0 已 broken**（`is_accelerate_available` ImportError），Python 中 `from optimum.intel import ...` 会报错，但 `uv run optimum-cli export openvino` CLI 仍可用
- 根 venv 有 `openvino_genai`（LLMPipeline），qwen35 venv 没有

## uv 包管理

- **uv** 是 Rust 实现的 Python 包管理器，替代 pip + venv
- **Windows 路径**: `C:\Users\taowen\.local\bin\uv.exe`
- **WSL2 中调用**: 必须通过 `powershell.exe -Command` 调用 Windows 侧的 uv.exe
- **项目级 venv**: `uv run python ...` 自动使用项目根目录的 `.venv/`
- **子项目 venv**: `uv run --project qwen35 python ...` 使用 `qwen35/.venv/`
- **添加依赖**: `uv add <package>` 更新 `pyproject.toml` + `uv.lock`
- **锁文件**: `uv.lock` 是确定性锁文件，类似 `package-lock.json`

## 本地 OpenVINO Fork

三个仓库在 `/mnt/c/Apps/` 下，用于跟踪上游进展和做本地实验：

| 仓库 | 路径 | 分支 | 说明 |
|------|------|------|------|
| **openvino** | `/mnt/c/Apps/openvino` | master | NPUW 插件有 GDN 适配补丁 |
| **openvino.genai** | `/mnt/c/Apps/openvino.genai` | master | LLMPipeline 等高层 API |
| **optimum-intel** | `/mnt/c/Apps/optimum-intel` | main | 已有 GDN linear attention 导出支持 (#1619) |

### openvino NPUW GDN 适配

本地 fork 的 `src/plugins/intel_npu/src/plugin/npuw/` 有以下 Qwen3.5 GDN 相关修改：
- **状态保留**: `RemoveEmptyKVTensors` 跳过 GDN 的 `conv`/`recurrent` 状态（初始非空，不能删）
- **精度保护**: `cvt_kvcache_to_low_precision` 排除 GDN 状态，保持 FP32
- **静态维度**: `reshape_to_static` 保留 GDN 状态的固定维度（如 D_k=128），不当作动态 KV seq_len
- **Passthrough copy**: prefill→generate 阶段对固定大小的 GDN 状态做全量拷贝，不做 KV slice

### 上游进展

- **optimum-intel PR #1634**: 官方 Qwen3.5 导出支持（`RecurrentAttentionCellOp`、双状态管理、Stateful IR）
- **openvino issue #34532**: `ScatterUpdate` 在 Loop 内强制降 FP16 — 影响 GPU 上的 GDN 递归精度（Open）

## 从 WSL2 调用 Windows 命令

```bash
powershell.exe -Command 'cd C:\Apps\translatorle; C:\Users\taowen\.local\bin\uv.exe run python -m app'
```

设置环境变量：

```bash
powershell.exe -Command '$env:PYTHONIOENCODING="utf-8"; cd C:\Apps\translatorle; C:\Users\taowen\.local\bin\uv.exe run python ...'
```

## 项目结构

| 目录 | 说明 |
|------|------|
| `asr/` | 流式语音识别（Qwen3-ASR 0.6B/1.7B） |
| `hymt/` | 机器翻译（HY-MT1.5-1.8B） |
| `qwen3/` | Qwen3 标准 Transformer benchmark（NPU/GPU/CPU 性能测试） |
| `qwen35/` | Qwen3.5 通用推理（GDN 混合架构，含 VL 视觉语言） |
| `app/` | PySide6 桌面 GUI |
| `models/` | OpenVINO IR 模型文件（不入 git） |

各模块有独立的 CLAUDE.md，详见 `asr/CLAUDE.md`、`hymt/CLAUDE.md`、`app/CLAUDE.md`、`qwen35/CLAUDE.md`。

## 设备分配

| 组件 | 0.6B 模式 | 1.7B 模式（默认） |
|------|----------|------------------|
| ASR Encoder | NPU | NPU |
| ASR Decoder | NPU | NPU |
| MT (HY-MT) | NPU/GPU | NPU/GPU |

- 0.6B: ASR 全在 NPU（folding 正常），MT 在 GPU，互不干扰
- 1.7B: decoder 在 NPU 运行，使用 `NPUW_FOLD=NO` 绕过 folding assert（inputs_embeds 3D 入口导致层不均匀匹配失败）。编译时间 ~460s，RTF 0.30-0.32x，输出与 GPU 完全一致
- MT 模型 INT4_SYM 已验证可在 NPU 运行（LLMPipeline），当前默认 GPU 以避免与 ASR 争用 NPU 带宽

## NPU 量化规则

| 模型规模 | 推荐精度 | 要点 |
|---------|---------|------|
| ≤1B | FP16 | 小模型 INT4 反而更慢（反量化开销 > 带宽节省） |
| 1-2B | INT4_SYM (group=128) | **必须 SYM**，ASYM 回退慢路径（1.2 vs 29 tok/s） |
| Dynamic Quant | 不推荐（≤2B） | 小模型已饱和带宽，只增加开销 |

关键注意：
- NPU **不支持 INT4_ASYM**，optimum-intel 导出时必须指定 `--symmetric`
- optimum-intel 对 >1B 模型**自动应用 INT8_ASYM**，必须 `load_in_8bit=False` 手动控制
- `nncf.compress_weights()` 可直接对已导出的 IR 模型做压缩，无需原始 PyTorch 模型
