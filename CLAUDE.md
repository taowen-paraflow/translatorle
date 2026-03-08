# Translatorle — 语音识别 + 机器翻译桌面应用

## 环境

- **OS**: Windows 11 + WSL2
- **Python**: 3.12, 用 `uv` 管理依赖
- **uv 路径**: `C:\Users\taowen\.local\bin\uv.exe`
- **推理硬件**: Intel Core Ultra 7 258V (Lunar Lake) — NPU + GPU

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
| `app/` | PySide6 桌面 GUI |
| `models/` | OpenVINO IR 模型文件（不入 git） |

各模块有独立的 CLAUDE.md，详见 `asr/CLAUDE.md`、`hymt/CLAUDE.md`、`app/CLAUDE.md`。

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
