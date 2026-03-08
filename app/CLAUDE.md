# App 模块 — PySide6 桌面应用（录音 → 转写 → 翻译）

基于 PySide6 的桌面 GUI，集成 `asr/`（流式语音识别）和 `hymt/`（机器翻译）模块。

## 快速开始

```powershell
$env:PYTHONIOENCODING = "utf-8"
uv run python -m app
```

启动后等待 ASR + MT 引擎加载完（1.7B 约 40s），状态栏变为 "Ready" 即可使用。

## 使用流程

1. 从工具栏右侧选择 ASR 模型（默认 1.7B）
2. 点击 **Record** 开始录音，对麦克风说话
3. 实时转写显示在 Transcription 区域；完整句子自动发送到 MT 翻译
4. 点击 **Stop** 停止录音，剩余文本完成翻译

## 架构

```
sounddevice → AudioCapture.queue → ASRWorker(QThread) → UI → MTWorker(QThread) → UI
```

### 数据流

```
用户点 Record
  → MTWorker.start_session(target_lang)：KV cache 会话开始
  → AudioCapture.start()：16kHz/mono/float32, 100ms blocks
  → ASRWorker 读 queue → engine.feed() → text_updated 信号
  → MainWindow 检测句子边界（。！？等标点）
  → 完整句子 → MTWorker 逐句翻译 → 增量更新翻译区
用户点 Stop
  → AudioCapture.stop() → ASRWorker drain + finish → session_finished
  → 剩余文本发 MT → MTWorker.finish_session()
```

### 设备分配

设备由 `ASRModelConfig` 决定（见 `asr/config.py`）：
- **0.6B**: ASR(enc+dec) → NPU, MT → GPU
- **1.7B**: ASR(enc+dec) → NPU（decoder 使用 `NPUW_FOLD=NO`）, MT → GPU

ASR 和 MT 分别在各自的 QThread 中初始化和运行。

## 文件结构

| 文件 | 类 | 职责 |
|------|----|------|
| `__main__.py` | — | 入口：QApplication + MainWindow |
| `main_window.py` | `MainWindow` | UI 布局 + 信号连接 + 状态管理 |
| `audio_capture.py` | `AudioCapture` | sounddevice 麦克风录音封装 |
| `asr_worker.py` | `ASRWorker(QThread)` | ASR 推理线程 |
| `mt_worker.py` | `MTWorker(QThread)` | MT 推理线程 |

## 模块接口

### AudioCapture

```python
cap = AudioCapture()        # 内部创建 queue.Queue
cap.start()                 # 开始录音
cap.stop()                  # 停止录音
```

### ASRWorker

```python
worker = ASRWorker(audio_queue, model_name=None)

# 信号
worker.engine_ready            # 引擎加载完成
worker.text_updated(str)       # 实时转写文本
worker.session_finished(str, str)  # (最终文本, 语言)
worker.model_reloaded          # 模型切换完成
worker.error(str)              # 错误信息

# 命令
worker.start_session()         # 开始 ASR 会话
worker.stop_session()          # 结束会话
worker.reload_model(name)      # 切换 ASR 模型（"0.6B" / "1.7B"）
worker.shutdown()              # 退出线程
```

### MTWorker

```python
worker = MTWorker()

# 信号
worker.engine_ready                     # 引擎加载完成
worker.sentence_translated(str, str)    # (原文, 译文)
worker.error(str)

# 命令
worker.start_session(target_lang)                # 开始 KV cache 会话
worker.finish_session()                          # 释放 KV cache
worker.translate_sentence(sentence, target_lang) # 逐句翻译
worker.shutdown()
```

## UI 布局

```
┌───────────────────────────────────────────────────┐
│  Translatorle                                     │
├───────────────────────────────────────────────────┤
│  [Record]  Translate to: [Chinese ▼]   ASR Model: [1.7B ▼] │
├───────────────────────────────────────────────────┤
│  Transcription                                    │
│  ┌───────────────────────────────────────────────┐│
│  │ (实时语音识别文本)                             ││
│  └───────────────────────────────────────────────┘│
│  Translation                                      │
│  ┌───────────────────────────────────────────────┐│
│  │ (翻译结果)                                    ││
│  └───────────────────────────────────────────────┘│
│  Status: Ready                                    │
└───────────────────────────────────────────────────┘
```

## 设计决策

1. **sounddevice 而非 Qt Audio** — 直接支持 16kHz float32，无需重采样
2. **Engine 在 worker 线程初始化** — 避免 OpenVINO/NPUW 线程安全问题
3. **Command queue 模式** — Worker 用 `queue.Queue` 接收命令，比 Qt 信号更简单可控
4. **边 ASR 边翻译** — 句子完成即送翻译，无需等录音结束
5. **模型热切换** — 通过 `reload_model` 支持运行时切换 ASR 模型，录音中禁止切换
