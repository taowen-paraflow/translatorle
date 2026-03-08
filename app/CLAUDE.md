# App 模块 — PySide6 桌面应用（录音 → 转写 → 翻译）

基于 PySide6 的桌面 GUI，集成 `asr/`（流式语音识别）和 `hymt/`（机器翻译）模块，运行在 Intel NPU 上。

## 快速开始

```powershell
# 安装依赖（首次）
uv sync

# 启动应用
$env:PYTHONIOENCODING = "utf-8"
uv run python -m app
```

启动后状态栏显示 "Loading engines..."，等 ASR + MT 引擎加载完（约 8-15s）变为 "Ready" 即可使用。

## 使用流程

1. 点击 **Record** 开始录音，对麦克风说话
2. 实时转写文本显示在 Transcription 区域；每当 ASR 输出完整句子（以。！？等标点结尾），自动发送到 MT 翻译，Translation 区域增量更新
3. 点击 **Stop** 停止录音，剩余未翻译文本发送到 MT 完成翻译
4. 可手动切换目标语言后点 **Translate** 重新翻译全文

## 架构

```
sounddevice callback → AudioCapture.queue → ASRWorker(QThread) --signal→ UI → MTWorker(QThread) --signal→ UI
```

### 数据流

```
用户点 Record
  → MTWorker.start_session(target_lang)：开始 KV cache 会话
  → AudioCapture.start()：sounddevice 16kHz/mono/float32, 100ms blocks
  → 音频块入 queue.Queue
  → ASRWorker(NPU) 从 queue 读取，调用 engine.feed() → text_updated 信号
  → MainWindow 检测句子边界（。！？等标点）
  → 完整句子 → MTWorker(GPU) 逐句翻译（KV cache 自动维护上下文）→ sentence_translated 信号增量更新翻译区
  → 接近 token 上限时自动 reset_session() 重建 KV cache
用户点 Stop
  → AudioCapture.stop()
  → ASRWorker drain + engine.finish() → session_finished 信号
  → 剩余未翻译文本发送到 MT → 翻译区显示完整结果
  → MTWorker.finish_session()：释放 KV cache
```

### 设备分配

- ASR（encoder + decoder） → NPU
- HY-MT → GPU

ASR 和 MT 运行在不同硬件上，因此可以并发推理：录音期间 ASR 在 NPU 上持续转写，MT 在 GPU 上逐句翻译已完成的句子，无需等录音结束。两个引擎分别在各自的 QThread 中初始化。

## 文件结构

| 文件 | 类 | 职责 |
|------|----|------|
| `__main__.py` | — | 入口：创建 QApplication + MainWindow |
| `main_window.py` | `MainWindow` | UI 布局 + 信号连接 + 状态管理 |
| `audio_capture.py` | `AudioCapture` | sounddevice 麦克风录音封装 |
| `asr_worker.py` | `ASRWorker(QThread)` | ASR 推理线程 |
| `mt_worker.py` | `MTWorker(QThread)` | MT 推理线程 |

## 模块接口

### AudioCapture

```python
cap = AudioCapture()        # 内部创建 queue.Queue
cap.start()                 # 开始录音，音频块(np.float32)进入 cap.queue
cap.stop()                  # 停止录音
AudioCapture.is_available() # 检查 sounddevice 是否可用
```

参数：16kHz, mono, float32, blocksize=1600 (100ms)。

### ASRWorker

```python
worker = ASRWorker(audio_queue)  # 传入 AudioCapture.queue
worker.start()                    # 启动线程，加载 ASREngine(NPU)

# 信号
worker.engine_ready          # 引擎加载完成
worker.text_updated(str)     # 实时转写文本更新
worker.session_finished(str, str)  # (最终文本, 检测语言)
worker.error(str)            # 错误信息

# 命令（主线程调用，通过内部 command queue 传递）
worker.start_session()       # 开始新 ASR 会话
worker.stop_session()        # 结束会话（drain + finish）
worker.shutdown()            # 退出线程
```

### MTWorker

```python
worker = MTWorker()
worker.start()               # 启动线程，加载 MTEngine(GPU)

# 信号
worker.engine_ready          # 引擎加载完成
worker.sentence_translated(str, str)  # (原文, 译文) — 逐句增量翻译
worker.translation_done(str) # 全文翻译结果（Translate 按钮触发）
worker.error(str)            # 错误信息

# 命令
worker.start_session(target_lang)                # 开始 KV cache 会话
worker.finish_session()                          # 结束会话，释放 KV cache
worker.translate_sentence(sentence, target_lang) # 录音中逐句翻译（会话模式自动复用 KV cache）
worker.translate(text, target_lang)              # 全文翻译（手动 Translate 按钮）
worker.shutdown()
```

## UI 布局

```
┌──────────────────────────────────────┐
│  Translatorle                        │
├──────────────────────────────────────┤
│  [Record]  Translate to: [Chinese ▼] │
│            [Translate]               │
├──────────────────────────────────────┤
│  Transcription                       │
│  ┌──────────────────────────────────┐│
│  │ (实时语音识别文本)                ││
│  └──────────────────────────────────┘│
│  Translation                         │
│  ┌──────────────────────────────────┐│
│  │ (翻译结果)                       ││
│  └──────────────────────────────────┘│
│  Status: Ready                       │
└──────────────────────────────────────┘
```

## 设计决策

1. **sounddevice 而非 Qt Audio** — 直接支持 16kHz float32，无需重采样
2. **Engine 在 worker 线程初始化** — 避免 OpenVINO/NPUW 线程安全问题
3. **Command queue 模式** — Worker 用 `queue.Queue` 接收命令，比 Qt 信号更简单可控
4. **AudioCapture 自持 queue** — MainWindow 通过 `audio.queue` 传给 ASRWorker，避免外部 queue 与内部 queue 不一致
5. **边 ASR 边翻译** — ASR(NPU) 和 MT(GPU) 在不同硬件上并发运行，句子完成即送翻译，无需等录音结束
6. **Record 按钮 checkable** — 切换录音/停止状态，录音结束到 session_finished 期间禁用防止重复点击

## 依赖

在 `pyproject.toml` 中添加：
- `PySide6>=6.7` — Qt GUI 框架
- `sounddevice` — 麦克风录音

运行环境：Windows 11, Python 3.12, Intel NPU, `uv` 管理依赖。

## 已知警告（无害）

启动时可能出现：
- `XPU device count is zero!` — torch XPU 初始化提示，不影响 NPU 推理
