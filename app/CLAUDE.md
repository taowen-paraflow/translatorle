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
2. 实时转写文本显示在 Transcription 区域
3. 点击 **Stop** 停止录音，自动检测语言并翻译
4. 翻译结果显示在 Translation 区域
5. 可手动切换目标语言后点 **Translate** 重新翻译

## 架构

```
sounddevice callback → AudioCapture.queue → ASRWorker(QThread) --signal→ UI → MTWorker(QThread) --signal→ UI
```

### 数据流

```
用户点 Record
  → AudioCapture.start()：sounddevice 16kHz/mono/float32, 100ms blocks
  → 音频块入 queue.Queue
  → ASRWorker 从 queue 读取，调用 engine.feed() → text_updated 信号更新 UI
用户点 Stop
  → AudioCapture.stop()
  → ASRWorker drain + engine.finish() → session_finished 信号
  → 自动调用 MTWorker.translate() → translation_done 信号更新 UI
```

### NPU 防冲突

录音期间只跑 ASR 推理，Translate 按钮禁用。停止录音后才触发 MT 推理。两个引擎分别在各自的 QThread 中初始化，避免 OpenVINO/NPUW 线程安全问题。

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
worker.start()               # 启动线程，加载 MTEngine(NPU)

# 信号
worker.engine_ready          # 引擎加载完成
worker.translation_done(str) # 翻译结果
worker.error(str)            # 错误信息

# 命令
worker.translate(text, target_lang)  # "Chinese", "English", "Japanese" 等
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
5. **录音停止后自动翻译** — 根据 ASR 检测的语言自动设置翻译方向（中→英 / 其他→中）
6. **Record 按钮 checkable** — 切换录音/停止状态，录音结束到 session_finished 期间禁用防止重复点击

## 依赖

在 `pyproject.toml` 中添加：
- `PySide6>=6.7` — Qt GUI 框架
- `sounddevice` — 麦克风录音

运行环境：Windows 11, Python 3.12, Intel NPU, `uv` 管理依赖。

## 已知警告（无害）

启动时可能出现：
- `XPU device count is zero!` — torch XPU 初始化提示，不影响 NPU 推理
- `incorrect regex pattern` — Qwen3 tokenizer 已知 warning，不影响功能
