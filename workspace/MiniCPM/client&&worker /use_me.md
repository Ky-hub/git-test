# MiniCPMO45 Gateway 全双工诊断日志使用说明

---

## 一、功能概述

在 `/ws/duplex/{session_id}` 和 `/v1/realtime` 两个全双工入口增加了逐消息诊断日志。  
每个 WebSocket 会话独立生成一个 JSONL 日志文件，记录：

- 每条消息的收发时间戳（毫秒级）
- 距上一条消息的间隔时间 `delta_ms`
- 音频/视频数据长度
- 是否为空包
- Worker 推理状态（`is_listen`、`end_of_turn`、`kv_cache_length`）

用于排查前端卡顿、Worker 无响应、空包风暴等问题。

---

## 二、日志文件位置

默认路径：
```bash
{gateway.py所在目录}/data/logs/duplex_trace/
```

文件命名规则：

| 入口 | 文件名示例 |
|------|-----------|
| `/ws/duplex/abc123` | `abc123.jsonl` |
| `/v1/realtime` | `rt_1751445600000.jsonl` |

如需修改路径，编辑 `gateway.py` 中两个函数的：
```python
trace_dir = os.path.join(os.path.dirname(__file__), "data", "logs", "duplex_trace")
# 改为任意绝对或相对路径，如：
# trace_dir = "/var/log/minicpmo/trace"
```

目录会自动创建，无需手动建。

---

## 三、实时查看日志

### 1. 查看最新写入的会话
```bash
ls -lt data/logs/duplex_trace/ | head -n 5
```

### 2. 实时追踪某个会话
```bash
tail -f data/logs/duplex_trace/abc123.jsonl | jq .
```

### 3. 查看历史会话最后 20 条
```bash
tail -n 20 data/logs/duplex_trace/abc123.jsonl | jq .
```

### 4. 过滤卡顿点（delta_ms > 1000ms）
```bash
cat data/logs/duplex_trace/abc123.jsonl | jq 'select(.delta_ms > 1000)'
```

### 5. 只看空包
```bash
cat data/logs/duplex_trace/abc123.jsonl | jq 'select(.is_empty == true)'
```

---

## 四、日志字段说明

每行一条 JSON，字段如下：

| 字段 | 类型 | 说明 |
|------|------|------|
| `ts` | string | ISO 8601 时间戳（毫秒） |
| `unix_ts` | float | Unix 时间戳（秒，6位小数） |
| `direction` | string | `C->W`（前端→Worker）或 `W->C`（Worker→前端） |
| `session_id` | string | 会话 ID |
| `msg_type` | string | 消息类型：`audio_chunk`、`prepare`、`result`、`pause`、`resume`、`stop` 等 |
| `raw_bytes` | int | 原始 JSON 字符串的字节数 |
| `delta_ms` | float | **距上一条消息的时间间隔（毫秒）**，卡顿排查核心指标 |
| `audio_len` | int | Base64 音频字符串长度（0 表示无音频） |
| `video_frame_count` | int | 视频帧数量 |
| `video_total_len` | int | 所有视频帧 Base64 总长度 |
| `is_empty` | bool | **true 表示音频+视频+文本全空** |
| `text_preview` | string | Worker 返回文本的前 80 字（仅 `W->C`） |
| `is_listen` | bool | Worker 是否处于"收听/思考"状态（仅 `W->C`） |
| `end_of_turn` | bool | 是否为当前轮次结束（仅 `W->C`） |
| `kv_cache_length` | int | KV Cache 长度（仅 `W->C`） |
| `force_listen` | bool | 前端是否强制触发收听（仅 `C->W`） |

---

## 五、卡顿排查速查表

| 现象 | 日志特征 | 根因定位 |
|------|---------|---------|
| **前端画面卡死，无 AI 回复** | `C->W` 正常发送，但 `W->C` 长时间缺失，最后一条 `W->C` 是 `is_listen=true` | Worker 推理阻塞，检查 GPU 负载、KV Cache 是否过长 |
| **前端画面卡死，AI 不说话** | `W->C` 持续返回 `is_empty=true`，`audio_len=0`，`text_preview=""` | Worker 生成空结果，检查模型输入或推理异常 |
| **前端卡顿，但日志显示高频发送** | `C->W` 密集出现，`delta_ms` < 50ms，`is_empty=true` 居多 | 前端 VAD/采集逻辑异常，发送大量空音频包 |
| **视频全双工时只有画面没有声音** | `C->W` 中 `video_frame_count>0` 但 `audio_len=0` | 麦克风未采集到数据，检查浏览器麦克风权限 |
| **交互延迟越来越大** | `delta_ms` 逐渐从 200ms 涨到 2000ms+ | 网络延迟或 Worker 队列堆积 |
| **会话突然断开** | 日志突然中断，无 `stop` 消息 | 浏览器刷新/切后台、Worker 崩溃、或 Gateway 超时 |
| **AI 回复被截断** | `W->C` 中 `kv_cache_length` 接近或超过 8192 | 上下文满了触发强制结束，需清理会话或缩短对话 |

---

## 六、典型排查流程

### 步骤 1：确认前端走哪个入口
```bash
ls -lt data/logs/duplex_trace/ | head -n 5
```
看最新文件是 `{session_id}.jsonl` 还是 `rt_xxx.jsonl`。

### 步骤 2：观察消息间隔
```bash
tail -f data/logs/duplex_trace/xxx.jsonl | jq '{ts, direction, delta_ms, msg_type, is_empty}'
```
正常情况：
- 前端发送音频：`delta_ms` 约 100~300ms（取决于采集帧率）
- Worker 返回结果：`delta_ms` 约 100~800ms（取决于推理速度）

### 步骤 3：定位卡顿点
```bash
cat data/logs/duplex_trace/xxx.jsonl | jq 'select(.delta_ms > 2000)'
```
找到 `delta_ms` 突然跳变的时刻，查看前后几条消息的方向和内容。

### 步骤 4：检查空包
```bash
cat data/logs/duplex_trace/xxx.jsonl | jq 'select(.is_empty == true and .direction == "C->W")' | wc -l
```
如果空包数量占总消息 50% 以上，前端采集逻辑有问题。

### 步骤 5：检查 Worker 状态
```bash
cat data/logs/duplex_trace/xxx.jsonl | jq 'select(.direction == "W->C") | {ts, is_listen, end_of_turn, kv_cache_length, text_preview}'
```
看 Worker 是否卡在 `is_listen=true` 不返回，或 `kv_cache_length` 是否异常增长。

---

## 七、注意事项

1. **日志不记录完整 Base64 内容**，只记录长度。如需保存完整音频/视频帧，需额外修改 `_write_trace` 函数。
2. **日志文件不会自动清理**，长期运行需配合 `logrotate` 或定时脚本清理 `data/logs/duplex_trace/`。
3. **高并发时磁盘 I/O**：每个消息都写盘，如果 QPS 很高，建议把 `trace_dir` 放在 SSD 或 tmpfs 上。
4. **日志写入是异步的**（通过 `asyncio.to_thread`），不会阻塞 WebSocket 透传，但极端高负载下仍可能轻微影响延迟。
5. **两个入口日志独立**：`/ws/duplex` 和 `/v1/realtime` 各写各的文件，排查时别找错文件。

---

## 八、快速命令备忘

```bash
# 看最新 5 个会话
ls -lt data/logs/duplex_trace/ | head -n 6

# 实时追踪
tail -f data/logs/duplex_trace/SESSION_ID.jsonl | jq .

# 只看 Worker 返回
cat data/logs/duplex_trace/SESSION_ID.jsonl | jq 'select(.direction == "W->C")'

# 只看前端发送
cat data/logs/duplex_trace/SESSION_ID.jsonl | jq 'select(.direction == "C->W")'

# 找卡顿点（间隔 > 1秒）
cat data/logs/duplex_trace/SESSION_ID.jsonl | jq 'select(.delta_ms > 1000)'

# 统计空包比例
total=$(wc -l < data/logs/duplex_trace/SESSION_ID.jsonl)
empty=$(cat data/logs/duplex_trace/SESSION_ID.jsonl | jq -s '[.[] | select(.is_empty == true)] | length')
echo "空包比例: $empty / $total"

# 统计消息方向比例
cat data/logs/duplex_trace/SESSION_ID.jsonl | jq -s 'group_by(.direction) | map({direction: .[0].direction, count: length})'
```

---

重启 Gateway 后生效。出现卡顿时，按上述流程拉取对应 `session_id` 的日志即可定位问题。
