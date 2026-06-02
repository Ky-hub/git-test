@app.websocket("/v1/realtime")
async def realtime_ws(ws: WebSocket):
    """OpenAI Realtime-style WebSocket 代理

    Protocol translation gateway:
      Client speaks OpenAI Realtime events → translates to old protocol → Worker
      Worker speaks old protocol → translates to OpenAI events → Client

    Uses the same FIFO queue + Worker allocation as /ws/duplex.
    """
    session_id = f"rt_{int(datetime.now().timestamp()*1000)}"
    mode = ws.query_params.get("mode", "video")
    max_duration_s = 300 if mode == "video" else 600

    if worker_pool is None:
        await ws.close(code=1013, reason="Service not ready")
        return

    session_id = _sanitize_session_id(session_id)
    await ws.accept()

    try:
        duplex_type = "omni_duplex" if mode == "video" else "audio_duplex"
        ticket, future = worker_pool.enqueue(duplex_type, session_id=session_id)
    except WorkerPool.QueueFullError:
        await ws.send_json({
            "type": "error",
            "error": {"code": "queue_full", "message": "Queue full", "type": "server_error"},
        })
        await ws.close(code=1013, reason="Queue full")
        return

    worker: Optional[WorkerConnection] = None
    if future.done():
        worker = future.result()
    else:
        try:
            await ws.send_json({
                "type": "session.queued",
                "position": ticket.position,
                "estimated_wait_s": ticket.estimated_wait_s,
                "ticket_id": ticket.ticket_id,
                "queue_length": worker_pool.queue_length,
            })
            while not future.done():
                try:
                    worker = await asyncio.wait_for(asyncio.shield(future), timeout=3.0)
                    break
                except asyncio.TimeoutError:
                    updated = worker_pool.get_ticket(ticket.ticket_id)
                    if updated:
                        await ws.send_json({
                            "type": "session.queue_update",
                            "position": updated.position,
                            "estimated_wait_s": updated.estimated_wait_s,
                            "queue_length": worker_pool.queue_length,
                        })
                except asyncio.CancelledError:
                    worker_pool.cancel(ticket.ticket_id)
                    return
        except (WebSocketDisconnect, Exception) as e:
            logger.info(f"Realtime WS disconnected during queue: session={session_id}, cancelling ({e})")
            worker_pool.cancel(ticket.ticket_id)
            return
        if worker is None and future.done():
            worker = future.result()

    if worker is None:
        await ws.send_json({
            "type": "error",
            "error": {"code": "worker_busy", "message": "No worker available", "type": "server_error"},
        })
        await ws.close(code=1013, reason="No worker available")
        return

    await ws.send_json({"type": "session.queue_done"})
    logger.info(f"Realtime WS connected: session={session_id} → {worker.worker_id}")

    worker.mark_busy(GatewayWorkerStatus.DUPLEX_ACTIVE, duplex_type, session_id=session_id)
    task_start = datetime.now()
    session_closed = asyncio.Event()

    # ============ 全双工诊断日志（每个 session 一个文件） ============
    trace_dir = os.path.join(os.path.dirname(__file__), "data", "logs", "duplex_trace")
    os.makedirs(trace_dir, exist_ok=True)
    trace_path = os.path.join(trace_dir, f"{session_id}.jsonl")
    trace_lock = asyncio.Lock()
    last_log_time = {"t": time.time()}

    def _sync_append_local(path: str, line: str) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)

    async def _write_trace(direction: str, raw: str) -> None:
        """记录收发时间戳、数据长度、空包标记"""
        async with trace_lock:
            now = time.time()
            delta_ms = (now - last_log_time["t"]) * 1000
            last_log_time["t"] = now

            try:
                msg = json.loads(raw)
            except Exception:
                msg = {}

            record = {
                "ts": datetime.now().isoformat(timespec="milliseconds"),
                "unix_ts": round(now, 6),
                "direction": direction,
                "session_id": session_id,
                "msg_type": msg.get("type", "unknown"),
                "raw_bytes": len(raw.encode("utf-8")),
                "delta_ms": round(delta_ms, 2),
            }

            if direction == "C->W":
                # 客户端原始消息（OpenAI 格式或原生格式）
                audio_b64 = msg.get("audio") or msg.get("audio_base64") or ""
                frames = msg.get("video_frames") or msg.get("frame_base64_list") or []
                record["audio_len"] = len(audio_b64) if isinstance(audio_b64, str) else 0
                record["video_frame_count"] = len(frames)
                record["video_total_len"] = sum(len(f) for f in frames if isinstance(f, str))
                record["is_empty"] = (record["audio_len"] == 0 and record["video_frame_count"] == 0)
                if msg.get("force_listen"):
                    record["force_listen"] = True
                if msg.get("max_slice_nums"):
                    record["max_slice_nums"] = msg["max_slice_nums"]
            else:
                # Worker 返回（内部协议格式）
                record["text_preview"] = (msg.get("text", "") or "")[:80]
                audio_data = msg.get("audio_data") or ""
                record["audio_len"] = len(audio_data) if isinstance(audio_data, str) else 0
                record["is_listen"] = msg.get("is_listen", False)
                record["end_of_turn"] = msg.get("end_of_turn", False)
                record["kv_cache_length"] = msg.get("kv_cache_length", 0)
                record["is_empty"] = (record["audio_len"] == 0 and not record["text_preview"])

            line = json.dumps(record, ensure_ascii=False) + "\n"
            await asyncio.to_thread(_sync_append_local, trace_path, line)
    # =====================================================================

    worker_ws = None

    try:
        import websockets
        ws_url = f"ws://{worker.host}:{worker.port}/ws/duplex?session_id={session_id}"

        max_retries = 5
        for attempt in range(max_retries):
            try:
                worker_ws = await websockets.connect(ws_url, open_timeout=5)
                break
            except Exception as conn_err:
                if attempt < max_retries - 1:
                    logger.warning(f"Realtime WS connect to {worker.worker_id} failed (attempt {attempt + 1}): {conn_err}")
                    await asyncio.sleep(1.0)
                else:
                    raise

        async def session_timeout_watchdog():
            """Total session duration watchdog."""
            await asyncio.sleep(max_duration_s)
            if session_closed.is_set():
                return
            logger.info(f"Realtime session timeout ({max_duration_s}s): session={session_id}")
            try:
                await ws.send_json({"type": "session.closed", "reason": "timeout"})
            except Exception:
                pass
            session_closed.set()

        async def client_to_worker():
            """Client (OpenAI Realtime) → Worker (old protocol): translate on the fly"""
            try:
                async for raw in ws.iter_text():
                    await _write_trace("C->W", raw)   # ← 埋点：记录客户端原始消息

                    msg = json.loads(raw)
                    msg_type = msg.get("type", "")

                    if msg_type == "session.update":
                        session_cfg = msg.get("session", {})
                        worker_msg = {
                            "type": "prepare",
                            "system_prompt": session_cfg.get("instructions", "You are a helpful assistant."),
                            "deferred_finalize": True,
                            "max_slice_nums": session_cfg.get("max_slice_nums", 1),
                        }
                        if session_cfg.get("ref_audio"):
                            worker_msg["ref_audio_base64"] = session_cfg["ref_audio"]
                        if session_cfg.get("tts_ref_audio"):
                            worker_msg["tts_ref_audio_base64"] = session_cfg["tts_ref_audio"]
                        if session_cfg.get("voice_config"):
                            worker_msg["config"] = session_cfg["voice_config"]
                        await worker_ws.send(json.dumps(worker_msg))

                    elif msg_type == "input_audio_buffer.append":
                        worker_msg = {
                            "type": "audio_chunk",
                            "audio_base64": msg.get("audio", ""),
                        }
                        if msg.get("force_listen"):
                            worker_msg["force_listen"] = True
                        if msg.get("video_frames"):
                            worker_msg["frame_base64_list"] = msg["video_frames"]
                        if msg.get("max_slice_nums"):
                            worker_msg["max_slice_nums"] = msg["max_slice_nums"]
                        await worker_ws.send(json.dumps(worker_msg))

                    elif msg_type == "session.close":
                        await worker_ws.send(json.dumps({"type": "stop"}))

            except WebSocketDisconnect:
                pass

        async def worker_to_client():
            """Worker (old protocol) → Client (OpenAI Realtime): translate on the fly"""
            try:
                async for raw in worker_ws:
                    await _write_trace("W->C", raw)   # ← 埋点：记录 Worker 原始消息

                    msg = json.loads(raw)
                    msg_type = msg.get("type", "")

                    if msg_type == "prepared":
                        await ws.send_json({
                            "type": "session.created",
                            "session_id": session_id,
                            "prompt_length": msg.get("prompt_length", 0),
                        })

                    elif msg_type == "result":
                        kv_len = msg.get("kv_cache_length", 0)
                        if msg.get("is_listen"):
                            await ws.send_json({
                                "type": "response.listen",
                                "kv_cache_length": kv_len,
                            })
                        else:
                            await ws.send_json({
                                "type": "response.output_audio.delta",
                                "text": msg.get("text", ""),
                                "audio": msg.get("audio_data"),
                                "end_of_turn": msg.get("end_of_turn", False),
                                "kv_cache_length": kv_len,
                            })
                        if kv_len >= 8192 and not session_closed.is_set():
                            logger.info(f"Realtime context full (kv={kv_len}): session={session_id}")
                            await ws.send_json({"type": "session.closed", "reason": "context_full"})
                            session_closed.set()

                    elif msg_type == "stopped":
                        await ws.send_json({"type": "session.closed", "reason": "stopped"})

                    elif msg_type == "timeout":
                        await ws.send_json({"type": "session.closed", "reason": "timeout"})

                    elif msg_type == "error":
                        await ws.send_json({
                            "type": "error",
                            "error": {
                                "code": "inference_failed",
                                "message": msg.get("error", "Unknown error"),
                                "type": "server_error",
                            },
                        })

                    else:
                        await ws.send_text(raw)

            except Exception:
                pass

        done, pending = await asyncio.wait(
            [
                asyncio.create_task(client_to_worker()),
                asyncio.create_task(worker_to_client()),
                asyncio.create_task(session_timeout_watchdog()),
            ],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()

    except Exception as e:
        logger.error(f"Realtime WS error: {e}", exc_info=True)
    finally:
        if worker_ws:
            try:
                await worker_ws.close()
            except Exception:
                pass

        if worker:
            duration = (datetime.now() - task_start).total_seconds() if task_start else 0
            worker_pool.release_worker(
                worker,
                request_type=duplex_type,
                duration_s=duration,
            )
            logger.info(f"Realtime WS ended: session={session_id}, Worker released ({duration:.1f}s)")
