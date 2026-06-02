@app.websocket("/ws/duplex/{session_id}")
async def duplex_ws(ws: WebSocket, session_id: str):
    """Duplex WebSocket 代理

    先 accept WS（以便推送排队状态），然后入 FIFO 队列等待 Worker。
    Duplex 独占一个 Worker，直到用户挂断或暂停超时。
    """
    duplex_app = "audio_duplex" if session_id.startswith("adx_") else "omni"
    if not app_registry.is_enabled(duplex_app):
        await ws.close(code=1008, reason=f"{duplex_app} is currently disabled")
        return

    if worker_pool is None:
        await ws.close(code=1013, reason="Service not ready")
        return

    session_id = _sanitize_session_id(session_id)

    # 先 accept，这样排队期间可以推送状态
    await ws.accept()

    # 入队
    try:
        duplex_type = "audio_duplex" if session_id.startswith("adx_") else "omni_duplex"
        ticket, future = worker_pool.enqueue(duplex_type, session_id=session_id)
    except WorkerPool.QueueFullError:
        await ws.send_json({
            "type": "error",
            "error": f"Queue full ({worker_pool.max_queue_size} requests)",
        })
        await ws.close(code=1013, reason="Queue full")
        return

    # 等待 Worker 分配（排队期间检测前端断连，断连时取消 ticket）
    worker: Optional[WorkerConnection] = None
    if future.done():
        worker = future.result()
    else:
        try:
            await ws.send_json({
                "type": "queued",
                "position": ticket.position,
                "estimated_wait_s": ticket.estimated_wait_s,
                "ticket_id": ticket.ticket_id,
                "queue_length": worker_pool.queue_length,
            })
            while not future.done():
                try:
                    worker = await asyncio.wait_for(
                        asyncio.shield(future), timeout=3.0
                    )
                    break
                except asyncio.TimeoutError:
                    updated = worker_pool.get_ticket(ticket.ticket_id)
                    if updated:
                        await ws.send_json({
                            "type": "queue_update",
                            "position": updated.position,
                            "estimated_wait_s": updated.estimated_wait_s,
                            "queue_length": worker_pool.queue_length,
                        })
                except asyncio.CancelledError:
                    worker_pool.cancel(ticket.ticket_id)
                    return
        except (WebSocketDisconnect, Exception) as e:
            logger.info(f"Duplex WS disconnected during queue wait: session={session_id}, cancelling ticket {ticket.ticket_id} ({e})")
            worker_pool.cancel(ticket.ticket_id)
            return
        if worker is None and future.done():
            worker = future.result()

    if worker is None:
        await ws.send_json({"type": "error", "error": "No worker available"})
        await ws.close(code=1013, reason="No worker available")
        return

    # 通知前端排队完成
    await ws.send_json({"type": "queue_done"})
    logger.info(f"Duplex WS connected: session={session_id} → {worker.worker_id}")

    worker.mark_busy(GatewayWorkerStatus.DUPLEX_ACTIVE, duplex_type, session_id=session_id)
    task_start = datetime.now()

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
                audio_b64 = msg.get("audio_base64") or msg.get("audio") or ""
                frames = msg.get("frame_base64_list") or msg.get("video_frames") or []
                record["audio_len"] = len(audio_b64) if isinstance(audio_b64, str) else 0
                record["video_frame_count"] = len(frames)
                record["video_total_len"] = sum(len(f) for f in frames if isinstance(f, str))
                record["is_empty"] = (record["audio_len"] == 0 and record["video_frame_count"] == 0)
                if msg.get("force_listen"):
                    record["force_listen"] = True
                if msg.get("max_slice_nums"):
                    record["max_slice_nums"] = msg["max_slice_nums"]
            else:
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

        # Worker 可能在清理上一个 Duplex session（GPU 显存释放等），
        # 短暂重试确保 Worker 准备就绪
        max_retries = 5
        for attempt in range(max_retries):
            try:
                worker_ws = await websockets.connect(ws_url, open_timeout=5)
                break
            except Exception as conn_err:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Duplex WS connect to {worker.worker_id} failed (attempt {attempt + 1}): "
                        f"{conn_err}, retrying in 1s..."
                    )
                    await asyncio.sleep(1.0)
                else:
                    raise

        diag_log_path = os.path.join("tmp", f"diag_{session_id}.jsonl")

        async def client_to_worker():
            """Client → Worker"""
            try:
                async for raw in ws.iter_text():
                    await _write_trace("C->W", raw)

                    msg = json.loads(raw)

                    if msg.get("type") == "client_diagnostic":
                        await _write_diagnostic(diag_log_path, msg)
                        continue

                    if msg.get("type") == "pause":
                        worker.update_duplex_status(GatewayWorkerStatus.DUPLEX_PAUSED)
                    elif msg.get("type") == "resume":
                        worker.update_duplex_status(GatewayWorkerStatus.DUPLEX_ACTIVE)
                    elif msg.get("type") == "stop":
                        pass

                    await worker_ws.send(raw)
            except WebSocketDisconnect:
                pass

        async def worker_to_client():
            """Worker → Client"""
            try:
                async for raw in worker_ws:
                    await _write_trace("W->C", raw)
                    await ws.send_text(raw)
            except Exception:
                pass

        done, pending = await asyncio.wait(
            [
                asyncio.create_task(client_to_worker()),
                asyncio.create_task(worker_to_client()),
            ],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()

    except Exception as e:
        logger.error(f"Duplex WS error: {e}", exc_info=True)
    finally:
        if worker_ws:
            try:
                await worker_ws.close()
            except Exception:
                pass

        if worker:
            duration = (datetime.now() - task_start).total_seconds() if task_start else 0
            worker_pool.release_worker(worker, request_type=duplex_type, duration_s=duration)
            logger.info(f"Duplex WS ended: session={session_id}, type={duplex_type}, Worker released ({duration:.1f}s)")
