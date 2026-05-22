# lkperf.py
"""
LiveKit Agent 专用性能探针（生产完整版）
- 精确时间戳：start_us / end_us 微秒级
- 嵌套识别：ContextVar 调用栈，自动 parent_id
- 线程安全：asyncio / 多线程混合（含线程池 fallback）
- 内存防溢出：deque 滑动窗口
- 后台持久化：按日期分目录，按小时合并 JSONL
- 配置驱动 + 热重载：lkperf.json
- 日志感知：所有 patch/flush 打印到 stderr
- 类批量探针：白名单/黑名单，自动过滤危险方法
- Tags 标签：支持 block/wrap/patch_class 维度标记

环境变量：
    LKPERF_CONFIG=/path/to/lkperf.json

配置文件搜索顺序：
    1. $LKPERF_CONFIG
    2. ./lkperf.json
    3. ~/.lkperf.json

配置示例（lkperf.json）：
{
    "enabled": true,
    "mem_limit": 100000,
    "log_dir": "./logs",
    "flush_sec": 5,
    "check_config_sec": 3,
    "log_level": "INFO",
    "patches": [
        {"module": "my_models", "functions": ["FaceDetector.forward"]}
    ],
    "class_patches": [
        {"module": "my_models", "class": "FaceDetector", "include": ["forward", "preprocess"], "tags": ["model"]}
    ]
}
"""

import os
import sys
import time
import uuid
import asyncio
import functools
import contextvars
import importlib
import json
import threading
from dataclasses import dataclass, field
from collections import defaultdict, deque
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Dict, List, Callable, Any


# ========== 0. 日志感知层 ==========
class __LKPerfLogger:
    LEVELS = {"DEBUG": 0, "INFO": 1, "WARN": 2, "ERROR": 3, "SILENT": 99}

    def __init__(self, level: str = "INFO"):
        self._level = self.LEVELS.get(level, 1)

    def set_level(self, level: str) -> None:
        self._level = self.LEVELS.get(level, 1)

    def _log(self, level: str, msg: str) -> None:
        if self.LEVELS.get(level, 1) < self._level:
            return
        ts = time.strftime("%H:%M:%S")
        print(f"[LKPerf][{ts}][{level}] {msg}", file=sys.stderr, flush=True)

    def debug(self, msg: str) -> None: self._log("DEBUG", msg)
    def info(self, msg: str) -> None: self._log("INFO", msg)
    def warn(self, msg: str) -> None: self._log("WARN", msg)
    def error(self, msg: str) -> None: self._log("ERROR", msg)


_lkp_logger = __LKPerfLogger("INFO")


# ========== 1. 配置管理 + 热重载 ==========
@dataclass
class LKPerfConfig:
    enabled: bool = True
    mem_limit: int = 100000
    log_dir: Optional[str] = None
    flush_sec: int = 5
    check_config_sec: int = 3
    log_level: str = "INFO"
    patches: List[Dict[str, Any]] = field(default_factory=list)
    class_patches: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_file(cls, path: str) -> "LKPerfConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in raw.items() if k in valid_keys})

    def to_summary(self) -> str:
        return (f"enabled={self.enabled}, mem_limit={self.mem_limit}, "
                f"log_dir={self.log_dir}, flush_sec={self.flush_sec}, "
                f"patches={len(self.patches)}, class_patches={len(self.class_patches)}")


class __LKPerfConfigWatcher:
    __slots__ = ("_path", "_config", "_lock", "_last_mtime", "_thread")

    def __init__(self):
        self._path: Optional[str] = None
        self._config: LKPerfConfig = LKPerfConfig()
        self._lock: threading.RLock = threading.RLock()
        self._last_mtime: float = 0.0
        self._thread: Optional[threading.Thread] = None

    def _find_config_path(self) -> Optional[str]:
        candidates = [
            os.getenv("LKPERF_CONFIG"),
            "./lkperf.json",
            os.path.expanduser("~/.lkperf.json"),
        ]
        for c in candidates:
            if c and Path(c).exists():
                return c
        return None

    def load(self) -> LKPerfConfig:
        path = self._find_config_path()
        if path is None:
            _lkp_logger.info("No config file found, using defaults")
            return LKPerfConfig()

        try:
            mtime = Path(path).stat().st_mtime
            cfg = LKPerfConfig.from_file(path)
            self._path = path
            self._last_mtime = mtime
            _lkp_logger.info(f"Config loaded from {path}: {cfg.to_summary()}")
            return cfg
        except Exception as e:
            _lkp_logger.error(f"Failed to load {path}: {e}")
            return self._config

    def start_monitor(self, registry: "__LKPerfRegistry") -> None:
        def watcher() -> None:
            while True:
                time.sleep(self._config.check_config_sec)
                if self._path is None:
                    continue
                try:
                    mtime = Path(self._path).stat().st_mtime
                    if mtime != self._last_mtime:
                        _lkp_logger.info(f"Config change detected: {self._path}")
                        new_cfg = self.load()
                        with self._lock:
                            self._config = new_cfg
                        registry._apply_config(new_cfg)
                        _lkp_apply_patches_from_config(new_cfg)
                        _lkp_apply_class_patches_from_config(new_cfg)
                except Exception as e:
                    _lkp_logger.error(f"Config monitor error: {e}")

        self._thread = threading.Thread(target=watcher, daemon=True)
        self._thread.start()

    @property
    def config(self) -> LKPerfConfig:
        with self._lock:
            return self._config


# ========== 2. 协程级上下文 + 线程 fallback ==========
# FIX 1: 默认值改为 None，避免多线程/多协程共享同一个空列表
_lkp_uid_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("_lkp_uid", default="")
_lkp_room_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("_lkp_room", default="")
_lkp_stack_ctx: contextvars.ContextVar[Optional[List[str]]] = contextvars.ContextVar("_lkp_stack", default=None)

# FIX 2: 线程本地存储，用于线程池场景 fallback
_lkp_thread_stack: threading.local = threading.local()

def _lkp_get_stack() -> List[str]:
    """获取当前调用栈，优先 ContextVar，回退 thread local。"""
    stack = _lkp_stack_ctx.get()
    if stack is not None:
        return stack
    stack = getattr(_lkp_thread_stack, "stack", None)
    if stack is not None:
        return stack
    return []

def _lkp_set_stack(stack: List[str]) -> None:
    """设置当前调用栈，同时更新 ContextVar 和 thread local。"""
    _lkp_stack_ctx.set(stack)
    _lkp_thread_stack.stack = stack


def lkp_bind(room: str, uid: str) -> None:
    _lkp_room_ctx.set(room)
    _lkp_uid_ctx.set(uid)
    _lkp_set_stack([])


# ========== 3. 数据模型（含 tags） ==========
@dataclass
class LKPerfSpan:
    name: str
    duration_ms: float
    start_us: int = 0
    end_us: int = 0
    uid: str = ""
    room: str = ""
    trace_id: str = ""
    parent_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


# ========== 4. 注册中心 ==========
class __LKPerfRegistry:
    __slots__ = (
        "_enabled", "_spans", "_lock",
        "_log_dir", "_flush_sec", "_flush_buf", "_flush_lock", "_flush_thread",
        "_config",
    )

    def __init__(self, config: LKPerfConfig):
        self._apply_config(config, init=True)

    def _apply_config(self, cfg: LKPerfConfig, init: bool = False) -> None:
        self._enabled = cfg.enabled
        _lkp_logger.set_level(cfg.log_level)

        mem_limit = cfg.mem_limit
        if init:
            self._spans: deque[LKPerfSpan] = deque(maxlen=mem_limit)
            self._lock: threading.RLock = threading.RLock()
        else:
            with self._lock:
                old = list(self._spans)
                self._spans = deque(old, maxlen=mem_limit)

        new_log_dir = Path(cfg.log_dir) if cfg.log_dir else None
        if init:
            self._log_dir: Optional[Path] = new_log_dir
            self._flush_sec: int = cfg.flush_sec
            self._flush_buf: List[Dict[str, Any]] = []
            self._flush_lock: threading.Lock = threading.Lock()
            self._flush_thread: Optional[threading.Thread] = None
            if self._log_dir:
                self._log_dir.mkdir(parents=True, exist_ok=True)
                self._start_flush_daemon()
        else:
            self._flush_sec = cfg.flush_sec
            if new_log_dir and (not self._log_dir or self._log_dir != new_log_dir):
                _lkp_logger.warn("log_dir change requires process restart to take full effect")

        self._config = cfg
        _lkp_logger.info(f"Registry config applied: {cfg.to_summary()}")

    def _record(self, name: str, duration_sec: float,
                start_us: int,
                trace_id: Optional[str] = None,
                parent_id: Optional[str] = None,
                tags: Optional[List[str]] = None,
                **meta) -> None:
        if not self._enabled:
            return

        span = LKPerfSpan(
            name=name,
            duration_ms=duration_sec * 1000.0,
            start_us=start_us,
            end_us=int(time.time() * 1_000_000),
            uid=_lkp_uid_ctx.get(),
            room=_lkp_room_ctx.get(),
            trace_id=trace_id or str(uuid.uuid4())[:8],
            parent_id=parent_id,
            tags=tags or [],
            meta=meta,
        )

        with self._lock:
            self._spans.append(span)

        if self._log_dir:
            with self._flush_lock:
                self._flush_buf.append({
                    "ts": time.time(),
                    "start_us": span.start_us,
                    "end_us": span.end_us,
                    "name": span.name,
                    "ms": span.duration_ms,
                    "uid": span.uid,
                    "room": span.room,
                    "trace": span.trace_id,
                    "parent": span.parent_id,
                    "tags": span.tags,
                    "meta": span.meta,
                })

    @contextmanager
    def _block(self, name: str, tags: Optional[List[str]] = None, **meta):
        if not self._enabled:
            yield
            return

        # FIX 3: 使用统一的栈获取接口
        stack = _lkp_get_stack()

        parent_id = stack[-1] if stack else None
        trace_id = str(uuid.uuid4())[:8]
        _lkp_set_stack(stack + [trace_id])

        t0 = time.perf_counter()
        start_us = int(time.time() * 1_000_000)
        try:
            yield
        finally:
            _lkp_set_stack(stack)
            self._record(name, time.perf_counter() - t0,
                         start_us=start_us,
                         trace_id=trace_id,
                         parent_id=parent_id,
                         tags=tags,
                         **meta)

    def _snapshot(self) -> List[LKPerfSpan]:
        with self._lock:
            return list(self._spans)

    def _report(self, uid: Optional[str], room: Optional[str],
                tag: Optional[str], top_n: int) -> str:
        pool = [
            s for s in self._snapshot()
            if (uid is None or s.uid == uid)
            and (room is None or s.room == room)
        ]
        if tag:
            pool = [s for s in pool if tag in s.tags]

        if not pool:
            return f"[LKPerf] No data for uid={uid or 'ALL'} room={room or 'ALL'} tag={tag or 'ALL'}"

        groups: Dict[str, List[float]] = defaultdict(list)
        for s in pool:
            groups[s.name].append(s.duration_ms)

        lines = [
            f"\n=== LKPerf Report [uid={uid or 'ALL'} room={room or 'ALL'} tag={tag or 'ALL'}] ===",
            f"Total spans: {len(pool)}",
        ]
        stats = []
        for name, vals in groups.items():
            avg = sum(vals) / len(vals)
            p99 = sorted(vals)[int(len(vals) * 0.99)] if len(vals) > 1 else vals[0]
            stats.append((name, len(vals), avg, max(vals), p99))
        stats.sort(key=lambda x: x[2], reverse=True)

        for name, cnt, avg, mx, p99 in stats[:top_n]:
            lines.append(
                f"{name:35s} cnt={cnt:>3} avg={avg:>7.2f}ms "
                f"p99={p99:>7.2f}ms max={mx:>7.2f}ms"
            )
        return "\n".join(lines)

    def _summary_by_user(self) -> List[Dict[str, Any]]:
        users = defaultdict(lambda: {"frames": 0, "total_ms": 0.0, "max_ms": 0.0})
        for s in self._snapshot():
            if "frame" in s.name:
                key = (s.room, s.uid)
                users[key]["frames"] += 1
                users[key]["total_ms"] += s.duration_ms
                users[key]["max_ms"] = max(users[key]["max_ms"], s.duration_ms)

        return [
            {
                "room": k[0],
                "uid": k[1],
                "frames": v["frames"],
                "avg_ms": round(v["total_ms"] / v["frames"], 2) if v["frames"] else 0,
                "max_ms": round(v["max_ms"], 2),
            }
            for k, v in sorted(users.items(), key=lambda x: x[1]["total_ms"], reverse=True)
        ]

    def _tree(self, uid: Optional[str] = None, room: Optional[str] = None,
              last_n_frames: int = 1) -> str:
        pool = [
            s for s in self._snapshot()
            if (uid is None or s.uid == uid) and (room is None or s.room == room)
        ]
        if not pool:
            return "[LKPerf] No data for tree"

        pool.reverse()
        # FIX 4: 放宽根节点选择，先尝试 "frame" 名称，否则回退到所有 parent_id is None
        roots = [s for s in pool if s.parent_id is None and "frame" in s.name]
        if not roots:
            roots = [s for s in pool if s.parent_id is None]
        if not roots:
            # 如果连 parent_id is None 的都没有，说明所有 span 都是孤儿节点
            # 使用最早的几个 span 作为根节点
            roots = pool[:last_n_frames] if pool else []

        children_map: Dict[str, List[LKPerfSpan]] = defaultdict(list)
        for s in pool:
            if s.parent_id:
                children_map[s.parent_id].append(s)

        lines: List[str] = []

        def dfs(node: LKPerfSpan, depth: int = 0) -> None:
            indent = "│   " * (depth - 1) + ("├── " if depth > 0 else "")
            tag_str = f" tags={node.tags}" if node.tags else ""
            lines.append(
                f"{indent}{node.name} ({node.duration_ms:.2f}ms){tag_str} "
                f"[trace:{node.trace_id}]"
            )
            for child in children_map.get(node.trace_id, []):
                dfs(child, depth + 1)

        for root in roots[:last_n_frames]:
            dfs(root)
            lines.append("")

        return "\n".join(lines)

    def _reset(self) -> None:
        with self._lock:
            self._spans.clear()
        _lkp_logger.info("Memory spans cleared")

    def _dump(self, path: str) -> None:
        snapshot = self._snapshot()
        payload = [
            {
                "name": s.name,
                "ms": s.duration_ms,
                "start_us": s.start_us,
                "end_us": s.end_us,
                "uid": s.uid,
                "room": s.room,
                "trace": s.trace_id,
                "parent": s.parent_id,
                "tags": s.tags,
                "meta": s.meta,
            }
            for s in snapshot
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _lkp_logger.info(f"Dumped {len(snapshot)} spans to {path}")

    def _start_flush_daemon(self) -> None:
        def worker() -> None:
            while True:
                sleep_sec = self._flush_sec
                time.sleep(sleep_sec)

                batch: List[Dict[str, Any]] = []
                with self._flush_lock:
                    if self._flush_buf:
                        batch = self._flush_buf.copy()
                        self._flush_buf.clear()

                if not batch:
                    continue

                day_dir = self._log_dir / time.strftime('%Y%m%d')
                day_dir.mkdir(exist_ok=True)
                fname = day_dir / f"lkperf_{time.strftime('%H')}.jsonl"

                try:
                    with open(fname, "a", encoding="utf-8") as f:
                        for item in batch:
                            f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    _lkp_logger.info(f"Flushed {len(batch)} spans -> {fname}")
                except Exception as e:
                    _lkp_logger.error(f"Flush failed: {e}")

        self._flush_thread = threading.Thread(target=worker, daemon=True)
        self._flush_thread.start()


# ========== 5. 包装器（支持 tags + 调用栈修复） ==========
class __LKPerfWrapper:
    __slots__ = ()

    @staticmethod
    def _make_timer(func: Callable, name: Optional[str] = None,
                    tags: Optional[List[str]] = None) -> Callable:
        label = name or f"{func.__module__}.{func.__qualname__}"

        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def _async_wrapper(*args: Any, **kwargs: Any):
                # FIX 5: 接入调用栈，正确设置 parent_id / trace_id
                stack = _lkp_get_stack()
                parent_id = stack[-1] if stack else None
                trace_id = str(uuid.uuid4())[:8]
                _lkp_set_stack(stack + [trace_id])

                t0 = time.perf_counter()
                start_us = int(time.time() * 1_000_000)
                try:
                    return await func(*args, **kwargs)
                finally:
                    _lkp_set_stack(stack)
                    _lkp_reg._record(
                        label, time.perf_counter() - t0,
                        start_us=start_us,
                        trace_id=trace_id,
                        parent_id=parent_id,
                        tags=tags,
                    )
            return _async_wrapper

        @functools.wraps(func)
        def _sync_wrapper(*args: Any, **kwargs: Any):
            # FIX 6: 接入调用栈，正确设置 parent_id / trace_id
            stack = _lkp_get_stack()
            parent_id = stack[-1] if stack else None
            trace_id = str(uuid.uuid4())[:8]
            _lkp_set_stack(stack + [trace_id])

            t0 = time.perf_counter()
            start_us = int(time.time() * 1_000_000)
            try:
                return func(*args, **kwargs)
            finally:
                _lkp_set_stack(stack)
                _lkp_reg._record(
                    label, time.perf_counter() - t0,
                    start_us=start_us,
                    trace_id=trace_id,
                    parent_id=parent_id,
                    tags=tags,
                )
        return _sync_wrapper


# ========== 6. 全局初始化 ==========
_lkp_watcher = __LKPerfConfigWatcher()
_lkp_cfg = _lkp_watcher.load()
_lkp_reg = __LKPerfRegistry(_lkp_cfg)
_lkp_watcher.start_monitor(_lkp_reg)


# ========== 7. Patch 逻辑 ==========
def _lkp_apply_patches_from_config(cfg: LKPerfConfig) -> None:
    for item in cfg.patches:
        mod = item.get("module")
        funcs = item.get("functions", [])
        if mod and funcs:
            lkp_patch(mod, funcs, quiet=False)


def _lkp_apply_class_patches_from_config(cfg: LKPerfConfig) -> None:
    for item in cfg.class_patches:
        mod = item.get("module")
        cls = item.get("class")
        inc = item.get("include")
        exc = item.get("exclude")
        tags = item.get("tags")
        if mod and cls:
            lkp_patch_class(mod, cls, include=inc, exclude=exc, tags=tags, quiet=False)


def lkp_patch(module_path: str, func_names: List[str], *, quiet: bool = True) -> None:
    try:
        mod = importlib.import_module(module_path)
    except Exception as e:
        if not quiet:
            _lkp_logger.error(f"Skip {module_path}: {e}")
        return

    patched_count = 0
    for fn in func_names:
        if not hasattr(mod, fn):
            if not quiet:
                _lkp_logger.warn(f"{module_path}.{fn} not found")
            continue
        orig = getattr(mod, fn)
        if getattr(orig, "__lkp_probed__", False):
            continue
        wrapped = __LKPerfWrapper._make_timer(orig, name=f"{module_path}.{fn}")
        wrapped.__lkp_probed__ = True  # type: ignore
        setattr(mod, fn, wrapped)
        patched_count += 1
        if not quiet:
            _lkp_logger.info(f"Patched {module_path}.{fn}")

    if not quiet and patched_count == 0:
        _lkp_logger.warn(f"No new functions patched in {module_path}")


# 危险方法黑名单
_LKP_UNSAFE_NAMES = {
    "__init__", "__new__", "__del__",
    "__getattribute__", "__setattr__", "__delattr__",
    "__enter__", "__exit__", "__aenter__", "__aexit__",
    "__repr__", "__str__", "__format__",
    "__call__", "__class_getitem__", "__mro_entries__",
    "__get__", "__set__", "__delete__",
    "__len__", "__getitem__", "__setitem__", "__delitem__",
    "__iter__", "__next__", "__aiter__", "__anext__",
    "__await__",
    "__hash__", "__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__",
    "__bool__", "__sizeof__",
    "__getstate__", "__setstate__", "__reduce__", "__reduce_ex__",
    "__subclasshook__", "__instancecheck__", "__subclasscheck__",
}


def lkp_patch_class(
    module_path: str,
    class_name: str,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    quiet: bool = True
) -> None:
    try:
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
    except Exception as e:
        if not quiet:
            _lkp_logger.error(f"Skip {module_path}.{class_name}: {e}")
        return

    exclude_set = set(exclude or [])
    exclude_set.update(_LKP_UNSAFE_NAMES)

    candidates = include if include else list(cls.__dict__.keys())
    patched_count = 0
    skipped: List[tuple] = []

    for name in candidates:
        if name in exclude_set:
            continue

        member = cls.__dict__.get(name)
        if member is None:
            continue
        if isinstance(member, property):
            continue

        is_static = isinstance(member, staticmethod)
        is_class = isinstance(member, classmethod)

        try:
            if is_static or is_class:
                orig_func = member.__func__
                if not callable(orig_func) or getattr(orig_func, "__lkp_probed__", False):
                    continue
                wrapped = __LKPerfWrapper._make_timer(
                    orig_func,
                    name=f"{module_path}.{class_name}.{name}",
                    tags=tags
                )
                wrapped.__lkp_probed__ = True  # type: ignore
                if is_static:
                    setattr(cls, name, staticmethod(wrapped))
                else:
                    setattr(cls, name, classmethod(wrapped))
            else:
                if not callable(member) or getattr(member, "__lkp_probed__", False):
                    continue
                wrapped = __LKPerfWrapper._make_timer(
                    member,
                    name=f"{module_path}.{class_name}.{name}",
                    tags=tags
                )
                wrapped.__lkp_probed__ = True  # type: ignore
                setattr(cls, name, wrapped)

            patched_count += 1
            if not quiet:
                _lkp_logger.info(f"Patched {module_path}.{class_name}.{name}")
        except Exception as e:
            skipped.append((name, str(e)))
            _lkp_logger.warn(f"Failed to patch {name}: {e}")

    if not quiet:
        _lkp_logger.info(
            f"Patched {patched_count} methods in {module_path}.{class_name}, "
            f"skipped {len(skipped)}"
        )
        if skipped:
            _lkp_logger.info(f"Skipped details: {skipped}")


# 启动时自动应用配置
_lkp_apply_patches_from_config(_lkp_cfg)
_lkp_apply_class_patches_from_config(_lkp_cfg)


# ========== 8. 公共 API ==========

@contextmanager
def lkp_block(name: str, *, tags: Optional[List[str]] = None, **meta):
    with _lkp_reg._block(name, tags=tags, **meta):
        yield


def lkp_wrap(func: Optional[Callable] = None, *, tags: Optional[List[str]] = None):
    if func is None:
        return functools.partial(lkp_wrap, tags=tags)
    return __LKPerfWrapper._make_timer(func, tags=tags)


def lkp_report(uid: Optional[str] = None, room: Optional[str] = None,
               tag: Optional[str] = None, top_n: int = 20) -> str:
    return _lkp_reg._report(uid, room, tag, top_n)


def lkp_tree(uid: Optional[str] = None, room: Optional[str] = None,
             last_n_frames: int = 1) -> str:
    return _lkp_reg._tree(uid, room, last_n_frames)


def lkp_summary() -> List[Dict[str, Any]]:
    return _lkp_reg._summary_by_user()


def lkp_reset() -> None:
    _lkp_reg._reset()


def lkp_enable(yes: bool = True) -> None:
    _lkp_reg._enabled = yes
    _lkp_logger.info(f"Probe enabled={yes}")


def lkp_dump(path: str) -> None:
    _lkp_reg._dump(path)


def lkp_reload(config_path: Optional[str] = None) -> None:
    if config_path:
        cfg = LKPerfConfig.from_file(config_path)
        _lkp_watcher._config = cfg
    else:
        cfg = _lkp_watcher.load()
    _lkp_reg._apply_config(cfg)
    _lkp_apply_patches_from_config(cfg)
    _lkp_apply_class_patches_from_config(cfg)
    _lkp_logger.info("Manual reload completed")


# ========== 9. 线程池辅助函数 ==========
def lkp_run_in_executor(executor, fn: Callable, *args):
    """
    在线程池中执行任务，自动传递 LKPerf 调用栈上下文。
    用法: result = await lkp_run_in_executor(executor, cpu_bound_func, arg1, arg2)
    """
    import asyncio
    loop = asyncio.get_running_loop()
    # 捕获当前上下文
    stack = _lkp_get_stack()
    room = _lkp_room_ctx.get()
    uid = _lkp_uid_ctx.get()

    def _wrapper(*a):
        # 在新线程中恢复上下文
        _lkp_room_ctx.set(room)
        _lkp_uid_ctx.set(uid)
        _lkp_set_stack(list(stack))  # 复制一份，避免共享引用
        return fn(*a)

    return loop.run_in_executor(executor, functools.partial(_wrapper, *args))


__all__ = [
    "lkp_bind",
    "lkp_block",
    "lkp_wrap",
    "lkp_patch",
    "lkp_patch_class",
    "lkp_report",
    "lkp_tree",
    "lkp_summary",
    "lkp_reset",
    "lkp_enable",
    "lkp_dump",
    "lkp_reload",
    "lkp_run_in_executor",
    "LKPerfConfig",
    "LKPerfSpan",
]
