#!/usr/bin/env python3
import os
import sys
import json
import time
import threading
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from flask import Flask, send_from_directory, jsonify, request

def load_config():
    for p in ["./config.json", "../config.json"]:
        path = Path(p).expanduser().resolve()
        if path.exists():
            with open(path) as f:
                return json.load(f)
    for p in ["./lkperf.json", "../lkperf.json", "~/lkperf.json"]:
        path = Path(p).expanduser()
        if path.exists():
            with open(path) as f:
                cfg = json.load(f)
            return {
                "log_dir": str(Path(cfg.get("log_dir", "./logs")).expanduser().resolve()),
                "host": "0.0.0.0", "port": 5000,
                "title": "LKPerf Viewer", "cache_ttl": 60
            }
    return {"log_dir": "./logs", "host": "0.0.0.0", "port": 5000,
            "title": "LKPerf Viewer", "cache_ttl": 60}

CONFIG = load_config()
LOG_DIR = Path(CONFIG["log_dir"]).expanduser().resolve()
HOST = CONFIG.get("host", "0.0.0.0")
PORT = CONFIG.get("port", 5000)
TITLE = CONFIG.get("title", "LKPerf Viewer")

app = Flask(__name__)

class SpanLoader:
    def __init__(self):
        self._cache = {}
        self._lock = threading.Lock()

    def load_day(self, date_str):
        with self._lock:
            if date_str in self._cache:
                return self._cache[date_str]
        spans = []
        day_dir = LOG_DIR / date_str
        if day_dir.exists():
            for jsonl in sorted(day_dir.glob("*.jsonl")):
                try:
                    with open(jsonl, "r", encoding="utf-8") as f:
                        for line in f:
                            if not line.strip():
                                continue
                            spans.append(json.loads(line))
                except Exception:
                    continue
        spans.sort(key=lambda x: x.get("start_us", 0))
        with self._lock:
            self._cache[date_str] = spans
        return spans

    def clear(self):
        with self._lock:
            self._cache.clear()

loader = SpanLoader()

FRONTEND_DIR = Path(__file__).parent / "frontend"

@app.route("/")
def index():
    html_path = FRONTEND_DIR / "index.html"
    if html_path.exists():
        content = html_path.read_text(encoding="utf-8")
        content = content.replace("LKPerf Dashboard", TITLE)
        return content
    return "Frontend not found", 404

@app.route("/js/<path:path>")
def send_js(path):
    return send_from_directory(FRONTEND_DIR / "js", path)

@app.route("/css/<path:path>")
def send_css(path):
    return send_from_directory(FRONTEND_DIR / "css", path)

# ========== 通用过滤（全链路统一 UTC） ==========

def _us_to_minute(start_us):
    if not start_us:
        return 0
    dt = datetime.utcfromtimestamp(start_us / 1e6)
    return dt.hour * 60 + dt.minute

def _node_matches(s, name=None, tag=None, room=None):
    if room and s.get("room") != room:
        return False
    if tag and tag not in s.get("tags", []):
        return False
    if name and name not in s.get("name", ""):
        return False
    return True

def _filter_spans(spans, start_min=0, end_min=1439, room=None, tag=None, name=None):
    filtered = []
    for s in spans:
        minute = _us_to_minute(s.get("start_us", 0))
        if minute < start_min or minute > end_min:
            continue
        if room and s.get("room") != room:
            continue
        if tag and tag not in s.get("tags", []):
            continue
        if name and name not in s.get("name", ""):
            continue
        filtered.append(s)
    return filtered

# ========== API ==========

@app.route("/api/dates")
def api_dates():
    result = []
    if not LOG_DIR.exists():
        return jsonify(result)
    for day_dir in sorted(LOG_DIR.iterdir(), reverse=True):
        if not day_dir.is_dir() or not day_dir.name.isdigit() or len(day_dir.name) != 8:
            continue
        total_lines = 0
        total_size = 0
        for jsonl in day_dir.glob("*.jsonl"):
            try:
                size = jsonl.stat().st_size
                with open(jsonl, "rb") as f:
                    sample = b"".join(f.readline() for _ in range(100))
                    avg = len(sample) / max(1, sample.count(b"\n")) if sample else 200
                    total_lines += int(size / max(avg, 1))
                    total_size += size
            except Exception:
                pass
        if total_lines > 0:
            result.append({
                "date": day_dir.name,
                "total_lines": total_lines,
                "size_mb": round(total_size / 1024 / 1024, 2),
            })
    return jsonify(result)

@app.route("/api/day_hourly")
def api_day_hourly():
    date = request.args.get("date", datetime.utcnow().strftime("%Y%m%d"))
    spans = loader.load_day(date)
    hours = [{"hour": h, "count": 0, "total_ms": 0.0, "names": set()} for h in range(24)]
    for s in spans:
        dt = datetime.utcfromtimestamp(s.get("start_us", 0) / 1e6)
        h = dt.hour
        hours[h]["count"] += 1
        hours[h]["total_ms"] += s.get("ms", 0)
        hours[h]["names"].add(s.get("name", "unknown"))
    for h in hours:
        h["names"] = list(h["names"])[:5]
        h["total_ms"] = round(h["total_ms"], 2)
    return jsonify({"date": date, "hours": hours, "total": len(spans)})

@app.route("/api/filters")
def api_filters():
    date = request.args.get("date")
    start_min = request.args.get("start_min", 0, type=int)
    end_min = request.args.get("end_min", 1439, type=int)
    if not date:
        return jsonify({"tags": [], "rooms": [], "names": []})
    spans = loader.load_day(date)
    filtered = _filter_spans(spans, start_min, end_min)
    tags, rooms, names = set(), set(), set()
    for s in filtered:
        tags.update(s.get("tags", []))
        rooms.add(s.get("room", ""))
        names.add(s.get("name", "unknown"))
    return jsonify({
        "tags": sorted(t for t in tags if t),
        "rooms": sorted(r for r in rooms if r),
        "names": sorted(n for n in names if n),
    })

@app.route("/api/stats")
def api_stats():
    date = request.args.get("date")
    start_min = request.args.get("start_min", 0, type=int)
    end_min = request.args.get("end_min", 1439, type=int)
    room = request.args.get("room", "").strip()
    tag = request.args.get("tag", "").strip()
    name_filter = request.args.get("name", "").strip()
    if not date:
        return jsonify([])
    spans = loader.load_day(date)
    filtered = _filter_spans(spans, start_min, end_min, room or None, tag or None, name_filter or None)
    stats = defaultdict(lambda: {"cnt": 0, "vals": [], "total_ms": 0.0})
    grand = 0.0
    for s in filtered:
        name = s.get("name", "unknown")
        ms = s.get("ms", 0)
        stats[name]["cnt"] += 1
        stats[name]["vals"].append(ms)
        stats[name]["total_ms"] += ms
        grand += ms
    result = []
    for name, v in stats.items():
        vals = sorted(v["vals"])
        n = len(vals)
        result.append({
            "name": name, "cnt": n,
            "avg_ms": round(v["total_ms"] / n, 2) if n else 0,
            "min_ms": round(min(vals), 2) if n else 0,
            "max_ms": round(max(vals), 2) if n else 0,
            "p50": round(vals[int(n*0.5)], 2) if n else 0,
            "p95": round(vals[int(n*0.95)], 2) if n else 0,
            "p99": round(vals[int(n*0.99)], 2) if n else 0,
            "total_ms": round(v["total_ms"], 2),
            "pct": round(v["total_ms"] / grand * 100, 2) if grand else 0,
        })
    result.sort(key=lambda x: x["total_ms"], reverse=True)
    return jsonify(result)

@app.route("/api/raw_spans")
def api_raw_spans():
    date = request.args.get("date")
    offset = request.args.get("offset", 0, type=int)
    limit = request.args.get("limit", 100, type=int)
    name_filter = request.args.get("name", "").strip()
    start_min = request.args.get("start_min", 0, type=int)
    end_min = request.args.get("end_min", 1439, type=int)
    room = request.args.get("room", "").strip()
    tag = request.args.get("tag", "").strip()
    if not date:
        return jsonify({"total": 0, "spans": []})
    spans = loader.load_day(date)
    filtered = _filter_spans(spans, start_min, end_min, room or None, tag or None, name_filter or None)
    total = len(filtered)
    page = filtered[offset:offset + limit]
    return jsonify({
        "total": total,
        "offset": offset,
        "limit": limit,
        "has_more": offset + limit < total,
        "spans": page,
    })

@app.route("/api/traces")
def api_traces():
    date = request.args.get("date")
    start_min = request.args.get("start_min", 0, type=int)
    end_min = request.args.get("end_min", 1439, type=int)
    room = request.args.get("room", "").strip()
    tag = request.args.get("tag", "").strip()
    name_filter = request.args.get("name", "").strip()
    limit = request.args.get("limit", 0, type=int)
    if not date:
        return jsonify([])
    spans = loader.load_day(date)
    filtered = _filter_spans(spans, start_min, end_min)

    children_map = defaultdict(list)
    for s in filtered:
        p = s.get("parent")
        if p:
            children_map[p].append(s)

    traces_in_filtered = {s.get("trace") for s in filtered}
    roots = [s for s in filtered if s.get("parent") is None or s.get("parent") not in traces_in_filtered]

    has_filter = bool(name_filter or tag or room)
    if has_filter:
        def _tree_has_match(node_trace):
            node = next((s for s in filtered if s.get("trace") == node_trace), None)
            if not node:
                return False
            if _node_matches(node, name_filter, tag, room):
                return True
            for child in children_map.get(node_trace, []):
                if _tree_has_match(child.get("trace")):
                    return True
            return False
        for r in roots:
            r["_matches"] = _tree_has_match(r.get("trace"))
        roots.sort(key=lambda x: (not x.get("_matches", False), -x.get("start_us", 0)))
    else:
        roots.sort(key=lambda x: x.get("start_us", 0), reverse=True)

    parent_counts = defaultdict(int)
    for s in filtered:
        p = s.get("parent")
        if p and p in traces_in_filtered:
            parent_counts[p] += 1

    result = []
    for r in (roots[:limit] if limit > 0 else roots):
        tid = r.get("trace")
        result.append({
            "trace": tid,
            "name": r.get("name", "unknown"),
            "ms": round(r.get("ms", 0), 2),
            "start_us": r.get("start_us", 0),
            "uid": r.get("uid", ""),
            "room": r.get("room", ""),
            "tags": r.get("tags", []),
            "children_count": parent_counts.get(tid, 0),
            "matches_filter": r.get("_matches", False) if has_filter else False,
        })
    return jsonify(result)

@app.route("/api/trace_tree")
def api_trace_tree():
    date = request.args.get("date")
    root_trace = request.args.get("root_trace", "").strip()
    room = request.args.get("room", "").strip()
    tag = request.args.get("tag", "").strip()
    name_filter = request.args.get("name", "").strip()
    if not date or not root_trace:
        return jsonify({"root_trace": root_trace, "tree": {}})
    spans = loader.load_day(date)
    root = next((s for s in spans if s.get("trace") == root_trace), None)
    if not root:
        return jsonify({"root_trace": root_trace, "tree": {}})

    children_map = defaultdict(list)
    for s in spans:
        p = s.get("parent")
        if p:
            children_map[p].append(s)

    related = [root]
    queue = [root_trace]
    visited = {root_trace}
    while queue:
        current = queue.pop(0)
        for child in children_map.get(current, []):
            ct = child.get("trace")
            if ct and ct not in visited:
                visited.add(ct)
                related.append(child)
                queue.append(ct)

    local_children_map = defaultdict(list)
    for s in related:
        p = s.get("parent")
        if p:
            local_children_map[p].append(s)

    def build_tree(node):
        tid = node.get("trace")
        children = [build_tree(c) for c in local_children_map.get(tid, [])]
        child_sum = sum(c.get("ms", 0) for c in local_children_map.get(tid, []))
        matches = _node_matches(node, name_filter or None, tag or None, room or None)
        desc_matches = any(c.get("matches_filter") or c.get("descendant_matches") for c in children)
        return {
            "trace": tid,
            "name": node.get("name", "unknown"),
            "ms": round(node.get("ms", 0), 2),
            "start_us": node.get("start_us", 0),
            "uid": node.get("uid", ""),
            "room": node.get("room", ""),
            "tags": node.get("tags", []),
            "children": children,
            "children_ms_sum": round(child_sum, 2),
            "matches_filter": matches,
            "descendant_matches": desc_matches,
        }

    tree = build_tree(root)

    def mark_ascendant(node, parent_in_path=False):
        in_path = parent_in_path or node.get("matches_filter") or node.get("descendant_matches")
        node["ascendant_matches"] = parent_in_path
        node["in_match_path"] = in_path
        for child in node.get("children", []):
            mark_ascendant(child, in_path)

    if name_filter or tag or room:
        mark_ascendant(tree, False)
    else:
        tree["in_match_path"] = True
        def set_all_in_path(n):
            n["in_match_path"] = True
            for c in n.get("children", []):
                set_all_in_path(c)
        set_all_in_path(tree)

    return jsonify({"root_trace": root_trace, "tree": tree})

@app.route("/api/timeline")
def api_timeline():
    date = request.args.get("date")
    start_min = request.args.get("start_min", 0, type=int)
    end_min = request.args.get("end_min", 1439, type=int)
    room = request.args.get("room", "").strip()
    tag = request.args.get("tag", "").strip()
    name_filter = request.args.get("name", "").strip()
    if not date:
        return jsonify([])
    spans = loader.load_day(date)
    filtered = _filter_spans(spans, start_min, end_min, room or None, tag or None, name_filter or None)
    result = []
    for s in filtered:
        result.append({
            "trace": s.get("trace"),
            "parent": s.get("parent"),
            "name": s.get("name", "unknown"),
            "start_us": s.get("start_us", 0),
            "end_us": s.get("end_us", 0),
            "ms": s.get("ms", 0),
            "uid": s.get("uid", ""),
            "room": s.get("room", ""),
            "tags": s.get("tags", []),
        })
    return jsonify(result)

@app.route("/api/tags")
def api_tags():
    spans = loader.load_day(datetime.utcnow().strftime("%Y%m%d"))
    tags = set()
    for s in spans:
        tags.update(s.get("tags", []))
    return jsonify(sorted(tags))

@app.route("/api/names")
def api_names():
    spans = loader.load_day(datetime.utcnow().strftime("%Y%m%d"))
    return jsonify(sorted(set(s.get("name", "unknown") for s in spans)))

if __name__ == "__main__":
    print(f"[Dashboard] log_dir={LOG_DIR}", file=sys.stderr)
    app.run(host=HOST, port=PORT, threaded=True)
