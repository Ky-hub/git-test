#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MsgPack Web Viewer"""

import json
import os
import traceback
from pathlib import Path

import msgpack
from flask import Flask, jsonify, render_template, request, send_file

app = Flask(__name__, static_folder="static", static_url_path="/static")

# ========== 加载配置 ==========
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


def load_config():
    if not os.path.exists(CONFIG_PATH):
        default = {
            "msgpack_dir": "./msgpack_data",
            "host": "0.0.0.0",
            "port": 5000,
            "debug": True
        }
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(default, f, indent=2)
        print(f"[INFO] 创建默认配置: {CONFIG_PATH}")
        return default
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


CFG = load_config()

# 解析 msgpack_dir：优先绝对路径，相对路径基于项目根目录
_raw_dir = CFG.get("msgpack_dir", "./msgpack_data")
if os.path.isabs(_raw_dir):
    BASE_DIR = Path(_raw_dir).resolve()
else:
    # 相对路径基于项目根目录（app.py 所在目录）
    BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / _raw_dir
    BASE_DIR = BASE_DIR.resolve()

print(f"[INFO] msgpack 数据目录: {BASE_DIR}")


def ensure_dir():
    if not BASE_DIR.exists():
        BASE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] 创建目录: {BASE_DIR}")


# ========== 工具函数 ==========

def list_data_files(root: Path):
    files = []
    if not root.exists():
        return files
    for ext in ("*.msgpack", "*.json"):
        for p in root.rglob(ext):
            rel = p.relative_to(root).as_posix()
            files.append({
                "path": rel,
                "name": p.name,
                "size": p.stat().st_size,
                "mtime": p.stat().st_mtime,
            })
    files.sort(key=lambda x: x["path"])
    return files


def decode_bytes(obj):
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8")
        except UnicodeDecodeError:
            return {
                "__bytes__": True,
                "length": len(obj),
                "hex": obj[:64].hex() + ("..." if len(obj) > 64 else "")
            }
    elif isinstance(obj, dict):
        return {k: decode_bytes(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [decode_bytes(v) for v in obj]
    elif isinstance(obj, tuple):
        return [decode_bytes(v) for v in obj]
    return obj


def unpack_data(path: Path):
    suffix = path.suffix.lower()

    # JSON 文件
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return decode_bytes(data)

    # msgpack 文件
    with open(path, "rb") as f:
        raw = f.read()

    # 尝试单条
    try:
        data = msgpack.unpackb(raw, strict_map_key=False)
        return decode_bytes(data)
    except Exception:
        pass

    # 尝试流式
    results = []
    try:
        unpacker = msgpack.Unpacker(raw_bytes=raw, strict_map_key=False)
        for item in unpacker:
            results.append(decode_bytes(item))
    except Exception as e:
        return {"__error__": True, "message": str(e), "raw_hex_preview": raw[:128].hex()}

    if len(results) == 1:
        return results[0]
    return {"__multi__": True, "count": len(results), "items": results}


# ========== 路由 ==========

@app.route("/")
def index():
    return render_template("index.html", base_dir=str(BASE_DIR))


@app.route("/api/health")
def api_health():
    ensure_dir()
    files = list_data_files(BASE_DIR)
    return jsonify({
        "status": "ok",
        "base_dir": str(BASE_DIR),
        "base_dir_exists": BASE_DIR.exists(),
        "file_count": len(files),
    })


@app.route("/api/files")
def api_files():
    try:
        ensure_dir()
        files = list_data_files(BASE_DIR)
        return jsonify({"base_dir": str(BASE_DIR), "files": files})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/view")
def api_view():
    rel = request.args.get("path", "").strip()
    if not rel:
        return jsonify({"error": "缺少 path 参数"}), 400

    target = (BASE_DIR / rel).resolve()
    if not str(target).startswith(str(BASE_DIR)):
        return jsonify({"error": "非法路径"}), 403
    if not target.exists():
        return jsonify({"error": "文件不存在"}), 404
    if not target.is_file():
        return jsonify({"error": "不是文件"}), 400

    try:
        data = unpack_data(target)
        return jsonify({
            "path": rel,
            "name": target.name,
            "size": target.stat().st_size,
            "data": data,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"解析失败: {str(e)}", "traceback": traceback.format_exc()}), 500


@app.route("/api/download")
def api_download():
    rel = request.args.get("path", "").strip()
    if not rel:
        return jsonify({"error": "缺少 path 参数"}), 400
    target = (BASE_DIR / rel).resolve()
    if not str(target).startswith(str(BASE_DIR)):
        return jsonify({"error": "非法路径"}), 403
    if not target.exists():
        return jsonify({"error": "文件不存在"}), 404
    return send_file(target, as_attachment=True, download_name=target.name)


# ========== 启动 ==========

if __name__ == "__main__":
    ensure_dir()
    host = CFG.get("host", "0.0.0.0")
    port = CFG.get("port", 5000)
    debug = CFG.get("debug", True)
    print(f"=" * 50)
    print(f"[MsgPack Viewer] 启动中...")
    print(f"  访问地址: http://{host}:{port}/")
    print(f"  数据目录: {BASE_DIR}")
    print(f"  健康检查: http://{host}:{port}/api/health")
    print(f"  把 .msgpack 文件放到数据目录即可浏览")
    print(f"=" * 50)
    app.run(host=host, port=port, debug=debug)
