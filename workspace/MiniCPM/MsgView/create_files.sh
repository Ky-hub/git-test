#!/bin/bash
# MsgPack Chat Viewer - 创建项目文件结构（空文件）

set -e

PROJECT_DIR="${1:-msgpack_viewer}"

echo "创建项目目录: $PROJECT_DIR"

mkdir -p "$PROJECT_DIR"
mkdir -p "$PROJECT_DIR/static/css"
mkdir -p "$PROJECT_DIR/static/js"
mkdir -p "$PROJECT_DIR/templates"
mkdir -p "$PROJECT_DIR/msgpack_data"

touch "$PROJECT_DIR/app.py"
touch "$PROJECT_DIR/config.json"
touch "$PROJECT_DIR/README.md"
touch "$PROJECT_DIR/templates/index.html"
touch "$PROJECT_DIR/static/css/style.css"
touch "$PROJECT_DIR/static/js/app.js"

echo "文件创建完成:"
find "$PROJECT_DIR" -type f | sort
echo ""
echo "下一步:"
echo "  1. 编辑 config.json 配置 msgpack_dir"
echo "  2. 把代码写入各文件"
echo "  3. pip install flask msgpack"
echo "  4. python app.py"
