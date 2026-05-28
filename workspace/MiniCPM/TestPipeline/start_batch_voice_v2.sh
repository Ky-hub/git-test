#!/usr/bin/env bash
# ============================================================
# MiniCPM-o 音色克隆批量推理 Gateway v2 启动脚本
# 剧本配置 + 本地目录 + 无状态 Worker
# ============================================================

set -euo pipefail

GATEWAY_PORT="${GATEWAY_PORT:-10024}"
WORKER_ADDRESSES="${WORKER_ADDRESSES:-localhost:22400,localhost:22401}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-300}"
HOST="${HOST:-0.0.0.0}"
PYTHON_CMD="${PYTHON_CMD:-python}"

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
GATEWAY_SCRIPT="${PROJECT_DIR}/gateway_batch_voice_v2.py"
LOG_DIR="${PROJECT_DIR}/logs"
LOG_FILE="${LOG_DIR}/gateway_batch_voice_v2_$(date +%Y%m%d_%H%M%S).log"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}   $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()   { echo -e "${RED}[ERR]${NC}  $*" >&2; }

info "启动 MiniCPM-o Batch Voice Gateway v2 ..."
info "项目目录: ${PROJECT_DIR}"
info "端口: ${GATEWAY_PORT}"
info "Workers: ${WORKER_ADDRESSES}"

if ! command -v "${PYTHON_CMD}" &>/dev/null; then
    err "未找到 Python: ${PYTHON_CMD}"
    exit 1
fi
ok "Python: $(command -v ${PYTHON_CMD}) ($(${PYTHON_CMD} --version))"

if [[ ! -f "${GATEWAY_SCRIPT}" ]]; then
    err "脚本未找到: ${GATEWAY_SCRIPT}"
    exit 1
fi
ok "Gateway: ${GATEWAY_SCRIPT}"

STATIC_PAGE="${PROJECT_DIR}/static/batch_voice_v2.html"
CONFIG_FILE="${PROJECT_DIR}/config/voice_batches.yaml"
if [[ ! -f "${STATIC_PAGE}" ]]; then
    warn "前端页面未找到: ${STATIC_PAGE}"
else
    ok "前端页面: ${STATIC_PAGE}"
fi
if [[ ! -f "${CONFIG_FILE}" ]]; then
    warn "剧本配置未找到: ${CONFIG_FILE} (可手动创建)"
else
    ok "剧本配置: ${CONFIG_FILE}"
fi

mkdir -p "${LOG_DIR}" data/assets/ref_audio config
ok "目录检查完成"

info "启动中..."
info "访问: http://${HOST}:${GATEWAY_PORT}/"

export PYTHONPATH="${PROJECT_DIR}"

nohup "${PYTHON_CMD}" "${GATEWAY_SCRIPT}" \
    --port "${GATEWAY_PORT}" \
    --host "${HOST}" \
    --workers "${WORKER_ADDRESSES}" \
    --timeout "${REQUEST_TIMEOUT}" \
    >> "${LOG_FILE}" 2>&1 &

PID=$!
echo $PID > "${LOG_DIR}/gateway_batch_voice_v2.pid"

sleep 1

if kill -0 "$PID" 2>/dev/null; then
    ok "启动成功! PID: ${PID}"
    info "日志: tail -f ${LOG_FILE}"
    info "停止: ./stop_batch_voice_v2.sh"
else
    err "启动失败: ${LOG_FILE}"
    exit 1
fi
