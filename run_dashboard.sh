#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/scripts/resolve_python.sh"

# PyArrow 25 在部分 Intel macOS 环境中使用 mimalloc 处理字符串去重时可能
# 触发原生层 SIGSEGV。必须在 Streamlit/PyArrow 导入前切换到系统内存池。
export ARROW_DEFAULT_MEMORY_POOL="${ARROW_DEFAULT_MEMORY_POOL:-system}"

exec "$PYTHON_BIN" -m streamlit run dashboard/app.py "$@"
