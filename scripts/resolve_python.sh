#!/usr/bin/env bash

# Resolve the Python 3.11 runtime used by local project scripts.
# This file is sourced by run_dashboard.sh and rebuild_demo.sh.

if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
elif [[ "${CONDA_DEFAULT_ENV:-}" == "ai-monitor" && -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
  PYTHON_BIN="$CONDA_PREFIX/bin/python"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  CONDA_PYTHON="$CONDA_BASE/envs/ai-monitor/bin/python"
  if [[ -x "$CONDA_PYTHON" ]]; then
    PYTHON_BIN="$CONDA_PYTHON"
  fi
fi

if [[ -z "${PYTHON_BIN:-}" ]] && command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
fi

if [[ -z "${PYTHON_BIN:-}" ]] || ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 11))' 2>/dev/null; then
  echo "未找到项目所需的 Python 3.11 环境。" >&2
  echo "请先运行：conda env create -f environment.local.yml" >&2
  echo "或者创建项目虚拟环境：python3.11 -m venv .venv" >&2
  exit 1
fi

if ! "$PYTHON_BIN" -c 'import streamlit' 2>/dev/null; then
  echo "Python 3.11 环境中尚未安装项目依赖。" >&2
  echo "请运行：$PYTHON_BIN -m pip install -r requirements.txt" >&2
  exit 1
fi

export PYTHON_BIN

