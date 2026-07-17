#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/scripts/resolve_python.sh"

export ARROW_DEFAULT_MEMORY_POOL="${ARROW_DEFAULT_MEMORY_POOL:-system}"

"$PYTHON_BIN" src/generate_synthetic_v2.py
"$PYTHON_BIN" src/build_features.py
"$PYTHON_BIN" src/composite_rule_engine.py
"$PYTHON_BIN" src/model_benchmark.py
"$PYTHON_BIN" src/fusion_rule_engine.py
"$PYTHON_BIN" src/probe_runner.py
"$PYTHON_BIN" src/detect_probe_alerts.py
"$PYTHON_BIN" src/capability_calibration.py
"$PYTHON_BIN" src/model_operations.py
"$PYTHON_BIN" src/model_profile.py
"$PYTHON_BIN" src/model_health_risk.py

"$PYTHON_BIN" scripts/check_deployment.py
echo "Demo data and routing-decision artifacts rebuilt successfully."
