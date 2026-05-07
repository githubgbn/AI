#!/bin/bash
# 一键运行所有测试

set -e

cd "$(dirname "$0")/.."

echo "================================"
echo "运行所有测试"
echo "================================"

echo ""
echo "[1/3] 语法检查..."
python3 -m py_compile boiling_water_processor.py demo_local.py demo_agent.py
echo "✅ 语法检查通过"

echo ""
echo "[2/3] 单元测试..."
python3 tests/test_unit.py
echo "✅ 单元测试通过"

echo ""
echo "[3/3] 端到端测试..."
python3 tests/test_end_to_end.py
echo "✅ 端到端测试通过"

echo ""
echo "================================"
echo "全部测试通过!"
echo "================================"
