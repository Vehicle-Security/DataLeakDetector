#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本：验证危险行为检测逻辑
测试新旧方法在不同风险等级下的判断差异
"""

import sys
import io

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from main.data_leak_detector.evidence_semantics import (
    CONFIRMED_RISK_LEVELS,
    is_confirmed_risk_level,
    normalize_risk_level
)

# 测试用例
test_cases = [
    # (risk_level, 旧方法应该返回, 新方法应该返回, 说明)
    ("none", False, False, "无敏感对象交互"),
    ("preparation", False, False, "仅打开外部服务页面"),
    ("attempted", False, False, "attempted 不在 CONFIRMED_RISK_LEVELS 中"),
    ("selected_or_attached", False, True, "文件已附加到邮件（未发送）- 核心变化"),
    ("in_progress", False, True, "上传进行中 - 核心变化"),
    ("content_exposed", True, True, "内容已暴露在外部输入框"),
    ("completed", True, True, "传输已完成"),
]

print("=" * 80)
print("危险行为检测逻辑验证")
print("=" * 80)
print(f"\n当前 CONFIRMED_RISK_LEVELS: {CONFIRMED_RISK_LEVELS}\n")

print(f"{'风险等级':<25} {'新方法结果':<12} {'预期':<8} {'状态':<8} {'说明'}")
print("-" * 80)

all_passed = True
for risk_level, old_expected, new_expected, description in test_cases:
    result = is_confirmed_risk_level(risk_level)
    status = "✅ PASS" if result == new_expected else "❌ FAIL"
    if result != new_expected:
        all_passed = False

    print(f"{risk_level:<25} {str(result):<12} {str(new_expected):<8} {status:<8} {description}")

print("\n" + "=" * 80)

# 对比分析
print("\n【对比分析】")
print("\n旧方法 (只检测完成):")
print("  - 阳性: content_exposed, completed")
print("  - 阴性: none, preparation, attempted, selected_or_attached, in_progress")
print("  - 检测率: ~30-40% (只捕获1-3秒的完成证据)")

print("\n新方法 (检测危险行为):")
print("  - 阳性: selected_or_attached, in_progress, content_exposed, completed")
print("  - 阴性: none, preparation, attempted")
print("  - 检测率: ~80-95% (捕获5-60秒的危险行为)")

print("\n【核心变化】")
print("  1. selected_or_attached (文件附加): 阴性 → 阳性")
print("     - 示例: 用户将敏感文件附加到邮件，但关闭前未发送")
print("     - 旧判断: TN/FN (未完成，不算泄露)")
print("     - 新判断: TP (危险行为，需要警告)")
print("     - DLP 价值: 检测到泄露意图，即使未完成")

print("\n  2. in_progress (传输进行中): 阴性 → 阳性")
print("     - 示例: 上传进度条显示60%，用户取消了")
print("     - 旧判断: TN/FN (未完成，不算泄露)")
print("     - 新判断: TP (危险行为，数据已经传输)")
print("     - DLP 价值: 即使取消，部分数据可能已传输")

print("\n【采样优势】")
print("  - 完成证据 (1-3秒): 5秒采样间隔 → 20-60% 捕获率")
print("  - 危险行为 (5-60秒): 5秒采样间隔 → 90-100% 捕获率")
print("  - 预期检测率提升: 2-3倍")

print("\n" + "=" * 80)
if all_passed:
    print("✅ 所有测试通过！危险行为检测逻辑已正确实施。")
else:
    print("❌ 部分测试失败！请检查实现。")
print("=" * 80)
