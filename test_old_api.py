#!/usr/bin/env python3
"""Test old pay-as-you-go API key."""

import os
import sys

# Use old pay-as-you-go config
api_key = "sk-1102995c430c46e69dde0bc8ef628c66"
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
model = "qwen-vl-plus"  # qwen3.6-plus equivalent

print(f"Testing old pay-as-you-go API key...")
print(f"Base URL: {base_url}")
print(f"Model: {model}")
print()

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage

    print("Creating ChatOpenAI client...")
    llm = ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        max_retries=2,
        timeout=30,
    )

    print("Sending test request...")
    response = llm.invoke([HumanMessage(content="请用中文回复'测试成功'")])

    print(f"\nSUCCESS! Response: {response.content}")
    print("\n旧的按量付费 API Key 可用！")
    sys.exit(0)

except Exception as e:
    error_msg = str(e)
    print(f"\nERROR: {type(e).__name__}")
    print(f"Details: {error_msg}")

    if "insufficient_quota" in error_msg or "exhausted" in error_msg:
        print("\n结论：旧 API Key 配额也已耗尽")
    elif "invalid" in error_msg.lower() or "authentication" in error_msg.lower():
        print("\n结论：旧 API Key 已失效")
    else:
        print("\n结论：其他错误，可能是网络或配置问题")

    sys.exit(1)
