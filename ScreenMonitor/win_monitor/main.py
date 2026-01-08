# -*- coding: utf-8 -*-
"""
main.py - Win Monitor 入口点

启动监控服务的简洁入口
"""

import os
import sys
import signal

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.config_loader import ConfigLoader
from services.recorder_service import RecorderService
from core.monitors.engine import Engine


def main():
    """主函数"""
    print("=" * 50)
    print("  Win Monitor - 屏幕监控服务")
    print("  架构: Sensor（传感器模式）")
    print("=" * 50)
    print()
    
    # 1. 加载配置
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    config = ConfigLoader(config_path)
    
    # 2. 创建录制服务
    recorder = RecorderService(fps=10)
    
    # 3. 创建引擎
    output_dir = os.path.join(os.path.dirname(__file__), "recordings")
    engine = Engine(
        config_loader=config,
        recorder_service=recorder,
        output_dir=output_dir
    )
    
    # 4. 设置信号处理（优雅退出）
    def signal_handler(signum, frame):
        print("\n\n⚠️ 收到退出信号，正在停止...")
        engine.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 5. 启动引擎
    engine.start()
    
    print()
    print("📍 监控已启动，按 Ctrl+C 停止")
    print()
    
    # 6. 保持运行
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断，正在停止...")
        engine.stop()


if __name__ == "__main__":
    main()