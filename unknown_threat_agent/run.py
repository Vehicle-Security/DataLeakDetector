#!/usr/bin/env python3
"""
启动脚本 - 用于直接运行 Unknown Threat Agent
"""

import sys
from pathlib import Path

# 将父目录添加到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入并运行主函数
from unknown_threat_agent.main import main

if __name__ == "__main__":
    main()
