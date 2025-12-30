# -*- coding: utf-8 -*-
import json
import os
import time
import threading
import traceback
from datetime import datetime
from collections import defaultdict

class ErrorLogger:
    """错误日志记录器"""

    def __init__(self, config):
        self.enabled = config.get("error_handling.enable_error_tracking", True)
        self.error_log_file = config.get("error_handling.error_log_file", "errors.log")
        self.max_size = config.get("error_handling.max_error_log_size", 1048576)
        self.error_counts = defaultdict(int)

    def log_error(self, error_type, message, exception=None):
        """记录错误"""
        if not self.enabled:
            return

        self.error_counts[error_type] += 1
        
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "message": message,
            "count": self.error_counts[error_type]
        }

        if exception:
            error_entry["exception"] = str(exception)
            error_entry["traceback"] = traceback.format_exc()

        try:
            # 检查文件大小并轮转
            if os.path.exists(self.error_log_file):
                if os.path.getsize(self.error_log_file) > self.max_size:
                    backup = f"{self.error_log_file}.{int(time.time())}"
                    os.rename(self.error_log_file, backup)

            with open(self.error_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(error_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[ERROR] 错误日志写入失败: {e}")

    def get_statistics(self):
        """获取错误统计"""
        return dict(self.error_counts)


class LogEngine:
    """日志引擎 - 优化版"""

    def __init__(self, config):
        self.log_dir = "logs"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.max_size = config.get("monitor_settings.max_log_size", 10485760)
        self.show_full_paths = config.get("log_enrichment.show_full_paths", True)
        self.enable_colors = config.get("log_enrichment.enable_console_colors", True)
        self.current_log_file = self._get_log_filename()
        
        self._lock = threading.Lock()

    def _get_log_filename(self):
        """生成基于时间的日志文件名"""
        timestamp = datetime.now().strftime("%Y%m%d_%H")
        return os.path.join(self.log_dir, f"monitor_{timestamp}.json")

    def write(self, data):
        """写入日志 - 线程安全"""
        with self._lock:
            filename = self._get_log_filename()

            try:
                with open(filename, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")

                self._console_print(data)
            except Exception as e:
                print(f"[ERROR] 日志写入失败: {e}")

    def _console_print(self, data):
        """控制台彩色输出 - 优化简洁版"""
        # ANSI 颜色代码
        class Colors:
            RESET = "\033[0m"
            GREEN = "\033[92m"   # Create
            YELLOW = "\033[93m"  # Modify
            RED = "\033[91m"     # Delete
            CYAN = "\033[96m"    # Move/Rename
            BLUE = "\033[94m"    # Info
            GRAY = "\033[90m"    # Details

        event_type = data['event_type']
        
        # 定义图标和颜色
        style_map = {
            "created":  (Colors.GREEN,  "✚"),
            "modified": (Colors.YELLOW, "⚡"),
            "deleted":  (Colors.RED,    "✖"),
            "moved":    (Colors.CYAN,   "➜"),
            "renamed":  (Colors.CYAN,   "➜")
        }
        
        color, icon = style_map.get(event_type, (Colors.RESET, "?"))
        
        # 时间戳 (仅显示时间部分)
        timestamp = data['timestamp'].split('T')[1][:12] if 'T' in data['timestamp'] else data['timestamp']
        
        # 核心信息行
        file_path = data['file_path']
        print(f"{color}{icon} [{timestamp}] {event_type.upper():<8} {Colors.RESET} {file_path}")
        
        # 详细信息行 (组合显示)
        details = []
        
        # 1. 目标路径 (如果是移动/重命名)
        if data.get('destination_path'):
            details.append(f"{Colors.CYAN}➜ {data['destination_path']}{Colors.RESET}")
            
        # 2. 进程信息
        proc_info = data.get('process_info', {})
        proc_name = proc_info.get('process_name')
        if proc_name and proc_name != 'Unknown':
            details.append(f"{Colors.BLUE}Proc:{Colors.RESET} {proc_name}")
            
        # 3. 用户信息
        user_info = data.get('user_info', {})
        username = user_info.get('username')
        if username and username != 'Unknown':
             details.append(f"{Colors.BLUE}User:{Colors.RESET} {username}")

        # 4. 文件大小
        if data.get('file_size') is not None and data['file_size'] > 0:
            size = data['file_size']
            size_str = f"{size/1024/1024:.2f} MB" if size > 1048576 else f"{size/1024:.2f} KB"
            details.append(f"{Colors.GRAY}({size_str}){Colors.RESET}")

        # 打印第二行 (如果有详细信息)
        if details:
            print(f"   {' | '.join(details)}")
            
        # 视觉分隔 (可选，这里用空行代替横线以保持简洁)
        # print(f"{Colors.GRAY}{'-'*50}{Colors.RESET}")
