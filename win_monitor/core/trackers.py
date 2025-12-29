# -*- coding: utf-8 -*-
import time
from datetime import datetime
import psutil
from core.platform import get_platform

class ProcessTracker:
    """进程追踪器 - 获取文件操作的进程信息"""

    def __init__(self, config):
        self.enabled = config.get("advanced.enable_handle_tracking", False)
        self._cache = {}
        self._cache_timeout = config.get("resource_management.handle_cache_timeout", 300)

    def get_process_by_file(self, file_path):
        """通过文件路径获取操作进程（简化版）"""
        # 注意: 完整的文件句柄追踪需要复杂的平台API调用
        # 这里返回当前前台进程作为近似
        try:
            active_info = get_platform().get_active_window()
            if active_info and active_info.get('pid'):
                return self._get_process_info(active_info['pid'])
        except Exception:
            pass
        return None

    def _get_process_info(self, pid):
        """获取进程详细信息"""
        now = time.time()
        
        # 检查缓存
        if pid in self._cache:
            timestamp, info = self._cache[pid]
            if now - timestamp < self._cache_timeout:
                return info

        try:
            proc = psutil.Process(pid)
            info = {
                "pid": pid,
                "name": proc.name(),
                "exe": proc.exe(),
                "cmdline": " ".join(proc.cmdline()) if proc.cmdline() else "",
                "username": proc.username(),
                "create_time": datetime.fromtimestamp(proc.create_time()).isoformat()
            }
            
            # 获取父进程信息
            try:
                parent = proc.parent()
                if parent:
                    info["parent_pid"] = parent.pid
                    info["parent_name"] = parent.name()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

            self._cache[pid] = (now, info)
            return info
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            return None


class WindowSpy:
    """窗口追踪器 """

    def __init__(self, stats_collector=None):
        self._cache = {}
        self._cache_ttl = 1.0
        self.stats = stats_collector

    def get_active_window_info(self):
        """获取当前前台窗口的详细信息"""
        try:
            # 使用平台抽象层获取窗口信息
            active_info = get_platform().get_active_window()
            if not active_info:
                if self.stats:
                    self.stats.record_cache_miss()
                return None

            # 简单的缓存策略
            # 注意：这里我们无法通过句柄缓存，因为Mac平台可能不返回句柄
            # 但我们可以缓存最近一次的结果作为短时间内的快照（如果需要）
            
            # 构建标准返回格式
            info = {
                "window_handle": str(active_info.get("hwnd", 0)),
                "window_title": active_info.get("title", ""),
                "window_class": active_info.get("class", ""),
                "pid": active_info.get("pid", 0),
                "process_name": active_info.get("process", "Unknown"),
                "process_path": "",
                "username": "Unknown",
                "cmdline": ""
            }
            
            # 补充进程详情
            if info["pid"]:
                try:
                    proc = psutil.Process(info["pid"])
                    info["process_path"] = proc.exe()
                    info["username"] = proc.username()
                    info["cmdline"] = " ".join(proc.cmdline())
                except:
                    pass

            return info

        except Exception as e:
            return {"error": str(e)}
