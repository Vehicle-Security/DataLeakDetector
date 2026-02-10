# -*- coding: utf-8 -*-
"""
sensor.py - 窗口/进程传感器
职责：获取原始数据（进程名、窗口标题）
不做任何判断逻辑，只负责采集

对应架构角色：Sensor（传感器）

注意：此模块已合并原 trackers.py 中的 ProcessTracker 和 WindowSpy
"""

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any

try:
    import win32gui
    import win32process
    import psutil
except ImportError as e:
    print(f"[ERROR] 缺少必要依赖: {e}")
    print("请运行: pip install pywin32 psutil")
    raise


@dataclass
class WindowData:
    """窗口数据 - 传感器输出的原始数据"""
    window_handle: int
    window_title: str
    window_class: str
    process_id: int
    process_name: str
    process_path: str


class Sensor:
    """
    传感器 - 获取当前活动窗口和进程信息
    
    只负责采集数据，不做任何规则判断
    包含进程信息缓存以降低 CPU 占用
    """
    
    def __init__(self):
        self._process_cache: Dict[int, tuple] = {}
        self._cache_ttl = 5.0  # 缓存 5 秒
    
    def get_active_window(self) -> Optional[WindowData]:
        """
        获取当前活动窗口信息
        
        优化：只查询前台窗口的 PID，不遍历所有进程
        使用缓存避免频繁创建 psutil.Process
        
        Returns:
            WindowData 或 None（如果获取失败）
        """
        try:
            # 获取前台窗口句柄
            hwnd = win32gui.GetForegroundWindow()
            if not hwnd:
                return None
            
            # 获取窗口信息
            window_title = win32gui.GetWindowText(hwnd)
            window_class = win32gui.GetClassName(hwnd)
            
            # 获取窗口所属进程的 PID
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            
            # 从缓存获取进程信息
            process_name = ""
            process_path = ""
            
            info = self._get_process_info_with_cache(pid)
            if info:
                process_name = info.get("name", "")
                process_path = info.get("exe", "")
            
            return WindowData(
                window_handle=hwnd,
                window_title=window_title,
                window_class=window_class,
                process_id=pid,
                process_name=process_name,
                process_path=process_path
            )
        
        except Exception:
            return None
    
    def get_process_name(self, pid: int) -> str:
        """根据 PID 获取进程名（带缓存）"""
        info = self._get_process_info_with_cache(pid)
        return info.get("name", "") if info else ""

    def _get_process_info_with_cache(self, pid: int) -> Optional[Dict[str, str]]:
        """获取进程信息（带缓存）"""
        now = time.time()
        
        # 定期清理缓存 (简单策略: 每访问 100 次或缓存超过 1000 条时清理)
        if len(self._process_cache) > 1000:
            self._cleanup_cache()
            
        # 检查缓存
        if pid in self._process_cache:
            timestamp, info = self._process_cache[pid]
            if now - timestamp < self._cache_ttl:
                return info
        
        # 缓存未命中，查询 psutil
        try:
            process = psutil.Process(pid)
            info = {
                "name": process.name(),
                "exe": process.exe()
            }
            self._process_cache[pid] = (now, info)
            return info
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None
            
    def _cleanup_cache(self):
        """清理过期缓存"""
        now = time.time()
        # 移除过期的条目
        expired = [pid for pid, (ts, _) in self._process_cache.items() 
                   if now - ts > self._cache_ttl * 2]  # 给点宽限
        for pid in expired:
            del self._process_cache[pid]



# ============================================================
# 以下类从 trackers.py 合并
# ============================================================

class ProcessTracker:
    """进程追踪器 - 获取文件操作的进程信息（带缓存）"""

    def __init__(self, config=None):
        config = config or {}
        self.enabled = config.get("advanced.enable_handle_tracking", False)
        self._cache: Dict[int, tuple] = {}
        self._cache_timeout = config.get("resource_management.handle_cache_timeout", 300)

    def get_process_by_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """通过文件路径获取操作进程（简化版）"""
        # 注意: 完整的文件句柄追踪需要复杂的平台 API 调用
        # 这里返回当前前台进程作为近似
        try:
            sensor = Sensor()
            window_data = sensor.get_active_window()
            if window_data:
                return self._get_process_info(window_data.process_id)
        except Exception:
            pass
        return None

    def _get_process_info(self, pid: int) -> Optional[Dict[str, Any]]:
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
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None


class WindowSpy:
    """窗口追踪器 - 获取窗口详细信息（字典格式）"""

    def __init__(self, stats_collector=None):
        self._cache: Dict = {}
        self._cache_ttl = 1.0
        self.stats = stats_collector

    def get_active_window_info(self) -> Optional[Dict[str, Any]]:
        """获取当前前台窗口的详细信息（字典格式）"""
        try:
            sensor = Sensor()
            window_data = sensor.get_active_window()
            
            if not window_data:
                if self.stats:
                    self.stats.record_cache_miss()
                return None

            # 构建标准返回格式
            info = {
                "window_handle": str(window_data.window_handle),
                "window_title": window_data.window_title,
                "window_class": window_data.window_class,
                "pid": window_data.process_id,
                "process_name": window_data.process_name,
                "process_path": window_data.process_path,
                "username": "Unknown",
                "cmdline": ""
            }
            
            # 补充进程详情
            if info["pid"]:
                try:
                    proc = psutil.Process(info["pid"])
                    info["username"] = proc.username()
                    info["cmdline"] = " ".join(proc.cmdline())
                except Exception:
                    pass

            return info

        except Exception as e:
            return {"error": str(e)}

