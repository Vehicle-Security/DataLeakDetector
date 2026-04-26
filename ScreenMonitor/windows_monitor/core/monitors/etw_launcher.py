# -*- coding: utf-8 -*-
"""
etw_launcher.py - EtwMonitor.exe 启动器
职责：管理 EtwMonitor.exe 的生命周期，并将其日志转换为 core 格式

功能：
1. 启动/停止 EtwMonitor.exe 子进程
2. 读取 EtwMonitor 输出的 JSON 日志
3. 转换格式并合并到 core 的 logs.json
"""

import os
import json
import glob
import ntpath
import signal
import subprocess
import time
from datetime import datetime
from typing import Optional, List, Dict, Any

from ..utils import app_logger
from ..logging.json_io import atomic_write_json, load_json_file, read_text_with_fallback
from ..logging.log_contract import (
    build_browser_file_access_event,
    normalize_app_name,
    normalize_event_entry,
    normalize_timestamp_text,
)


class EtwLauncher:
    """
    EtwMonitor.exe 启动器
    
    EtwMonitor.exe 会在其工作目录下创建 logs/session_YYYYMMDD_HHMMSS.json
    本类负责：
    1. 启动 EtwMonitor.exe
    2. 停止时优雅关闭进程
    3. 读取其输出的日志文件
    4. 转换格式并合并到 session 的 logs.json
    """
    
    def __init__(self):
        self._process: Optional[subprocess.Popen] = None
        self._start_time: Optional[datetime] = None
        self._etw_bin_dir: Optional[str] = None
        self._shell_launched: bool = False  # 标记是否通过 ShellExecute 启动
        self._output_dir: Optional[str] = None  # 输出目录
        
        # 定位 EtwMonitor.exe
        self._find_etw_monitor()
    
    def _find_etw_monitor(self):
        """定位 EtwMonitor.exe 的路径"""
        # 相对于 core/monitors/etw_launcher.py
        # 路径: core/C++ETW/bin/EtwMonitor.exe
        current_dir = os.path.dirname(os.path.abspath(__file__))
        core_dir = os.path.dirname(current_dir)  # core/
        
        etw_bin_dir = os.path.join(core_dir, "C++ETW", "bin")
        
        # 优先使用 V2 版本 (支持优雅退出)
        etw_exe_path = os.path.join(etw_bin_dir, "EtwMonitorV2.exe")
        if not os.path.exists(etw_exe_path):
             etw_exe_path = os.path.join(etw_bin_dir, "EtwMonitor.exe")
        
        if os.path.exists(etw_exe_path):
            self._etw_bin_dir = etw_bin_dir
            self._etw_exe_path = etw_exe_path
            app_logger.info(f"📍 找到 EtwMonitor.exe: {etw_exe_path}")
        else:
            self._etw_bin_dir = None
            self._etw_exe_path = None
            app_logger.warning(f"⚠️ 未找到 EtwMonitor.exe: {etw_exe_path}")
            
    @property
    def is_available(self) -> bool:
        """检查 EtwMonitor.exe 是否可用"""
        return self._etw_exe_path is not None and os.path.exists(self._etw_exe_path)
    
    @property
    def is_running(self) -> bool:
        """检查 EtwMonitor.exe 是否正在运行"""
        if self._process is not None:
            return self._process.poll() is None
        # 通过 ShellExecute 启动时，检查进程名
        if self._shell_launched:
            return self._is_etw_process_running()
        return False
    
    def _is_etw_process_running(self) -> bool:
        """检查 EtwMonitor.exe 进程是否在运行"""
        try:
            import subprocess
            # 检查 V2 或普通版本
            result = subprocess.run(
                ['tasklist', '/FI', 'IMAGENAME eq EtwMonitorV2.exe', '/NH'],
                capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW
            )
            if 'EtwMonitorV2.exe' in result.stdout:
                return True
                
            result = subprocess.run(
                ['tasklist', '/FI', 'IMAGENAME eq EtwMonitor.exe', '/NH'],
                capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW
            )
            return 'EtwMonitor.exe' in result.stdout
        except:
            return False
    
    def start(self, output_dir: Optional[str] = None) -> bool:
        """
        启动 EtwMonitor.exe (以管理员权限)
        
        Args:
            output_dir: 日志输出目录，如果不指定则使用默认的 logs 目录
        
        Returns:
            True 如果启动成功
        """
        if not self.is_available:
            app_logger.warning("EtwMonitor.exe 不可用，跳过启动")
            return False
        
        if self.is_running:
            app_logger.warning("EtwMonitor.exe 已在运行")
            return True
        
        try:
            # 记录启动时间（用于后续查找日志文件）
            self._start_time = datetime.now()
            self._output_dir = output_dir
            
            # 构建命令行参数
            args = [self._etw_exe_path]
            if output_dir:
                args.append(output_dir)
                app_logger.info(f"📂 ETW 日志将写入: {output_dir}")
            
            if os.name == 'nt':
                # Windows: 使用 ShellExecuteW 以管理员权限启动
                import ctypes
                
                # 首先检查是否已有管理员权限
                try:
                    is_admin = ctypes.windll.shell32.IsUserAnAdmin()
                except:
                    is_admin = False
                
                if is_admin:
                    # 已有管理员权限，直接启动
                    self._process = subprocess.Popen(
                        args,
                        cwd=self._etw_bin_dir,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                    )
                    app_logger.info(f"🚀 EtwMonitor.exe 已启动 (PID: {self._process.pid})")
                else:
                    # 没有管理员权限，使用 ShellExecuteW 请求提升
                    # 注意：这会弹出 UAC 提示
                    cmd_args = output_dir if output_dir else ""
                    result = ctypes.windll.shell32.ShellExecuteW(
                        None,           # hwnd
                        "runas",        # 操作：以管理员身份运行
                        self._etw_exe_path,  # 程序
                        cmd_args,       # 参数（输出目录）
                        self._etw_bin_dir,   # 工作目录
                        1               # SW_SHOWNORMAL
                    )
                    # ShellExecuteW 返回值 > 32 表示成功
                    if result > 32:
                        app_logger.info(f"🚀 EtwMonitor.exe 已请求管理员权限启动 (ShellExecute 返回: {result})")
                        # 为了让 is_running 工作，我们需要找到进程
                        # 但 ShellExecuteW 不直接返回进程句柄
                        # 设置一个标志表示已启动
                        self._process = None  # 无法直接获取进程对象
                        self._shell_launched = True
                    else:
                        app_logger.error(f"❌ 启动 EtwMonitor.exe 失败：ShellExecute 返回 {result}")
                        return False
                        
                return True
            else:
                # 非 Windows 系统
                self._process = subprocess.Popen(
                    args,
                    cwd=self._etw_bin_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                app_logger.info(f"🚀 EtwMonitor.exe 已启动 (PID: {self._process.pid})")
                return True
            
        except PermissionError:
            app_logger.error("❌ 启动 EtwMonitor.exe 失败：需要管理员权限")
            return False
        except Exception as e:
            app_logger.error(f"❌ 启动 EtwMonitor.exe 失败: {e}")
            return False

    def _signal_stop_event(self) -> bool:
        """发送停止信号 (Global\\EtwMonitorStopEvent)"""
        try:
            import ctypes
            EVENT_MODIFY_STATE = 0x0002
            # Open Named Event
            hEvent = ctypes.windll.kernel32.OpenEventW(EVENT_MODIFY_STATE, False, "Global\\EtwMonitorStopEvent")
            if hEvent:
                # Signal Event
                ctypes.windll.kernel32.SetEvent(hEvent)
                ctypes.windll.kernel32.CloseHandle(hEvent)
                app_logger.info("📡 已发送停止信号 (Global\\EtwMonitorStopEvent)")
                return True
        except Exception as e:
            app_logger.warning(f"⚠️ 发送停止信号失败: {e}")
        return False
    
    def stop(self) -> bool:
        """
        停止 EtwMonitor.exe
        优先尝试发送 Named Event 信号，等待优雅退出。
        超时则强制杀死。
        
        Returns:
            True 如果成功停止
        """
        if not self.is_running:
            return True
        
        # 1. 发送停止信号 (优雅退出)
        if self._signal_stop_event():
            # 等待进程退出 (给它一点时间刷新缓冲区)
            wait_time = 0
            while self.is_running and wait_time < 5:
                time.sleep(0.5)
                wait_time += 0.5
            
            if not self.is_running:
                app_logger.info("✅ EtwMonitor.exe 已响应信号并优雅退出")
                self._process = None
                self._shell_launched = False
                return True
        
        # 2. 如果还在运行，尝试 terminate/kill
        try:
            if self._process is not None:
                # 有进程句柄的情况
                if os.name == 'nt':
                    self._process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    self._process.terminate()
                
                # 等待进程退出（最多 5 秒）
                try:
                    self._process.wait(timeout=5)
                    app_logger.info("✅ EtwMonitor.exe 已正常停止")
                except subprocess.TimeoutExpired:
                    # 强制杀死
                    self._process.kill()
                    self._process.wait()
                    app_logger.warning("⚠️ EtwMonitor.exe 强制终止 (可能丢失部分日志)")
            elif self._shell_launched:
                # ShellExecute 启动的情况，使用 taskkill
                import subprocess as sp
                result = sp.run(
                    ['taskkill', '/F', '/IM', 'EtwMonitorV2.exe'], # 尝试 V2
                    capture_output=True, text=True,
                    creationflags=sp.CREATE_NO_WINDOW
                )
                if result.returncode != 0:
                     # 尝试旧名称
                     sp.run(
                        ['taskkill', '/F', '/IM', 'EtwMonitor.exe'],
                        capture_output=True, text=True,
                        creationflags=sp.CREATE_NO_WINDOW
                    )
                
                app_logger.info("✅ EtwMonitor.exe 已通过 taskkill 停止")
            
            return True
            
        except Exception as e:
            app_logger.error(f"❌ 停止 EtwMonitor.exe 失败: {e}")
            return False
        finally:
            self._process = None
            self._shell_launched = False
    
    def get_log_files(self) -> List[str]:
        """
        获取 EtwMonitor 输出的日志文件
        
        Returns:
            日志文件路径列表
        """
        # 1. 如果指定了输出目录，优先在输出目录查找
        if self._output_dir and os.path.exists(self._output_dir):
            logs_dir = self._output_dir
            # 查找所有 etw_session_*.json 文件
            pattern = os.path.join(logs_dir, "etw_session_*.json")
            files = glob.glob(pattern)
            
            # 如果没找到，尝试旧模式
            if not files:
                 pattern = os.path.join(logs_dir, "session_*.json")
                 files = glob.glob(pattern)
                 
            if files:
                # 过滤掉非本次会话的文件 (如果有启动时间)
                if self._start_time:
                    filtered = []
                    for f in files:
                        try:
                            # 只要修改时间在启动时间之后（允许 2 秒误差）
                            mtime = datetime.fromtimestamp(os.path.getmtime(f))
                            if mtime >= self._start_time.replace(microsecond=0):
                                filtered.append(f)
                            # 或者，如果是 session 目录下的文件，可能就是我们创建的
                            # 但最好还是检查时间，或者我们能更精确地匹配文件名
                            # 目前 etw_session_<timestamp>.json 可能会有秒级误差
                        except Exception:
                            pass
                    return filtered if filtered else files
                return files

        # 2. 如果没找到，或者是默认启动，尝试在 bin/logs 目录查找
        if not self._etw_bin_dir:
            return []
        
        logs_dir = os.path.join(self._etw_bin_dir, "logs")
        if not os.path.exists(logs_dir):
            return []
        
        # 查找所有 etw_session_*.json 文件
        pattern = os.path.join(logs_dir, "etw_session_*.json")
        files = glob.glob(pattern)
        
        # 兼容旧的文件名模式
        if not files:
            pattern = os.path.join(logs_dir, "session_*.json")
            files = glob.glob(pattern)
        
        # 如果有启动时间，只返回启动后创建的文件
        if self._start_time and files:
            filtered = []
            for f in files:
                try:
                    mtime = datetime.fromtimestamp(os.path.getmtime(f))
                    # 只要修改时间在启动时间之后（允许 1 秒误差）
                    if mtime >= self._start_time.replace(microsecond=0):
                        filtered.append(f)
                except Exception:
                    pass
            return filtered
        
        return files
    
    def convert_and_merge(self, target_logs_path: str) -> int:
        """
        转换 EtwMonitor 日志格式并合并到目标 logs.json
        
        Args:
            target_logs_path: 目标 logs.json 的完整路径
            
        Returns:
            合并的事件数量
        """
        import shutil
        
        log_files = self.get_log_files()
        if not log_files:
            app_logger.info("📭 没有找到 EtwMonitor 日志文件")
            return 0
        
        merged_count = 0
        converted_events = []
        
        # 获取 session 目录的 logs 目录
        target_logs_dir = os.path.dirname(target_logs_path)
        
        # 读取并转换所有 EtwMonitor 日志
        for log_file in log_files:
            try:
                # 只有当文件不在目标目录时才复制
                log_file_dir = os.path.dirname(log_file)
                if os.path.normpath(log_file_dir) != os.path.normpath(target_logs_dir):
                    # 复制原始 ETW 日志到 session 目录 (保持原文件名)
                    etw_log_name = os.path.basename(log_file)
                    etw_log_dest = os.path.join(target_logs_dir, etw_log_name)
                    shutil.copy2(log_file, etw_log_dest)
                    app_logger.info(f"📋 已复制 ETW 日志到: {etw_log_dest}")
                
                # 读取并转换事件
                events = self._read_etw_log(log_file)
                for event in events:
                    converted = self._convert_event(event)
                    if converted:
                        converted_events.append(converted)
                        merged_count += 1
            except Exception as e:
                app_logger.warning(f"⚠️ 读取 EtwMonitor 日志失败 ({log_file}): {e}")
        
        if not converted_events:
            return 0
        
        # 读取现有的 logs.json
        existing_events = []
        if os.path.exists(target_logs_path):
            try:
                existing_events = load_json_file(target_logs_path, default=[])
                if not isinstance(existing_events, list):
                    existing_events = []
            except Exception as e:
                app_logger.warning(f"⚠️ 读取现有日志失败: {e}")
                existing_events = []
        
        # 合并并按时间排序
        all_events = existing_events + converted_events
        all_events.sort(key=lambda x: x.get("timestamp", ""))
        
        # 写回文件
        try:
            atomic_write_json(target_logs_path, all_events)
            app_logger.info(f"✅ 已合并 {merged_count} 条 EtwMonitor 事件到 logs.json")
        except Exception as e:
            app_logger.error(f"❌ 保存合并日志失败: {e}")
        
        return merged_count
    
    def _read_etw_log(self, log_file: str) -> List[Dict[str, Any]]:
        """读取 EtwMonitor 日志文件（支持 JSON Array 和 JSON Lines 格式）"""
        events = []
        content = read_text_with_fallback(log_file).strip()
        if not content:
            return events

        # 尝试 JSON Array 格式（EtwMonitor.exe 默认输出格式）
        if content.startswith('['):
            try:
                parsed = json.loads(content)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass

        # 回退到 JSON Lines 格式
        for line in content.splitlines():
            line = line.strip()
            if line:
                try:
                    event = json.loads(line)
                    events.append(event)
                except json.JSONDecodeError:
                    continue
        return events
    
    def _convert_event(self, etw_event: Dict[str, Any]) -> Dict[str, Any]:
        """
        将 EtwMonitor 事件转换为 core 格式
        
        EtwMonitor 格式:
        {
            "timestamp": "2026-01-25 21:15:00",
            "process": "chrome.exe",
            "pid": 1234,
            "path": "C:\\Users\\...\\file.docx"
        }
        
        Core 格式:
        {
            "timestamp": "2026-01-25T21:15:00.000",
            "event_type": "created",
            "file_path": "...",
            "file_name": "...",
            "process_info": {...},
            ...
        }
        """
        operation = str(etw_event.get("operation", "") or "").strip().lower()
        if operation in {"closed", "close", "cleanup"}:
            path = str(etw_event.get("path", "") or "").strip()
            process_name = str(etw_event.get("process", "") or "").strip()
            basename = ntpath.basename(path.rstrip("\\/"))
            _, ext = ntpath.splitext(basename)
            event = {
                "timestamp": normalize_timestamp_text(etw_event.get("timestamp", "")),
                "event_type": "closed",
                "file_path": path,
                "file_name": basename,
                "file_size": 0,
                "file_extension": ext,
                "process_info": {
                    "pid": str(etw_event.get("pid", 0) or ""),
                    "process_name": process_name,
                    "process_path": "",
                    "cmdline": "",
                },
                "window_info": {
                    "window_handle": "",
                    "window_title": "",
                    "window_class": "",
                },
                "user_info": {
                    "username": os.environ.get("USERNAME", "Unknown"),
                    "hostname": os.environ.get("COMPUTERNAME", "Unknown"),
                },
                "disk_info": {
                    "drive_letter": path[:2] if len(path) >= 2 and path[1] == ":" else "",
                    "disk_type": "Fixed",
                },
                "app_name": normalize_app_name(process_name),
                "extra": {
                    "raw_operation": "closed",
                    "category": "browser_file_access",
                    "source": "etw_monitor",
                    "end_time_source": "cpp_etw_close",
                },
            }
            return normalize_event_entry(event, drop_invalid_file_event=True)

        converted = build_browser_file_access_event(
            raw_timestamp=etw_event.get("timestamp", ""),
            process_name=etw_event.get("process", ""),
            pid=etw_event.get("pid", 0),
            file_path=etw_event.get("path", ""),
            username=os.environ.get("USERNAME", "Unknown"),
            hostname=os.environ.get("COMPUTERNAME", "Unknown"),
        )
        if converted and operation in {"opened", "open", "read"}:
            converted["event_type"] = "opened"
            converted.setdefault("extra", {})["raw_operation"] = f"browser_file_access_{operation}"
        return converted
    
    def _normalize_app_name(self, process_name: str) -> str:
        """规范化应用名称"""
        return normalize_app_name(process_name)


# 单例实例
_etw_launcher: Optional[EtwLauncher] = None


def get_etw_launcher() -> EtwLauncher:
    """获取 EtwLauncher 单例"""
    global _etw_launcher
    if _etw_launcher is None:
        _etw_launcher = EtwLauncher()
    return _etw_launcher
